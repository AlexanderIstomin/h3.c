/* Run LTX-2.5's embeddings connector on the released weights.
 *
 * The last stage of the conditioning path, and the only one that lives in the
 * *DiT* file rather than the text encoder: 8 bidirectional blocks per stream,
 * 2.0 GB, taking the aggregated features from the Gemma tower and producing
 * what the DiT's cross-attention actually reads.
 *
 * It runs here as its own binary rather than as another phase of the text
 * runner because of where its weights are. The DiT is 21.5 GB and the text
 * encoder 15.4 GB on a 32 GB machine, so the two files cannot both be
 * resident, and the entry point is two-phase by construction. This is the
 * second phase: the text runner writes its aggregated features out, this reads
 * them back with only the DiT file open.
 *
 * The structure was verified at toy dimensions against LTX's own
 * Embeddings1DConnector in test_ltx_conditioning.c. What is new here is the
 * released width and depth, and three things that only exist at full scale:
 *
 *   - the span must be a multiple of 128, the register count, because the
 *     reference asserts it. 77 real tokens become a span of 128 with the rest
 *     taken by learnable registers;
 *   - the registers are model weights, not scaffolding. Attention is
 *     bidirectional over the whole span with no mask at all -- the reference
 *     zeroes the mask the moment it substitutes them -- so valid tokens attend
 *     to registers and their content shapes the result; and
 *   - the frequencies are float64 by config (`frequencies_precision`), spread
 *     geometrically from 1 to theta over the whole inner width and then split
 *     across heads, so no two of the 32 heads rotate alike.
 *
 * Two conventions differ from the Gemma tower deliberately, and both are
 * silent if confused: every norm between the residuals is parameter-free, and
 * the query/key norms span the full 4096 rather than one 128-wide head.
 *
 * The check is check_ltx_connector.py, which runs the vendored
 * Embeddings1DConnector on the same input with the same released weights.
 *
 * usage: h3_real_ltx_connector_test DIT.safetensors STATES.bin [OUT.bin] */

#include "h3_gpu.h"
#include "h3_safetensors.h"
#include "h3_weights.h"

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum {
    VIDEO_DIM = 4096,
    VIDEO_HEAD = 128,
    AUDIO_DIM = 2048,
    AUDIO_HEAD = 64,
    HEADS = 32,
    BLOCKS = 8,
    REGISTERS = 128,
    MAX_POS = 4096,
    CONVROT_GROUP = 256,
    FF_MULTIPLE = 4,
    STATES_MAGIC = 0x4C545854
};

#define CONNECTOR_EPSILON 1e-6f

static const double THETA = 10000.0;

static h3_weight_store *store = NULL;
static h3_gpu *gpu = NULL;

static void fail(const char *format, ...)
    __attribute__((format(printf, 1, 2), noreturn));

static void fail(const char *format, ...) {
    va_list arguments;
    va_start(arguments, format);
    fprintf(stderr, "FAIL: ");
    vfprintf(stderr, format, arguments);
    fprintf(stderr, "\n");
    va_end(arguments);
    exit(1);
}

static void require(int condition, const char *message) {
    if (!condition) fail("%s", message);
}

#define GPU_OP(call, what) \
    do { if (!(call)) fail("%s: %s", (what), h3_gpu_error(gpu)); } while (0)

static double now(void) {
    struct timespec moment;
    clock_gettime(CLOCK_MONOTONIC, &moment);
    return (double)moment.tv_sec + (double)moment.tv_nsec * 1e-9;
}

static float from_bf16(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static uint16_t to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += UINT32_C(0x7fff) + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

/* ------------------------------------------------------------------ shapes */

typedef struct {
    const char *name;
    const char *prefix;
    uint32_t dim;
    uint32_t head_dim;
    uint32_t ff_dim;
} stream;

static const stream STREAMS[2] = {
    {"video", "model.diffusion_model.video_embeddings_connector.",
     VIDEO_DIM, VIDEO_HEAD, VIDEO_DIM * FF_MULTIPLE},
    {"audio", "model.diffusion_model.audio_embeddings_connector.",
     AUDIO_DIM, AUDIO_HEAD, AUDIO_DIM * FF_MULTIPLE}
};

/* ----------------------------------------------------------------- loading */

static h3_gpu_tensor *load_bf16(const char *name, int ndim,
                                uint64_t first, uint64_t second) {
    uint64_t shape[2] = {first, second};
    char error[512];
    h3_gpu_tensor *tensor = h3_weight_load_bf16(store, gpu, name, ndim, shape,
                                                error, sizeof(error));
    if (!tensor) fail("cannot load %s: %s", name, error);
    return tensor;
}

typedef struct {
    h3_gpu_tensor *weight;
    h3_gpu_tensor *scales;
    h3_gpu_tensor *bias;
} projection;

/* Every projection in the connector carries a bias, unlike the DiT's video
 * feed-forward, whose ff_bias is false. */
static void load_projection(const char *prefix, const char *suffix,
                            uint64_t out_dim, uint64_t in_dim,
                            projection *into) {
    char name[224], bias[224];
    snprintf(name, sizeof(name), "%s%s.weight", prefix, suffix);
    snprintf(bias, sizeof(bias), "%s%s.bias", prefix, suffix);
    char error[512];
    if (!h3_weight_load_i8_linear(store, gpu, name, out_dim, in_dim,
                                  &into->weight, &into->scales,
                                  error, sizeof(error)))
        fail("cannot load %s: %s", name, error);
    uint32_t group = 0;
    if (!h3_weight_i8_linear_convrot_group(store, name, &group,
                                           error, sizeof(error)))
        fail("cannot read the ConvRot marker for %s: %s", name, error);
    if (group != CONVROT_GROUP)
        fail("%s has ConvRot group %u, expected %d", name, group, CONVROT_GROUP);
    into->bias = load_bf16(bias, 1, out_dim, 0);
}

static void free_projection(projection *which) {
    h3_gpu_tensor_free(which->weight);
    h3_gpu_tensor_free(which->scales);
    h3_gpu_tensor_free(which->bias);
    memset(which, 0, sizeof(*which));
}

typedef struct {
    projection query, key, value, output, ff_in, ff_out;
    h3_gpu_tensor *query_norm;
    h3_gpu_tensor *key_norm;
    h3_gpu_tensor *gate_weight;
    h3_gpu_tensor *gate_bias;
} block_weights;

static void load_block(const stream *which, int index, block_weights *weights) {
    char prefix[224];
    snprintf(prefix, sizeof(prefix), "%stransformer_1d_blocks.%d.",
             which->prefix, index);
    memset(weights, 0, sizeof(*weights));
    load_projection(prefix, "attn1.to_q", which->dim, which->dim, &weights->query);
    load_projection(prefix, "attn1.to_k", which->dim, which->dim, &weights->key);
    load_projection(prefix, "attn1.to_v", which->dim, which->dim, &weights->value);
    load_projection(prefix, "attn1.to_out.0", which->dim, which->dim,
                    &weights->output);
    load_projection(prefix, "ff.net.0.proj", which->ff_dim, which->dim,
                    &weights->ff_in);
    load_projection(prefix, "ff.net.2", which->dim, which->ff_dim,
                    &weights->ff_out);
    char name[224];
    snprintf(name, sizeof(name), "%sattn1.q_norm.weight", prefix);
    weights->query_norm = load_bf16(name, 1, which->dim, 0);
    snprintf(name, sizeof(name), "%sattn1.k_norm.weight", prefix);
    weights->key_norm = load_bf16(name, 1, which->dim, 0);
    /* The gate is a small dense projection, not quantized: one logit per head
     * from the block's *normed input*, taken before any rotation. */
    snprintf(name, sizeof(name), "%sattn1.to_gate_logits.weight", prefix);
    weights->gate_weight = load_bf16(name, 2, HEADS, which->dim);
    snprintf(name, sizeof(name), "%sattn1.to_gate_logits.bias", prefix);
    weights->gate_bias = load_bf16(name, 1, HEADS, 0);
}

static void free_block(block_weights *weights) {
    free_projection(&weights->query);
    free_projection(&weights->key);
    free_projection(&weights->value);
    free_projection(&weights->output);
    free_projection(&weights->ff_in);
    free_projection(&weights->ff_out);
    h3_gpu_tensor_free(weights->query_norm);
    h3_gpu_tensor_free(weights->key_norm);
    h3_gpu_tensor_free(weights->gate_weight);
    h3_gpu_tensor_free(weights->gate_bias);
    memset(weights, 0, sizeof(*weights));
}

/* -------------------------------------------------------------------- rope */

/* LTX's split rope over one axis, which is not the tower's construction at
 * any point. The inner width's frequencies are spread geometrically from 1 to
 * theta -- theta^(i / (count - 1)) -- scaled by pi/2, and positions are mapped
 * onto [-1, 1] through max_pos rather than used directly. The result is then
 * split across heads, so head h owns frequencies h * half through h * half +
 * half - 1 and no two heads rotate alike.
 *
 * Laid out [head][token][half] to match the kernel's head stride. Computed in
 * double because the released config asks for float64 frequencies. */
static void rope_tables(h3_gpu_tensor **cosine, h3_gpu_tensor **sine,
                        uint32_t span, uint32_t dim, uint32_t head_dim) {
    const uint32_t half = head_dim / 2;
    const uint32_t count = dim / 2;
    require(count == HEADS * half, "the frequency count does not fill the heads");
    float *cos_table = malloc((size_t)count * span * sizeof(*cos_table));
    float *sin_table = malloc((size_t)count * span * sizeof(*sin_table));
    require(cos_table && sin_table, "cannot allocate the rotary tables");
    for (uint32_t index = 0; index < count; index++) {
        const double exponent = count > 1 ?
            (double)index / (double)(count - 1) : 0.0;
        const double frequency = pow(THETA, exponent) * M_PI / 2.0;
        const uint32_t head = index / half;
        const uint32_t slot = index % half;
        for (uint32_t token = 0; token < span; token++) {
            const double fractional = (double)token / (double)MAX_POS;
            const double angle = frequency * (fractional * 2.0 - 1.0);
            const size_t at = (size_t)head * span * half + (size_t)token * half + slot;
            cos_table[at] = (float)cos(angle);
            sin_table[at] = (float)sin(angle);
        }
    }
    *cosine = h3_gpu_tensor_from_f32(gpu, cos_table, (size_t)count * span);
    *sine = h3_gpu_tensor_from_f32(gpu, sin_table, (size_t)count * span);
    require(*cosine && *sine, "cannot upload the rotary tables");
    free(cos_table);
    free(sin_table);
}

/* ------------------------------------------------------------------- block */

typedef struct {
    h3_gpu_tensor *normed;
    h3_gpu_tensor *query;
    h3_gpu_tensor *key;
    h3_gpu_tensor *value;
    h3_gpu_tensor *heads;
    h3_gpu_tensor *logits;
    h3_gpu_tensor *inner;
    h3_gpu_tensor *ones;
    h3_gpu_tensor *cosine;
    h3_gpu_tensor *sine;
} scratch;

static int linear_i8(h3_gpu_tensor *output, const h3_gpu_tensor *input,
                     const projection *weight, uint32_t rows,
                     uint32_t in_dim, uint32_t out_dim) {
    return h3_gpu_linear_i8_weight_bf16(gpu, output, input, weight->weight,
                                        weight->scales, weight->bias, rows,
                                        in_dim, out_dim);
}

static void run_block(const stream *which, const block_weights *weight,
                      scratch *space, h3_gpu_tensor *hidden, uint32_t span) {
    const uint32_t dim = which->dim;
    const uint32_t head_dim = which->head_dim;
    const uint32_t stride = span * (head_dim / 2);

    /* Parameter-free: the connector's norms carry no learned scale at all,
     * where every norm in the Gemma tower does. */
    GPU_OP(h3_gpu_rms_norm_bf16(gpu, space->normed, hidden, space->ones,
                                span, dim, CONNECTOR_EPSILON), "attention norm");
    /* The gate reads the normed input, before the rotation the quantized
     * projections need -- it is dense, so it never sees a rotated activation. */
    GPU_OP(h3_gpu_linear_bf16(gpu, space->logits, space->normed,
                              weight->gate_weight, weight->gate_bias,
                              span, dim, HEADS), "gate logits");
    GPU_OP(h3_gpu_convrot_bf16(gpu, space->normed, space->normed, span, dim,
                               CONVROT_GROUP), "Q/K/V ConvRot");
    GPU_OP(linear_i8(space->query, space->normed, &weight->query, span, dim, dim),
           "query projection");
    GPU_OP(linear_i8(space->key, space->normed, &weight->key, span, dim, dim),
           "key projection");
    GPU_OP(linear_i8(space->value, space->normed, &weight->value, span, dim, dim),
           "value projection");
    /* Over the full inner width, not per head. The tower's are per head, and
     * the two disagree by a factor that varies head to head. */
    GPU_OP(h3_gpu_rms_norm_bf16(gpu, space->query, space->query,
                                weight->query_norm, span, dim,
                                CONNECTOR_EPSILON), "query norm");
    GPU_OP(h3_gpu_rms_norm_bf16(gpu, space->key, space->key, weight->key_norm,
                                span, dim, CONNECTOR_EPSILON), "key norm");
    GPU_OP(h3_gpu_rope_rows_bf16(gpu, space->query, space->cosine, space->sine,
                                 span, HEADS, head_dim, stride), "query rope");
    GPU_OP(h3_gpu_rope_rows_bf16(gpu, space->key, space->cosine, space->sine,
                                 span, HEADS, head_dim, stride), "key rope");
    /* Bidirectional over the whole span: the registers have already taken the
     * padded slots and the reference zeroes its mask when it does that, so
     * there is nothing left to mask. The value is neither normed nor rotated. */
    GPU_OP(h3_gpu_sdpa_bf16(gpu, space->heads, space->query, space->key,
                            space->value, span, HEADS, head_dim,
                            1.0f / sqrtf((float)head_dim)), "attention");
    GPU_OP(h3_gpu_head_gate_bf16(gpu, space->heads, space->logits, span,
                                 HEADS, head_dim), "head gate");
    GPU_OP(h3_gpu_convrot_bf16(gpu, space->heads, space->heads, span, dim,
                               CONVROT_GROUP), "output ConvRot");
    GPU_OP(linear_i8(space->normed, space->heads, &weight->output, span,
                     dim, dim), "output projection");
    GPU_OP(h3_gpu_add_bf16(gpu, hidden, hidden, space->normed, span * dim),
           "attention residual");

    GPU_OP(h3_gpu_rms_norm_bf16(gpu, space->normed, hidden, space->ones,
                                span, dim, CONNECTOR_EPSILON),
           "feed-forward norm");
    GPU_OP(h3_gpu_convrot_bf16(gpu, space->normed, space->normed, span, dim,
                               CONVROT_GROUP), "feed-forward ConvRot");
    GPU_OP(linear_i8(space->inner, space->normed, &weight->ff_in, span, dim,
                     which->ff_dim), "feed-forward in");
    /* Plain GELU at 4x, not a gated pair: net.0.proj widens, net.2 consumes
     * all of it. There is no w1/w3 here to fuse or to confuse. */
    GPU_OP(h3_gpu_gelu_bf16(gpu, space->inner, space->inner,
                            span * which->ff_dim, 1), "GELU");
    GPU_OP(h3_gpu_convrot_bf16(gpu, space->inner, space->inner, span,
                               which->ff_dim, CONVROT_GROUP),
           "feed-forward out ConvRot");
    GPU_OP(linear_i8(space->normed, space->inner, &weight->ff_out, span,
                     which->ff_dim, dim), "feed-forward out");
    GPU_OP(h3_gpu_add_bf16(gpu, hidden, hidden, space->normed, span * dim),
           "feed-forward residual");
}

/* ------------------------------------------------------------------ stream */

static void read_bf16_as_f32(const h3_gpu_tensor *tensor, float *out,
                             size_t count) {
    uint16_t *staged = malloc(count * sizeof(*staged));
    require(staged != NULL, "cannot allocate a readback buffer");
    require(h3_gpu_tensor_read_bf16(tensor, staged, count),
            "cannot read a GPU tensor");
    for (size_t index = 0; index < count; index++)
        out[index] = from_bf16(staged[index]);
    free(staged);
}

/* Padded slots take the registers, tiled to cover the span -- the reference
 * repeats the table span/128 times, so slot s takes register s % 128. Valid
 * slots keep their aggregated features. */
static void run_stream(const stream *which, const float *features,
                       uint32_t tokens, uint32_t span, float *out) {
    printf("  %s: %u valid of %u, %u blocks\n", which->name, tokens, span, BLOCKS);
    fflush(stdout);
    const double began = now();
    const uint32_t dim = which->dim;
    const size_t state = (size_t)span * dim;

    char name[224];
    snprintf(name, sizeof(name), "%slearnable_registers", which->prefix);
    h3_gpu_tensor *registers = load_bf16(name, 2, REGISTERS, dim);
    float *host_registers = malloc((size_t)REGISTERS * dim * sizeof(*host_registers));
    require(host_registers != NULL, "cannot allocate the registers");
    read_bf16_as_f32(registers, host_registers, (size_t)REGISTERS * dim);
    h3_gpu_tensor_free(registers);

    uint16_t *staged = malloc(state * sizeof(*staged));
    require(staged != NULL, "cannot allocate the connector input");
    for (uint32_t token = 0; token < span; token++) {
        const float *source = token < tokens ?
            features + (size_t)token * dim :
            host_registers + (size_t)(token % REGISTERS) * dim;
        for (uint32_t index = 0; index < dim; index++)
            staged[(size_t)token * dim + index] = to_bf16(source[index]);
    }
    free(host_registers);
    h3_gpu_tensor *hidden = h3_gpu_tensor_from_bf16(gpu, staged, state);
    require(hidden != NULL, "cannot upload the connector input");
    free(staged);

    scratch space;
    memset(&space, 0, sizeof(space));
    space.normed = h3_gpu_tensor_new_bf16(gpu, state);
    space.query = h3_gpu_tensor_new_bf16(gpu, state);
    space.key = h3_gpu_tensor_new_bf16(gpu, state);
    space.value = h3_gpu_tensor_new_bf16(gpu, state);
    space.heads = h3_gpu_tensor_new_bf16(gpu, state);
    space.logits = h3_gpu_tensor_new_bf16(gpu, (size_t)span * HEADS);
    space.inner = h3_gpu_tensor_new_bf16(gpu, (size_t)span * which->ff_dim);
    require(space.normed && space.query && space.key && space.value &&
            space.heads && space.logits && space.inner,
            "cannot allocate connector scratch");
    uint16_t *ones = malloc(dim * sizeof(*ones));
    require(ones != NULL, "cannot allocate the norm weight");
    for (uint32_t index = 0; index < dim; index++) ones[index] = 0x3f80;
    space.ones = h3_gpu_tensor_from_bf16(gpu, ones, dim);
    require(space.ones != NULL, "cannot upload the norm weight");
    free(ones);
    rope_tables(&space.cosine, &space.sine, span, dim, which->head_dim);

    for (int index = 0; index < BLOCKS; index++) {
        block_weights weights;
        load_block(which, index, &weights);
        GPU_OP(h3_gpu_begin(gpu), "begin block");
        run_block(which, &weights, &space, hidden, span);
        GPU_OP(h3_gpu_submit(gpu), "submit block");
        free_block(&weights);
    }
    /* connector_norm_output: one more parameter-free norm on the way out. */
    GPU_OP(h3_gpu_begin(gpu), "begin output norm");
    GPU_OP(h3_gpu_rms_norm_bf16(gpu, space.normed, hidden, space.ones,
                                span, dim, CONNECTOR_EPSILON), "output norm");
    GPU_OP(h3_gpu_submit(gpu), "submit output norm");
    read_bf16_as_f32(space.normed, out, state);

    double peak = 0.0;
    for (size_t index = 0; index < state; index++) {
        require(isfinite(out[index]), "a connector output is not finite");
        if (fabs((double)out[index]) > peak) peak = fabs((double)out[index]);
    }
    printf("  ok  %-6s [%u, %u] peak %.4f, %.2f s\n", which->name, span, dim,
           peak, now() - began);

    h3_gpu_tensor_free(hidden);
    h3_gpu_tensor_free(space.normed); h3_gpu_tensor_free(space.query);
    h3_gpu_tensor_free(space.key); h3_gpu_tensor_free(space.value);
    h3_gpu_tensor_free(space.heads); h3_gpu_tensor_free(space.logits);
    h3_gpu_tensor_free(space.inner); h3_gpu_tensor_free(space.ones);
    h3_gpu_tensor_free(space.cosine); h3_gpu_tensor_free(space.sine);
}

/* ------------------------------------------------------------------- main */

int main(int argc, char **argv) {
    if (argc < 3 || argc > 4) {
        fprintf(stderr, "usage: %s DIT.safetensors STATES.bin [OUT.bin]\n",
                argv[0]);
        return 2;
    }
    char error[512];
    store = h3_weight_store_open(argv[1], error, sizeof(error));
    if (!store) fail("cannot open the DiT checkpoint: %s", error);

    /* The text runner's dump: its header, its token ids, its forty-nine hidden
     * states, then the two aggregated features this consumes. The states are
     * skipped by seeking rather than read. */
    FILE *file = fopen(argv[2], "rb");
    if (!file) fail("cannot open %s", argv[2]);
    uint32_t header[6];
    require(fread(header, sizeof(header), 1, file) == 1, "cannot read the dump");
    if (header[0] != STATES_MAGIC) fail("%s is not a state dump", argv[2]);
    const uint32_t tokens = header[1], states = header[2], hidden = header[3];
    if (header[4] != VIDEO_DIM || header[5] != AUDIO_DIM) {
        fail("the dump carries %u/%u features, expected %d/%d",
             header[4], header[5], VIDEO_DIM, AUDIO_DIM);
    }
    require(fseek(file, (long)(4 * tokens) +
                        (long)(4 * (size_t)states * tokens * hidden), SEEK_CUR) == 0,
            "cannot seek past the hidden states");
    float *video_features = malloc((size_t)tokens * VIDEO_DIM * sizeof(float));
    float *audio_features = malloc((size_t)tokens * AUDIO_DIM * sizeof(float));
    require(video_features && audio_features, "cannot allocate the features");
    require(fread(video_features, sizeof(float), (size_t)tokens * VIDEO_DIM,
                  file) == (size_t)tokens * VIDEO_DIM,
            "cannot read the video features");
    require(fread(audio_features, sizeof(float), (size_t)tokens * AUDIO_DIM,
                  file) == (size_t)tokens * AUDIO_DIM,
            "cannot read the audio features");
    fclose(file);

    /* The reference asserts span % 128 == 0 outright, so this is the model's
     * constraint rather than a convenience. Rounding up is what a pipeline
     * padding its prompt would do. */
    const uint32_t span = ((tokens + REGISTERS - 1) / REGISTERS) * REGISTERS;
    require(span % REGISTERS == 0, "the span is not a multiple of the registers");

    gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) fail("cannot create the Metal context: %s", error);

    printf("LTX-2.5 embeddings connector on the released DiT, %u tokens "
           "padded to %u:\n", tokens, span);
    float *video = malloc((size_t)span * VIDEO_DIM * sizeof(*video));
    float *audio = malloc((size_t)span * AUDIO_DIM * sizeof(*audio));
    require(video && audio, "cannot allocate the connector outputs");
    run_stream(&STREAMS[0], video_features, tokens, span, video);
    run_stream(&STREAMS[1], audio_features, tokens, span, audio);

    if (argc > 3) {
        FILE *out = fopen(argv[3], "wb");
        if (!out) fail("cannot open %s for writing", argv[3]);
        const uint32_t written[4] = {STATES_MAGIC, span, VIDEO_DIM, AUDIO_DIM};
        require(fwrite(written, sizeof(written), 1, out) == 1,
                "cannot write the output header");
        require(fwrite(video, sizeof(float), (size_t)span * VIDEO_DIM, out) ==
                    (size_t)span * VIDEO_DIM, "cannot write the video output");
        require(fwrite(audio, sizeof(float), (size_t)span * AUDIO_DIM, out) ==
                    (size_t)span * AUDIO_DIM, "cannot write the audio output");
        require(fclose(out) == 0, "cannot close the output");
        printf("  wrote %s for the connector check\n", argv[3]);
    }

    free(video); free(audio);
    free(video_features); free(audio_features);
    h3_gpu_free(gpu);
    h3_weight_store_free(store);
    printf("\nok: both connector streams ran on the released weights, %d "
           "blocks each over a %u-token span with %d learnable registers\n",
           BLOCKS, span, REGISTERS);
    return 0;
}
