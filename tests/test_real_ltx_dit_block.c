/* Run a released LTX-2.5 DiT block at full width against LTX's own.
 *
 * test_ltx_dit_block.c already checks this block's shape against
 * BasicAVTransformerBlock at toy dimensions. What it cannot reach is the
 * released widths, the released weights, and the modulation wiring as the
 * checkpoint actually stores it. Those are where this port's remaining risk
 * lives, and all three are silent when wrong.
 *
 * Six attentions and two feed-forwards. In order:
 *
 *   video self-attention, then video cross-attention to the text context;
 *   the same two for audio; audio into video and video into audio; then the
 *   two feed-forwards.
 *
 * The modulation is the part worth reading carefully. A block owns six tables
 * and the run supplies six timestep tensors, and they do not pair up the way
 * the shapes suggest:
 *
 *   - the nine-row table drives self-attention from slots 0-2, the feed-forward
 *     from 3-5 and the text cross-attention from 6-8, each as shift, scale,
 *     gate in that order;
 *   - the two-row prompt table modulates the *context* rather than the
 *     queries, and it is an affine with no norm in front of it -- unlike every
 *     other modulation here, which normalizes first;
 *   - the cross-modal table has five rows against a four-slot timestep and a
 *     one-slot gate timestep. Rows 0-1 are audio-into-video, rows 2-3 are
 *     video-into-audio, row 4 is the gate for both, and the order inside a
 *     pair is scale then shift where everywhere else it is shift then scale.
 *
 * And one ordering trap that no shape can catch: video-into-audio reads the
 * video stream *as it was before* audio-into-video modified it. The reference
 * snapshots both streams before either direction runs, and running them in
 * sequence off the live tensors would let the first direction bias the second.
 *
 * Everything the block consumes but does not own arrives from the fixture --
 * timesteps, rope tables, contexts. The timestep embedder, the 3D rope and the
 * patchify/head have their own checks, and folding them in here would mean a
 * failure could have come from anywhere.
 *
 * usage: h3_real_ltx_dit_block_test DIT.safetensors ANCHOR.safetensors */

#include "h3_gpu.h"
#include "h3_safetensors.h"
#include "h3_weights.h"

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    VIDEO_DIM = 4096,
    AUDIO_DIM = 2048,
    HEADS = 32,
    VIDEO_HEAD = VIDEO_DIM / HEADS,
    AUDIO_HEAD = AUDIO_DIM / HEADS,
    VIDEO_FF = VIDEO_DIM * 4,
    AUDIO_FF = AUDIO_DIM * 4,
    ADA_SLOTS = 9,
    PROMPT_SLOTS = 2,
    CROSS_SLOTS = 4,
    CROSS_TABLE_SLOTS = 5,
    GATE_SLOTS = 1,
    CONVROT_GROUP = 256,
    BLOCKS = 2,
    VIDEO_ROWS = 48,
    AUDIO_ROWS = 24,
    TEXT_ROWS = 32
};

#define BLOCK_EPSILON 1e-6f

static int failures = 0;
static h3_weight_store *store = NULL;
static h3_weight_store *anchor = NULL;
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

/* ----------------------------------------------------------------- fixture */

static float *load_anchor(const char *name, size_t expected) {
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(anchor, name, &header);
    if (!tensor) fail("the anchor has no tensor %s", name);
    if (tensor->dtype != H3_DTYPE_F32) fail("anchor %s is not F32", name);
    size_t elements = (size_t)h3_st_tensor_elements(tensor);
    if (elements != expected)
        fail("anchor %s has %zu elements, expected %zu", name, elements, expected);
    float *values = malloc(elements * sizeof(*values));
    require(values != NULL, "cannot allocate an anchor tensor");
    char error[512];
    require(h3_st_read_data(header, tensor, values, elements * sizeof(*values),
                            error, sizeof(error)),
            "cannot read an anchor tensor");
    return values;
}

static h3_gpu_tensor *upload(const float *values, size_t count) {
    uint16_t *staged = malloc(count * sizeof(*staged));
    require(staged != NULL, "cannot allocate a staging buffer");
    for (size_t index = 0; index < count; index++)
        staged[index] = to_bf16(values[index]);
    h3_gpu_tensor *tensor = h3_gpu_tensor_from_bf16(gpu, staged, count);
    require(tensor != NULL, "cannot upload a tensor");
    free(staged);
    return tensor;
}

static h3_gpu_tensor *upload_anchor(const char *name, size_t count) {
    float *values = load_anchor(name, count);
    h3_gpu_tensor *tensor = upload(values, count);
    free(values);
    return tensor;
}

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

/* The engine against the F32 reference, with the reference's own BF16 pass
 * printed beside it so the bound is a measurement rather than a choice. */
static void compare(const char *label, const float *actual,
                    const float *expected, const float *floor, size_t count,
                    double tolerance) {
    double worst = 0.0, spread = 0.0, peak = 0.0;
    size_t worst_at = 0;
    for (size_t index = 0; index < count; index++) {
        if (!isfinite(actual[index])) {
            fprintf(stderr, "FAIL %-24s produced %f at %zu\n",
                    label, actual[index], index);
            failures++;
            return;
        }
        double delta = fabs((double)actual[index] - (double)expected[index]);
        if (delta > worst) { worst = delta; worst_at = index; }
        double reference = fabs((double)floor[index] - (double)expected[index]);
        if (reference > spread) spread = reference;
        double magnitude = fabs((double)expected[index]);
        if (magnitude > peak) peak = magnitude;
    }
    const double relative = worst / peak;
    if (relative > tolerance) {
        fprintf(stderr, "FAIL %-24s %.3e = %.2e of peak, floor %.2e, at %zu "
                        "(%.5f vs %.5f)\n", label, worst, relative,
                spread / peak, worst_at, actual[worst_at], expected[worst_at]);
        failures++;
    } else {
        printf("  ok  %-24s %.3e = %.2e of peak %.3f, reference's own BF16 "
               "floor %.2e\n", label, worst, relative, peak, spread / peak);
    }
}

/* ----------------------------------------------------------------- weights */

typedef struct {
    h3_gpu_tensor *weight;
    h3_gpu_tensor *scales;
    h3_gpu_tensor *bias;
} projection;

static h3_gpu_tensor *load_bf16(const char *name, int ndim,
                                uint64_t first, uint64_t second) {
    uint64_t shape[2] = {first, second};
    char error[512];
    h3_gpu_tensor *tensor = h3_weight_load_bf16(store, gpu, name, ndim, shape,
                                                error, sizeof(error));
    if (!tensor) fail("cannot load %s: %s", name, error);
    return tensor;
}

/* Reads a small F32 table into host floats: the modulation tables are added to
 * per-token timesteps on the way into a BF16 buffer, so they never become GPU
 * tensors of their own. */
static float *load_table(const char *name, size_t expected) {
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(store, name, &header);
    if (!tensor) fail("the checkpoint has no %s", name);
    if (tensor->dtype != H3_DTYPE_F32) fail("%s is not F32", name);
    if ((size_t)h3_st_tensor_elements(tensor) != expected)
        fail("%s has %llu elements, expected %zu", name,
             (unsigned long long)h3_st_tensor_elements(tensor), expected);
    float *values = malloc(expected * sizeof(*values));
    require(values != NULL, "cannot allocate a modulation table");
    char error[512];
    require(h3_st_read_data(header, tensor, values, expected * sizeof(*values),
                            error, sizeof(error)),
            "cannot read a modulation table");
    return values;
}

static void load_projection(const char *prefix, const char *suffix,
                            uint64_t out_dim, uint64_t in_dim, int bias,
                            projection *into) {
    char name[256];
    snprintf(name, sizeof(name), "%s%s.weight", prefix, suffix);
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
    if (bias) {
        char bias_name[256];
        snprintf(bias_name, sizeof(bias_name), "%s%s.bias", prefix, suffix);
        into->bias = load_bf16(bias_name, 1, out_dim, 0);
    }
}

static void free_projection(projection *which) {
    h3_gpu_tensor_free(which->weight);
    h3_gpu_tensor_free(which->scales);
    h3_gpu_tensor_free(which->bias);
    memset(which, 0, sizeof(*which));
}

typedef struct {
    const char *name;
    uint32_t query_in;
    uint32_t kv_in;
    uint32_t inner;
    uint32_t out_dim;
    uint32_t gate_in;
    projection query, key, value, output;
    h3_gpu_tensor *query_norm, *key_norm, *gate_weight, *gate_bias;
} attention;

static void load_attention(const char *block, attention *into) {
    char prefix[256];
    snprintf(prefix, sizeof(prefix), "%s%s.", block, into->name);
    load_projection(prefix, "to_q", into->inner, into->query_in, 1, &into->query);
    load_projection(prefix, "to_k", into->inner, into->kv_in, 1, &into->key);
    load_projection(prefix, "to_v", into->inner, into->kv_in, 1, &into->value);
    load_projection(prefix, "to_out.0", into->out_dim, into->inner, 1,
                    &into->output);
    char name[256];
    snprintf(name, sizeof(name), "%sq_norm.weight", prefix);
    into->query_norm = load_bf16(name, 1, into->inner, 0);
    snprintf(name, sizeof(name), "%sk_norm.weight", prefix);
    into->key_norm = load_bf16(name, 1, into->inner, 0);
    snprintf(name, sizeof(name), "%sto_gate_logits.weight", prefix);
    into->gate_weight = load_bf16(name, 2, HEADS, into->gate_in);
    snprintf(name, sizeof(name), "%sto_gate_logits.bias", prefix);
    into->gate_bias = load_bf16(name, 1, HEADS, 0);
}

static void free_attention(attention *which) {
    free_projection(&which->query); free_projection(&which->key);
    free_projection(&which->value); free_projection(&which->output);
    h3_gpu_tensor_free(which->query_norm);
    h3_gpu_tensor_free(which->key_norm);
    h3_gpu_tensor_free(which->gate_weight);
    h3_gpu_tensor_free(which->gate_bias);
}

/* --------------------------------------------------------------- modulation */

/* One buffer per (stream, purpose), holding table + per-token timestep so the
 * AdaLN and gate kernels can read a slot straight out of it. The table is F32
 * in the checkpoint and the timesteps arrive per token, which is exactly what
 * h3_add_row_bf16 was written for. */
static h3_gpu_tensor *build_modulation(const char *table_name,
                                       const char *timestep_name,
                                       uint32_t rows, uint32_t width,
                                       uint32_t table_slots,
                                       uint32_t first_slot,
                                       uint32_t used_slots) {
    float *table = load_table(table_name, (size_t)table_slots * width);
    float *timesteps = load_anchor(timestep_name,
                                   (size_t)rows * used_slots * width);
    h3_gpu_tensor *stamp = upload(timesteps, (size_t)rows * used_slots * width);
    h3_gpu_tensor *out = h3_gpu_tensor_new_bf16(
        gpu, (size_t)rows * used_slots * width);
    require(out != NULL, "cannot allocate a modulation buffer");
    h3_gpu_tensor *row = h3_gpu_tensor_from_f32(
        gpu, table + (size_t)first_slot * width, (size_t)used_slots * width);
    require(row != NULL, "cannot upload a modulation table");
    GPU_OP(h3_gpu_begin(gpu), "begin modulation");
    GPU_OP(h3_gpu_add_row_bf16(gpu, out, stamp, row, rows, used_slots * width),
           "modulation");
    GPU_OP(h3_gpu_submit(gpu), "submit modulation");
    /* That the buffer really is its table rows plus the per-token timestep,
     * elementwise. This pins the add and the timestep's row/slot alignment,
     * and it deliberately does not pin `first_slot`: it recomputes the
     * expectation from the same parameter, so a wrong slot agrees with itself
     * here. That one is caught by the block comparison instead, which is why
     * its tolerance has to stay near the measured floor. */
    uint16_t *staged = malloc((size_t)used_slots * width * sizeof(*staged));
    require(staged != NULL, "cannot allocate a modulation probe");
    require(h3_gpu_tensor_read_bf16(out, staged, (size_t)used_slots * width),
            "cannot read a modulation buffer");
    for (uint32_t slot = 0; slot < used_slots; slot++) {
        for (uint32_t index = 0; index < width; index++) {
            const size_t at = (size_t)slot * width + index;
            const float wanted = table[(size_t)(first_slot + slot) * width + index] +
                                 timesteps[at];
            if (fabsf(from_bf16(staged[at]) - wanted) >
                    0.02f * (fabsf(wanted) + 1.0f)) {
                fail("%s slot %u is not table row %u plus its timestep "
                     "(%f against %f at %u)", table_name, slot,
                     first_slot + slot, (double)from_bf16(staged[at]),
                     (double)wanted, index);
            }
        }
    }
    free(staged);

    h3_gpu_tensor_free(stamp);
    h3_gpu_tensor_free(row);
    free(table);
    free(timesteps);
    return out;
}

/* The context modulation is the one affine here with no norm in front of it,
 * and h3_gpu_adaln_bf16 always normalizes, so it runs on the host. The context
 * is a few hundred rows at most and this is once per block per stream, which
 * is nothing beside the block's own arithmetic -- a kernel would be premature.
 *
 * Order is shift then scale, matching the nine-row table and not the
 * cross-modal one. */
static h3_gpu_tensor *modulate_context(const char *table_name,
                                       const char *context_name,
                                       const char *timestep_name,
                                       uint32_t rows, uint32_t width) {
    float *table = load_table(table_name, (size_t)PROMPT_SLOTS * width);
    float *context = load_anchor(context_name, (size_t)rows * width);
    float *timesteps = load_anchor(timestep_name,
                                   (size_t)rows * PROMPT_SLOTS * width);
    float *out = malloc((size_t)rows * width * sizeof(*out));
    require(out != NULL, "cannot allocate a modulated context");
    for (uint32_t row = 0; row < rows; row++) {
        const float *stamp = timesteps + (size_t)row * PROMPT_SLOTS * width;
        for (uint32_t index = 0; index < width; index++) {
            const float shift = table[index] + stamp[index];
            const float scale = table[width + index] + stamp[width + index];
            out[(size_t)row * width + index] =
                context[(size_t)row * width + index] * (1.0f + scale) + shift;
        }
    }
    h3_gpu_tensor *tensor = upload(out, (size_t)rows * width);
    free(table); free(context); free(timesteps); free(out);
    return tensor;
}

/* ---------------------------------------------------------------- attention */

typedef struct {
    h3_gpu_tensor *query, *key, *value, *heads, *logits;
    h3_gpu_tensor *rotated_query, *rotated_kv, *inner;
    h3_gpu_tensor *ones_video, *ones_audio, *row_map_video, *row_map_audio;
} scratch;

/* One gated attention, general enough for all six: the query and the
 * key/value may come from different streams with different lengths, and each
 * side carries its own rotary table or none at all.
 *
 * The gate reads the attention's *input* before the Hadamard rotation, since
 * to_gate_logits is dense and never sees a rotated activation. */
static void run_attention(const attention *a, h3_gpu_tensor *out,
                          const h3_gpu_tensor *query_input, uint32_t query_rows,
                          const h3_gpu_tensor *kv_input, uint32_t kv_rows,
                          const h3_gpu_tensor *query_cos,
                          const h3_gpu_tensor *query_sin,
                          const h3_gpu_tensor *key_cos,
                          const h3_gpu_tensor *key_sin,
                          scratch *space) {
    const uint32_t head_dim = a->inner / HEADS;
    GPU_OP(h3_gpu_linear_bf16(gpu, space->logits, query_input, a->gate_weight,
                              a->gate_bias, query_rows, a->gate_in, HEADS),
           "gate logits");
    GPU_OP(h3_gpu_convrot_bf16(gpu, space->rotated_query, query_input,
                               query_rows, a->query_in, CONVROT_GROUP),
           "query ConvRot");
    const h3_gpu_tensor *rotated_kv = space->rotated_query;
    if (kv_input != query_input) {
        GPU_OP(h3_gpu_convrot_bf16(gpu, space->rotated_kv, kv_input, kv_rows,
                                   a->kv_in, CONVROT_GROUP), "context ConvRot");
        rotated_kv = space->rotated_kv;
    }
    GPU_OP(h3_gpu_linear_i8_weight_bf16(gpu, space->query, space->rotated_query,
                                        a->query.weight, a->query.scales,
                                        a->query.bias, query_rows,
                                        a->query_in, a->inner),
           "query projection");
    GPU_OP(h3_gpu_linear_i8_weight_bf16(gpu, space->key, rotated_kv,
                                        a->key.weight, a->key.scales,
                                        a->key.bias, kv_rows, a->kv_in,
                                        a->inner), "key projection");
    GPU_OP(h3_gpu_linear_i8_weight_bf16(gpu, space->value, rotated_kv,
                                        a->value.weight, a->value.scales,
                                        a->value.bias, kv_rows, a->kv_in,
                                        a->inner), "value projection");
    /* Over the full inner width, as the connector's are and unlike Gemma's. */
    GPU_OP(h3_gpu_rms_norm_bf16(gpu, space->query, space->query, a->query_norm,
                                query_rows, a->inner, BLOCK_EPSILON),
           "query norm");
    GPU_OP(h3_gpu_rms_norm_bf16(gpu, space->key, space->key, a->key_norm,
                                kv_rows, a->inner, BLOCK_EPSILON), "key norm");
    if (query_cos)
        GPU_OP(h3_gpu_rope_rows_bf16(gpu, space->query, query_cos, query_sin,
                                     query_rows, HEADS, head_dim,
                                     query_rows * (head_dim / 2)), "query rope");
    if (key_cos)
        GPU_OP(h3_gpu_rope_rows_bf16(gpu, space->key, key_cos, key_sin, kv_rows,
                                     HEADS, head_dim, kv_rows * (head_dim / 2)),
               "key rope");
    GPU_OP(h3_gpu_sdpa_cross_bf16(gpu, space->heads, space->query, space->key,
                                  space->value, query_rows, kv_rows, HEADS,
                                  head_dim, 1.0f / sqrtf((float)head_dim)),
           "attention");
    GPU_OP(h3_gpu_head_gate_bf16(gpu, space->heads, space->logits, query_rows,
                                 HEADS, head_dim), "head gate");
    GPU_OP(h3_gpu_convrot_bf16(gpu, space->heads, space->heads, query_rows,
                               a->inner, CONVROT_GROUP), "output ConvRot");
    GPU_OP(h3_gpu_linear_i8_weight_bf16(gpu, out, space->heads,
                                        a->output.weight, a->output.scales,
                                        a->output.bias, query_rows, a->inner,
                                        a->out_dim), "output projection");
}

/* ------------------------------------------------------------------- block */

typedef struct {
    attention attn1, attn2, audio_attn1, audio_attn2, a2v, v2a;
    projection ff_in, ff_out, audio_ff_in, audio_ff_out;
    h3_gpu_tensor *video_modulation, *audio_modulation;
    h3_gpu_tensor *video_cross, *audio_cross;
    h3_gpu_tensor *video_gate, *audio_gate;
    h3_gpu_tensor *video_context, *audio_context;
} block_weights;

static void load_block(int index, block_weights *weights) {
    char block[128];
    snprintf(block, sizeof(block),
             "model.diffusion_model.transformer_blocks.%d.", index);
    memset(weights, 0, sizeof(*weights));
    weights->attn1 = (attention){"attn1", VIDEO_DIM, VIDEO_DIM, VIDEO_DIM,
                                 VIDEO_DIM, VIDEO_DIM, {0}, {0}, {0}, {0},
                                 NULL, NULL, NULL, NULL};
    weights->attn2 = (attention){"attn2", VIDEO_DIM, VIDEO_DIM, VIDEO_DIM,
                                 VIDEO_DIM, VIDEO_DIM, {0}, {0}, {0}, {0},
                                 NULL, NULL, NULL, NULL};
    weights->audio_attn1 = (attention){"audio_attn1", AUDIO_DIM, AUDIO_DIM,
                                       AUDIO_DIM, AUDIO_DIM, AUDIO_DIM,
                                       {0}, {0}, {0}, {0}, NULL, NULL, NULL, NULL};
    weights->audio_attn2 = (attention){"audio_attn2", AUDIO_DIM, AUDIO_DIM,
                                       AUDIO_DIM, AUDIO_DIM, AUDIO_DIM,
                                       {0}, {0}, {0}, {0}, NULL, NULL, NULL, NULL};
    /* The cross-modal pair are the asymmetric ones: query from one stream,
     * keys from the other, and a working width that is neither side's. */
    weights->a2v = (attention){"audio_to_video_attn", VIDEO_DIM, AUDIO_DIM,
                               AUDIO_DIM, VIDEO_DIM, VIDEO_DIM,
                               {0}, {0}, {0}, {0}, NULL, NULL, NULL, NULL};
    weights->v2a = (attention){"video_to_audio_attn", AUDIO_DIM, VIDEO_DIM,
                               AUDIO_DIM, AUDIO_DIM, AUDIO_DIM,
                               {0}, {0}, {0}, {0}, NULL, NULL, NULL, NULL};
    load_attention(block, &weights->attn1);
    load_attention(block, &weights->attn2);
    load_attention(block, &weights->audio_attn1);
    load_attention(block, &weights->audio_attn2);
    load_attention(block, &weights->a2v);
    load_attention(block, &weights->v2a);
    /* The video feed-forward has no bias and the audio one does. */
    load_projection(block, "ff.net.0.proj", VIDEO_FF, VIDEO_DIM, 0,
                    &weights->ff_in);
    load_projection(block, "ff.net.2", VIDEO_DIM, VIDEO_FF, 0, &weights->ff_out);
    load_projection(block, "audio_ff.net.0.proj", AUDIO_FF, AUDIO_DIM, 1,
                    &weights->audio_ff_in);
    load_projection(block, "audio_ff.net.2", AUDIO_DIM, AUDIO_FF, 1,
                    &weights->audio_ff_out);

    char name[256];
    snprintf(name, sizeof(name), "%sscale_shift_table", block);
    weights->video_modulation = build_modulation(
        name, "video_timesteps", VIDEO_ROWS, VIDEO_DIM, ADA_SLOTS, 0, ADA_SLOTS);
    snprintf(name, sizeof(name), "%saudio_scale_shift_table", block);
    weights->audio_modulation = build_modulation(
        name, "audio_timesteps", AUDIO_ROWS, AUDIO_DIM, ADA_SLOTS, 0, ADA_SLOTS);
    /* Rows 0-3 of the five, against a four-slot timestep. */
    snprintf(name, sizeof(name), "%sscale_shift_table_a2v_ca_video", block);
    weights->video_cross = build_modulation(
        name, "video_cross_timestep", VIDEO_ROWS, VIDEO_DIM,
        CROSS_TABLE_SLOTS, 0, CROSS_SLOTS);
    snprintf(name, sizeof(name), "%sscale_shift_table_a2v_ca_audio", block);
    weights->audio_cross = build_modulation(
        name, "audio_cross_timestep", AUDIO_ROWS, AUDIO_DIM,
        CROSS_TABLE_SLOTS, 0, CROSS_SLOTS);
    /* And row 4 alone, against the one-slot gate timestep. */
    snprintf(name, sizeof(name), "%sscale_shift_table_a2v_ca_video", block);
    weights->video_gate = build_modulation(
        name, "video_gate_timestep", VIDEO_ROWS, VIDEO_DIM,
        CROSS_TABLE_SLOTS, CROSS_SLOTS, GATE_SLOTS);
    snprintf(name, sizeof(name), "%sscale_shift_table_a2v_ca_audio", block);
    weights->audio_gate = build_modulation(
        name, "audio_gate_timestep", AUDIO_ROWS, AUDIO_DIM,
        CROSS_TABLE_SLOTS, CROSS_SLOTS, GATE_SLOTS);

    snprintf(name, sizeof(name), "%sprompt_scale_shift_table", block);
    weights->video_context = modulate_context(
        name, "video_context", "video_prompt_timestep", TEXT_ROWS, VIDEO_DIM);
    snprintf(name, sizeof(name), "%saudio_prompt_scale_shift_table", block);
    weights->audio_context = modulate_context(
        name, "audio_context", "audio_prompt_timestep", TEXT_ROWS, AUDIO_DIM);
}

static void free_block(block_weights *weights) {
    free_attention(&weights->attn1); free_attention(&weights->attn2);
    free_attention(&weights->audio_attn1); free_attention(&weights->audio_attn2);
    free_attention(&weights->a2v); free_attention(&weights->v2a);
    free_projection(&weights->ff_in); free_projection(&weights->ff_out);
    free_projection(&weights->audio_ff_in);
    free_projection(&weights->audio_ff_out);
    h3_gpu_tensor_free(weights->video_modulation);
    h3_gpu_tensor_free(weights->audio_modulation);
    h3_gpu_tensor_free(weights->video_cross);
    h3_gpu_tensor_free(weights->audio_cross);
    h3_gpu_tensor_free(weights->video_gate);
    h3_gpu_tensor_free(weights->audio_gate);
    h3_gpu_tensor_free(weights->video_context);
    h3_gpu_tensor_free(weights->audio_context);
    memset(weights, 0, sizeof(*weights));
}

typedef struct {
    h3_gpu_tensor *cos, *sin;
} rope;

typedef struct {
    h3_gpu_tensor *video, *audio;
    h3_gpu_tensor *video_pre, *audio_pre;
    h3_gpu_tensor *video_scaled, *audio_scaled;
    h3_gpu_tensor *video_branch, *audio_branch;
    rope video_pe, audio_pe, video_cross_pe, audio_cross_pe;
} state;

/* Self-attention, then the text cross-attention, for one stream. Slots 0-2 and
 * 6-8 of the nine-row modulation, each as shift, scale, gate. */
static void self_and_text(const attention *self, const attention *text,
                          h3_gpu_tensor *x, h3_gpu_tensor *scaled,
                          h3_gpu_tensor *branch,
                          const h3_gpu_tensor *modulation,
                          const h3_gpu_tensor *context,
                          const h3_gpu_tensor *ones,
                          const h3_gpu_tensor *row_map,
                          const rope *pe, uint32_t rows, uint32_t dim,
                          scratch *space) {
    GPU_OP(h3_gpu_adaln_bf16(gpu, scaled, x, ones, modulation, row_map, rows,
                             dim, ADA_SLOTS, 0, 1, BLOCK_EPSILON),
           "self-attention modulation");
    run_attention(self, branch, scaled, rows, scaled, rows,
                  pe->cos, pe->sin, pe->cos, pe->sin, space);
    GPU_OP(h3_gpu_gate_bf16(gpu, x, x, branch, modulation, row_map, rows, dim,
                            ADA_SLOTS, 2), "self-attention residual");
    /* The text cross-attention carries no rotation on either side. */
    GPU_OP(h3_gpu_adaln_bf16(gpu, scaled, x, ones, modulation, row_map, rows,
                             dim, ADA_SLOTS, 6, 7, BLOCK_EPSILON),
           "text cross-attention modulation");
    run_attention(text, branch, scaled, rows, context, TEXT_ROWS,
                  NULL, NULL, NULL, NULL, space);
    GPU_OP(h3_gpu_gate_bf16(gpu, x, x, branch, modulation, row_map, rows, dim,
                            ADA_SLOTS, 8), "text cross-attention residual");
}

static void feed_forward(const projection *in, const projection *out,
                         h3_gpu_tensor *x, h3_gpu_tensor *scaled,
                         h3_gpu_tensor *branch,
                         const h3_gpu_tensor *modulation,
                         const h3_gpu_tensor *ones,
                         const h3_gpu_tensor *row_map, uint32_t rows,
                         uint32_t dim, uint32_t width, scratch *space) {
    GPU_OP(h3_gpu_adaln_bf16(gpu, scaled, x, ones, modulation, row_map, rows,
                             dim, ADA_SLOTS, 3, 4, BLOCK_EPSILON),
           "feed-forward modulation");
    GPU_OP(h3_gpu_convrot_bf16(gpu, scaled, scaled, rows, dim, CONVROT_GROUP),
           "feed-forward ConvRot");
    GPU_OP(h3_gpu_linear_i8_weight_bf16(gpu, space->inner, scaled, in->weight,
                                        in->scales, in->bias, rows, dim, width),
           "feed-forward in");
    GPU_OP(h3_gpu_gelu_bf16(gpu, space->inner, space->inner, rows * width, 1),
           "feed-forward activation");
    GPU_OP(h3_gpu_convrot_bf16(gpu, space->inner, space->inner, rows, width,
                               CONVROT_GROUP), "feed-forward out ConvRot");
    GPU_OP(h3_gpu_linear_i8_weight_bf16(gpu, branch, space->inner, out->weight,
                                        out->scales, out->bias, rows, width,
                                        dim), "feed-forward out");
    GPU_OP(h3_gpu_gate_bf16(gpu, x, x, branch, modulation, row_map, rows, dim,
                            ADA_SLOTS, 5), "feed-forward residual");
}

static void run_block(const block_weights *weight, state *s, scratch *space) {
    self_and_text(&weight->attn1, &weight->attn2, s->video, s->video_scaled,
                  s->video_branch, weight->video_modulation,
                  weight->video_context, space->ones_video,
                  space->row_map_video, &s->video_pe, VIDEO_ROWS, VIDEO_DIM,
                  space);
    self_and_text(&weight->audio_attn1, &weight->audio_attn2, s->audio,
                  s->audio_scaled, s->audio_branch, weight->audio_modulation,
                  weight->audio_context, space->ones_audio,
                  space->row_map_audio, &s->audio_pe, AUDIO_ROWS, AUDIO_DIM,
                  space);

    /* Both streams are snapshotted before either direction runs. Reading the
     * live tensors instead would let audio-into-video bias video-into-audio,
     * and the result would still be perfectly well formed. */
    GPU_OP(h3_gpu_copy_bf16(gpu, s->video_pre, 0, s->video, 0,
                            (size_t)VIDEO_ROWS * VIDEO_DIM), "video snapshot");
    GPU_OP(h3_gpu_copy_bf16(gpu, s->audio_pre, 0, s->audio, 0,
                            (size_t)AUDIO_ROWS * AUDIO_DIM), "audio snapshot");

    /* Audio into video. Scale then shift, which is the opposite order to the
     * nine-row table's, and the gate is row 4 of a different timestep. */
    GPU_OP(h3_gpu_adaln_bf16(gpu, s->video_scaled, s->video_pre,
                             space->ones_video, weight->video_cross,
                             space->row_map_video, VIDEO_ROWS, VIDEO_DIM,
                             CROSS_SLOTS, 1, 0, BLOCK_EPSILON),
           "a2v video modulation");
    GPU_OP(h3_gpu_adaln_bf16(gpu, s->audio_scaled, s->audio_pre,
                             space->ones_audio, weight->audio_cross,
                             space->row_map_audio, AUDIO_ROWS, AUDIO_DIM,
                             CROSS_SLOTS, 1, 0, BLOCK_EPSILON),
           "a2v audio modulation");
    run_attention(&weight->a2v, s->video_branch, s->video_scaled, VIDEO_ROWS,
                  s->audio_scaled, AUDIO_ROWS, s->video_cross_pe.cos,
                  s->video_cross_pe.sin, s->audio_cross_pe.cos,
                  s->audio_cross_pe.sin, space);
    GPU_OP(h3_gpu_gate_bf16(gpu, s->video, s->video, s->video_branch,
                            weight->video_gate, space->row_map_video,
                            VIDEO_ROWS, VIDEO_DIM, GATE_SLOTS, 0),
           "a2v residual");

    /* Video into audio, off the snapshots. Rows 2-3 of the same tables. */
    GPU_OP(h3_gpu_adaln_bf16(gpu, s->audio_scaled, s->audio_pre,
                             space->ones_audio, weight->audio_cross,
                             space->row_map_audio, AUDIO_ROWS, AUDIO_DIM,
                             CROSS_SLOTS, 3, 2, BLOCK_EPSILON),
           "v2a audio modulation");
    GPU_OP(h3_gpu_adaln_bf16(gpu, s->video_scaled, s->video_pre,
                             space->ones_video, weight->video_cross,
                             space->row_map_video, VIDEO_ROWS, VIDEO_DIM,
                             CROSS_SLOTS, 3, 2, BLOCK_EPSILON),
           "v2a video modulation");
    run_attention(&weight->v2a, s->audio_branch, s->audio_scaled, AUDIO_ROWS,
                  s->video_scaled, VIDEO_ROWS, s->audio_cross_pe.cos,
                  s->audio_cross_pe.sin, s->video_cross_pe.cos,
                  s->video_cross_pe.sin, space);
    GPU_OP(h3_gpu_gate_bf16(gpu, s->audio, s->audio, s->audio_branch,
                            weight->audio_gate, space->row_map_audio,
                            AUDIO_ROWS, AUDIO_DIM, GATE_SLOTS, 0),
           "v2a residual");

    feed_forward(&weight->ff_in, &weight->ff_out, s->video, s->video_scaled,
                 s->video_branch, weight->video_modulation, space->ones_video,
                 space->row_map_video, VIDEO_ROWS, VIDEO_DIM, VIDEO_FF, space);
    feed_forward(&weight->audio_ff_in, &weight->audio_ff_out, s->audio,
                 s->audio_scaled, s->audio_branch, weight->audio_modulation,
                 space->ones_audio, space->row_map_audio, AUDIO_ROWS,
                 AUDIO_DIM, AUDIO_FF, space);
}

/* ------------------------------------------------------------------- main */

static rope load_rope(const char *name, uint32_t rows, uint32_t head_dim) {
    char full[128];
    rope out;
    snprintf(full, sizeof(full), "%s_cos", name);
    out.cos = NULL;
    float *values = load_anchor(full, (size_t)HEADS * rows * (head_dim / 2));
    out.cos = h3_gpu_tensor_from_f32(gpu, values,
                                     (size_t)HEADS * rows * (head_dim / 2));
    free(values);
    snprintf(full, sizeof(full), "%s_sin", name);
    values = load_anchor(full, (size_t)HEADS * rows * (head_dim / 2));
    out.sin = h3_gpu_tensor_from_f32(gpu, values,
                                     (size_t)HEADS * rows * (head_dim / 2));
    free(values);
    require(out.cos && out.sin, "cannot upload a rotary table");
    return out;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s DIT.safetensors ANCHOR.safetensors\n",
                argv[0]);
        return 2;
    }
    char error[512];
    store = h3_weight_store_open(argv[1], error, sizeof(error));
    if (!store) fail("cannot open the DiT checkpoint: %s", error);
    anchor = h3_weight_store_open(argv[2], error, sizeof(error));
    if (!anchor) fail("cannot open the anchor: %s", error);
    gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) fail("cannot create the Metal context: %s", error);

    printf("A released LTX-2.5 DiT block against BasicAVTransformerBlock, "
           "%d video rows, %d audio, %d context:\n",
           VIDEO_ROWS, AUDIO_ROWS, TEXT_ROWS);

    scratch space;
    memset(&space, 0, sizeof(space));
    const size_t widest = (size_t)VIDEO_ROWS * VIDEO_DIM;
    space.query = h3_gpu_tensor_new_bf16(gpu, widest);
    space.key = h3_gpu_tensor_new_bf16(gpu, widest);
    space.value = h3_gpu_tensor_new_bf16(gpu, widest);
    space.heads = h3_gpu_tensor_new_bf16(gpu, widest);
    space.logits = h3_gpu_tensor_new_bf16(gpu, (size_t)VIDEO_ROWS * HEADS);
    space.rotated_query = h3_gpu_tensor_new_bf16(gpu, widest);
    space.rotated_kv = h3_gpu_tensor_new_bf16(gpu, widest);
    space.inner = h3_gpu_tensor_new_bf16(gpu, (size_t)VIDEO_ROWS * VIDEO_FF);
    require(space.query && space.key && space.value && space.heads &&
            space.logits && space.rotated_query && space.rotated_kv &&
            space.inner, "cannot allocate attention scratch");
    uint16_t *ones = malloc(VIDEO_DIM * sizeof(*ones));
    require(ones != NULL, "cannot allocate a norm weight");
    for (int index = 0; index < VIDEO_DIM; index++) ones[index] = 0x3f80;
    space.ones_video = h3_gpu_tensor_from_bf16(gpu, ones, VIDEO_DIM);
    space.ones_audio = h3_gpu_tensor_from_bf16(gpu, ones, AUDIO_DIM);
    free(ones);
    uint32_t *identity = malloc(VIDEO_ROWS * sizeof(*identity));
    require(identity != NULL, "cannot allocate a row map");
    for (int index = 0; index < VIDEO_ROWS; index++)
        identity[index] = (uint32_t)index;
    space.row_map_video = h3_gpu_tensor_from_u32(gpu, identity, VIDEO_ROWS);
    space.row_map_audio = h3_gpu_tensor_from_u32(gpu, identity, AUDIO_ROWS);
    free(identity);
    require(space.ones_video && space.ones_audio && space.row_map_video &&
            space.row_map_audio, "cannot allocate modulation helpers");

    state s;
    memset(&s, 0, sizeof(s));
    s.video_pre = h3_gpu_tensor_new_bf16(gpu, (size_t)VIDEO_ROWS * VIDEO_DIM);
    s.audio_pre = h3_gpu_tensor_new_bf16(gpu, (size_t)AUDIO_ROWS * AUDIO_DIM);
    s.video_scaled = h3_gpu_tensor_new_bf16(gpu, (size_t)VIDEO_ROWS * VIDEO_DIM);
    s.audio_scaled = h3_gpu_tensor_new_bf16(gpu, (size_t)AUDIO_ROWS * AUDIO_DIM);
    s.video_branch = h3_gpu_tensor_new_bf16(gpu, (size_t)VIDEO_ROWS * VIDEO_DIM);
    s.audio_branch = h3_gpu_tensor_new_bf16(gpu, (size_t)AUDIO_ROWS * AUDIO_DIM);
    require(s.video_pre && s.audio_pre && s.video_scaled && s.audio_scaled &&
            s.video_branch && s.audio_branch, "cannot allocate block state");
    s.video_pe = load_rope("video_pe", VIDEO_ROWS, VIDEO_HEAD);
    s.audio_pe = load_rope("audio_pe", AUDIO_ROWS, AUDIO_HEAD);
    /* Both cross-modal directions run at the audio head dim, so the video side
     * of that pair is audio-shaped too. */
    s.video_cross_pe = load_rope("video_cross_pe", VIDEO_ROWS, AUDIO_HEAD);
    s.audio_cross_pe = load_rope("audio_cross_pe", AUDIO_ROWS, AUDIO_HEAD);

    s.video = upload_anchor("vx", (size_t)VIDEO_ROWS * VIDEO_DIM);
    s.audio = upload_anchor("ax", (size_t)AUDIO_ROWS * AUDIO_DIM);

    float *video_out = malloc((size_t)VIDEO_ROWS * VIDEO_DIM * sizeof(float));
    float *audio_out = malloc((size_t)AUDIO_ROWS * AUDIO_DIM * sizeof(float));
    require(video_out && audio_out, "cannot allocate the readback");

    for (int index = 0; index < BLOCKS; index++) {
        block_weights weights;
        load_block(index, &weights);
        GPU_OP(h3_gpu_begin(gpu), "begin block");
        run_block(&weights, &s, &space);
        GPU_OP(h3_gpu_submit(gpu), "submit block");
        free_block(&weights);
        read_bf16_as_f32(s.video, video_out, (size_t)VIDEO_ROWS * VIDEO_DIM);
        read_bf16_as_f32(s.audio, audio_out, (size_t)AUDIO_ROWS * AUDIO_DIM);
        for (int stream = 0; stream < 2; stream++) {
            const char *tag = stream ? "audio" : "video";
            const size_t count = stream ? (size_t)AUDIO_ROWS * AUDIO_DIM
                                        : (size_t)VIDEO_ROWS * VIDEO_DIM;
            char name[64], coarse[80], label[64];
            snprintf(name, sizeof(name), "block%d_%s", index, tag);
            snprintf(coarse, sizeof(coarse), "block%d_%s_bf16", index, tag);
            snprintf(label, sizeof(label), "block %d %s", index, tag);
            float *expected = load_anchor(name, count);
            float *floor = load_anchor(coarse, count);
            /* Deliberately close to the floor, and the four numbers it has to
             * separate are worth writing down because nothing else recovers
             * them: the engine runs at 5.4e-03, 9.0e-03, 6.1e-03 and 9.1e-03
             * against reference BF16 spreads of 1.1e-02, 6.8e-03, 1.2e-02 and
             * 7.0e-03. Reading the cross-modal gate one table row off puts the
             * first of those at 1.6e-02 and changes nothing else.
             *
             * So the useful range is narrow, and every loose choice above it
             * costs a real mutation: 5e-02 let both that and a dropped
             * feed-forward bias (3.9e-02) through. A tolerance picked for
             * comfort is a tolerance picked to pass. */
            compare(label, stream ? audio_out : video_out, expected, floor,
                    count, 1.2e-2);
            free(expected);
            free(floor);
        }
    }

    free(video_out); free(audio_out);
    h3_gpu_free(gpu);
    h3_weight_store_free(store);
    h3_weight_store_free(anchor);
    if (failures) {
        fprintf(stderr, "\n%d comparisons failed\n", failures);
        return 1;
    }
    printf("\nok: %d released DiT blocks reproduce BasicAVTransformerBlock at "
           "full width, across six attentions, both feed-forwards and all six "
           "modulation tables\n", BLOCKS);
    return 0;
}
