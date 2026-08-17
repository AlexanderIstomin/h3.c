/* A multi-step denoise of LTX-2.5's DiT, with a partly-conditioned latent.
 *
 * The end-to-end anchor runs one forward pass at one sigma. Three things stay
 * invisible until there is a *loop* and a *mask*, and this covers all three.
 *
 * **Velocity is not the latent update.** The model emits a velocity; the
 * sampler consumes a denoised prediction. `x - v * t` is the conversion, and
 * getting its sign or its factor wrong still yields a plausible tensor.
 *
 * **That `t` is per token.** It is `sigma * denoise_mask`, so a token whose
 * mask is zero gets `x - v * 0` and comes back *exactly* as it went in,
 * whatever the model predicted for it. That single line is the whole
 * conditioning mechanism -- image-to-video, keyframes and a held audio track
 * are all it. A driver that substitutes the scalar sigma denoises the
 * conditioning away, and a from-noise generation never notices.
 *
 * **The eight AdaLN modules do not read the same thing.** Three inputs are
 * spread across them, and under a single scalar sigma all three coincide, so
 * no single-step test can separate them. The MODULES table at the bottom of
 * this file is that mapping, and the sharp entry is the cross-modal gate: the
 * gate on what video takes *from* audio is driven by *audio's* noise level,
 * the opposite of what the symmetric reading of the name suggests.
 *
 * The fixture arranges for all three to be observable at once. Video denoises
 * with a conditioning first frame; the audio track is *frozen* -- mask all
 * zero and, per LTX's own `LatentState.frozen`, its scalar noise level forced
 * to zero too, which is what makes the two streams' sigmas differ and the
 * gate's cross-reading visible. Freezing is not skipping: a frozen stream
 * still runs every block, because the cross-modal attention reads its hidden
 * states. Only its latent update is suppressed.
 *
 * Anchored on two blocks against LTX's own, then runnable at the full 48.
 * Weights load and free a block at a time, so this holds about 400 MB -- but
 * that also means a step reloads the whole stack, which is what dominates a
 * long run and is reported separately below.
 *
 * usage: h3_real_ltx_dit_sample_test DIT.safetensors ANCHOR.safetensors [BLOCKS] */

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
    AUDIO_DIM = 2048,
    HEADS = 32,
    VIDEO_HEAD = VIDEO_DIM / HEADS,
    AUDIO_HEAD = AUDIO_DIM / HEADS,
    VIDEO_FF = VIDEO_DIM * 4,
    AUDIO_FF = AUDIO_DIM * 4,
    LATENT = 128,
    TIMESTEP_FEATURES = 256,
    ADA_SLOTS = 9,
    PROMPT_SLOTS = 2,
    CROSS_SLOTS = 4,
    CROSS_TABLE_SLOTS = 5,
    GATE_SLOTS = 1,
    HEAD_SLOTS = 2,
    CONVROT_GROUP = 256,
    TOTAL_BLOCKS = 48,
    ANCHOR_BLOCKS = 2,
    FRAMES = 2, HEIGHT = 4, WIDTH = 4,
    VIDEO_ROWS = FRAMES * HEIGHT * WIDTH,
    AUDIO_ROWS = 16,
    TEXT_ROWS = 32,
    VIDEO_AXES = 3,
    AUDIO_AXES = 1,
    STEPS = 4,
    /* Two modulation rows per module: one for tokens the mask holds at zero,
     * one for the tokens actually being denoised. The mask is binary, which is
     * asserted rather than assumed -- a fractional mask would need a row per
     * distinct value, and the row map already supports that. */
    LEVELS = 2,
    /* The first latent frame is the conditioning image. */
    CONDITIONED = HEIGHT * WIDTH
};

#define BLOCK_EPSILON 1e-6f
#define TIMESTEP_MULTIPLIER 1000.0
#define MAX_PERIOD 10000.0

/* Deviation from the reference, as a fraction of each tensor's own peak.
 *
 * Two numbers bound this. The reference run against itself with its weights
 * rounded through BF16 -- `gen_ltx_dit_sample.py --floor` -- moves by 5.6e-05
 * to 6.1e-04, growing with the step count. That is only the *weight* floor:
 * the reference still computes in F32 activations while the engine is BF16
 * throughout, which is what the remaining factor of thirty is.
 *
 * Measured here: 3.0e-03 at step 0 rising to 1.67e-02 at step 3, the growth
 * being error carried forward by the loop. So this sits just above the worst
 * of them and no higher -- a comfortable bound is one that hides a real
 * mutation, which is how 5e-2 concealed two of them on the block anchor. */
#define TOLERANCE 2e-2

/* The modulation buffers are compared before anything consumes them, so this
 * bounds the timestep path alone: a sinusoid and three small matrices, in F32
 * on both sides. Nothing accumulates, so it is orders tighter than the one
 * above -- and it has to be, since the wirings it exists to separate produce
 * differences of 100% here and 0.001% at the far end. */
#define MODULATION_TOLERANCE 2e-5

/* Tokens whose latent never moves -- the held frame, and the frozen audio
 * track -- get a tighter bound than the ones being denoised, because nothing
 * accumulates for them: their deviation is flat at 3.4e-03 to 3.6e-03 across
 * all four steps where the free tokens climb from 4.5e-03 to 1.7e-02. Keeping
 * them under the loose bound would waste the one place the head's per-token
 * modulation is observable at all: ignoring it moves them by 7.4e-03. */
#define HELD_TOLERANCE 5e-3

static const double VIDEO_MAX_POS[VIDEO_AXES] = {20.0, 2048.0, 2048.0};
static const double AUDIO_MAX_POS[AUDIO_AXES] = {20.0};
static const double ROPE_THETA = 10000.0;

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

/* ------------------------------------------------------------------ loading */

static float *read_tensor(const h3_weight_store *from, const char *name,
                          size_t expected) {
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(from, name, &header);
    if (!tensor) fail("no tensor %s", name);
    size_t elements = (size_t)h3_st_tensor_elements(tensor);
    if (elements != expected)
        fail("%s has %zu elements, expected %zu", name, elements, expected);
    float *values = malloc(elements * sizeof(*values));
    require(values != NULL, "cannot allocate a tensor");
    char error[512];
    if (tensor->dtype == H3_DTYPE_F32) {
        require(h3_st_read_data(header, tensor, values,
                                elements * sizeof(*values), error, sizeof(error)),
                "cannot read an F32 tensor");
    } else if (tensor->dtype == H3_DTYPE_BF16) {
        uint16_t *raw = malloc(elements * sizeof(*raw));
        require(raw != NULL, "cannot allocate a staging buffer");
        require(h3_st_read_data(header, tensor, raw, elements * sizeof(*raw),
                                error, sizeof(error)),
                "cannot read a BF16 tensor");
        for (size_t index = 0; index < elements; index++)
            values[index] = from_bf16(raw[index]);
        free(raw);
    } else {
        fail("%s is %s", name, h3_dtype_name(tensor->dtype));
    }
    return values;
}

static float *weight_of(const char *name, size_t expected) {
    char full[224];
    snprintf(full, sizeof(full), "model.diffusion_model.%s", name);
    return read_tensor(store, full, expected);
}

static float *golden(const char *name, size_t expected) {
    return read_tensor(anchor, name, expected);
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

static void compare(const char *label, const float *actual,
                    const float *expected, size_t count, double tolerance) {
    double worst = 0.0, peak = 0.0;
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
        double magnitude = fabs((double)expected[index]);
        if (magnitude > peak) peak = magnitude;
    }
    const double relative = worst / peak;
    if (relative > tolerance) {
        fprintf(stderr, "FAIL %-24s %.3e = %.2e of peak, at %zu (%.5f vs %.5f)\n",
                label, worst, relative, worst_at, actual[worst_at],
                expected[worst_at]);
        failures++;
    } else {
        printf("  ok  %-24s %.3e = %.2e of peak %.4f\n",
               label, worst, relative, peak);
    }
}

/* -------------------------------------------------------- the timestep path */

static float silu(float value) { return value / (1.0f + expf(-value)); }

static void linear_rows(float *out, const float *in, const float *weight,
                        const float *bias, uint32_t rows, uint32_t in_dim,
                        uint32_t out_dim) {
    for (uint32_t row = 0; row < rows; row++) {
        for (uint32_t column = 0; column < out_dim; column++) {
            float sum = bias ? bias[column] : 0.0f;
            const float *w = weight + (size_t)column * in_dim;
            const float *x = in + (size_t)row * in_dim;
            for (uint32_t index = 0; index < in_dim; index++)
                sum = fmaf(x[index], w[index], sum);
            out[(size_t)row * out_dim + column] = sum;
        }
    }
}

/* The argument reaches 732 radians at sigma 0.732, so this is F32 whatever the
 * rest of the model is. Cosine first, and the exponent divides by half rather
 * than half - 1. */
static void sinusoid(float *out, double timestep) {
    const int half = TIMESTEP_FEATURES / 2;
    for (int index = 0; index < half; index++) {
        const double exponent = -log(MAX_PERIOD) * (double)index / (double)half;
        const double angle = timestep * exp(exponent);
        out[index] = (float)cos(angle);
        out[half + index] = (float)sin(angle);
    }
}

/* Which of the four noise quantities a module is driven by. Under a scalar
 * sigma with no mask all four collapse to the same number, which is why this
 * distinction only shows up in a run like this one. */
typedef enum {
    VIDEO_TOKENS,   /* video's per-token timesteps, sigma * mask */
    AUDIO_TOKENS,
    VIDEO_SIGMA,    /* video's scalar noise level, zero when frozen */
    AUDIO_SIGMA
} timestep_source;

typedef struct {
    const char *name;
    uint32_t width, slots;
    timestep_source source;
} adaln_module;

/* Evaluate one module at `count` timesteps in a single pass over its weights.
 * The projection is 4096x36864 for the largest of them, so reading it once per
 * step rather than once per timestep is the difference between a few hundred
 * megabytes of I/O and a few gigabytes.
 *
 * `modulation` comes back as [level][slot][width] and `embedded` as
 * [level][width], which is the layout h3_adaln_bf16 indexes: it reads
 * `row_map[token] * slots * width` as the base of a token's modulation. */
static void run_adaln(const adaln_module *module, const double *timesteps,
                      uint32_t count, double multiplier, float **modulation,
                      float **embedded) {
    char name[224];
    const size_t width = module->width;
    float *features = malloc((size_t)count * TIMESTEP_FEATURES * sizeof(*features));
    require(features != NULL, "cannot allocate the timestep features");
    for (uint32_t level = 0; level < count; level++)
        sinusoid(features + (size_t)level * TIMESTEP_FEATURES,
                 timesteps[level] * multiplier);
    snprintf(name, sizeof(name), "%s.emb.timestep_embedder.linear_1.weight",
             module->name);
    float *first = weight_of(name, width * TIMESTEP_FEATURES);
    snprintf(name, sizeof(name), "%s.emb.timestep_embedder.linear_1.bias",
             module->name);
    float *first_bias = weight_of(name, module->width);
    float *hidden = malloc((size_t)count * width * sizeof(*hidden));
    require(hidden != NULL, "cannot allocate a timestep hidden state");
    linear_rows(hidden, features, first, first_bias, count, TIMESTEP_FEATURES,
                module->width);
    for (size_t index = 0; index < (size_t)count * width; index++)
        hidden[index] = silu(hidden[index]);
    free(first); free(first_bias); free(features);
    snprintf(name, sizeof(name), "%s.emb.timestep_embedder.linear_2.weight",
             module->name);
    float *second = weight_of(name, width * width);
    snprintf(name, sizeof(name), "%s.emb.timestep_embedder.linear_2.bias",
             module->name);
    float *second_bias = weight_of(name, module->width);
    *embedded = malloc((size_t)count * width * sizeof(**embedded));
    require(*embedded != NULL, "cannot allocate an embedded timestep");
    linear_rows(*embedded, hidden, second, second_bias, count, module->width,
                module->width);
    free(second); free(second_bias); free(hidden);
    float *activated = malloc((size_t)count * width * sizeof(*activated));
    require(activated != NULL, "cannot allocate the activated timestep");
    for (size_t index = 0; index < (size_t)count * width; index++)
        activated[index] = silu((*embedded)[index]);
    snprintf(name, sizeof(name), "%s.linear.weight", module->name);
    float *projection = weight_of(name, (size_t)module->slots * width * width);
    snprintf(name, sizeof(name), "%s.linear.bias", module->name);
    float *projection_bias = weight_of(name, (size_t)module->slots * width);
    *modulation = malloc((size_t)count * module->slots * width *
                         sizeof(**modulation));
    require(*modulation != NULL, "cannot allocate a modulation");
    linear_rows(*modulation, activated, projection, projection_bias, count,
                module->width, module->slots * module->width);
    free(projection); free(projection_bias); free(activated);
}

/* ------------------------------------------------------------------- rope */

/* [head][row][half / heads], which is the layout the rotation kernel's head
 * stride walks. The axes interleave -- slot = frequency * axes + axis -- and
 * the padding sits in front. */
static void rope_tables(h3_gpu_tensor **cosine, h3_gpu_tensor **sine,
                        const float *grid, uint32_t rows, uint32_t dim,
                        int axes, const double *max_pos) {
    const uint32_t half = dim / 2;
    const uint32_t per_axis = dim / (2 * (uint32_t)axes);
    const uint32_t pad = half - per_axis * (uint32_t)axes;
    const uint32_t per_head = half / HEADS;
    float *flat_cos = calloc((size_t)rows * half, sizeof(*flat_cos));
    float *flat_sin = calloc((size_t)rows * half, sizeof(*flat_sin));
    require(flat_cos && flat_sin, "cannot allocate rotary tables");
    for (uint32_t row = 0; row < rows; row++) {
        for (uint32_t slot = 0; slot < pad; slot++)
            flat_cos[(size_t)row * half + slot] = 1.0f;
        for (int axis = 0; axis < axes; axis++) {
            const float *pair = grid + ((size_t)axis * rows + row) * 2;
            const float middle = (pair[0] + pair[1]) / 2.0f;
            const float position = middle / (float)max_pos[axis] * 2.0f - 1.0f;
            for (uint32_t index = 0; index < per_axis; index++) {
                const double exponent = per_axis > 1 ?
                    (double)index / (double)(per_axis - 1) : 0.0;
                /* Rounded to F32 on purpose: the reference computes these in
                 * double and hands back a float32 tensor, so keeping the
                 * double is more accurate and disagrees. */
                const float frequency =
                    (float)(pow(ROPE_THETA, exponent) * M_PI / 2.0);
                const float angle = frequency * position;
                const size_t at = (size_t)row * half + pad +
                                  (size_t)index * (uint32_t)axes + (uint32_t)axis;
                flat_cos[at] = cosf(angle);
                flat_sin[at] = sinf(angle);
            }
        }
    }
    float *head_cos = malloc((size_t)rows * half * sizeof(*head_cos));
    float *head_sin = malloc((size_t)rows * half * sizeof(*head_sin));
    require(head_cos && head_sin, "cannot allocate head-major rotary tables");
    for (uint32_t head = 0; head < HEADS; head++)
        for (uint32_t row = 0; row < rows; row++)
            for (uint32_t index = 0; index < per_head; index++) {
                const size_t to = ((size_t)head * rows + row) * per_head + index;
                const size_t from = (size_t)row * half + head * per_head + index;
                head_cos[to] = flat_cos[from];
                head_sin[to] = flat_sin[from];
            }
    *cosine = h3_gpu_tensor_from_f32(gpu, head_cos, (size_t)rows * half);
    *sine = h3_gpu_tensor_from_f32(gpu, head_sin, (size_t)rows * half);
    require(*cosine && *sine, "cannot upload rotary tables");
    free(flat_cos); free(flat_sin); free(head_cos); free(head_sin);
}

/* ------------------------------------------------------------------ weights */

typedef struct { h3_gpu_tensor *weight, *scales, *bias; } projection;

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
        fail("%s has ConvRot group %u", name, group);
    if (bias) {
        char bias_name[256];
        snprintf(bias_name, sizeof(bias_name), "%s%s.bias", prefix, suffix);
        uint64_t shape[1] = {out_dim};
        into->bias = h3_weight_load_bf16(store, gpu, bias_name, 1, shape,
                                         error, sizeof(error));
        if (!into->bias) fail("cannot load %s: %s", bias_name, error);
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
    uint32_t query_in, kv_in, inner, out_dim, gate_in;
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
    char name[256], error[512];
    uint64_t inner[1] = {into->inner}, heads[1] = {HEADS};
    uint64_t gate[2] = {HEADS, into->gate_in};
    snprintf(name, sizeof(name), "%sq_norm.weight", prefix);
    into->query_norm = h3_weight_load_bf16(store, gpu, name, 1, inner, error,
                                           sizeof(error));
    snprintf(name, sizeof(name), "%sk_norm.weight", prefix);
    into->key_norm = h3_weight_load_bf16(store, gpu, name, 1, inner, error,
                                         sizeof(error));
    snprintf(name, sizeof(name), "%sto_gate_logits.weight", prefix);
    into->gate_weight = h3_weight_load_bf16(store, gpu, name, 2, gate, error,
                                            sizeof(error));
    snprintf(name, sizeof(name), "%sto_gate_logits.bias", prefix);
    into->gate_bias = h3_weight_load_bf16(store, gpu, name, 1, heads, error,
                                          sizeof(error));
    require(into->query_norm && into->key_norm && into->gate_weight &&
            into->gate_bias, "cannot load an attention's dense weights");
}

static void free_attention(attention *which) {
    free_projection(&which->query); free_projection(&which->key);
    free_projection(&which->value); free_projection(&which->output);
    h3_gpu_tensor_free(which->query_norm); h3_gpu_tensor_free(which->key_norm);
    h3_gpu_tensor_free(which->gate_weight); h3_gpu_tensor_free(which->gate_bias);
}

/* --------------------------------------------------------------- modulation */

/* The block's static table added to every level of the module's output. The
 * table is per block and shared by all levels; the timestep part is what
 * varies, so the sum is [level][slot][width] -- the layout the row map
 * indexes. Built on the host and uploaded once per block.
 *
 * `timestep` is the whole [level][table_slots][width] output of the module,
 * from which `used_slots` are taken starting at `first_slot`; the two
 * cross-modal consumers split one five-row table between them that way. */
static h3_gpu_tensor *modulation_of(const char *block, const char *table_name,
                                    const float *timestep, uint32_t width,
                                    uint32_t table_slots, uint32_t first_slot,
                                    uint32_t used_slots, uint32_t module_slots,
                                    uint32_t levels) {
    char name[256];
    snprintf(name, sizeof(name), "%s%s", block, table_name);
    float *table = weight_of(name + strlen("model.diffusion_model."),
                             (size_t)table_slots * width);
    float *sum = malloc((size_t)levels * used_slots * width * sizeof(*sum));
    require(sum != NULL, "cannot allocate a modulation");
    for (uint32_t level = 0; level < levels; level++)
        for (uint32_t slot = 0; slot < used_slots; slot++)
            for (uint32_t index = 0; index < width; index++)
                sum[((size_t)level * used_slots + slot) * width + index] =
                    table[(size_t)(first_slot + slot) * width + index] +
                    timestep[((size_t)level * module_slots + slot) * width +
                             index];
    h3_gpu_tensor *tensor = upload(sum, (size_t)levels * used_slots * width);
    free(table);
    free(sum);
    return tensor;
}

/* The context modulation: the one affine with no norm in front of it, so the
 * AdaLN kernel cannot serve it. The context is a few hundred rows and this
 * runs once per block, which is nothing beside the block itself. */
static h3_gpu_tensor *modulate_context(const char *block, const char *table_name,
                                       const float *context,
                                       const float *timestep, uint32_t rows,
                                       uint32_t width) {
    char name[256];
    snprintf(name, sizeof(name), "%s%s", block, table_name);
    float *table = weight_of(name + strlen("model.diffusion_model."),
                             (size_t)PROMPT_SLOTS * width);
    float *out = malloc((size_t)rows * width * sizeof(*out));
    require(out != NULL, "cannot allocate a modulated context");
    for (uint32_t row = 0; row < rows; row++)
        for (uint32_t index = 0; index < width; index++) {
            const float shift = table[index] + timestep[index];
            const float scale = table[width + index] + timestep[width + index];
            out[(size_t)row * width + index] =
                context[(size_t)row * width + index] * (1.0f + scale) + shift;
        }
    h3_gpu_tensor *tensor = upload(out, (size_t)rows * width);
    free(table);
    free(out);
    return tensor;
}

/* ------------------------------------------------------------------- block */

typedef struct {
    h3_gpu_tensor *query, *key, *value, *heads, *logits;
    h3_gpu_tensor *rotated_query, *rotated_kv, *inner;
    h3_gpu_tensor *ones_video, *ones_audio, *row_map_video, *row_map_audio;
} scratch;

static void run_attention(const attention *a, h3_gpu_tensor *out,
                          const h3_gpu_tensor *query_input, uint32_t query_rows,
                          const h3_gpu_tensor *kv_input, uint32_t kv_rows,
                          const h3_gpu_tensor *query_cos,
                          const h3_gpu_tensor *query_sin,
                          const h3_gpu_tensor *key_cos,
                          const h3_gpu_tensor *key_sin, scratch *space) {
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
                                        a->query.bias, query_rows, a->query_in,
                                        a->inner), "query projection");
    GPU_OP(h3_gpu_linear_i8_weight_bf16(gpu, space->key, rotated_kv,
                                        a->key.weight, a->key.scales,
                                        a->key.bias, kv_rows, a->kv_in,
                                        a->inner), "key projection");
    GPU_OP(h3_gpu_linear_i8_weight_bf16(gpu, space->value, rotated_kv,
                                        a->value.weight, a->value.scales,
                                        a->value.bias, kv_rows, a->kv_in,
                                        a->inner), "value projection");
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

typedef struct {
    attention attn1, attn2, audio_attn1, audio_attn2, a2v, v2a;
    projection ff_in, ff_out, audio_ff_in, audio_ff_out;
    h3_gpu_tensor *video_modulation, *audio_modulation;
    h3_gpu_tensor *video_cross, *audio_cross, *video_gate, *audio_gate;
    h3_gpu_tensor *video_context, *audio_context;
} block_weights;

typedef struct {
    float *video_timestep, *audio_timestep;
    float *video_prompt, *audio_prompt;
    float *video_cross, *audio_cross, *video_gate, *audio_gate;
    float *video_embedded, *audio_embedded;
    float *video_context, *audio_context;
} conditioning;

static void load_block(int index, const conditioning *from,
                       block_weights *weights) {
    char block[160];
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
                                       AUDIO_DIM, AUDIO_DIM, AUDIO_DIM, {0},
                                       {0}, {0}, {0}, NULL, NULL, NULL, NULL};
    weights->audio_attn2 = (attention){"audio_attn2", AUDIO_DIM, AUDIO_DIM,
                                       AUDIO_DIM, AUDIO_DIM, AUDIO_DIM, {0},
                                       {0}, {0}, {0}, NULL, NULL, NULL, NULL};
    weights->a2v = (attention){"audio_to_video_attn", VIDEO_DIM, AUDIO_DIM,
                               AUDIO_DIM, VIDEO_DIM, VIDEO_DIM, {0}, {0}, {0},
                               {0}, NULL, NULL, NULL, NULL};
    weights->v2a = (attention){"video_to_audio_attn", AUDIO_DIM, VIDEO_DIM,
                               AUDIO_DIM, AUDIO_DIM, AUDIO_DIM, {0}, {0}, {0},
                               {0}, NULL, NULL, NULL, NULL};
    load_attention(block, &weights->attn1);
    load_attention(block, &weights->attn2);
    load_attention(block, &weights->audio_attn1);
    load_attention(block, &weights->audio_attn2);
    load_attention(block, &weights->a2v);
    load_attention(block, &weights->v2a);
    load_projection(block, "ff.net.0.proj", VIDEO_FF, VIDEO_DIM, 0, &weights->ff_in);
    load_projection(block, "ff.net.2", VIDEO_DIM, VIDEO_FF, 0, &weights->ff_out);
    load_projection(block, "audio_ff.net.0.proj", AUDIO_FF, AUDIO_DIM, 1,
                    &weights->audio_ff_in);
    load_projection(block, "audio_ff.net.2", AUDIO_DIM, AUDIO_FF, 1,
                    &weights->audio_ff_out);

    weights->video_modulation = modulation_of(
        block, "scale_shift_table", from->video_timestep, VIDEO_DIM,
        ADA_SLOTS, 0, ADA_SLOTS, ADA_SLOTS, LEVELS);
    weights->audio_modulation = modulation_of(
        block, "audio_scale_shift_table", from->audio_timestep, AUDIO_DIM,
        ADA_SLOTS, 0, ADA_SLOTS, ADA_SLOTS, LEVELS);
    weights->video_cross = modulation_of(
        block, "scale_shift_table_a2v_ca_video", from->video_cross, VIDEO_DIM,
        CROSS_TABLE_SLOTS, 0, CROSS_SLOTS, CROSS_SLOTS, LEVELS);
    weights->audio_cross = modulation_of(
        block, "scale_shift_table_a2v_ca_audio", from->audio_cross, AUDIO_DIM,
        CROSS_TABLE_SLOTS, 0, CROSS_SLOTS, CROSS_SLOTS, LEVELS);
    weights->video_gate = modulation_of(
        block, "scale_shift_table_a2v_ca_video", from->video_gate, VIDEO_DIM,
        CROSS_TABLE_SLOTS, CROSS_SLOTS, GATE_SLOTS, GATE_SLOTS, LEVELS);
    weights->audio_gate = modulation_of(
        block, "scale_shift_table_a2v_ca_audio", from->audio_gate, AUDIO_DIM,
        CROSS_TABLE_SLOTS, CROSS_SLOTS, GATE_SLOTS, GATE_SLOTS, LEVELS);
    weights->video_context = modulate_context(
        block, "prompt_scale_shift_table", from->video_context,
        from->video_prompt, TEXT_ROWS, VIDEO_DIM);
    weights->audio_context = modulate_context(
        block, "audio_prompt_scale_shift_table", from->audio_context,
        from->audio_prompt, TEXT_ROWS, AUDIO_DIM);
}

static void free_block(block_weights *weights) {
    free_attention(&weights->attn1); free_attention(&weights->attn2);
    free_attention(&weights->audio_attn1); free_attention(&weights->audio_attn2);
    free_attention(&weights->a2v); free_attention(&weights->v2a);
    free_projection(&weights->ff_in); free_projection(&weights->ff_out);
    free_projection(&weights->audio_ff_in); free_projection(&weights->audio_ff_out);
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

typedef struct { h3_gpu_tensor *cos, *sin; } rope;

typedef struct {
    h3_gpu_tensor *video, *audio, *video_pre, *audio_pre;
    h3_gpu_tensor *video_scaled, *audio_scaled;
    h3_gpu_tensor *video_branch, *audio_branch;
    rope video_pe, audio_pe, video_cross_pe, audio_cross_pe;
} state;

static void self_and_text(const attention *self, const attention *text,
                          h3_gpu_tensor *x, h3_gpu_tensor *scaled,
                          h3_gpu_tensor *branch,
                          const h3_gpu_tensor *modulation,
                          const h3_gpu_tensor *context,
                          const h3_gpu_tensor *ones,
                          const h3_gpu_tensor *row_map, const rope *pe,
                          uint32_t rows, uint32_t dim, scratch *space) {
    GPU_OP(h3_gpu_adaln_bf16(gpu, scaled, x, ones, modulation, row_map, rows,
                             dim, ADA_SLOTS, 0, 1, BLOCK_EPSILON),
           "self-attention modulation");
    run_attention(self, branch, scaled, rows, scaled, rows,
                  pe->cos, pe->sin, pe->cos, pe->sin, space);
    GPU_OP(h3_gpu_gate_bf16(gpu, x, x, branch, modulation, row_map, rows, dim,
                            ADA_SLOTS, 2), "self-attention residual");
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
    /* Snapshot both streams: video-into-audio must read the video stream as it
     * was before audio-into-video changed it. */
    GPU_OP(h3_gpu_copy_bf16(gpu, s->video_pre, 0, s->video, 0,
                            (size_t)VIDEO_ROWS * VIDEO_DIM), "video snapshot");
    GPU_OP(h3_gpu_copy_bf16(gpu, s->audio_pre, 0, s->audio, 0,
                            (size_t)AUDIO_ROWS * AUDIO_DIM), "audio snapshot");
    /* Scale then shift, the opposite order to the nine-row table's. */
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

/* -------------------------------------------------------------- the head */

/* LayerNorm -- the one norm here that subtracts a mean -- then the two-row
 * table plus the *same* embedded timestep in both rows, then the projection
 * back down to the latent width.
 *
 * The embedded timestep is per token here too: `_process_output` indexes it
 * `embedded_timestep[:, :, None]`, so a conditioning token's head modulation
 * is the one built at timestep zero, not the one the rest of the frame uses. */
static void output_head(const float *x, const float *table,
                        const float *embedded, const uint32_t *level_of,
                        const float *weight, const float *bias, float *out,
                        uint32_t rows, uint32_t dim) {
    float *scratch_row = malloc(dim * sizeof(*scratch_row));
    require(scratch_row != NULL, "cannot allocate a head row");
    for (uint32_t row = 0; row < rows; row++) {
        const float *source = x + (size_t)row * dim;
        const float *level = embedded + (size_t)level_of[row] * dim;
        double mean = 0.0;
        for (uint32_t index = 0; index < dim; index++) mean += source[index];
        mean /= (double)dim;
        double variance = 0.0;
        for (uint32_t index = 0; index < dim; index++) {
            const double centred = (double)source[index] - mean;
            variance += centred * centred;
        }
        const double inverse = 1.0 / sqrt(variance / (double)dim + 1e-6);
        for (uint32_t index = 0; index < dim; index++) {
            const float shift = table[index] + level[index];
            const float scale = table[dim + index] + level[index];
            scratch_row[index] =
                (float)(((double)source[index] - mean) * inverse) *
                (1.0f + scale) + shift;
        }
        linear_rows(out + (size_t)row * LATENT, scratch_row, weight, bias, 1,
                    dim, LATENT);
    }
    free(scratch_row);
}

/* ----------------------------------------------------------- the sampler */

/* `x - v * t`, at each token's own timestep. A token held at zero is returned
 * bit for bit, which is the whole of conditioning. */
static void to_denoised(float *out, const float *sample, const float *velocity,
                        const float *timestep, uint32_t rows, uint32_t width) {
    for (uint32_t row = 0; row < rows; row++)
        for (uint32_t index = 0; index < width; index++) {
            const size_t at = (size_t)row * width + index;
            out[at] = sample[at] - velocity[at] * timestep[row];
        }
}

/* The Euler step, on the *scalar* schedule sigma rather than the per-token
 * timestep -- the mask has already done its work, because a held token's
 * denoised prediction equals its sample and the velocity below is then zero.
 * The two are not interchangeable: `sigma` here is never zero, while a held
 * token's timestep always is. */
static void euler_step(float *sample, const float *denoised, double sigma,
                       double next, size_t count) {
    const double delta = next - sigma;
    for (size_t index = 0; index < count; index++) {
        const double velocity =
            ((double)sample[index] - (double)denoised[index]) / sigma;
        sample[index] = (float)((double)sample[index] + velocity * delta);
    }
}

/* ------------------------------------------------------------------- main */

/* Which noise quantity drives each module. Everything above is machinery; this
 * table is the finding. Note the last two rows: each gate reads the *cross*
 * stream's sigma, so the gate on audio-into-video asks how noisy the *audio*
 * is. Swap them and a run where both streams share a sigma agrees exactly. */
static const adaln_module MODULES[] = {
    {"adaln_single", VIDEO_DIM, ADA_SLOTS, VIDEO_TOKENS},
    {"audio_adaln_single", AUDIO_DIM, ADA_SLOTS, AUDIO_TOKENS},
    {"prompt_adaln_single", VIDEO_DIM, PROMPT_SLOTS, VIDEO_SIGMA},
    {"audio_prompt_adaln_single", AUDIO_DIM, PROMPT_SLOTS, AUDIO_SIGMA},
    {"av_ca_video_scale_shift_adaln_single", VIDEO_DIM, CROSS_SLOTS, VIDEO_TOKENS},
    {"av_ca_audio_scale_shift_adaln_single", AUDIO_DIM, CROSS_SLOTS, AUDIO_TOKENS},
    {"av_ca_a2v_gate_adaln_single", VIDEO_DIM, GATE_SLOTS, AUDIO_SIGMA},
    {"av_ca_v2a_gate_adaln_single", AUDIO_DIM, GATE_SLOTS, VIDEO_SIGMA}
};


int main(int argc, char **argv) {
    if (argc < 3 || argc > 4) {
        fprintf(stderr, "usage: %s DIT.safetensors ANCHOR.safetensors [BLOCKS]\n",
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
    const int blocks = argc > 3 ? atoi(argv[3]) : ANCHOR_BLOCKS;
    if (blocks < 1 || blocks > TOTAL_BLOCKS)
        fail("block count %d is outside 1..%d", blocks, TOTAL_BLOCKS);

    /* ----------------------------------------------------------- the fixture */

    float *schedule = golden("sigmas", STEPS + 1);
    float *video_latent = golden("video_latent", (size_t)VIDEO_ROWS * LATENT);
    float *audio_latent = golden("audio_latent", (size_t)AUDIO_ROWS * LATENT);
    float *video_mask = golden("video_denoise_mask", VIDEO_ROWS);
    float *audio_mask = golden("audio_denoise_mask", AUDIO_ROWS);
    float *keyframes = golden("keyframes_mask", VIDEO_ROWS);
    float *video_context = golden("video_context", (size_t)TEXT_ROWS * VIDEO_DIM);
    float *audio_context = golden("audio_context", (size_t)TEXT_ROWS * AUDIO_DIM);

    /* The mask decides which modulation row each token reads, so it also has to
     * be a value this many rows can express. Binary is what LTX's conditioning
     * builders produce; anything else would need a row per distinct value. */
    uint32_t video_level[VIDEO_ROWS], audio_level[AUDIO_ROWS];
    int video_frozen = 1, audio_frozen = 1;
    for (int token = 0; token < VIDEO_ROWS; token++) {
        if (video_mask[token] != 0.0f && video_mask[token] != 1.0f)
            fail("the video mask is %f at %d, which is neither held nor free",
                 video_mask[token], token);
        video_level[token] = video_mask[token] != 0.0f ? 1u : 0u;
        if (video_level[token]) video_frozen = 0;
    }
    for (int token = 0; token < AUDIO_ROWS; token++) {
        if (audio_mask[token] != 0.0f && audio_mask[token] != 1.0f)
            fail("the audio mask is %f at %d", audio_mask[token], token);
        audio_level[token] = audio_mask[token] != 0.0f ? 1u : 0u;
        if (audio_level[token]) audio_frozen = 0;
    }

    /* A frozen stream's scalar noise level is zero whatever the schedule says.
     * Freezing and an all-zero mask are the same condition, which is why one
     * flag drives both here. */
    double video_sigma[STEPS], audio_sigma[STEPS];
    int sigmas_differ = 0, some_level_differs = 0;
    for (int step = 0; step < STEPS; step++) {
        video_sigma[step] = video_frozen ? 0.0 : (double)schedule[step];
        audio_sigma[step] = audio_frozen ? 0.0 : (double)schedule[step];
        if (video_sigma[step] != audio_sigma[step]) sigmas_differ = 1;
        if (video_sigma[step] != 0.0 || audio_sigma[step] != 0.0)
            some_level_differs = 1;
    }
    /* Two things this fixture has to actually switch on, checked rather than
     * assumed -- a fixture that fails to exercise a branch passes every
     * mutation of it. */
    require(sigmas_differ, "the two streams share a sigma at every step, so "
            "swapping the cross-modal gates would agree exactly");
    require(some_level_differs, "every timestep is zero, so the two modulation "
            "levels are identical and the row map is untested");
    printf("LTX-2.5 DiT, %d steps, %d of %d blocks; video %d tokens "
           "(%d held), audio %d (%s):\n", STEPS, blocks, TOTAL_BLOCKS,
           VIDEO_ROWS, VIDEO_ROWS - (int)(VIDEO_ROWS - CONDITIONED),
           AUDIO_ROWS, audio_frozen ? "frozen" : "denoised");
    printf("  schedule");
    for (int step = 0; step <= STEPS; step++) printf(" %.4f", schedule[step]);
    printf("\n");
    const double began = now();

    /* ------------------------------------- every module at every timestep */

    /* One pass over each module's weights covers the whole run: the timesteps
     * it will ever be asked for are known up front, and its projection is up
     * to 4096x36864, which is not worth reading four times. */
    float *modulation[8] = {0}, *embedded[8] = {0};
    for (int index = 0; index < 8; index++) {
        double values[STEPS * LEVELS];
        for (int step = 0; step < STEPS; step++)
            for (int level = 0; level < LEVELS; level++) {
                double value = 0.0;
                switch (MODULES[index].source) {
                /* sigma * mask, and the mask is the level */
                case VIDEO_TOKENS: value = level ? video_sigma[step] : 0.0; break;
                case AUDIO_TOKENS: value = level ? audio_sigma[step] : 0.0; break;
                /* Scalar: both levels hold the same row, so whichever the row
                 * map picks is the same answer. */
                case VIDEO_SIGMA: value = video_sigma[step]; break;
                case AUDIO_SIGMA: value = audio_sigma[step]; break;
                }
                values[step * LEVELS + level] = value;
            }
        run_adaln(&MODULES[index], values, STEPS * LEVELS, TIMESTEP_MULTIPLIER,
                  &modulation[index], &embedded[index]);
    }

    /* Each module's output held against the reference's, directly.
     *
     * This has to be direct rather than inferred from the end of the pipeline.
     * Measured on the reference (`gen_ltx_dit_sample.py --probe`), swapping
     * the two cross-modal gates moves the final velocity by 4.7e-03 of peak
     * and driving the cross scale-shift from the scalar sigma moves it by
     * 1.3e-05 -- against an engine whose own BF16 deviation is 1.7e-02. Both
     * are invisible downstream at any tolerance that the honest run passes,
     * because the cross-modal branch is a small correction on the residual.
     * Here they are total: a module fed the wrong quantity produces an
     * unrelated tensor, and the bound below is the BF16 rounding of one
     * matrix-vector product rather than a whole pipeline's accumulation. */
    static const char *const TAGS[8] = {
        "video_ada", "audio_ada", "video_prompt", "audio_prompt",
        "video_cross", "audio_cross", "video_gate", "audio_gate"
    };
    for (int step = 0; step < STEPS; step++) {
        for (int index = 0; index < 8; index++) {
            const size_t span = (size_t)MODULES[index].slots *
                                MODULES[index].width;
            char name[64], label[64];
            snprintf(name, sizeof(name), "step%d_%s", step, TAGS[index]);
            float *expected = golden(name, LEVELS * span);
            snprintf(label, sizeof(label), "step %d %s", step, TAGS[index]);
            compare(label, modulation[index] + (size_t)step * LEVELS * span,
                    expected, LEVELS * span, MODULATION_TOLERANCE);
            free(expected);
        }
        for (int stream = 0; stream < 2; stream++) {
            const size_t width = stream ? AUDIO_DIM : VIDEO_DIM;
            char name[64], label[64];
            snprintf(name, sizeof(name), "step%d_%s_embedded", step,
                     stream ? "audio" : "video");
            float *expected = golden(name, LEVELS * width);
            snprintf(label, sizeof(label), "step %d %s embedded", step,
                     stream ? "audio" : "video");
            compare(label, embedded[stream] + (size_t)step * LEVELS * width,
                    expected, LEVELS * width, MODULATION_TOLERANCE);
            free(expected);
        }
    }

    /* ------------------------------------------------- weights held across */

    float *video_patch_weight = weight_of("patchify_proj.weight",
                                          (size_t)VIDEO_DIM * LATENT);
    float *video_patch_bias = weight_of("patchify_proj.bias", VIDEO_DIM);
    float *audio_patch_weight = weight_of("audio_patchify_proj.weight",
                                          (size_t)AUDIO_DIM * LATENT);
    float *audio_patch_bias = weight_of("audio_patchify_proj.bias", AUDIO_DIM);
    float *keyframe_marker = weight_of("keyframes_abs_pos_embedding", VIDEO_DIM);
    float *video_head_table = weight_of("scale_shift_table",
                                        (size_t)HEAD_SLOTS * VIDEO_DIM);
    float *audio_head_table = weight_of("audio_scale_shift_table",
                                        (size_t)HEAD_SLOTS * AUDIO_DIM);
    float *video_head_weight = weight_of("proj_out.weight",
                                         (size_t)LATENT * VIDEO_DIM);
    float *video_head_bias = weight_of("proj_out.bias", LATENT);
    float *audio_head_weight = weight_of("audio_proj_out.weight",
                                         (size_t)LATENT * AUDIO_DIM);
    float *audio_head_bias = weight_of("audio_proj_out.bias", LATENT);

    /* ------------------------------------------------------------- rope */

    state s;
    memset(&s, 0, sizeof(s));
    {
        float *video_grid = malloc((size_t)VIDEO_AXES * VIDEO_ROWS * 2 *
                                   sizeof(*video_grid));
        require(video_grid != NULL, "cannot allocate the video grid");
        for (int token = 0; token < VIDEO_ROWS; token++) {
            const int frame = token / (HEIGHT * WIDTH);
            const int rest = token % (HEIGHT * WIDTH);
            const int starts[VIDEO_AXES] = {frame, rest / WIDTH, rest % WIDTH};
            for (int axis = 0; axis < VIDEO_AXES; axis++) {
                const size_t at = ((size_t)axis * VIDEO_ROWS + token) * 2;
                video_grid[at] = (float)starts[axis];
                video_grid[at + 1] = (float)(starts[axis] + 1);
            }
        }
        float *audio_grid = malloc((size_t)AUDIO_ROWS * 2 * sizeof(*audio_grid));
        require(audio_grid != NULL, "cannot allocate the audio grid");
        for (int token = 0; token < AUDIO_ROWS; token++) {
            audio_grid[(size_t)token * 2] = (float)token;
            audio_grid[(size_t)token * 2 + 1] = (float)(token + 1);
        }
        rope_tables(&s.video_pe.cos, &s.video_pe.sin, video_grid, VIDEO_ROWS,
                    VIDEO_DIM, VIDEO_AXES, VIDEO_MAX_POS);
        rope_tables(&s.audio_pe.cos, &s.audio_pe.sin, audio_grid, AUDIO_ROWS,
                    AUDIO_DIM, AUDIO_AXES, AUDIO_MAX_POS);
        rope_tables(&s.video_cross_pe.cos, &s.video_cross_pe.sin, video_grid,
                    VIDEO_ROWS, AUDIO_DIM, VIDEO_AXES, VIDEO_MAX_POS);
        rope_tables(&s.audio_cross_pe.cos, &s.audio_cross_pe.sin, audio_grid,
                    AUDIO_ROWS, AUDIO_DIM, AUDIO_AXES, AUDIO_MAX_POS);
        free(video_grid); free(audio_grid);
    }

    /* ----------------------------------------------------------- scratch */

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
    /* The row map is the denoise mask: a held token reads the modulation built
     * at timestep zero, a free one the modulation built at this step's sigma. */
    space.row_map_video = h3_gpu_tensor_from_u32(gpu, video_level, VIDEO_ROWS);
    space.row_map_audio = h3_gpu_tensor_from_u32(gpu, audio_level, AUDIO_ROWS);
    require(space.ones_video && space.ones_audio && space.row_map_video &&
            space.row_map_audio, "cannot allocate modulation helpers");

    s.video_pre = h3_gpu_tensor_new_bf16(gpu, (size_t)VIDEO_ROWS * VIDEO_DIM);
    s.audio_pre = h3_gpu_tensor_new_bf16(gpu, (size_t)AUDIO_ROWS * AUDIO_DIM);
    s.video_scaled = h3_gpu_tensor_new_bf16(gpu, (size_t)VIDEO_ROWS * VIDEO_DIM);
    s.audio_scaled = h3_gpu_tensor_new_bf16(gpu, (size_t)AUDIO_ROWS * AUDIO_DIM);
    s.video_branch = h3_gpu_tensor_new_bf16(gpu, (size_t)VIDEO_ROWS * VIDEO_DIM);
    s.audio_branch = h3_gpu_tensor_new_bf16(gpu, (size_t)AUDIO_ROWS * AUDIO_DIM);
    require(s.video_pre && s.audio_pre && s.video_scaled && s.audio_scaled &&
            s.video_branch && s.audio_branch, "cannot allocate block state");

    /* ------------------------------------------------------------ the loop */

    float *video_start = malloc((size_t)VIDEO_ROWS * LATENT * sizeof(float));
    float *audio_start = malloc((size_t)AUDIO_ROWS * LATENT * sizeof(float));
    float *video_tokens = malloc((size_t)VIDEO_ROWS * VIDEO_DIM * sizeof(float));
    float *audio_tokens = malloc((size_t)AUDIO_ROWS * AUDIO_DIM * sizeof(float));
    float *video_out = malloc((size_t)VIDEO_ROWS * VIDEO_DIM * sizeof(float));
    float *audio_out = malloc((size_t)AUDIO_ROWS * AUDIO_DIM * sizeof(float));
    float *video_velocity = malloc((size_t)VIDEO_ROWS * LATENT * sizeof(float));
    float *audio_velocity = malloc((size_t)AUDIO_ROWS * LATENT * sizeof(float));
    float *video_denoised = malloc((size_t)VIDEO_ROWS * LATENT * sizeof(float));
    float *audio_denoised = malloc((size_t)AUDIO_ROWS * LATENT * sizeof(float));
    require(video_start && audio_start && video_tokens && audio_tokens &&
            video_out && audio_out && video_velocity && audio_velocity &&
            video_denoised && audio_denoised, "cannot allocate the loop state");
    memcpy(video_start, video_latent, (size_t)VIDEO_ROWS * LATENT * sizeof(float));
    memcpy(audio_start, audio_latent, (size_t)AUDIO_ROWS * LATENT * sizeof(float));

    double block_seconds = 0.0, load_seconds = 0.0;
    for (int step = 0; step < STEPS; step++) {
        conditioning cond;
        memset(&cond, 0, sizeof(cond));
        float *slice[8];
        for (int index = 0; index < 8; index++)
            slice[index] = modulation[index] +
                (size_t)step * LEVELS * MODULES[index].slots * MODULES[index].width;
        cond.video_timestep = slice[0];
        cond.audio_timestep = slice[1];
        cond.video_prompt = slice[2];
        cond.audio_prompt = slice[3];
        cond.video_cross = slice[4];
        cond.audio_cross = slice[5];
        cond.video_gate = slice[6];
        cond.audio_gate = slice[7];
        cond.video_embedded = embedded[0] + (size_t)step * LEVELS * VIDEO_DIM;
        cond.audio_embedded = embedded[1] + (size_t)step * LEVELS * AUDIO_DIM;
        cond.video_context = video_context;
        cond.audio_context = audio_context;

        /* ------------------------------------------------------- patchify */

        linear_rows(video_tokens, video_latent, video_patch_weight,
                    video_patch_bias, VIDEO_ROWS, LATENT, VIDEO_DIM);
        for (int row = 0; row < VIDEO_ROWS; row++)
            if (keyframes[row] > 0.0f)
                for (int index = 0; index < VIDEO_DIM; index++)
                    video_tokens[(size_t)row * VIDEO_DIM + index] +=
                        keyframe_marker[index];
        linear_rows(audio_tokens, audio_latent, audio_patch_weight,
                    audio_patch_bias, AUDIO_ROWS, LATENT, AUDIO_DIM);
        s.video = upload(video_tokens, (size_t)VIDEO_ROWS * VIDEO_DIM);
        s.audio = upload(audio_tokens, (size_t)AUDIO_ROWS * AUDIO_DIM);

        /* --------------------------------------------------------- blocks */

        for (int index = 0; index < blocks; index++) {
            block_weights weights;
            const double loading = now();
            load_block(index, &cond, &weights);
            load_seconds += now() - loading;
            const double running = now();
            GPU_OP(h3_gpu_begin(gpu), "begin block");
            run_block(&weights, &s, &space);
            GPU_OP(h3_gpu_submit(gpu), "submit block");
            block_seconds += now() - running;
            free_block(&weights);
        }

        /* ----------------------------------------------------- the head */

        read_bf16_as_f32(s.video, video_out, (size_t)VIDEO_ROWS * VIDEO_DIM);
        read_bf16_as_f32(s.audio, audio_out, (size_t)AUDIO_ROWS * AUDIO_DIM);
        output_head(video_out, video_head_table, cond.video_embedded,
                    video_level, video_head_weight, video_head_bias,
                    video_velocity, VIDEO_ROWS, VIDEO_DIM);
        output_head(audio_out, audio_head_table, cond.audio_embedded,
                    audio_level, audio_head_weight, audio_head_bias,
                    audio_velocity, AUDIO_ROWS, AUDIO_DIM);
        h3_gpu_tensor_free(s.video); s.video = NULL;
        h3_gpu_tensor_free(s.audio); s.audio = NULL;

        /* --------------------------------------------------- the sampler */

        float video_timestep[VIDEO_ROWS], audio_timestep[AUDIO_ROWS];
        for (int token = 0; token < VIDEO_ROWS; token++)
            video_timestep[token] = (float)(video_sigma[step] * video_mask[token]);
        for (int token = 0; token < AUDIO_ROWS; token++)
            audio_timestep[token] = (float)(audio_sigma[step] * audio_mask[token]);
        to_denoised(video_denoised, video_latent, video_velocity,
                    video_timestep, VIDEO_ROWS, LATENT);
        to_denoised(audio_denoised, audio_latent, audio_velocity,
                    audio_timestep, AUDIO_ROWS, LATENT);
        euler_step(video_latent, video_denoised, schedule[step],
                   schedule[step + 1], (size_t)VIDEO_ROWS * LATENT);
        euler_step(audio_latent, audio_denoised, schedule[step],
                   schedule[step + 1], (size_t)AUDIO_ROWS * LATENT);

        /* ------------------------------------------------------- checks */

        /* The audio *latent* comparison below is close to a tautology -- both
         * sides leave a frozen track alone, so both report zero. It is the
         * audio *velocity* that checks the audio path really ran, and it is
         * the one that carries a real deviation. */
        char name[64], label[64];
        if (blocks == ANCHOR_BLOCKS) {
            snprintf(name, sizeof(name), "step%d_video_velocity", step);
            float *expected = golden(name, (size_t)VIDEO_ROWS * LATENT);
            /* Held and free tokens compared against their *own* peaks. Taking
             * one peak across all 32 dilutes anything that only touches the 16
             * held ones -- and the head's per-token modulation is exactly
             * that. Measured on the reference, ignoring it moves the held
             * tokens by 7.4e-03 of their own peak and the whole tensor by
             * 4.5e-03, which the single bound below would not resolve. */
            snprintf(label, sizeof(label), "step %d velocity, held", step);
            compare(label, video_velocity, expected,
                    (size_t)CONDITIONED * LATENT, HELD_TOLERANCE);
            snprintf(label, sizeof(label), "step %d velocity, free", step);
            compare(label, video_velocity + (size_t)CONDITIONED * LATENT,
                    expected + (size_t)CONDITIONED * LATENT,
                    (size_t)(VIDEO_ROWS - CONDITIONED) * LATENT, TOLERANCE);
            free(expected);
            snprintf(name, sizeof(name), "step%d_audio_velocity", step);
            snprintf(label, sizeof(label), "step %d audio velocity", step);
            expected = golden(name, (size_t)AUDIO_ROWS * LATENT);
            /* Frozen, so its input never changes and its deviation does
             * not accumulate either. */
            compare(label, audio_velocity, expected,
                    (size_t)AUDIO_ROWS * LATENT, HELD_TOLERANCE);
            free(expected);
            snprintf(name, sizeof(name), "step%d_video_latent", step);
            snprintf(label, sizeof(label), "step %d video latent", step);
            expected = golden(name, (size_t)VIDEO_ROWS * LATENT);
            compare(label, video_latent, expected,
                    (size_t)VIDEO_ROWS * LATENT, TOLERANCE);
            free(expected);
            snprintf(name, sizeof(name), "step%d_audio_latent", step);
            snprintf(label, sizeof(label), "step %d audio latent", step);
            expected = golden(name, (size_t)AUDIO_ROWS * LATENT);
            compare(label, audio_latent, expected,
                    (size_t)AUDIO_ROWS * LATENT, TOLERANCE);
            free(expected);
        }

        /* Held tokens must be returned bit for bit, not merely close: the
         * arithmetic that produces them is `x - v * 0`, which is exact for
         * every finite v. Anything else means the timestep reaching them was
         * not zero. */
        double moved = 0.0, free_moved = 0.0;
        for (int token = 0; token < VIDEO_ROWS; token++)
            for (int index = 0; index < LATENT; index++) {
                const size_t at = (size_t)token * LATENT + index;
                const double delta = fabs((double)video_latent[at] -
                                          (double)video_start[at]);
                if (video_mask[token] == 0.0f) { if (delta > moved) moved = delta; }
                else if (delta > free_moved) free_moved = delta;
            }
        for (size_t at = 0; at < (size_t)AUDIO_ROWS * LATENT; at++) {
            const double delta = fabs((double)audio_latent[at] -
                                      (double)audio_start[at]);
            if (audio_frozen && delta > moved) moved = delta;
        }
        if (moved != 0.0) {
            fprintf(stderr, "FAIL step %d: a held token moved by %.3e; its "
                    "timestep was not zero\n", step, moved);
            failures++;
        } else if (free_moved == 0.0) {
            fprintf(stderr, "FAIL step %d: no free token moved, so the step "
                    "did nothing\n", step);
            failures++;
        } else {
            printf("  ok  %-24s held 0, free moved %.3f\n",
                   "conditioning survived", free_moved);
        }
    }

    printf("\n%d steps of %d blocks: %.2f s in blocks, %.2f s loading "
           "weights, %.2f s total\n", STEPS, blocks, block_seconds,
           load_seconds, now() - began);

    /* --------------------------------------------------------------- free */

    for (int index = 0; index < 8; index++) {
        free(modulation[index]);
        free(embedded[index]);
    }
    free(schedule); free(video_latent); free(audio_latent);
    free(video_mask); free(audio_mask); free(keyframes);
    free(video_context); free(audio_context);
    free(video_patch_weight); free(video_patch_bias);
    free(audio_patch_weight); free(audio_patch_bias); free(keyframe_marker);
    free(video_head_table); free(audio_head_table);
    free(video_head_weight); free(video_head_bias);
    free(audio_head_weight); free(audio_head_bias);
    free(video_start); free(audio_start);
    free(video_tokens); free(audio_tokens);
    free(video_out); free(audio_out);
    free(video_velocity); free(audio_velocity);
    free(video_denoised); free(audio_denoised);
    h3_gpu_free(gpu);
    h3_weight_store_free(store);
    h3_weight_store_free(anchor);
    if (failures) {
        fprintf(stderr, "\n%d comparisons failed\n", failures);
        return 1;
    }
    printf("ok: %d denoising steps over %d blocks -- per-token timesteps, the "
           "velocity-to-denoised conversion and the Euler step, with the "
           "conditioning frame and the frozen audio track untouched\n",
           STEPS, blocks);
    return 0;
}
