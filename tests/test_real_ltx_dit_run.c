/* LTX-2.5's DiT end to end: latents and a sigma to a velocity prediction.
 *
 * The block and the top level each have their own anchor. This is the one that
 * covers what neither can -- the seams between them. A block's modulation
 * comes from the eight AdaLN modules, its rotary tables from a positions grid,
 * and its input from patchify, and each of those hand-offs is a place to be
 * consistently wrong in a way that looks right from both sides.
 *
 * Anchored on two blocks against LTX's own, then run at the full 48. Only the
 * loop bound differs, so what the long run adds is depth and a timing, not new
 * arithmetic. Weights load and free a block at a time, so this holds about
 * 400 MB rather than the tower's 18.5 GB.
 *
 * With a scalar sigma every token shares one modulation row, which is what the
 * reference broadcasts and what the row map here expresses. Per-token
 * timesteps would widen those buffers and change nothing else.
 *
 * usage: h3_real_ltx_dit_run_test DIT.safetensors ANCHOR.safetensors [BLOCKS] */

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
    AUDIO_AXES = 1
};

#define BLOCK_EPSILON 1e-6f
#define TIMESTEP_MULTIPLIER 1000.0
#define MAX_PERIOD 10000.0

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

typedef struct { const char *name; uint32_t width, slots; } adaln_module;

static void run_adaln(const adaln_module *module, double sigma,
                      double multiplier, float **modulation, float **embedded) {
    char name[224];
    float features[TIMESTEP_FEATURES];
    sinusoid(features, sigma * multiplier);
    snprintf(name, sizeof(name), "%s.emb.timestep_embedder.linear_1.weight",
             module->name);
    float *first = weight_of(name, (size_t)module->width * TIMESTEP_FEATURES);
    snprintf(name, sizeof(name), "%s.emb.timestep_embedder.linear_1.bias",
             module->name);
    float *first_bias = weight_of(name, module->width);
    float *hidden = malloc(module->width * sizeof(*hidden));
    require(hidden != NULL, "cannot allocate a timestep hidden state");
    linear_rows(hidden, features, first, first_bias, 1, TIMESTEP_FEATURES,
                module->width);
    for (uint32_t index = 0; index < module->width; index++)
        hidden[index] = silu(hidden[index]);
    free(first); free(first_bias);
    snprintf(name, sizeof(name), "%s.emb.timestep_embedder.linear_2.weight",
             module->name);
    float *second = weight_of(name, (size_t)module->width * module->width);
    snprintf(name, sizeof(name), "%s.emb.timestep_embedder.linear_2.bias",
             module->name);
    float *second_bias = weight_of(name, module->width);
    *embedded = malloc(module->width * sizeof(**embedded));
    require(*embedded != NULL, "cannot allocate an embedded timestep");
    linear_rows(*embedded, hidden, second, second_bias, 1, module->width,
                module->width);
    free(second); free(second_bias); free(hidden);
    float *activated = malloc(module->width * sizeof(*activated));
    require(activated != NULL, "cannot allocate the activated timestep");
    for (uint32_t index = 0; index < module->width; index++)
        activated[index] = silu((*embedded)[index]);
    snprintf(name, sizeof(name), "%s.linear.weight", module->name);
    float *projection = weight_of(
        name, (size_t)module->slots * module->width * module->width);
    snprintf(name, sizeof(name), "%s.linear.bias", module->name);
    float *projection_bias = weight_of(name, (size_t)module->slots * module->width);
    *modulation = malloc((size_t)module->slots * module->width * sizeof(**modulation));
    require(*modulation != NULL, "cannot allocate a modulation");
    linear_rows(*modulation, activated, projection, projection_bias, 1,
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

/* One row, broadcast to every token by the row map. The table is F32 in the
 * checkpoint and the module's output is one vector, so their sum is built on
 * the host and uploaded once per block. */
static h3_gpu_tensor *modulation_of(const char *block, const char *table_name,
                                    const float *timestep, uint32_t width,
                                    uint32_t table_slots, uint32_t first_slot,
                                    uint32_t used_slots) {
    char name[256];
    snprintf(name, sizeof(name), "%s%s", block, table_name);
    float *table = weight_of(name + strlen("model.diffusion_model."),
                             (size_t)table_slots * width);
    float *sum = malloc((size_t)used_slots * width * sizeof(*sum));
    require(sum != NULL, "cannot allocate a modulation");
    for (uint32_t slot = 0; slot < used_slots; slot++)
        for (uint32_t index = 0; index < width; index++)
            sum[(size_t)slot * width + index] =
                table[(size_t)(first_slot + slot) * width + index] +
                timestep[(size_t)slot * width + index];
    h3_gpu_tensor *tensor = upload(sum, (size_t)used_slots * width);
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
        ADA_SLOTS, 0, ADA_SLOTS);
    weights->audio_modulation = modulation_of(
        block, "audio_scale_shift_table", from->audio_timestep, AUDIO_DIM,
        ADA_SLOTS, 0, ADA_SLOTS);
    weights->video_cross = modulation_of(
        block, "scale_shift_table_a2v_ca_video", from->video_cross, VIDEO_DIM,
        CROSS_TABLE_SLOTS, 0, CROSS_SLOTS);
    weights->audio_cross = modulation_of(
        block, "scale_shift_table_a2v_ca_audio", from->audio_cross, AUDIO_DIM,
        CROSS_TABLE_SLOTS, 0, CROSS_SLOTS);
    weights->video_gate = modulation_of(
        block, "scale_shift_table_a2v_ca_video", from->video_gate, VIDEO_DIM,
        CROSS_TABLE_SLOTS, CROSS_SLOTS, GATE_SLOTS);
    weights->audio_gate = modulation_of(
        block, "scale_shift_table_a2v_ca_audio", from->audio_gate, AUDIO_DIM,
        CROSS_TABLE_SLOTS, CROSS_SLOTS, GATE_SLOTS);
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
 * back down to the latent width. */
static void output_head(const float *x, const float *table,
                        const float *embedded, const float *weight,
                        const float *bias, float *out, uint32_t rows,
                        uint32_t dim) {
    float *scratch_row = malloc(dim * sizeof(*scratch_row));
    require(scratch_row != NULL, "cannot allocate a head row");
    for (uint32_t row = 0; row < rows; row++) {
        const float *source = x + (size_t)row * dim;
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
            const float shift = table[index] + embedded[index];
            const float scale = table[dim + index] + embedded[index];
            scratch_row[index] =
                (float)(((double)source[index] - mean) * inverse) *
                (1.0f + scale) + shift;
        }
        linear_rows(out + (size_t)row * LATENT, scratch_row, weight, bias, 1,
                    dim, LATENT);
    }
    free(scratch_row);
}

/* ------------------------------------------------------------------- main */

static const adaln_module MODULES[] = {
    {"adaln_single", VIDEO_DIM, ADA_SLOTS},
    {"audio_adaln_single", AUDIO_DIM, ADA_SLOTS},
    {"prompt_adaln_single", VIDEO_DIM, PROMPT_SLOTS},
    {"audio_prompt_adaln_single", AUDIO_DIM, PROMPT_SLOTS},
    {"av_ca_video_scale_shift_adaln_single", VIDEO_DIM, CROSS_SLOTS},
    {"av_ca_audio_scale_shift_adaln_single", AUDIO_DIM, CROSS_SLOTS},
    {"av_ca_a2v_gate_adaln_single", VIDEO_DIM, GATE_SLOTS},
    {"av_ca_v2a_gate_adaln_single", AUDIO_DIM, GATE_SLOTS}
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

    float *sigma_values = golden("sigma", 1);
    const double sigma = sigma_values[0];
    free(sigma_values);
    printf("LTX-2.5 DiT, %d of %d blocks, %d video tokens (%dx%dx%d), %d audio, "
           "%d context, sigma %.4f:\n", blocks, TOTAL_BLOCKS, VIDEO_ROWS,
           FRAMES, HEIGHT, WIDTH, AUDIO_ROWS, TEXT_ROWS, sigma);
    const double began = now();

    /* --------------------------------------------- the eight AdaLN modules */

    conditioning cond;
    memset(&cond, 0, sizeof(cond));
    float *embedded[8] = {0};
    float *modulation[8] = {0};
    for (int index = 0; index < 8; index++) {
        /* The cross-modal modules carry their own multiplier; the released
         * config sets it to 1000, the same as the rest. */
        run_adaln(&MODULES[index], sigma, TIMESTEP_MULTIPLIER,
                  &modulation[index], &embedded[index]);
    }
    cond.video_timestep = modulation[0];
    cond.audio_timestep = modulation[1];
    cond.video_prompt = modulation[2];
    cond.audio_prompt = modulation[3];
    cond.video_cross = modulation[4];
    cond.audio_cross = modulation[5];
    cond.video_gate = modulation[6];
    cond.audio_gate = modulation[7];
    cond.video_embedded = embedded[0];
    cond.audio_embedded = embedded[1];
    cond.video_context = golden("video_context", (size_t)TEXT_ROWS * VIDEO_DIM);
    cond.audio_context = golden("audio_context", (size_t)TEXT_ROWS * AUDIO_DIM);

    /* --------------------------------------------------------- patchify */

    float *video_latent = golden("video_latent", (size_t)VIDEO_ROWS * LATENT);
    float *audio_latent = golden("audio_latent", (size_t)AUDIO_ROWS * LATENT);
    float *keyframes = golden("keyframes_mask", VIDEO_ROWS);
    float *video_tokens = malloc((size_t)VIDEO_ROWS * VIDEO_DIM * sizeof(float));
    float *audio_tokens = malloc((size_t)AUDIO_ROWS * AUDIO_DIM * sizeof(float));
    require(video_tokens && audio_tokens, "cannot allocate the patchified state");
    {
        float *weight = weight_of("patchify_proj.weight",
                                  (size_t)VIDEO_DIM * LATENT);
        float *bias = weight_of("patchify_proj.bias", VIDEO_DIM);
        linear_rows(video_tokens, video_latent, weight, bias, VIDEO_ROWS,
                    LATENT, VIDEO_DIM);
        free(weight); free(bias);
        float *marker = weight_of("keyframes_abs_pos_embedding", VIDEO_DIM);
        for (int row = 0; row < VIDEO_ROWS; row++)
            if (keyframes[row] > 0.0f)
                for (int index = 0; index < VIDEO_DIM; index++)
                    video_tokens[(size_t)row * VIDEO_DIM + index] += marker[index];
        free(marker);
        weight = weight_of("audio_patchify_proj.weight",
                           (size_t)AUDIO_DIM * LATENT);
        bias = weight_of("audio_patchify_proj.bias", AUDIO_DIM);
        linear_rows(audio_tokens, audio_latent, weight, bias, AUDIO_ROWS,
                    LATENT, AUDIO_DIM);
        free(weight); free(bias);
    }

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
        /* Both cross-modal directions run at the audio head dim, so the video
         * side of that pair is built at the audio width from the same grid. */
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
    /* Every token reads modulation row zero: with a scalar sigma there is only
     * one, which is exactly the broadcast the reference does. */
    uint32_t *zeros = calloc(VIDEO_ROWS, sizeof(*zeros));
    require(zeros != NULL, "cannot allocate a row map");
    space.row_map_video = h3_gpu_tensor_from_u32(gpu, zeros, VIDEO_ROWS);
    space.row_map_audio = h3_gpu_tensor_from_u32(gpu, zeros, AUDIO_ROWS);
    free(zeros);
    require(space.ones_video && space.ones_audio && space.row_map_video &&
            space.row_map_audio, "cannot allocate modulation helpers");

    s.video = upload(video_tokens, (size_t)VIDEO_ROWS * VIDEO_DIM);
    s.audio = upload(audio_tokens, (size_t)AUDIO_ROWS * AUDIO_DIM);
    s.video_pre = h3_gpu_tensor_new_bf16(gpu, (size_t)VIDEO_ROWS * VIDEO_DIM);
    s.audio_pre = h3_gpu_tensor_new_bf16(gpu, (size_t)AUDIO_ROWS * AUDIO_DIM);
    s.video_scaled = h3_gpu_tensor_new_bf16(gpu, (size_t)VIDEO_ROWS * VIDEO_DIM);
    s.audio_scaled = h3_gpu_tensor_new_bf16(gpu, (size_t)AUDIO_ROWS * AUDIO_DIM);
    s.video_branch = h3_gpu_tensor_new_bf16(gpu, (size_t)VIDEO_ROWS * VIDEO_DIM);
    s.audio_branch = h3_gpu_tensor_new_bf16(gpu, (size_t)AUDIO_ROWS * AUDIO_DIM);
    require(s.video_pre && s.audio_pre && s.video_scaled && s.audio_scaled &&
            s.video_branch && s.audio_branch, "cannot allocate block state");

    /* ---------------------------------------------------------- the blocks */

    float *video_out = malloc((size_t)VIDEO_ROWS * VIDEO_DIM * sizeof(float));
    float *audio_out = malloc((size_t)AUDIO_ROWS * AUDIO_DIM * sizeof(float));
    require(video_out && audio_out, "cannot allocate the readback");
    const double blocks_began = now();
    for (int index = 0; index < blocks; index++) {
        block_weights weights;
        load_block(index, &cond, &weights);
        GPU_OP(h3_gpu_begin(gpu), "begin block");
        run_block(&weights, &s, &space);
        GPU_OP(h3_gpu_submit(gpu), "submit block");
        free_block(&weights);
        if (index < ANCHOR_BLOCKS) {
            char name[64], label[64];
            read_bf16_as_f32(s.video, video_out, (size_t)VIDEO_ROWS * VIDEO_DIM);
            read_bf16_as_f32(s.audio, audio_out, (size_t)AUDIO_ROWS * AUDIO_DIM);
            snprintf(name, sizeof(name), "block%d_video", index);
            snprintf(label, sizeof(label), "block %d video", index);
            float *expected = golden(name, (size_t)VIDEO_ROWS * VIDEO_DIM);
            compare(label, video_out, expected,
                    (size_t)VIDEO_ROWS * VIDEO_DIM, 2e-2);
            free(expected);
            snprintf(name, sizeof(name), "block%d_audio", index);
            snprintf(label, sizeof(label), "block %d audio", index);
            expected = golden(name, (size_t)AUDIO_ROWS * AUDIO_DIM);
            compare(label, audio_out, expected,
                    (size_t)AUDIO_ROWS * AUDIO_DIM, 2e-2);
            free(expected);
        }
        if (blocks > ANCHOR_BLOCKS && (index + 1) % 12 == 0)
            printf("    %2d/%d blocks, %.1f s\n", index + 1, blocks,
                   now() - blocks_began);
    }
    const double block_seconds = now() - blocks_began;

    /* ------------------------------------------------------------ the head */

    read_bf16_as_f32(s.video, video_out, (size_t)VIDEO_ROWS * VIDEO_DIM);
    read_bf16_as_f32(s.audio, audio_out, (size_t)AUDIO_ROWS * AUDIO_DIM);
    for (int stream = 0; stream < 2; stream++) {
        const uint32_t rows = stream ? AUDIO_ROWS : VIDEO_ROWS;
        const uint32_t dim = stream ? AUDIO_DIM : VIDEO_DIM;
        const char *tag = stream ? "audio" : "video";
        char name[160];
        snprintf(name, sizeof(name), "%s%s", stream ? "audio_" : "",
                 "scale_shift_table");
        float *table = weight_of(name, (size_t)HEAD_SLOTS * dim);
        snprintf(name, sizeof(name), "%s%s", stream ? "audio_" : "", "proj_out");
        char weight_name[192], bias_name[192];
        snprintf(weight_name, sizeof(weight_name), "%s.weight", name);
        snprintf(bias_name, sizeof(bias_name), "%s.bias", name);
        float *weight = weight_of(weight_name, (size_t)LATENT * dim);
        float *bias = weight_of(bias_name, LATENT);
        float *velocity = malloc((size_t)rows * LATENT * sizeof(*velocity));
        require(velocity != NULL, "cannot allocate the velocity");
        output_head(stream ? audio_out : video_out, table,
                    stream ? cond.audio_embedded : cond.video_embedded,
                    weight, bias, velocity, rows, dim);
        if (blocks == ANCHOR_BLOCKS) {
            snprintf(name, sizeof(name), "%s_velocity", tag);
            float *expected = golden(name, (size_t)rows * LATENT);
            char label[64];
            snprintf(label, sizeof(label), "%s velocity", tag);
            compare(label, velocity, expected, (size_t)rows * LATENT, 3e-2);
            free(expected);
        } else {
            double peak = 0.0;
            for (size_t index = 0; index < (size_t)rows * LATENT; index++) {
                require(isfinite(velocity[index]), "a velocity is not finite");
                if (fabs((double)velocity[index]) > peak)
                    peak = fabs((double)velocity[index]);
            }
            printf("  ok  %-24s [%u, %d] finite, peak %.4f\n",
                   stream ? "audio velocity" : "video velocity", rows, LATENT,
                   peak);
        }
        free(table); free(weight); free(bias); free(velocity);
    }

    printf("\n%d blocks in %.2f s, %.1f ms a block; whole pass %.2f s\n",
           blocks, block_seconds, block_seconds * 1e3 / blocks, now() - began);

    for (int index = 0; index < 8; index++) {
        free(modulation[index]);
        free(embedded[index]);
    }
    free(cond.video_context); free(cond.audio_context);
    free(video_latent); free(audio_latent); free(keyframes);
    free(video_tokens); free(audio_tokens); free(video_out); free(audio_out);
    h3_gpu_free(gpu);
    h3_weight_store_free(store);
    h3_weight_store_free(anchor);
    if (failures) {
        fprintf(stderr, "\n%d comparisons failed\n", failures);
        return 1;
    }
    printf("ok: the DiT ran end to end -- patchify, %d blocks and the output "
           "head, from latents and a sigma to a velocity\n", blocks);
    return 0;
}
