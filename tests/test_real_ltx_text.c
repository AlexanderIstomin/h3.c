/* Run LTX-2.5's Gemma 4 text tower on the released INT8 ConvRot checkpoint.
 *
 * Every piece of this path already has a check against the vendor's own code
 * at toy dimensions -- the tokenizer, the Gemma block, the rotary tables, the
 * aggregation. What none of them cover is assembly: forty-eight layers wired
 * in order with the right per-layer type, the hidden states collected in the
 * order the aggregation expects, and ConvRot applied to each activation with
 * the group its marker gives. Those are the seams, and a wrong one is silent.
 *
 * Four phases, ordered so a mistake surfaces as early and as specifically as
 * it can:
 *
 *   1. resolve every tensor for all forty-eight layers without reading a
 *      payload, deriving each layer's type from its own shapes and insisting
 *      projections that share an input agree on a ConvRot group;
 *   2. reproduce the reference's Hadamard on a probe, which is the one
 *      construction the fixture and the engine would otherwise share blindly;
 *   3. run the first six layers -- five sliding and one global -- against
 *      activations captured from Hugging Face's own Gemma4Unified fed the
 *      same weights, dequantized; and
 *   4. run the whole tower and the aggregation, which is the deliverable.
 *
 * The fixture is built by gen_ltx_text_anchor.py, which undoes the
 * quantization rather than reimplementing it: it derives each torch weight as
 * (W_i8 * scale) @ H^T so the reference and the engine consume the same
 * numbers by different routes.
 *
 * Two facts this pins that are easy to get wrong and impossible to notice:
 *
 *   - the collected hidden states are the residual stream *before* each layer
 *     plus the final normed output, so state i is the output of layer i-1 and
 *     the last layer's raw output never appears. Forty-eight layers give
 *     forty-nine states, which is exactly the 188160 = 3840 x 49 the
 *     aggregation weight expects;
 *   - the embedding scale is sqrt(3840) rounded to BF16, 62.0 and not
 *     61.96773, because the reference multiplies by embed_scale.to(bf16).
 *     Using the exact value puts every token 0.05% out from the start.
 *
 * usage: h3_real_ltx_text_test ENCODER.safetensors ANCHOR.safetensors */

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
    HIDDEN = 3840,
    INTERMEDIATE = 15360,
    LAYERS = 48,
    VOCAB = 262144,
    QUERY_HEADS = 16,
    SLIDING_HEAD = 256,
    SLIDING_KV_HEADS = 8,
    GLOBAL_HEAD = 512,
    GLOBAL_KV_HEADS = 1,
    SLIDING_WINDOW = 1024,
    CONVROT_GROUP = 256,
    /* Five sliding layers, the first global one, and one more sliding layer
     * after it -- so the global layer's own output is compared rather than
     * only its effect on the final norm. A mutation pass showed that ending
     * on the global layer hid a rope that rotated the whole head. */
    ANCHOR_LAYERS = 7,
    HIDDEN_STATES = LAYERS + 1,
    VIDEO_DIM = 4096,
    AUDIO_DIM = 2048,
    AGGREGATE_INPUT = HIDDEN * HIDDEN_STATES
};

#define RMS_EPSILON 1e-6f
#define AGGREGATE_EPSILON 1e-6f

static const double SLIDING_THETA = 10000.0;
static const double GLOBAL_THETA = 1000000.0;
static const double GLOBAL_PARTIAL = 0.25;

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

static float round_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += UINT32_C(0x7fff) + ((bits >> 16) & 1u);
    bits &= UINT32_C(0xffff0000);
    memcpy(&value, &bits, sizeof(value));
    return value;
}

/* ------------------------------------------------------- phase 1: resolve */

/* Presence, dtype and shape without touching a payload. Everything below
 * assumes these hold, and checking all forty-eight layers up front costs
 * milliseconds where discovering it mid-forward costs a confusing failure
 * twelve gigabytes later. */
static const h3_st_tensor *expect(const char *name, h3_dtype dtype,
                                  int ndim, uint64_t first, uint64_t second) {
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(store, name, &header);
    if (!tensor) fail("the checkpoint has no tensor %s", name);
    if (tensor->dtype != dtype) {
        fail("%s is %s, expected %s", name, h3_dtype_name(tensor->dtype),
             h3_dtype_name(dtype));
    }
    if (tensor->ndim != ndim) fail("%s has %d dimensions, expected %d",
                                   name, tensor->ndim, ndim);
    if (tensor->shape[0] != first) {
        fail("%s is [%llu, ...], expected [%llu, ...]", name,
             (unsigned long long)tensor->shape[0], (unsigned long long)first);
    }
    if (ndim > 1 && tensor->shape[1] != second) {
        fail("%s is [%llu, %llu], expected [%llu, %llu]", name,
             (unsigned long long)tensor->shape[0],
             (unsigned long long)tensor->shape[1],
             (unsigned long long)first, (unsigned long long)second);
    }
    return tensor;
}

static int has(const char *name) {
    return h3_weight_find(store, name, NULL) != NULL;
}

static uint32_t convrot_group(const char *name) {
    uint32_t group = 0;
    char error[512];
    if (!h3_weight_i8_linear_convrot_group(store, name, &group,
                                           error, sizeof(error)))
        fail("cannot read the ConvRot marker for %s: %s", name, error);
    if (group != CONVROT_GROUP)
        fail("%s has ConvRot group %u, expected %d", name, group, CONVROT_GROUP);
    return group;
}

/* An I8 projection, its per-output-channel scales, and its marker. */
static void expect_projection(const char *prefix, const char *suffix,
                              uint64_t out_dim, uint64_t in_dim,
                              uint32_t *group) {
    char name[192];
    snprintf(name, sizeof(name), "%s%s.weight", prefix, suffix);
    expect(name, H3_DTYPE_I8, 2, out_dim, in_dim);
    char scales[192];
    snprintf(scales, sizeof(scales), "%s%s.weight_scale", prefix, suffix);
    expect(scales, H3_DTYPE_F32, 2, out_dim, 1);
    *group = convrot_group(name);
}

/* Layer types come from the checkpoint's own shapes rather than from the
 * config's layer_types list, so a disagreement between the two is caught
 * instead of inherited. The published list puts full_attention at every sixth
 * index; that is asserted rather than assumed. */
static int resolve_layer(int index) {
    char prefix[96];
    snprintf(prefix, sizeof(prefix), "model.layers.%d.", index);

    char query[192];
    snprintf(query, sizeof(query), "%sself_attn.q_proj.weight", prefix);
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(store, query, &header);
    if (!tensor) fail("the checkpoint has no layer %d", index);
    int global;
    if (tensor->shape[0] == (uint64_t)QUERY_HEADS * SLIDING_HEAD) global = 0;
    else if (tensor->shape[0] == (uint64_t)QUERY_HEADS * GLOBAL_HEAD) global = 1;
    else fail("layer %d has a query projection of %llu rows, which is neither "
              "%d sliding heads nor %d global ones", index,
              (unsigned long long)tensor->shape[0], QUERY_HEADS, QUERY_HEADS);

    char value[192];
    snprintf(value, sizeof(value), "%sself_attn.v_proj.weight", prefix);
    if (has(value) == global) {
        fail("layer %d looks %s by shape but %s a value projection", index,
             global ? "global" : "sliding", global ? "has" : "lacks");
    }
    if (global != (index % 6 == 5)) {
        fail("layer %d is %s, but the published layer_types put full_attention "
             "at every sixth index", index, global ? "global" : "sliding");
    }

    const uint32_t head = global ? GLOBAL_HEAD : SLIDING_HEAD;
    const uint32_t kv_heads = global ? GLOBAL_KV_HEADS : SLIDING_KV_HEADS;
    const uint64_t query_dim = (uint64_t)QUERY_HEADS * head;
    const uint64_t kv_dim = (uint64_t)kv_heads * head;

    for (int which = 0; which < 4; which++) {
        static const char *const norms[] = {
            "input_layernorm.weight", "post_attention_layernorm.weight",
            "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight"
        };
        char name[192];
        snprintf(name, sizeof(name), "%s%s", prefix, norms[which]);
        expect(name, H3_DTYPE_BF16, 1, HIDDEN, 0);
    }
    char name[192];
    snprintf(name, sizeof(name), "%sself_attn.q_norm.weight", prefix);
    expect(name, H3_DTYPE_BF16, 1, head, 0);
    snprintf(name, sizeof(name), "%sself_attn.k_norm.weight", prefix);
    expect(name, H3_DTYPE_BF16, 1, head, 0);
    snprintf(name, sizeof(name), "%slayer_scalar", prefix);
    expect(name, H3_DTYPE_BF16, 1, 1, 0);

    uint32_t q_group, k_group, v_group = 0, o_group, gate_group, up_group, down_group;
    expect_projection(prefix, "self_attn.q_proj", query_dim, HIDDEN, &q_group);
    expect_projection(prefix, "self_attn.k_proj", kv_dim, HIDDEN, &k_group);
    if (!global)
        expect_projection(prefix, "self_attn.v_proj", kv_dim, HIDDEN, &v_group);
    expect_projection(prefix, "self_attn.o_proj", HIDDEN, query_dim, &o_group);
    expect_projection(prefix, "mlp.gate_proj", INTERMEDIATE, HIDDEN, &gate_group);
    expect_projection(prefix, "mlp.up_proj", INTERMEDIATE, HIDDEN, &up_group);
    expect_projection(prefix, "mlp.down_proj", HIDDEN, INTERMEDIATE, &down_group);

    /* The rotation is applied to the activation, so every projection reading
     * the same activation must agree on it. One rotation then serves all of
     * them -- and if they disagreed, rotating once would quietly corrupt the
     * others rather than fail. */
    if (q_group != k_group || (!global && q_group != v_group)) {
        fail("layer %d Q/K%s ConvRot groups disagree: %u, %u, %u", index,
             global ? "" : "/V", q_group, k_group, v_group);
    }
    if (gate_group != up_group) {
        fail("layer %d gate/up ConvRot groups disagree: %u, %u", index,
             gate_group, up_group);
    }
    (void)o_group;
    (void)down_group;
    return global;
}

/* ---------------------------------------------------------- the fixture */

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

/* Deviation from the F32 reference, measured against that tensor's own peak
 * because these activations run from a fraction to seventeen hundred once the
 * final norm's weights (max 600) are applied, and one absolute bound cannot
 * mean anything across that range.
 *
 * `floor` is the same reference computed with its weights rounded to BF16 --
 * the identical code and the identical numbers, differing only in precision.
 * Printing it beside the engine's deviation is what makes these bounds
 * defensible: the engine is not being held to F32, it is being held to no
 * worse than the reference manages against itself when it takes the same class
 * of rounding the engine takes. A wiring mistake leaves that floor far behind;
 * a rounding difference does not. */
static void compare(const char *label, const float *actual,
                    const float *expected, const float *floor,
                    size_t count, double tolerance) {
    double worst = 0.0, spread = 0.0, peak = 0.0;
    size_t worst_at = 0;
    for (size_t index = 0; index < count; index++) {
        if (!isfinite(actual[index])) {
            fprintf(stderr, "FAIL %-28s produced %f at %zu\n",
                    label, actual[index], index);
            failures++;
            return;
        }
        double delta = fabs((double)actual[index] - (double)expected[index]);
        if (delta > worst) { worst = delta; worst_at = index; }
        if (floor) {
            double reference = fabs((double)floor[index] - (double)expected[index]);
            if (reference > spread) spread = reference;
        }
        double magnitude = fabs((double)expected[index]);
        if (magnitude > peak) peak = magnitude;
    }
    double relative = peak > 0.0 ? worst / peak : worst;
    if (relative > tolerance) {
        fprintf(stderr, "FAIL %-28s %.3e = %.2e of peak, floor %.2e, at %zu "
                        "(%.5f vs %.5f)\n",
                label, worst, relative, peak > 0.0 ? spread / peak : spread,
                worst_at, actual[worst_at], expected[worst_at]);
        failures++;
    } else {
        printf("  ok  %-28s %.3e = %.2e of peak %.3e, reference's own "
               "BF16 floor %.2e\n", label, worst, relative, peak,
               peak > 0.0 ? spread / peak : spread);
    }
}

/* The pair the comparison above needs: the F32 reference and the same states
 * with the reference's weights rounded to BF16. */
static void compare_state(const char *label, const char *name,
                          const float *actual, size_t count, double tolerance) {
    char coarse[80];
    snprintf(coarse, sizeof(coarse), "%s_bf16", name);
    float *expected = load_anchor(name, count);
    float *floor = load_anchor(coarse, count);
    compare(label, actual, expected, floor, count, tolerance);
    free(expected);
    free(floor);
}

/* ---------------------------------------------------------------- weights */

typedef struct {
    h3_gpu_tensor *weight;
    h3_gpu_tensor *scales;
} projection;

typedef struct {
    int global;
    uint32_t head_dim;
    uint32_t kv_heads;
    uint32_t query_dim;
    uint32_t kv_dim;
    float scalar;
    h3_gpu_tensor *input_norm;
    h3_gpu_tensor *post_attention_norm;
    h3_gpu_tensor *pre_ffn_norm;
    h3_gpu_tensor *post_ffn_norm;
    h3_gpu_tensor *query_norm;
    h3_gpu_tensor *key_norm;
    projection query, key, value, output, gate, up, down;
} layer_weights;

static h3_gpu_tensor *load_bf16(const char *name, int ndim,
                                uint64_t first, uint64_t second) {
    uint64_t shape[2] = {first, second};
    char error[512];
    h3_gpu_tensor *tensor = h3_weight_load_bf16(store, gpu, name, ndim, shape,
                                                error, sizeof(error));
    if (!tensor) fail("cannot load %s: %s", name, error);
    return tensor;
}

static void load_projection(const char *prefix, const char *suffix,
                            uint64_t out_dim, uint64_t in_dim,
                            projection *into) {
    char name[192];
    snprintf(name, sizeof(name), "%s%s.weight", prefix, suffix);
    char error[512];
    if (!h3_weight_load_i8_linear(store, gpu, name, out_dim, in_dim,
                                  &into->weight, &into->scales,
                                  error, sizeof(error)))
        fail("cannot load %s: %s", name, error);
}

static void free_projection(projection *which) {
    h3_gpu_tensor_free(which->weight);
    h3_gpu_tensor_free(which->scales);
    memset(which, 0, sizeof(*which));
}

static float read_scalar(const char *name) {
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(store, name, &header);
    if (!tensor) fail("the checkpoint has no %s", name);
    uint16_t raw = 0;
    char error[512];
    require(h3_st_read_data(header, tensor, &raw, sizeof(raw),
                            error, sizeof(error)),
            "cannot read a layer scalar");
    return from_bf16(raw);
}

static void load_layer(int index, layer_weights *weights) {
    char prefix[96];
    snprintf(prefix, sizeof(prefix), "model.layers.%d.", index);
    memset(weights, 0, sizeof(*weights));
    weights->global = index % 6 == 5;
    weights->head_dim = weights->global ? GLOBAL_HEAD : SLIDING_HEAD;
    weights->kv_heads = weights->global ? GLOBAL_KV_HEADS : SLIDING_KV_HEADS;
    weights->query_dim = QUERY_HEADS * weights->head_dim;
    weights->kv_dim = weights->kv_heads * weights->head_dim;

    char name[192];
#define NORM(field, suffix, width) do {                                       \
    snprintf(name, sizeof(name), "%s%s", prefix, suffix);                     \
    weights->field = load_bf16(name, 1, (width), 0);                          \
} while (0)
    NORM(input_norm, "input_layernorm.weight", HIDDEN);
    NORM(post_attention_norm, "post_attention_layernorm.weight", HIDDEN);
    NORM(pre_ffn_norm, "pre_feedforward_layernorm.weight", HIDDEN);
    NORM(post_ffn_norm, "post_feedforward_layernorm.weight", HIDDEN);
    NORM(query_norm, "self_attn.q_norm.weight", weights->head_dim);
    NORM(key_norm, "self_attn.k_norm.weight", weights->head_dim);
#undef NORM
    snprintf(name, sizeof(name), "%slayer_scalar", prefix);
    weights->scalar = read_scalar(name);

    load_projection(prefix, "self_attn.q_proj", weights->query_dim, HIDDEN,
                    &weights->query);
    load_projection(prefix, "self_attn.k_proj", weights->kv_dim, HIDDEN,
                    &weights->key);
    if (!weights->global)
        load_projection(prefix, "self_attn.v_proj", weights->kv_dim, HIDDEN,
                        &weights->value);
    load_projection(prefix, "self_attn.o_proj", HIDDEN, weights->query_dim,
                    &weights->output);
    load_projection(prefix, "mlp.gate_proj", INTERMEDIATE, HIDDEN, &weights->gate);
    load_projection(prefix, "mlp.up_proj", INTERMEDIATE, HIDDEN, &weights->up);
    load_projection(prefix, "mlp.down_proj", HIDDEN, INTERMEDIATE, &weights->down);
}

static void free_layer(layer_weights *weights) {
    h3_gpu_tensor_free(weights->input_norm);
    h3_gpu_tensor_free(weights->post_attention_norm);
    h3_gpu_tensor_free(weights->pre_ffn_norm);
    h3_gpu_tensor_free(weights->post_ffn_norm);
    h3_gpu_tensor_free(weights->query_norm);
    h3_gpu_tensor_free(weights->key_norm);
    free_projection(&weights->query);
    free_projection(&weights->key);
    if (!weights->global) free_projection(&weights->value);
    free_projection(&weights->output);
    free_projection(&weights->gate);
    free_projection(&weights->up);
    free_projection(&weights->down);
    memset(weights, 0, sizeof(*weights));
}

/* --------------------------------------------------------------- the rope */

/* One compact [tokens, head_dim / 2] pair of tables, shared across heads.
 *
 * The global layers rotate a quarter of their 512-wide head -- 64 of 256
 * frequency pairs -- and leave the rest at zero, which is an identity
 * rotation. The exponent divides by the *full* head dim rather than by the
 * rotated width, which is what `proportional` names: the rotated quarter keeps
 * its place in the ladder instead of being stretched across it. Dividing by
 * the rotated width is the obvious misreading and gives entirely different
 * frequencies while producing perfectly well-formed conditioning. */
static void rope_tables(h3_gpu_tensor **cosine, h3_gpu_tensor **sine,
                        uint32_t tokens, uint32_t head_dim,
                        double theta, double partial) {
    const uint32_t half = head_dim / 2;
    const uint32_t rotated = (uint32_t)(partial * (double)head_dim / 2.0);
    float *cos_table = calloc((size_t)tokens * half, sizeof(*cos_table));
    float *sin_table = calloc((size_t)tokens * half, sizeof(*sin_table));
    require(cos_table && sin_table, "cannot allocate rotary tables");
    for (uint32_t position = 0; position < tokens; position++) {
        for (uint32_t index = 0; index < half; index++) {
            double inverse = index < rotated ?
                1.0 / pow(theta, (double)(2 * index) / (double)head_dim) : 0.0;
            double angle = (double)position * inverse;
            cos_table[position * half + index] = (float)cos(angle);
            sin_table[position * half + index] = (float)sin(angle);
        }
    }
    *cosine = h3_gpu_tensor_from_f32(gpu, cos_table, (size_t)tokens * half);
    *sine = h3_gpu_tensor_from_f32(gpu, sin_table, (size_t)tokens * half);
    require(*cosine && *sine, "cannot upload rotary tables");
    free(cos_table);
    free(sin_table);
}

/* ------------------------------------------------------------- the layer */

typedef struct {
    h3_gpu_tensor *normed;
    h3_gpu_tensor *query;
    h3_gpu_tensor *key;
    h3_gpu_tensor *value;
    h3_gpu_tensor *heads;
    h3_gpu_tensor *projected;
    h3_gpu_tensor *gate;
    h3_gpu_tensor *up;
    h3_gpu_tensor *ones;
    h3_gpu_tensor *sliding_cos, *sliding_sin;
    h3_gpu_tensor *global_cos, *global_sin;
} scratch;

static void scratch_create(scratch *space, uint32_t tokens) {
    const size_t widest_query = (size_t)tokens * QUERY_HEADS * GLOBAL_HEAD;
    const size_t widest_kv = (size_t)tokens * SLIDING_KV_HEADS * SLIDING_HEAD;
    space->normed = h3_gpu_tensor_new_bf16(gpu, (size_t)tokens * HIDDEN);
    space->query = h3_gpu_tensor_new_bf16(gpu, widest_query);
    space->key = h3_gpu_tensor_new_bf16(gpu, widest_kv);
    space->value = h3_gpu_tensor_new_bf16(gpu, widest_kv);
    space->heads = h3_gpu_tensor_new_bf16(gpu, widest_query);
    space->projected = h3_gpu_tensor_new_bf16(gpu, (size_t)tokens * HIDDEN);
    space->gate = h3_gpu_tensor_new_bf16(gpu, (size_t)tokens * INTERMEDIATE);
    space->up = h3_gpu_tensor_new_bf16(gpu, (size_t)tokens * INTERMEDIATE);
    require(space->normed && space->query && space->key && space->value &&
            space->heads && space->projected && space->gate && space->up,
            "cannot allocate tower scratch");

    /* The value norm carries no learnable scale, so it is driven with ones
     * rather than a second kernel. 0x3f80 is 1.0f in BF16. */
    uint16_t *ones = malloc(GLOBAL_HEAD * sizeof(*ones));
    require(ones != NULL, "cannot allocate the value norm weight");
    for (int index = 0; index < GLOBAL_HEAD; index++) ones[index] = 0x3f80;
    space->ones = h3_gpu_tensor_from_bf16(gpu, ones, GLOBAL_HEAD);
    require(space->ones != NULL, "cannot upload the value norm weight");
    free(ones);

    rope_tables(&space->sliding_cos, &space->sliding_sin, tokens,
                SLIDING_HEAD, SLIDING_THETA, 1.0);
    rope_tables(&space->global_cos, &space->global_sin, tokens,
                GLOBAL_HEAD, GLOBAL_THETA, GLOBAL_PARTIAL);
}

static void scratch_free(scratch *space) {
    h3_gpu_tensor_free(space->normed);
    h3_gpu_tensor_free(space->query);
    h3_gpu_tensor_free(space->key);
    h3_gpu_tensor_free(space->value);
    h3_gpu_tensor_free(space->heads);
    h3_gpu_tensor_free(space->projected);
    h3_gpu_tensor_free(space->gate);
    h3_gpu_tensor_free(space->up);
    h3_gpu_tensor_free(space->ones);
    h3_gpu_tensor_free(space->sliding_cos);
    h3_gpu_tensor_free(space->sliding_sin);
    h3_gpu_tensor_free(space->global_cos);
    h3_gpu_tensor_free(space->global_sin);
    memset(space, 0, sizeof(*space));
}

static int linear_i8(h3_gpu_tensor *output, const h3_gpu_tensor *input,
                     const projection *weight, uint32_t rows,
                     uint32_t in_dim, uint32_t out_dim) {
    return h3_gpu_linear_i8_weight_bf16(gpu, output, input, weight->weight,
                                        weight->scales, NULL, rows,
                                        in_dim, out_dim);
}

/* One decoder layer, in the order Gemma4UnifiedTextDecoderLayer runs it:
 * a norm sandwich around each residual, GeGLU rather than SwiGLU, no
 * 1/sqrt(head_dim) anywhere, and a trained scalar on the way out. */
static void run_layer(const layer_weights *weight, scratch *space,
                      h3_gpu_tensor *hidden, uint32_t tokens) {
    const uint32_t head_dim = weight->head_dim;
    const uint32_t kv_heads = weight->kv_heads;
    const uint32_t query_dim = weight->query_dim;
    const uint32_t kv_dim = weight->kv_dim;
    const h3_gpu_tensor *cosine = weight->global ? space->global_cos
                                                 : space->sliding_cos;
    const h3_gpu_tensor *sine = weight->global ? space->global_sin
                                               : space->sliding_sin;

    GPU_OP(h3_gpu_rms_norm_bf16(gpu, space->normed, hidden, weight->input_norm,
                                tokens, HIDDEN, RMS_EPSILON), "input norm");
    GPU_OP(h3_gpu_convrot_bf16(gpu, space->normed, space->normed, tokens,
                               HIDDEN, CONVROT_GROUP), "Q/K/V ConvRot");
    GPU_OP(linear_i8(space->query, space->normed, &weight->query, tokens,
                     HIDDEN, query_dim), "query projection");
    GPU_OP(linear_i8(space->key, space->normed, &weight->key, tokens,
                     HIDDEN, kv_dim), "key projection");
    if (weight->global) {
        /* attention_k_eq_v: the value is the raw key projection, taken before
         * k_norm and before the rotation. */
        GPU_OP(h3_gpu_copy_bf16(gpu, space->value, 0, space->key, 0,
                                (size_t)tokens * kv_dim), "value from key");
    } else {
        GPU_OP(linear_i8(space->value, space->normed, &weight->value, tokens,
                         HIDDEN, kv_dim), "value projection");
    }
    GPU_OP(h3_gpu_head_rms_norm_bf16(gpu, space->query, weight->query_norm,
                                     tokens, QUERY_HEADS, head_dim,
                                     RMS_EPSILON), "query norm");
    GPU_OP(h3_gpu_head_rms_norm_bf16(gpu, space->key, weight->key_norm,
                                     tokens, kv_heads, head_dim,
                                     RMS_EPSILON), "key norm");
    GPU_OP(h3_gpu_rope_text_bf16(gpu, space->query, space->key, cosine, sine,
                                 tokens, QUERY_HEADS, kv_heads, head_dim, 0),
           "rotary");
    GPU_OP(h3_gpu_head_rms_norm_bf16(gpu, space->value, space->ones, tokens,
                                     kv_heads, head_dim, RMS_EPSILON),
           "value norm");
    /* Gemma 4 sets the softmax scale to 1.0 and folds nothing into the tables. */
    GPU_OP(h3_gpu_gqa_causal_bf16(gpu, space->heads, space->query, space->key,
                                  space->value, tokens, QUERY_HEADS, kv_heads,
                                  head_dim, 1.0f), "causal attention");
    GPU_OP(h3_gpu_convrot_bf16(gpu, space->heads, space->heads, tokens,
                               query_dim, CONVROT_GROUP), "output ConvRot");
    GPU_OP(linear_i8(space->projected, space->heads, &weight->output, tokens,
                     query_dim, HIDDEN), "output projection");
    GPU_OP(h3_gpu_rms_norm_bf16(gpu, space->projected, space->projected,
                                weight->post_attention_norm, tokens, HIDDEN,
                                RMS_EPSILON), "post-attention norm");
    GPU_OP(h3_gpu_add_bf16(gpu, hidden, hidden, space->projected,
                           tokens * HIDDEN), "attention residual");

    GPU_OP(h3_gpu_rms_norm_bf16(gpu, space->normed, hidden, weight->pre_ffn_norm,
                                tokens, HIDDEN, RMS_EPSILON), "pre-FFN norm");
    GPU_OP(h3_gpu_convrot_bf16(gpu, space->normed, space->normed, tokens,
                               HIDDEN, CONVROT_GROUP), "gate/up ConvRot");
    GPU_OP(linear_i8(space->gate, space->normed, &weight->gate, tokens,
                     HIDDEN, INTERMEDIATE), "MLP gate");
    GPU_OP(linear_i8(space->up, space->normed, &weight->up, tokens,
                     HIDDEN, INTERMEDIATE), "MLP up");
    GPU_OP(h3_gpu_gelu_mul_bf16(gpu, space->gate, space->gate, space->up,
                                tokens * INTERMEDIATE), "GeGLU");
    GPU_OP(h3_gpu_convrot_bf16(gpu, space->gate, space->gate, tokens,
                               INTERMEDIATE, CONVROT_GROUP), "down ConvRot");
    GPU_OP(linear_i8(space->projected, space->gate, &weight->down, tokens,
                     INTERMEDIATE, HIDDEN), "MLP down");
    GPU_OP(h3_gpu_rms_norm_bf16(gpu, space->projected, space->projected,
                                weight->post_ffn_norm, tokens, HIDDEN,
                                RMS_EPSILON), "post-FFN norm");
    GPU_OP(h3_gpu_add_bf16(gpu, hidden, hidden, space->projected,
                           tokens * HIDDEN), "MLP residual");
    GPU_OP(h3_gpu_scale_bf16(gpu, hidden, hidden, tokens * HIDDEN,
                             weight->scalar), "layer scalar");
}

/* ---------------------------------------------------------------- reading */

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

/* --------------------------------------------------------------- the tower */

/* The scaled embedding lookup, rounded the way the reference rounds it. */
static h3_gpu_tensor *embed(const int32_t *ids, uint32_t tokens) {
    printf("  loading the embedding table (2.0 GB)...\n");
    fflush(stdout);
    h3_gpu_tensor *table = load_bf16("model.embed_tokens.weight", 2,
                                     VOCAB, HIDDEN);
    uint32_t *raw = malloc((size_t)tokens * sizeof(*raw));
    require(raw != NULL, "cannot allocate the token ids");
    for (uint32_t index = 0; index < tokens; index++) {
        require(ids[index] >= 0 && ids[index] < VOCAB,
                "an anchor token id is out of range");
        raw[index] = (uint32_t)ids[index];
    }
    h3_gpu_tensor *token_ids = h3_gpu_tensor_from_u32(gpu, raw, tokens);
    require(token_ids != NULL, "cannot upload the token ids");
    free(raw);

    h3_gpu_tensor *hidden = h3_gpu_tensor_new_bf16(gpu, (size_t)tokens * HIDDEN);
    require(hidden != NULL, "cannot allocate the hidden state");
    GPU_OP(h3_gpu_begin(gpu), "begin embedding");
    GPU_OP(h3_gpu_embedding_bf16(gpu, hidden, table, token_ids, tokens,
                                 VOCAB, HIDDEN), "embedding lookup");
    /* sqrt(3840) is 61.96773, but the reference multiplies by a BF16 copy of
     * it, so the scale that actually runs is 62.0. Rounding here rather than
     * passing the exact value keeps this path on the reference's arithmetic. */
    GPU_OP(h3_gpu_scale_bf16(gpu, hidden, hidden, tokens * HIDDEN,
                             round_bf16(sqrtf((float)HIDDEN))),
           "embedding scale");
    GPU_OP(h3_gpu_submit(gpu), "submit embedding");

    h3_gpu_tensor_free(table);
    h3_gpu_tensor_free(token_ids);
    return hidden;
}

/* -------------------------------------------------------- the aggregation */

/* LTX's FeatureExtractorV2. Each hidden state is normalized per token over its
 * own 3840 channels, the 49 of them are laid out with the layer index *minor*
 * -- normed[token][d * 49 + l], which is what reshape(B, T, D * L) produces
 * from a [B, T, D, L] stack -- rescaled, and projected.
 *
 * The two streams differ only in their weights and their rescale, which is
 * sqrt(out_dim / 3840) and so is not the same number for a 4096-wide video
 * embedding as for a 2048-wide audio one. Reusing one for the other is wrong
 * by sqrt(2) and produces entirely plausible conditioning. */
static void aggregate_stream(const float *normed, uint32_t tokens,
                             const char *name, uint32_t out_dim,
                             float *out) {
    const float rescale = sqrtf((float)out_dim / (float)HIDDEN);
    const size_t count = (size_t)tokens * AGGREGATE_INPUT;
    uint16_t *staged = malloc(count * sizeof(*staged));
    require(staged != NULL, "cannot allocate the aggregation input");
    for (size_t index = 0; index < count; index++) {
        float value = round_bf16(normed[index] * rescale);
        uint32_t bits;
        memcpy(&bits, &value, sizeof(bits));
        staged[index] = (uint16_t)(bits >> 16);
    }
    h3_gpu_tensor *input = h3_gpu_tensor_from_bf16(gpu, staged, count);
    require(input != NULL, "cannot upload the aggregation input");
    free(staged);

    char weight_name[160], bias_name[160];
    snprintf(weight_name, sizeof(weight_name),
             "text_embedding_projection.%s.weight", name);
    snprintf(bias_name, sizeof(bias_name),
             "text_embedding_projection.%s.bias", name);
    h3_gpu_tensor *weight = load_bf16(weight_name, 2, out_dim, AGGREGATE_INPUT);
    h3_gpu_tensor *bias = load_bf16(bias_name, 1, out_dim, 0);
    h3_gpu_tensor *result = h3_gpu_tensor_new_bf16(gpu, (size_t)tokens * out_dim);
    require(result != NULL, "cannot allocate the aggregation output");
    GPU_OP(h3_gpu_begin(gpu), "begin aggregation");
    GPU_OP(h3_gpu_linear_bf16(gpu, result, input, weight, bias, tokens,
                              AGGREGATE_INPUT, out_dim), "aggregation");
    GPU_OP(h3_gpu_submit(gpu), "submit aggregation");
    read_bf16_as_f32(result, out, (size_t)tokens * out_dim);

    double peak = 0.0;
    for (size_t index = 0; index < (size_t)tokens * out_dim; index++) {
        require(isfinite(out[index]), "an aggregated feature is not finite");
        if (fabs((double)out[index]) > peak) peak = fabs((double)out[index]);
    }
    printf("  ok  %-28s [%u, %u] rescale %.5f, peak %.3f\n",
           name, tokens, out_dim, (double)rescale, peak);

    h3_gpu_tensor_free(input);
    h3_gpu_tensor_free(weight);
    h3_gpu_tensor_free(bias);
    h3_gpu_tensor_free(result);
}

static void aggregate(float *const *states, uint32_t tokens,
                      float **video, float **audio) {
    const size_t count = (size_t)tokens * AGGREGATE_INPUT;
    float *normed = malloc(count * sizeof(*normed));
    require(normed != NULL, "cannot allocate the normalized states");
    for (uint32_t token = 0; token < tokens; token++) {
        for (int layer = 0; layer < HIDDEN_STATES; layer++) {
            const float *row = states[layer] + (size_t)token * HIDDEN;
            double sum = 0.0;
            for (int index = 0; index < HIDDEN; index++)
                sum += (double)row[index] * (double)row[index];
            const float inverse = 1.0f /
                sqrtf((float)(sum / (double)HIDDEN) + AGGREGATE_EPSILON);
            float *out = normed + (size_t)token * AGGREGATE_INPUT;
            for (int index = 0; index < HIDDEN; index++)
                out[(size_t)index * HIDDEN_STATES + layer] = row[index] * inverse;
        }
    }
    *video = malloc((size_t)tokens * VIDEO_DIM * sizeof(**video));
    *audio = malloc((size_t)tokens * AUDIO_DIM * sizeof(**audio));
    require(*video && *audio, "cannot allocate the aggregated features");
    aggregate_stream(normed, tokens, "video_aggregate_embed", VIDEO_DIM, *video);
    aggregate_stream(normed, tokens, "audio_aggregate_embed", AUDIO_DIM, *audio);
    free(normed);
}

/* Everything the aggregation check needs, in a layout a numpy reader can take
 * in four lines: a header of six uint32 (magic, tokens, states, hidden, video,
 * audio), then the token ids as int32, then the hidden states as F32 in order,
 * then the two aggregated results. Not safetensors, because writing that from
 * here would be more code than the reader saves. */
static void write_states(const char *path, const int32_t *ids,
                         float *const *states, const float *video,
                         const float *audio, uint32_t tokens) {
    FILE *file = fopen(path, "wb");
    if (!file) fail("cannot open %s for writing", path);
    const uint32_t header[6] = {UINT32_C(0x4C545854), tokens, HIDDEN_STATES,
                                HIDDEN, VIDEO_DIM, AUDIO_DIM};
    const size_t state = (size_t)tokens * HIDDEN;
    require(fwrite(header, sizeof(header), 1, file) == 1, "cannot write a header");
    require(fwrite(ids, sizeof(*ids), tokens, file) == tokens,
            "cannot write the token ids");
    for (int index = 0; index < HIDDEN_STATES; index++)
        require(fwrite(states[index], sizeof(float), state, file) == state,
                "cannot write a hidden state");
    require(fwrite(video, sizeof(float), (size_t)tokens * VIDEO_DIM, file) ==
                (size_t)tokens * VIDEO_DIM, "cannot write the video features");
    require(fwrite(audio, sizeof(float), (size_t)tokens * AUDIO_DIM, file) ==
                (size_t)tokens * AUDIO_DIM, "cannot write the audio features");
    require(fclose(file) == 0, "cannot close the state dump");
    printf("  wrote %s for the aggregation check\n", path);
}

int main(int argc, char **argv) {
    if (argc < 3 || argc > 4) {
        fprintf(stderr, "usage: %s ENCODER.safetensors ANCHOR.safetensors "
                        "[STATES.bin]\n", argv[0]);
        return 2;
    }
    char error[512];
    store = h3_weight_store_open(argv[1], error, sizeof(error));
    if (!store) fail("cannot open the encoder: %s", error);
    anchor = h3_weight_store_open(argv[2], error, sizeof(error));
    if (!anchor) fail("cannot open the anchor: %s", error);

    /* ------------------------------------------------ 1: resolve everything */

    printf("Resolving all %d layers of the released Gemma tower:\n", LAYERS);
    int sliding = 0, global = 0;
    for (int index = 0; index < LAYERS; index++) {
        if (resolve_layer(index)) global++;
        else sliding++;
    }
    expect("model.embed_tokens.weight", H3_DTYPE_BF16, 2, VOCAB, HIDDEN);
    expect("model.norm.weight", H3_DTYPE_BF16, 1, HIDDEN, 0);
    expect("text_embedding_projection.video_aggregate_embed.weight",
           H3_DTYPE_BF16, 2, VIDEO_DIM, AGGREGATE_INPUT);
    expect("text_embedding_projection.video_aggregate_embed.bias",
           H3_DTYPE_BF16, 1, VIDEO_DIM, 0);
    expect("text_embedding_projection.audio_aggregate_embed.weight",
           H3_DTYPE_BF16, 2, AUDIO_DIM, AGGREGATE_INPUT);
    expect("text_embedding_projection.audio_aggregate_embed.bias",
           H3_DTYPE_BF16, 1, AUDIO_DIM, 0);
    printf("  ok  %d sliding and %d global layers, every projection ConvRot "
           "at group %d, and %d x %d aggregation inputs\n",
           sliding, global, CONVROT_GROUP, HIDDEN, HIDDEN_STATES);
    require(sliding == 40 && global == 8, "the layer type split is not 40/8");

    gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) fail("cannot create the Metal context: %s", error);

    /* ------------------------------------------------------ 2: the rotation */

    printf("\nThe fixture's Hadamard against the engine's kernel:\n");
    if (!h3_weight_find(anchor, "convrot_probe_in", NULL)) {
        printf("  --  the anchor carries no ConvRot probe, so this is a "
               "prompt run; skipping\n");
    } else {
        const size_t count = (size_t)3 * HIDDEN;
        float *probe_in = load_anchor("convrot_probe_in", count);
        float *probe_out = load_anchor("convrot_probe_out", count);
        uint16_t *staged = malloc(count * sizeof(*staged));
        require(staged != NULL, "cannot allocate the probe staging buffer");
        for (size_t index = 0; index < count; index++) {
            float rounded = round_bf16(probe_in[index]);
            uint32_t bits;
            memcpy(&bits, &rounded, sizeof(bits));
            staged[index] = (uint16_t)(bits >> 16);
        }
        h3_gpu_tensor *probe = h3_gpu_tensor_from_bf16(gpu, staged, count);
        require(probe != NULL, "cannot upload the ConvRot probe");
        GPU_OP(h3_gpu_begin(gpu), "begin probe");
        GPU_OP(h3_gpu_convrot_bf16(gpu, probe, probe, 3, HIDDEN, CONVROT_GROUP),
               "probe ConvRot");
        GPU_OP(h3_gpu_submit(gpu), "submit probe");
        float *actual = malloc(count * sizeof(*actual));
        require(actual != NULL, "cannot allocate the probe result");
        read_bf16_as_f32(probe, actual, count);
        /* Both sides sum 256 BF16 terms; the reference sums them in F32 and in
         * a different order, so this floor is rounding, not disagreement. */
        compare("H256 = H4^(x)4 / 16", actual, probe_out, NULL, count, 4e-3);
        free(probe_in); free(probe_out); free(actual); free(staged);
        h3_gpu_tensor_free(probe);
    }

    /* ------------------------------------------------------- 3: the anchor */

    const h3_st_header *ids_header = NULL;
    const h3_st_tensor *ids_tensor = h3_weight_find(anchor, "input_ids",
                                                    &ids_header);
    if (!ids_tensor) fail("the anchor has no input_ids");
    const uint32_t tokens = (uint32_t)h3_st_tensor_elements(ids_tensor);
    int32_t *ids = malloc((size_t)tokens * sizeof(*ids));
    require(ids != NULL, "cannot allocate the token ids");
    require(h3_st_read_data(ids_header, ids_tensor, ids,
                            (size_t)tokens * sizeof(*ids), error, sizeof(error)),
            "cannot read the anchor token ids");
    if (tokens > SLIDING_WINDOW) {
        fail("%u tokens exceeds the %d sliding window, which this runner does "
             "not apply yet", tokens, SLIDING_WINDOW);
    }

    /* A run driven by a real prompt supplies only `input_ids` -- there is no
     * golden tower output for an arbitrary caption, and generating one would
     * need the 12B model at F32. Phases 2 and 3 compare against a fixture, so
     * they are skipped in that case and the note says so; phase 4 is the part
     * that produces conditioning, and it is the same code either way. */
    const int anchored = h3_weight_find(anchor, "hidden_0", NULL) != NULL;
    if (!anchored)
        printf("\nthe anchor carries only input_ids, so this is a prompt run: "
               "phase 3's comparisons are skipped and phase 4 runs unchanged\n");

    /* Shared with phase 4, which reassigns `hidden` and reuses the scratch. */
    const size_t state = (size_t)tokens * HIDDEN;
    h3_gpu_tensor *hidden = NULL;
    scratch space;
    scratch_create(&space, tokens);

    if (anchored) {
    printf("\nThe first %d layers against Gemma4Unified on the same weights, "
           "%u tokens:\n", ANCHOR_LAYERS, tokens);
    float *actual = malloc(state * sizeof(*actual));
    require(actual != NULL, "cannot allocate the readback");

    hidden = embed(ids, tokens);
    read_bf16_as_f32(hidden, actual, state);
    compare_state("scaled embeddings", "hidden_0", actual, state, 3e-3);
    /* And directly against the BF16 reference, which is the only thing that
     * pins *which* embedding scale is right. Held to F32 alone, the exact
     * sqrt(3840) scores better than the BF16-rounded one the real pipeline
     * uses, so that comparison rewards the wrong answer. Here the engine and
     * the reference do the identical multiply, and anything but agreement to
     * the last bit means they do not. */
    {
        float *rounded = load_anchor("hidden_0_bf16", state);
        compare("embeddings against BF16", actual, rounded, NULL, state, 1e-7);
        free(rounded);
    }

    for (int index = 0; index < ANCHOR_LAYERS; index++) {
        layer_weights weights;
        load_layer(index, &weights);
        GPU_OP(h3_gpu_begin(gpu), "begin layer");
        run_layer(&weights, &space, hidden, tokens);
        GPU_OP(h3_gpu_submit(gpu), "submit layer");
        /* The reference records the residual stream *entering* each layer, so
         * the output of layer i is its state i + 1 -- and the last layer's raw
         * output is never recorded, only the normed one below. */
        if (index + 1 < ANCHOR_LAYERS) {
            char label[64], name[64];
            snprintf(label, sizeof(label), "after layer %d (%s)", index,
                     weights.global ? "global" : "sliding");
            snprintf(name, sizeof(name), "hidden_%d", index + 1);
            read_bf16_as_f32(hidden, actual, state);
            compare_state(label, name, actual, state, 1.5e-2);
        }
        free_layer(&weights);
    }

    {
        h3_gpu_tensor *final_norm = load_bf16("model.norm.weight", 1, HIDDEN, 0);
        h3_gpu_tensor *normed = h3_gpu_tensor_new_bf16(gpu, state);
        require(normed != NULL, "cannot allocate the final normed state");
        GPU_OP(h3_gpu_begin(gpu), "begin final norm");
        GPU_OP(h3_gpu_rms_norm_bf16(gpu, normed, hidden, final_norm,
                                    tokens, HIDDEN, RMS_EPSILON), "final norm");
        GPU_OP(h3_gpu_submit(gpu), "submit final norm");
        read_bf16_as_f32(normed, actual, state);
        /* Looser than the layers by design, and the floor says why: the final
         * norm's weights reach 600, so it takes a direction that agrees to a
         * few thousandths and stretches the disagreement with it. */
        char name[64];
        snprintf(name, sizeof(name), "hidden_%d", ANCHOR_LAYERS);
        compare_state("final normed output", name, actual, state, 4e-2);
        h3_gpu_tensor_free(final_norm);
        h3_gpu_tensor_free(normed);
    }

    h3_gpu_tensor_free(hidden);
    free(actual);

    }

    /* --------------------------------------------- 4: the whole conditioning */

    /* The deliverable: all forty-eight layers, the forty-nine hidden states,
     * and both aggregation projections. Weights are loaded and freed a layer
     * at a time, so this holds one layer rather than the tower's twelve
     * gigabytes and reads the file once.
     *
     * There is no full-depth reference to hold this to -- the tower does not
     * fit in torch at F32 on this machine -- so what runs here is asserted to
     * be finite and is written out for the aggregation to be checked against
     * LTX's own FeatureExtractorV2 separately. The wiring it repeats is the
     * wiring phase 3 anchored; what is new is only depth. */
    if (getenv("H3_LTX_TEXT_ANCHOR_ONLY")) {
        printf("\nH3_LTX_TEXT_ANCHOR_ONLY set: stopping before the full tower\n");
    } else {
        printf("\nThe whole tower, %d layers and %d hidden states:\n",
               LAYERS, HIDDEN_STATES);
        const double began = now();
        float **states = calloc(HIDDEN_STATES, sizeof(*states));
        require(states != NULL, "cannot allocate the hidden state table");
        for (int index = 0; index < HIDDEN_STATES; index++) {
            states[index] = malloc(state * sizeof(**states));
            require(states[index] != NULL, "cannot allocate a hidden state");
        }

        hidden = embed(ids, tokens);
        read_bf16_as_f32(hidden, states[0], state);
        for (int index = 0; index < LAYERS; index++) {
            layer_weights weights;
            load_layer(index, &weights);
            GPU_OP(h3_gpu_begin(gpu), "begin layer");
            run_layer(&weights, &space, hidden, tokens);
            GPU_OP(h3_gpu_submit(gpu), "submit layer");
            free_layer(&weights);
            /* State i + 1 is layer i's output, for every layer but the last:
             * the reference records the stream entering each layer, so layer
             * 47's raw output is never a state and only its normed form is. */
            if (index + 1 < LAYERS)
                read_bf16_as_f32(hidden, states[index + 1], state);
            if ((index + 1) % 12 == 0) {
                printf("    %2d/%d layers, %.1f s\n", index + 1, LAYERS,
                       now() - began);
                fflush(stdout);
            }
        }
        {
            h3_gpu_tensor *final_norm = load_bf16("model.norm.weight", 1,
                                                  HIDDEN, 0);
            h3_gpu_tensor *normed = h3_gpu_tensor_new_bf16(gpu, state);
            require(normed != NULL, "cannot allocate the final normed state");
            GPU_OP(h3_gpu_begin(gpu), "begin final norm");
            GPU_OP(h3_gpu_rms_norm_bf16(gpu, normed, hidden, final_norm,
                                        tokens, HIDDEN, RMS_EPSILON),
                   "final norm");
            GPU_OP(h3_gpu_submit(gpu), "submit final norm");
            read_bf16_as_f32(normed, states[LAYERS], state);
            h3_gpu_tensor_free(final_norm);
            h3_gpu_tensor_free(normed);
        }
        const double tower = now() - began;
        for (int index = 0; index < HIDDEN_STATES; index++)
            for (size_t at = 0; at < state; at++)
                require(isfinite(states[index][at]),
                        "a hidden state is not finite");
        printf("  ok  %d states, all finite, %.1f s\n", HIDDEN_STATES, tower);

        float *video = NULL, *audio = NULL;
        aggregate(states, tokens, &video, &audio);

        if (argc > 3) write_states(argv[3], ids, states, video, audio, tokens);
        for (int index = 0; index < HIDDEN_STATES; index++) free(states[index]);
        free(states);
        free(video);
        free(audio);
        h3_gpu_tensor_free(hidden);
    }

    scratch_free(&space);
    free(ids);
    h3_gpu_free(gpu);
    h3_weight_store_free(store);
    h3_weight_store_free(anchor);

    if (failures) {
        fprintf(stderr, "\n%d comparisons failed\n", failures);
        return 1;
    }
    printf("\nok: the released Gemma 4 tower resolves for all %d layers and "
           "reproduces Gemma4Unified's activations through its first %d, "
           "covering both sliding and value-less global attention\n",
           LAYERS, ANCHOR_LAYERS);
    return 0;
}
