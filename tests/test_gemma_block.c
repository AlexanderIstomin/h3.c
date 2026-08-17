/* Reproduce a Gemma 4 text tower on the host and check it against reference
 * activations from the published implementation.
 *
 * LTX-2.5 conditions on Gemma 4 12B, whose decoder differs from the Qwen3-VL
 * tower h3.c already runs in ways that all fail silently rather than loudly:
 *
 *   - four RMS norms per layer, not two, in a sandwich around each residual;
 *   - a trained per-layer scalar applied to the block output;
 *   - GeGLU with tanh-approximate GELU rather than SwiGLU;
 *   - embeddings scaled by sqrt(hidden_size);
 *   - no 1/sqrt(head_dim) softmax scaling anywhere;
 *   - two rotary tables, the global one partial and zero-padded; and
 *   - global layers that carry no value projection, reusing the raw key
 *     projection through a scale-free RMS norm instead.
 *
 * This is the F32 diagnosis path: it establishes that the architecture above
 * is understood correctly, and becomes the oracle a Metal implementation is
 * checked against. It deliberately does not touch the GPU.
 *
 * The fixture is a toy tower built from the reference implementation at small
 * dimensions, keeping one sliding layer and one global layer so both attention
 * forms are covered.
 *
 * usage: h3_gemma_block_test FIXTURE.safetensors */

#include "h3_safetensors.h"
#include "h3_weights.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    HIDDEN = 32,
    INTERMEDIATE = 64,
    QUERY_HEADS = 4,
    LAYERS = 2,
    TOKENS = 5,
    VOCAB = 64
};

#define NORM_EPS 1e-6f

typedef struct {
    int sliding;
    uint32_t head_dim;
    uint32_t kv_heads;
    double theta;
    double partial_rotary;   /* 1.0 rotates the whole head. */
    float layer_scalar;
} gemma_layer_spec;

/* Mirrors the fixture's layer_types: one sliding layer then one global layer,
 * the global one narrowing to a single key head of twice the width. */
static const gemma_layer_spec LAYERS_SPEC[LAYERS] = {
    {1, 8,  2, 10000.0,   1.00, 0.85f},
    {0, 16, 1, 1000000.0, 0.25, 0.95f}
};

static int failures = 0;

static void fail(const char *message) {
    fprintf(stderr, "FAIL: %s\n", message);
    exit(1);
}

static void require(int condition, const char *message) {
    if (!condition) fail(message);
}

/* ---------------------------------------------------------------- loading */

static h3_weight_store *store = NULL;

static float *load_f32(const char *name, size_t expected) {
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(store, name, &header);
    if (!tensor || !header) {
        fprintf(stderr, "FAIL: fixture has no tensor %s\n", name);
        exit(1);
    }
    if (tensor->dtype != H3_DTYPE_F32) {
        fprintf(stderr, "FAIL: %s is not F32\n", name);
        exit(1);
    }
    size_t elements = (size_t)h3_st_tensor_elements(tensor);
    if (elements != expected) {
        fprintf(stderr, "FAIL: %s has %zu elements, expected %zu\n",
                name, elements, expected);
        exit(1);
    }
    float *values = malloc(elements * sizeof(*values));
    char error[256];
    require(values != NULL, "cannot allocate a fixture tensor");
    require(h3_st_read_data(header, tensor, values,
                            elements * sizeof(*values), error, sizeof(error)),
            "cannot read a fixture tensor");
    return values;
}

static float *load_layer_f32(int layer, const char *suffix, size_t expected) {
    char name[160];
    snprintf(name, sizeof(name), "layers.%d.%s", layer, suffix);
    return load_f32(name, expected);
}

/* Optional: global layers have no value projection at all. */
static int has_tensor(const char *name) {
    const h3_st_header *header = NULL;
    return h3_weight_find(store, name, &header) != NULL;
}

/* ------------------------------------------------------------- primitives */

/* Gemma's RMS norm normalises in F32 as x * (mean(x^2) + eps)^-0.5, then
 * applies the weight directly. Note this is NOT the (1 + weight) form that
 * Gemma 2 and 3 used -- reusing that convention corrupts every norm. */
static void rms_norm(float *out, const float *in, const float *weight,
                     size_t rows, size_t width) {
    for (size_t row = 0; row < rows; row++) {
        const float *x = in + row * width;
        float sum = 0.0f;
        for (size_t index = 0; index < width; index++) sum += x[index] * x[index];
        float inverse = powf(sum / (float)width + NORM_EPS, -0.5f);
        for (size_t index = 0; index < width; index++) {
            float value = x[index] * inverse;
            out[row * width + index] = weight ? value * weight[index] : value;
        }
    }
}

/* Row-major [output_dim, input_dim] weights, no bias anywhere in this tower. */
static void linear(float *out, const float *in, const float *weight,
                   size_t rows, size_t input_dim, size_t output_dim) {
    for (size_t row = 0; row < rows; row++) {
        for (size_t column = 0; column < output_dim; column++) {
            float sum = 0.0f;
            const float *w = weight + column * input_dim;
            const float *x = in + row * input_dim;
            for (size_t k = 0; k < input_dim; k++) sum = fmaf(x[k], w[k], sum);
            out[row * output_dim + column] = sum;
        }
    }
}

static float gelu_tanh(float value) {
    const float root = 0.7978845608028654f;  /* sqrt(2/pi) */
    float cube = value * value * value;
    return 0.5f * value * (1.0f + tanhf(root * (value + 0.044715f * cube)));
}

/* Build one rotary table. Proportional RoPE fills only the first
 * partial * head_dim / 2 entries with real frequencies and leaves the rest at
 * zero, which is an identity rotation; the table still spans the whole head. */
static void rope_table(float *cos_out, float *sin_out,
                       const gemma_layer_spec *spec, size_t tokens) {
    size_t half = spec->head_dim / 2;
    size_t rotated = (size_t)(spec->partial_rotary * (double)spec->head_dim / 2.0);
    double *inverse = calloc(half, sizeof(*inverse));
    require(inverse != NULL, "cannot allocate a rotary table");
    for (size_t index = 0; index < rotated; index++) {
        inverse[index] = 1.0 / pow(spec->theta,
                                   (double)(2 * index) / (double)spec->head_dim);
    }
    for (size_t position = 0; position < tokens; position++) {
        for (size_t index = 0; index < half; index++) {
            double angle = (double)position * inverse[index];
            float c = (float)cos(angle);
            float s = (float)sin(angle);
            /* emb = cat(freqs, freqs), so each half repeats. */
            cos_out[position * spec->head_dim + index] = c;
            cos_out[position * spec->head_dim + half + index] = c;
            sin_out[position * spec->head_dim + index] = s;
            sin_out[position * spec->head_dim + half + index] = s;
        }
    }
    free(inverse);
}

/* q * cos + rotate_half(q) * sin, where rotate_half swaps the halves and
 * negates the second one. Applied per head over head_dim. */
static void apply_rope(float *values, const float *cos_table,
                       const float *sin_table, size_t tokens, size_t heads,
                       size_t head_dim) {
    size_t half = head_dim / 2;
    float *scratch = malloc(head_dim * sizeof(*scratch));
    require(scratch != NULL, "cannot allocate rotary scratch");
    for (size_t token = 0; token < tokens; token++) {
        const float *c = cos_table + token * head_dim;
        const float *s = sin_table + token * head_dim;
        for (size_t head = 0; head < heads; head++) {
            float *x = values + (token * heads + head) * head_dim;
            memcpy(scratch, x, head_dim * sizeof(*scratch));
            for (size_t index = 0; index < head_dim; index++) {
                float rotated = index < half ? -scratch[index + half]
                                             :  scratch[index - half];
                x[index] = scratch[index] * c[index] + rotated * s[index];
            }
        }
    }
    free(scratch);
}

/* ------------------------------------------------------------- attention */

/* Causal attention with no softmax scaling: Gemma 4 sets it to 1.0 and folds
 * nothing into the rotary tables either. Query heads share key/value heads in
 * groups, which is plain GQA for sliding layers and a 4:1 broadcast for the
 * global layer's single key head. */
static void attention(float *out, const float *query, const float *key,
                      const float *value, size_t tokens, size_t query_heads,
                      size_t kv_heads, size_t head_dim) {
    size_t group = query_heads / kv_heads;
    float *weights = malloc(tokens * sizeof(*weights));
    require(weights != NULL, "cannot allocate attention weights");
    for (size_t head = 0; head < query_heads; head++) {
        size_t kv_head = head / group;
        for (size_t token = 0; token < tokens; token++) {
            const float *q = query + (token * query_heads + head) * head_dim;
            float maximum = -INFINITY;
            for (size_t other = 0; other <= token; other++) {
                const float *k = key + (other * kv_heads + kv_head) * head_dim;
                float score = 0.0f;
                for (size_t index = 0; index < head_dim; index++)
                    score = fmaf(q[index], k[index], score);
                weights[other] = score;
                if (score > maximum) maximum = score;
            }
            float total = 0.0f;
            for (size_t other = 0; other <= token; other++) {
                weights[other] = expf(weights[other] - maximum);
                total += weights[other];
            }
            float *destination = out + (token * query_heads + head) * head_dim;
            for (size_t index = 0; index < head_dim; index++)
                destination[index] = 0.0f;
            for (size_t other = 0; other <= token; other++) {
                float share = weights[other] / total;
                const float *v = value + (other * kv_heads + kv_head) * head_dim;
                for (size_t index = 0; index < head_dim; index++)
                    destination[index] = fmaf(share, v[index], destination[index]);
            }
        }
    }
    free(weights);
}

/* ------------------------------------------------------------ comparison */

static void compare(const char *label, const float *actual,
                    const float *expected, size_t count, float tolerance) {
    double worst = 0.0;
    size_t worst_at = 0;
    for (size_t index = 0; index < count; index++) {
        double delta = fabs((double)actual[index] - (double)expected[index]);
        if (delta > worst) { worst = delta; worst_at = index; }
    }
    if (worst > tolerance) {
        fprintf(stderr,
                "FAIL %-28s worst |delta| %.3e at %zu (%.8f vs %.8f)\n",
                label, worst, worst_at, actual[worst_at], expected[worst_at]);
        failures++;
    } else {
        printf("  ok  %-28s worst |delta| %.2e\n", label, worst);
    }
}

/* ------------------------------------------------------------------ main */

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s FIXTURE.safetensors\n", argv[0]);
        return 2;
    }
    char error[512];
    store = h3_weight_store_open(argv[1], error, sizeof(error));
    if (!store) {
        fprintf(stderr, "FAIL: cannot open the fixture: %s\n", error);
        return 1;
    }

    printf("Gemma 4 tower on the host against reference activations:\n");

    /* Rotary tables first: if these are wrong every later comparison is, and
     * checking them separately says which half of the port is at fault. */
    for (int layer = 0; layer < LAYERS; layer++) {
        const gemma_layer_spec *spec = &LAYERS_SPEC[layer];
        const char *type = spec->sliding ? "sliding_attention" : "full_attention";
        char name[128];
        size_t elements = TOKENS * spec->head_dim;
        float *cos_table = calloc(elements, sizeof(*cos_table));
        float *sin_table = calloc(elements, sizeof(*sin_table));
        require(cos_table && sin_table, "cannot allocate rotary tables");
        rope_table(cos_table, sin_table, spec, TOKENS);
        snprintf(name, sizeof(name), "reference.%s_cos", type);
        float *reference_cos = load_f32(name, elements);
        snprintf(name, sizeof(name), "reference.%s_sin", type);
        float *reference_sin = load_f32(name, elements);
        snprintf(name, sizeof(name), "%s cos", type);
        compare(name, cos_table, reference_cos, elements, 1e-6f);
        snprintf(name, sizeof(name), "%s sin", type);
        compare(name, sin_table, reference_sin, elements, 1e-6f);
        free(reference_cos); free(reference_sin);
        free(cos_table); free(sin_table);
    }

    /* Scaled embedding lookup. */
    float *embed = load_f32("embed_tokens.weight", (size_t)VOCAB * HIDDEN);

    const h3_st_header *ids_header = NULL;
    const h3_st_tensor *ids_tensor = h3_weight_find(store, "reference.input_ids",
                                                    &ids_header);
    require(ids_tensor && ids_header, "fixture has no input ids");
    int32_t ids[TOKENS];
    require(h3_st_read_data(ids_header, ids_tensor, ids, sizeof(ids),
                            error, sizeof(error)),
            "cannot read the reference input ids");

    float *hidden = calloc(TOKENS * HIDDEN, sizeof(*hidden));
    require(hidden != NULL, "cannot allocate the hidden state");
    float embed_scale = sqrtf((float)HIDDEN);
    for (size_t token = 0; token < TOKENS; token++) {
        require(ids[token] >= 0 && ids[token] < VOCAB, "reference id is invalid");
        for (size_t index = 0; index < HIDDEN; index++) {
            hidden[token * HIDDEN + index] =
                embed[(size_t)ids[token] * HIDDEN + index] * embed_scale;
        }
    }
    float *reference = load_f32("reference.hidden_0", TOKENS * HIDDEN);
    compare("scaled embeddings", hidden, reference, TOKENS * HIDDEN, 2e-5f);
    free(reference);

    for (int layer = 0; layer < LAYERS; layer++) {
        const gemma_layer_spec *spec = &LAYERS_SPEC[layer];
        size_t head_dim = spec->head_dim;
        size_t inner = QUERY_HEADS * head_dim;
        size_t kv_inner = spec->kv_heads * head_dim;

        float *input_norm = load_layer_f32(layer, "input_layernorm.weight", HIDDEN);
        float *post_attention = load_layer_f32(layer, "post_attention_layernorm.weight", HIDDEN);
        float *pre_ffn = load_layer_f32(layer, "pre_feedforward_layernorm.weight", HIDDEN);
        float *post_ffn = load_layer_f32(layer, "post_feedforward_layernorm.weight", HIDDEN);
        float *q_weight = load_layer_f32(layer, "self_attn.q_proj.weight", inner * HIDDEN);
        float *k_weight = load_layer_f32(layer, "self_attn.k_proj.weight", kv_inner * HIDDEN);
        float *o_weight = load_layer_f32(layer, "self_attn.o_proj.weight", HIDDEN * inner);
        float *q_norm = load_layer_f32(layer, "self_attn.q_norm.weight", head_dim);
        float *k_norm = load_layer_f32(layer, "self_attn.k_norm.weight", head_dim);
        float *gate = load_layer_f32(layer, "mlp.gate_proj.weight", INTERMEDIATE * HIDDEN);
        float *up = load_layer_f32(layer, "mlp.up_proj.weight", INTERMEDIATE * HIDDEN);
        float *down = load_layer_f32(layer, "mlp.down_proj.weight", HIDDEN * INTERMEDIATE);

        char v_name[160];
        snprintf(v_name, sizeof(v_name), "layers.%d.self_attn.v_proj.weight", layer);
        int has_value = has_tensor(v_name);
        require(has_value == spec->sliding,
                "value projection presence does not match the layer type");
        float *v_weight = has_value ?
            load_layer_f32(layer, "self_attn.v_proj.weight", kv_inner * HIDDEN) : NULL;

        float *cos_table = calloc(TOKENS * head_dim, sizeof(*cos_table));
        float *sin_table = calloc(TOKENS * head_dim, sizeof(*sin_table));
        require(cos_table && sin_table, "cannot allocate layer rotary tables");
        rope_table(cos_table, sin_table, spec, TOKENS);

        float *normed = malloc(TOKENS * HIDDEN * sizeof(*normed));
        float *query = malloc(TOKENS * inner * sizeof(*query));
        float *key = malloc(TOKENS * kv_inner * sizeof(*key));
        float *value = malloc(TOKENS * kv_inner * sizeof(*value));
        float *heads = malloc(TOKENS * inner * sizeof(*heads));
        float *attn_out = malloc(TOKENS * HIDDEN * sizeof(*attn_out));
        float *gate_out = malloc(TOKENS * INTERMEDIATE * sizeof(*gate_out));
        float *up_out = malloc(TOKENS * INTERMEDIATE * sizeof(*up_out));
        float *mlp_out = malloc(TOKENS * HIDDEN * sizeof(*mlp_out));
        require(normed && query && key && value && heads && attn_out &&
                    gate_out && up_out && mlp_out,
                "cannot allocate layer scratch");

        rms_norm(normed, hidden, input_norm, TOKENS, HIDDEN);
        linear(query, normed, q_weight, TOKENS, HIDDEN, inner);
        linear(key, normed, k_weight, TOKENS, HIDDEN, kv_inner);
        if (has_value) {
            linear(value, normed, v_weight, TOKENS, HIDDEN, kv_inner);
        } else {
            /* attention_k_eq_v: the value is the raw key projection, captured
             * before k_norm and before RoPE. */
            memcpy(value, key, TOKENS * kv_inner * sizeof(*value));
        }
        rms_norm(query, query, q_norm, TOKENS * QUERY_HEADS, head_dim);
        rms_norm(key, key, k_norm, TOKENS * spec->kv_heads, head_dim);
        apply_rope(query, cos_table, sin_table, TOKENS, QUERY_HEADS, head_dim);
        apply_rope(key, cos_table, sin_table, TOKENS, spec->kv_heads, head_dim);
        /* The value norm has no learnable scale and no rotation. */
        rms_norm(value, value, NULL, TOKENS * spec->kv_heads, head_dim);

        attention(heads, query, key, value, TOKENS, QUERY_HEADS,
                  spec->kv_heads, head_dim);
        linear(attn_out, heads, o_weight, TOKENS, inner, HIDDEN);
        rms_norm(attn_out, attn_out, post_attention, TOKENS, HIDDEN);
        for (size_t index = 0; index < TOKENS * HIDDEN; index++)
            hidden[index] += attn_out[index];

        rms_norm(normed, hidden, pre_ffn, TOKENS, HIDDEN);
        linear(gate_out, normed, gate, TOKENS, HIDDEN, INTERMEDIATE);
        linear(up_out, normed, up, TOKENS, HIDDEN, INTERMEDIATE);
        for (size_t index = 0; index < TOKENS * INTERMEDIATE; index++)
            gate_out[index] = gelu_tanh(gate_out[index]) * up_out[index];
        linear(mlp_out, gate_out, down, TOKENS, INTERMEDIATE, HIDDEN);
        rms_norm(mlp_out, mlp_out, post_ffn, TOKENS, HIDDEN);
        for (size_t index = 0; index < TOKENS * HIDDEN; index++)
            hidden[index] += mlp_out[index];

        for (size_t index = 0; index < TOKENS * HIDDEN; index++)
            hidden[index] *= spec->layer_scalar;

        /* The reference records the final layer's output already normalised,
         * so only the earlier layers can be compared directly here. */
        if (layer + 1 < LAYERS) {
            char label[64];
            snprintf(label, sizeof(label), "after layer %d (%s)", layer,
                     spec->sliding ? "sliding" : "global");
            char name[64];
            snprintf(name, sizeof(name), "reference.hidden_%d", layer + 1);
            float *expected = load_f32(name, TOKENS * HIDDEN);
            compare(label, hidden, expected, TOKENS * HIDDEN, 2e-5f);
            free(expected);
        }

        free(input_norm); free(post_attention); free(pre_ffn); free(post_ffn);
        free(q_weight); free(k_weight); free(o_weight); free(v_weight);
        free(q_norm); free(k_norm); free(gate); free(up); free(down);
        free(cos_table); free(sin_table);
        free(normed); free(query); free(key); free(value); free(heads);
        free(attn_out); free(gate_out); free(up_out); free(mlp_out);
    }

    float *final_norm = load_f32("norm.weight", HIDDEN);
    rms_norm(hidden, hidden, final_norm, TOKENS, HIDDEN);
    float *expected_final = load_f32("reference.last_hidden_state", TOKENS * HIDDEN);
    compare("final normed output", hidden, expected_final, TOKENS * HIDDEN, 2e-5f);

    free(final_norm); free(expected_final); free(embed); free(hidden);
    h3_weight_store_free(store);

    if (failures) {
        fprintf(stderr, "\n%d Gemma tower comparisons failed\n", failures);
        return 1;
    }
    printf("ok: the Gemma 4 tower reproduces the reference activations, "
           "covering both sliding and value-less global attention\n");
    return 0;
}
