/* Check causal grouped attention on this GPU at Gemma 4's real head widths.
 *
 * h3.c's cooperative causal-GQA kernel is written for head dimensions up to
 * 128, which covers every MiniMax H3 shape. Gemma 4's text tower is wider:
 * its 40 sliding layers use 16 query heads over 8 key/value heads at head
 * dim 256, and its 8 global layers use 16 query heads over a single key/value
 * head at head dim 512. Those route to the shape-generic MPSGraph graph, and
 * this test is what says that route exists and is numerically right rather
 * than assuming it.
 *
 * Everything is synthetic, so no weights are needed. The host reference is the
 * same causal softmax the Gemma tower test uses. Each shape runs twice where it
 * matters: once at the conventional 1/sqrt(head_dim), which keeps the softmax
 * mixed and the comparison sensitive, and once at the scale 1.0 Gemma actually
 * uses, which on synthetic data of this amplitude saturates into a near-argmax
 * and so carries a wider tolerance. The heaviest attention weight is printed
 * with each result so a saturated, insensitive case cannot pass unnoticed. */

#include "h3_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *label;
    uint32_t tokens;
    uint32_t query_heads;
    uint32_t kv_heads;
    uint32_t head_dim;
    double tolerance;
} attention_case;

static const attention_case CASES[] = {
    /* An H3-shaped case, which the cooperative kernel still owns. */
    {"H3 DiT shape, 56 heads of 128",       64, 56, 56, 128, 6e-3},
    /* Gemma 4 12B, both layer families, at a realistic prompt length. */
    {"Gemma sliding, 16:8 heads of 256",   256, 16,  8, 256, 6e-3},
    {"Gemma global, 16:1 heads of 512",    256, 16,  1, 512, 6e-3},
    /* A short prompt exercises the same route with a different mask shape. */
    {"Gemma global, short prompt",           7, 16,  1, 512, 6e-3}
};

static int failures = 0;

static void fail(const char *message) {
    fprintf(stderr, "FAIL: %s\n", message);
    exit(1);
}

static void require(int condition, const char *message) {
    if (!condition) fail(message);
}

static uint16_t bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += UINT32_C(0x7fff) + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

static float bf16_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

/* Deterministic but properly decorrelated, and varying along every axis so a
 * transposed or broadcast-collapsed layout cannot pass by symmetry.
 *
 * A plain modular hash is not good enough here: its values stay structured
 * along the head dimension, which makes every query-key dot product large and
 * drives the softmax to a hard argmax. The comparison then agrees no matter
 * what the kernel does. SplitMix32 decorrelates the components, and the
 * amplitude is tied to 1/sqrt(head_dim) so the dot products stay near unit
 * scale at any width and the softmax keeps real spread. */
static float sample(uint32_t token, uint32_t head, uint32_t index,
                    uint32_t salt, uint32_t head_dim) {
    uint32_t state = token * 0x9e3779b9u + head * 0x85ebca6bu +
                     index * 0xc2b2ae35u + salt * 0x27d4eb2fu;
    state ^= state >> 16; state *= 0x7feb352du;
    state ^= state >> 15; state *= 0x846ca68bu;
    state ^= state >> 16;
    float unit = (float)(state >> 8) / (float)(1u << 24) * 2.0f - 1.0f;
    /* Choose the amplitude so query-key dot products land near a standard
     * deviation of 2-3 at Gemma's scale of 1.0. Too small and the softmax is
     * uniform, too large and it collapses to an argmax; either extreme makes
     * the comparison insensitive to what the kernel actually computes. */
    return unit * sqrtf(7.5f / sqrtf((float)head_dim));
}

static void fill(uint16_t *values, uint32_t tokens, uint32_t heads,
                 uint32_t head_dim, uint32_t salt) {
    for (uint32_t token = 0; token < tokens; token++)
        for (uint32_t head = 0; head < heads; head++)
            for (uint32_t index = 0; index < head_dim; index++)
                values[(token * heads + head) * head_dim + index] =
                    bf16(sample(token, head, index, salt, head_dim));
}

/* Causal softmax over BF16 inputs, accumulated in F32, with query heads
 * sharing key/value heads in contiguous groups. */
static void reference(float *out, const uint16_t *query, const uint16_t *key,
                      const uint16_t *value, uint32_t tokens,
                      uint32_t query_heads, uint32_t kv_heads,
                      uint32_t head_dim, float scale, double *peak) {
    uint32_t group = query_heads / kv_heads;
    double heaviest = 0.0;  /* mean largest weight on the last row */
    float *weights = malloc((size_t)tokens * sizeof(*weights));
    require(weights != NULL, "cannot allocate reference weights");
    for (uint32_t head = 0; head < query_heads; head++) {
        uint32_t kv_head = head / group;
        for (uint32_t token = 0; token < tokens; token++) {
            const uint16_t *q = query + ((size_t)token * query_heads + head) * head_dim;
            float maximum = -INFINITY;
            for (uint32_t other = 0; other <= token; other++) {
                const uint16_t *k = key + ((size_t)other * kv_heads + kv_head) * head_dim;
                float score = 0.0f;
                for (uint32_t index = 0; index < head_dim; index++)
                    score = fmaf(bf16_f32(q[index]), bf16_f32(k[index]), score);
                weights[other] = score * scale;
                score *= scale;
                if (score > maximum) maximum = score;
            }
            float total = 0.0f;
            for (uint32_t other = 0; other <= token; other++) {
                weights[other] = expf(weights[other] - maximum);
                total += weights[other];
            }
            /* Measure spread only on the final row. Row 0 attends to exactly
             * one key, so its largest weight is 1.0 by construction and says
             * nothing about whether the softmax is mixed. */
            if (token + 1 == tokens) {
                double largest = 0.0;
                for (uint32_t other = 0; other <= token; other++)
                    if (weights[other] / total > largest)
                        largest = weights[other] / total;
                heaviest += largest / (double)query_heads;
            }
            float *destination = out + ((size_t)token * query_heads + head) * head_dim;
            memset(destination, 0, head_dim * sizeof(*destination));
            for (uint32_t other = 0; other <= token; other++) {
                float share = weights[other] / total;
                const uint16_t *v = value + ((size_t)other * kv_heads + kv_head) * head_dim;
                for (uint32_t index = 0; index < head_dim; index++)
                    destination[index] = fmaf(share, bf16_f32(v[index]),
                                              destination[index]);
            }
        }
    }
    free(weights);
    if (peak) *peak = heaviest;
}

static void run_case(h3_gpu *gpu, const attention_case *item) {
    size_t query_count = (size_t)item->tokens * item->query_heads * item->head_dim;
    size_t kv_count = (size_t)item->tokens * item->kv_heads * item->head_dim;

    uint16_t *query_values = malloc(query_count * sizeof(*query_values));
    uint16_t *key_values = malloc(kv_count * sizeof(*key_values));
    uint16_t *value_values = malloc(kv_count * sizeof(*value_values));
    uint16_t *actual = malloc(query_count * sizeof(*actual));
    float *expected = malloc(query_count * sizeof(*expected));
    require(query_values && key_values && value_values && actual && expected,
            "cannot allocate attention buffers");

    fill(query_values, item->tokens, item->query_heads, item->head_dim, 1);
    fill(key_values, item->tokens, item->kv_heads, item->head_dim, 2);
    fill(value_values, item->tokens, item->kv_heads, item->head_dim, 3);

    h3_gpu_tensor *query = h3_gpu_tensor_from_bf16(gpu, query_values, query_count);
    h3_gpu_tensor *key = h3_gpu_tensor_from_bf16(gpu, key_values, kv_count);
    h3_gpu_tensor *value = h3_gpu_tensor_from_bf16(gpu, value_values, kv_count);
    h3_gpu_tensor *output = h3_gpu_tensor_new_bf16(gpu, query_count);
    require(query && key && value && output, "cannot allocate attention tensors");

    /* Gemma applies no 1/sqrt(head_dim) factor anywhere. */
    const float scale = 1.0f;
    require(h3_gpu_begin(gpu), "cannot begin the attention path");
    int supported = h3_gpu_gqa_causal_bf16(gpu, output, query, key, value,
                                           item->tokens, item->query_heads,
                                           item->kv_heads, item->head_dim, scale);
    if (!supported) {
        fprintf(stderr, "FAIL %-36s unsupported: %s\n", item->label,
                h3_gpu_error(gpu));
        failures++;
        h3_gpu_tensor_free(query); h3_gpu_tensor_free(key);
        h3_gpu_tensor_free(value); h3_gpu_tensor_free(output);
        free(query_values); free(key_values); free(value_values);
        free(actual); free(expected);
        return;
    }
    require(h3_gpu_submit(gpu), "attention submit failed");
    require(h3_gpu_tensor_read_bf16(output, actual, query_count),
            "cannot read the attention output");

    double peak = 0.0;
    reference(expected, query_values, key_values, value_values, item->tokens,
              item->query_heads, item->kv_heads, item->head_dim, scale, &peak);

    double worst = 0.0;
    size_t worst_at = 0;
    for (size_t index = 0; index < query_count; index++) {
        double delta = fabs((double)bf16_f32(actual[index]) -
                            (double)expected[index]);
        if (delta > worst) { worst = delta; worst_at = index; }
    }
    /* BF16 storage carries about three decimal digits, and the value average
     * is over at most `tokens` terms, so a few thousandths is the floor here. */
    double tolerance = item->tolerance;
    if (worst > tolerance) {
        fprintf(stderr, "FAIL %-36s worst |delta| %.3e at %zu (%.6f vs %.6f)\n",
                item->label, worst, worst_at,
                bf16_f32(actual[worst_at]), expected[worst_at]);
        failures++;
    } else if (peak > 0.9 || peak < 2.5 / (double)item->tokens) {
        /* Both extremes make the comparison a no-op: a saturated softmax
         * returns one value row, a uniform one returns their mean, and either
         * is reproduced by an implementation that is subtly wrong. */
        fprintf(stderr, "FAIL %-36s softmax is degenerate (heaviest %.4f "
                "against uniform %.4f), so the comparison proves little\n",
                item->label, peak, 1.0 / (double)item->tokens);
        failures++;
    } else {
        printf("  ok  %-36s worst |delta| %.2e  heaviest weight %.3f\n",
               item->label, worst, peak);
    }

    h3_gpu_tensor_free(query); h3_gpu_tensor_free(key);
    h3_gpu_tensor_free(value); h3_gpu_tensor_free(output);
    free(query_values); free(key_values); free(value_values);
    free(actual); free(expected);
}

int main(void) {
    char error[512];
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) {
        fprintf(stderr, "FAIL: cannot create Metal context: %s\n", error);
        return 1;
    }
    printf("causal grouped attention at Gemma 4 head widths:\n");
    size_t count = sizeof(CASES) / sizeof(*CASES);
    for (size_t index = 0; index < count; index++) run_case(gpu, &CASES[index]);
    h3_gpu_free(gpu);

    if (failures) {
        fprintf(stderr, "\n%d attention cases failed\n", failures);
        return 1;
    }
    printf("ok: %zu causal attention shapes match a host reference, including "
           "the 256- and 512-wide heads the Gemma tower needs\n", count);
    return 0;
}
