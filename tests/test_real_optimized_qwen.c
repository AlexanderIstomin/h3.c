#include "h3_text_encoder.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { QWEN_LAYERS = 50, QWEN_WIDTH = 5120 };

static void fail(const char *message) {
    fprintf(stderr, "FAIL tests/test_real_optimized_qwen.c: %s\n", message);
    exit(1);
}

static void progress(int completed, int total, void *opaque) {
    (void)opaque;
    if (completed == 1 || completed == total || completed % 5 == 0)
        fprintf(stderr, "optimized Qwen stream: %d/%d layers\n",
                completed, total);
}

static float bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 4) {
        fprintf(stderr, "usage: %s MODEL_ROOT [LAYERS [TOKENS]]\n", argv[0]);
        return 2;
    }
    int layers = argc >= 3 ? atoi(argv[2]) : 1;
    if (layers < 1 || layers > QWEN_LAYERS)
        fail("layer count must be in [1, 50]");
    int token_count = argc == 4 ? atoi(argv[3]) : 2;
    if (token_count < 1 || token_count > 512)
        fail("token count must be in [1, 512]");
    size_t path_size = strlen(argv[1]) + 128;
    char *path = malloc(path_size);
    if (!path) fail("cannot allocate optimized Qwen path");
    int length = snprintf(
        path, path_size, "%s/text_encoders/"
        "qwen3vl_32b_minimax_h3_int8_convrot.safetensors", argv[1]);
    if (length < 0 || (size_t)length >= path_size)
        fail("optimized Qwen path is too long");

    /* Ordinary vocabulary rows exercise causal attention. Two rows keep the
     * default 50-layer residency probe quick; an explicit count also covers
     * production-size prompt projection paths. */
    uint32_t *ids = malloc((size_t)token_count * sizeof(*ids));
    if (!ids) fail("cannot allocate token IDs");
    for (int index = 0; index < token_count; index++)
        ids[index] = (uint32_t)(index % 151936);
    char error[512];
    h3_text_embedding embedding;
    if (!h3_text_encode_layers_bf16(
            path, "h3_shaders.metal", ids, (size_t)token_count, layers,
            progress, NULL, &embedding, error, sizeof(error))) {
        fprintf(stderr, "FAIL: optimized Qwen encoding failed: %s\n", error);
        return 1;
    }
    if (embedding.tokens != (size_t)token_count ||
        embedding.width != QWEN_WIDTH || !embedding.values)
        fail("optimized Qwen output has the wrong shape");
    size_t count = embedding.tokens * embedding.width;
    double square_sum = 0.0;
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t index = 0; index < count; index++) {
        float value = bf16_to_f32(embedding.values[index]);
        if (!isfinite(value)) fail("optimized Qwen output is not finite");
        square_sum += (double)value * (double)value;
        hash ^= embedding.values[index] & UINT16_C(0xff);
        hash *= UINT64_C(1099511628211);
        hash ^= embedding.values[index] >> 8;
        hash *= UINT64_C(1099511628211);
    }
    if (!(square_sum > 0.0)) fail("optimized Qwen output is all zero");

    int prefetch_depth = 1;
    const char *depth_value = getenv("H3_QWEN_PREFETCH_DEPTH");
    if (depth_value && *depth_value) {
        prefetch_depth = atoi(depth_value);
        if (prefetch_depth < 1) prefetch_depth = 1;
        if (prefetch_depth > 6) prefetch_depth = 6;
    }
    uint64_t peak_limit = (uint64_t)(layers == 1 ? 1 : prefetch_depth + 1) *
        UINT64_C(650) * 1024 * 1024;
    if (embedding.gpu_stats.peak_live_bytes >= peak_limit)
        fail("optimized Qwen retained more than the bounded layer ring");
    printf("ok: %d optimized Qwen layer%s, %.3f GiB peak Metal residency, "
           "%.3f GiB cumulative allocations, output hash %016llx\n",
           layers, layers == 1 ? "" : "s",
           (double)embedding.gpu_stats.peak_live_bytes /
               (1024.0 * 1024.0 * 1024.0),
           (double)embedding.gpu_stats.allocated_bytes /
               (1024.0 * 1024.0 * 1024.0),
           (unsigned long long)hash);
    h3_text_embedding_free(&embedding);
    free(ids);
    free(path);
    return 0;
}
