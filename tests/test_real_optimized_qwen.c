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
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "usage: %s MODEL_ROOT [LAYERS]\n", argv[0]);
        return 2;
    }
    int layers = argc == 3 ? atoi(argv[2]) : 1;
    if (layers < 1 || layers > QWEN_LAYERS)
        fail("layer count must be in [1, 50]");
    size_t path_size = strlen(argv[1]) + 128;
    char *path = malloc(path_size);
    if (!path) fail("cannot allocate optimized Qwen path");
    int length = snprintf(
        path, path_size, "%s/text_encoders/"
        "qwen3vl_32b_minimax_h3_int8_convrot.safetensors", argv[1]);
    if (length < 0 || (size_t)length >= path_size)
        fail("optimized Qwen path is too long");

    /* Two ordinary vocabulary rows exercise causal attention while keeping the
     * full 50-layer residency probe quick enough for local development. */
    const uint32_t ids[] = {0, 1};
    char error[512];
    h3_text_embedding embedding;
    if (!h3_text_encode_layers_bf16(
            path, "h3_shaders.metal", ids, sizeof(ids) / sizeof(*ids), layers,
            progress, NULL, &embedding, error, sizeof(error))) {
        fprintf(stderr, "FAIL: optimized Qwen encoding failed: %s\n", error);
        return 1;
    }
    if (embedding.tokens != sizeof(ids) / sizeof(*ids) ||
        embedding.width != QWEN_WIDTH || !embedding.values)
        fail("optimized Qwen output has the wrong shape");
    size_t count = embedding.tokens * embedding.width;
    double square_sum = 0.0;
    for (size_t index = 0; index < count; index++) {
        float value = bf16_to_f32(embedding.values[index]);
        if (!isfinite(value)) fail("optimized Qwen output is not finite");
        square_sum += (double)value * (double)value;
    }
    if (!(square_sum > 0.0)) fail("optimized Qwen output is all zero");

    uint64_t peak_limit = layers == 1 ?
        UINT64_C(650) * 1024 * 1024 :
        UINT64_C(1300) * 1024 * 1024;
    if (embedding.gpu_stats.peak_live_bytes >= peak_limit)
        fail("optimized Qwen retained more than the bounded layer ring");
    printf("ok: %d optimized Qwen layer%s, %.3f GiB peak Metal residency, "
           "%.3f GiB cumulative allocations\n",
           layers, layers == 1 ? "" : "s",
           (double)embedding.gpu_stats.peak_live_bytes /
               (1024.0 * 1024.0 * 1024.0),
           (double)embedding.gpu_stats.allocated_bytes /
               (1024.0 * 1024.0 * 1024.0));
    h3_text_embedding_free(&embedding);
    free(path);
    return 0;
}
