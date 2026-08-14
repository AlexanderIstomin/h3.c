/* Probes whether the Qwen vision tower in the optimized single-file package
 * loads and encodes: the open question for reference and keyframe support. */
#include "h3_vision_encoder.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s TEXT_ENCODER\n", argv[0]);
        return 2;
    }
    enum { H = 64, W = 64 };
    float *pixels = malloc((size_t)3 * H * W * sizeof(*pixels));
    if (!pixels) return 1;
    for (int c = 0; c < 3; c++)
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                pixels[(size_t)(c * H + y) * W + x] =
                    0.5f + 0.4f * sinf((float)(x + y * (c + 1)) * 0.11f);

    char error[512];
    h3_vision_output output;
    memset(&output, 0, sizeof(output));
    if (!h3_vision_encode_bf16(argv[1], "h3_shaders.metal", pixels, 1, H, W,
                               NULL, NULL, &output, error, sizeof(error))) {
        fprintf(stderr, "FAIL: %s\n", error);
        return 1;
    }
    printf("grid %dx%d, %zu tokens x %u width\n", output.grid_h, output.grid_w,
           output.tokens, H3_VISION_OUTPUT_WIDTH);
    size_t count = output.tokens * H3_VISION_OUTPUT_WIDTH;
    double square_sum = 0.0;
    for (size_t i = 0; i < count; i++) {
        uint32_t bits = (uint32_t)output.merged[i] << 16;
        float v;
        memcpy(&v, &bits, sizeof(v));
        if (!isfinite(v)) { fprintf(stderr, "FAIL: non-finite output\n"); return 1; }
        square_sum += (double)v * v;
    }
    if (!(square_sum > 0.0)) { fprintf(stderr, "FAIL: all-zero output\n"); return 1; }
    printf("rms %.4f | deepstacks %s\n", sqrt(square_sum / (double)count),
           output.deepstack[0] && output.deepstack[1] && output.deepstack[2]
             ? "present" : "MISSING");
    printf("ok: the optimized package's vision tower encodes\n");
    return 0;
}
