/* Decodes one latent through the released fp16 decoder and the int8 ConvRot
 * decoder and compares the pixels. A quantized VAE that drifts here shows up
 * as visible artifacts in generated video, so the comparison is the gate. */
#include "h3_video_vae.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum {
    LATENT_TIME = 2,
    LATENT_HEIGHT = 16,
    LATENT_WIDTH = 16,
    LATENT_CHANNELS = 24,
    LATENT_ELEMENTS = LATENT_CHANNELS * LATENT_TIME * LATENT_HEIGHT * LATENT_WIDTH
};

static void fail(const char *message) {
    fprintf(stderr, "FAIL tests/test_int8_video_vae.c: %s\n", message);
    exit(1);
}

static double now_seconds(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0.0;
    return (double)value.tv_sec + (double)value.tv_nsec * 1e-9;
}

static double decode(const char *path, const float *latent,
                     h3_video_frames *frames) {
    char error[512];
    double started = now_seconds();
    if (!h3_video_vae_decode(path, "h3_shaders.metal", latent, LATENT_TIME,
                             LATENT_HEIGHT, LATENT_WIDTH, NULL, NULL, frames,
                             error, sizeof(error))) {
        fprintf(stderr, "decode failed for %s: %s\n", path, error);
        exit(1);
    }
    return now_seconds() - started;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s FP16_VAE INT8_VAE\n", argv[0]);
        return 2;
    }
    float *latent = malloc(LATENT_ELEMENTS * sizeof(*latent));
    if (!latent) fail("out of memory");
    for (size_t index = 0; index < LATENT_ELEMENTS; index++)
        latent[index] = (float)((int)(index % 13u) - 6) / 16.0f;

    h3_video_frames reference, quantized;
    memset(&reference, 0, sizeof(reference));
    memset(&quantized, 0, sizeof(quantized));
    double reference_seconds = decode(argv[1], latent, &reference);
    double quantized_seconds = decode(argv[2], latent, &quantized);

    if (reference.frames != quantized.frames ||
        reference.height != quantized.height ||
        reference.width != quantized.width)
        fail("decoders returned different shapes");

    size_t elements = (size_t)reference.frames * (size_t)reference.height *
                      (size_t)reference.width * 3;
    double sum = 0.0, square_sum = 0.0;
    float worst = 0.0f;
    for (size_t index = 0; index < elements; index++) {
        float a = reference.rgb[index], b = quantized.rgb[index];
        if (!isfinite(b) || b < 0.0f || b > 1.0f)
            fail("int8 decoder produced invalid RGB");
        float difference = fabsf(a - b);
        if (difference > worst) worst = difference;
        sum += difference;
        square_sum += (double)difference * difference;
    }
    double mean = sum / (double)elements;
    double rmse = sqrt(square_sum / (double)elements);
    double psnr = rmse > 0.0 ? 20.0 * log10(1.0 / rmse) : 99.0;

    printf("shape %dx%dx%d\n", reference.frames, reference.height,
           reference.width);
    printf("fp16 decode  %.2f s\nint8 decode  %.2f s  (%.2fx)\n",
           reference_seconds, quantized_seconds,
           quantized_seconds > 0.0 ? reference_seconds / quantized_seconds : 0.0);
    printf("mean |Δ| %.5f | worst |Δ| %.5f | PSNR %.1f dB\n",
           mean, (double)worst, psnr);

    h3_video_frames_free(&reference);
    h3_video_frames_free(&quantized);
    free(latent);
    /* Below roughly 30 dB the difference is visible; the released decoder is
     * the reference, so anything worse means the int8 path is wrong. */
    if (psnr < 30.0) fail("int8 decoder deviates too far from fp16");
    printf("ok: int8 ConvRot video VAE matches the released decoder\n");
    return 0;
}
