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
    LATENT_TIME = 7,   /* the 22-frame clip the app renders */
    LATENT_HEIGHT = 32,
    LATENT_WIDTH = 32,
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

/* The decoder object handles any canvas; the one-shot helper is limited to a
 * single spatial tile. Load is excluded from the timing so the number
 * reflects decode work. */
static double last_load_seconds = 0.0;

static double decode(const char *path, const float *latent,
                     h3_video_frames *frames) {
    char error[512];
    double load_started = now_seconds();
    h3_video_vae_decoder *decoder = h3_video_vae_decoder_load(
        path, "h3_shaders.metal", LATENT_HEIGHT, LATENT_WIDTH, NULL, NULL,
        error, sizeof(error));
    last_load_seconds = now_seconds() - load_started;
    if (!decoder) {
        fprintf(stderr, "load failed for %s: %s\n", path, error);
        exit(1);
    }
    double started = now_seconds();
    if (!h3_video_vae_decoder_decode(decoder, latent, LATENT_TIME, frames,
                                     error, sizeof(error))) {
        fprintf(stderr, "decode failed for %s: %s\n", path, error);
        exit(1);
    }
    double elapsed = now_seconds() - started;
    h3_video_vae_decoder_free(decoder);
    return elapsed;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s FP16_VAE INT8_VAE\n", argv[0]);
        return 2;
    }
    float *latent = malloc(LATENT_ELEMENTS * sizeof(*latent));
    if (!latent) fail("out of memory");
    /* A denoised latent is roughly unit-normal, not a small ramp; quantization
     * error depends on that dynamic range, so match it. */
    unsigned state = 12345u;
    for (size_t index = 0; index < LATENT_ELEMENTS; index++) {
        double sum = 0.0;
        for (int draw = 0; draw < 6; draw++) {
            state = state * 1103515245u + 12345u;
            sum += (double)((state >> 16) & 0x7fffu) / 32767.0;
        }
        latent[index] = (float)((sum - 3.0) * 1.4142);
    }

    h3_video_frames reference, quantized;
    memset(&reference, 0, sizeof(reference));
    memset(&quantized, 0, sizeof(quantized));
    double reference_seconds = decode(argv[1], latent, &reference);
    double reference_load = last_load_seconds;
    double quantized_seconds = decode(argv[2], latent, &quantized);
    double quantized_load = last_load_seconds;

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
    printf("fp16  load %.2f s + decode %.2f s = %.2f s\n",
           reference_load, reference_seconds, reference_load + reference_seconds);
    printf("int8  load %.2f s + decode %.2f s = %.2f s\n",
           quantized_load, quantized_seconds, quantized_load + quantized_seconds);
    printf("decode speed %.2fx | first render %.2fx | cached render %.2fx\n",
           quantized_seconds > 0.0 ? reference_seconds / quantized_seconds : 0.0,
           (quantized_load + quantized_seconds) > 0.0
             ? (reference_load + reference_seconds) /
               (quantized_load + quantized_seconds) : 0.0,
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
