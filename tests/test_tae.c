/* Loads the tiny preview decoder and decodes a synthetic latent, so a broken
 * dispatch surfaces in seconds instead of after a denoising step. The output
 * of a random latent is meaningless as a picture; the checks are that every
 * stage dispatches and the pixels come back finite and in range. */
#include "h3_tae.h"

#include <math.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

enum { LATENT_T = 7, LATENT_H = 28, LATENT_W = 28, CHANNELS = 24 };

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s TAEH3_SAFETENSORS\n", argv[0]);
        return 2;
    }
    char error[512];
    h3_tae *tae = h3_tae_load(argv[1], "h3_shaders.metal", error, sizeof(error));
    if (!tae) {
        fprintf(stderr, "FAIL load: %s\n", error);
        return 1;
    }
    if (!h3_tae_prepare(tae, LATENT_H, LATENT_W, error, sizeof(error))) {
        fprintf(stderr, "FAIL prepare: %s\n", error);
        return 1;
    }
    size_t elements = (size_t)CHANNELS * LATENT_T * LATENT_H * LATENT_W;
    float *latent = malloc(elements * sizeof(*latent));
    if (!latent) return 1;
    unsigned state = 20260814u;
    for (size_t index = 0; index < elements; index++) {
        state = state * 1103515245u + 12345u;
        latent[index] = ((float)((state >> 16) & 0x7fffu) / 16383.5f) - 1.0f;
    }
    h3_video_frames frames;
    int frame_index = -1;
    if (!h3_tae_decode_preview(tae, latent, LATENT_T, &frames, &frame_index,
                               error, sizeof(error))) {
        fprintf(stderr, "FAIL decode: %s\n", error);
        return 1;
    }
    printf("decoded %dx%d, representative frame %d\n", frames.width,
           frames.height, frame_index);
    if (frames.width != LATENT_W * 16 || frames.height != LATENT_H * 16) {
        fprintf(stderr, "FAIL: unexpected canvas\n");
        return 1;
    }
    size_t pixels = (size_t)frames.width * frames.height * 3;
    double sum = 0.0;
    float low = 1.0f, high = 0.0f;
    for (size_t index = 0; index < pixels; index++) {
        float value = frames.rgb[index];
        if (!isfinite(value) || value < 0.0f || value > 1.0f) {
            fprintf(stderr, "FAIL: pixel %zu out of range (%f)\n", index, value);
            return 1;
        }
        sum += value;
        if (value < low) low = value;
        if (value > high) high = value;
    }
    printf("mean %.4f, range [%.4f, %.4f]\n", sum / (double)pixels,
           (double)low, (double)high);

    /* Steady-state cost: what each denoising step actually pays once the
     * Metal library and graphs are built. */
    struct timespec started, ended;
    enum { REPEATS = 10 };
    clock_gettime(CLOCK_MONOTONIC, &started);
    for (int index = 0; index < REPEATS; index++) {
        h3_video_frames again;
        int frame_again = 0;
        if (!h3_tae_decode_preview(tae, latent, LATENT_T, &again, &frame_again,
                                   error, sizeof(error))) {
            fprintf(stderr, "FAIL repeat decode: %s\n", error);
            return 1;
        }
        h3_video_frames_free(&again);
    }
    clock_gettime(CLOCK_MONOTONIC, &ended);
    double seconds = (double)(ended.tv_sec - started.tv_sec) +
                     (double)(ended.tv_nsec - started.tv_nsec) * 1e-9;
    printf("steady-state decode %.0f ms per preview\n",
           seconds / REPEATS * 1000.0);
    if (high - low < 1e-6) {
        fprintf(stderr, "FAIL: constant output\n");
        return 1;
    }
    h3_video_frames_free(&frames);
    free(latent);
    h3_tae_free(tae);
    printf("ok: TAE decodes end to end\n");
    return 0;
}
