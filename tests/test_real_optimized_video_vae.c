#include "h3_video_vae.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

enum {
    LATENT_CHANNELS = 24,
    LATENT_TIME = 2,
    LATENT_HEIGHT = 1,
    LATENT_WIDTH = 1,
    LATENT_ELEMENTS = LATENT_CHANNELS * LATENT_TIME * LATENT_HEIGHT *
                      LATENT_WIDTH,
    OUTPUT_FRAMES = 5,
    OUTPUT_HEIGHT = 16,
    OUTPUT_WIDTH = 16
};

static void fail(const char *message) {
    fprintf(stderr, "FAIL tests/test_real_optimized_video_vae.c: %s\n",
            message);
    exit(1);
}

static void progress(int completed, int total, void *opaque) {
    (void)opaque;
    if (completed == 1 || completed == total || completed % 6 == 0)
        fprintf(stderr, "optimized VideoVAE stream: %d/%d blocks\n",
                completed, total);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s MODEL_ROOT\n", argv[0]);
        return 2;
    }
    char path[2048];
    int length = snprintf(
        path, sizeof(path), "%s/vae/minimax_h3_video_vae_fp16.safetensors",
        argv[1]);
    if (length < 0 || (size_t)length >= sizeof(path))
        fail("optimized VideoVAE path is too long");

    float latent[LATENT_ELEMENTS];
    for (size_t index = 0; index < LATENT_ELEMENTS; index++)
        latent[index] = (float)((int)(index % 13u) - 6) / 16.0f;

    char error[512];
    h3_video_frames frames;
    if (!h3_video_vae_decode(
            path, "h3_shaders.metal", latent, LATENT_TIME,
            LATENT_HEIGHT, LATENT_WIDTH, progress, NULL, &frames,
            error, sizeof(error))) {
        fprintf(stderr, "FAIL: optimized VideoVAE decode failed: %s\n", error);
        return 1;
    }
    if (frames.frames != OUTPUT_FRAMES || frames.height != OUTPUT_HEIGHT ||
        frames.width != OUTPUT_WIDTH || !frames.rgb)
        fail("optimized VideoVAE returned the wrong shape");

    size_t elements = (size_t)frames.frames * (size_t)frames.height *
                      (size_t)frames.width * 3;
    double square_sum = 0.0;
    float minimum = 1.0f, maximum = 0.0f;
    for (size_t index = 0; index < elements; index++) {
        float value = frames.rgb[index];
        if (!isfinite(value) || value < 0.0f || value > 1.0f)
            fail("optimized VideoVAE produced invalid RGB");
        if (value < minimum) minimum = value;
        if (value > maximum) maximum = value;
        square_sum += (double)value * (double)value;
    }
    if (!(square_sum > 0.0) || !(maximum > minimum))
        fail("optimized VideoVAE produced a degenerate frame");
    if (frames.gpu_stats.peak_live_bytes >=
        UINT64_C(1536) * 1024 * 1024)
        fail("optimized VideoVAE retained more than one expanded block");

    printf("ok: optimized VideoVAE decoded %d %dx%d frames, %.3f GiB peak "
           "Metal residency, %.3f GiB cumulative allocations\n",
           frames.frames, frames.width, frames.height,
           (double)frames.gpu_stats.peak_live_bytes /
               (1024.0 * 1024.0 * 1024.0),
           (double)frames.gpu_stats.allocated_bytes /
               (1024.0 * 1024.0 * 1024.0));
    h3_video_frames_free(&frames);
    return 0;
}
