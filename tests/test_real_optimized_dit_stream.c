#include "h3_dit.h"
#include "h3_host.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    DIT_LAYERS = 50,
    TEXT_TOKENS = 1,
    TEXT_WIDTH = 5120,
    LATENT_TIME = 1,
    LATENT_HEIGHT = 2,
    LATENT_WIDTH = 2,
    AUDIO_TIME = 1,
    VIDEO_ELEMENTS = 24 * LATENT_TIME * LATENT_HEIGHT * LATENT_WIDTH,
    AUDIO_ELEMENTS = 32 * 2 * AUDIO_TIME
};

static void fail(const char *message) {
    fprintf(stderr, "FAIL tests/test_real_optimized_dit_stream.c: %s\n",
            message);
    exit(1);
}

static uint16_t bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += UINT32_C(0x7fff) + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

static void progress(const char *phase, int completed, int total,
                     void *opaque) {
    (void)opaque;
    if (completed == total || completed == 1 || completed % 10 == 0)
        fprintf(stderr, "optimized DiT %s: %d/%d\n",
                phase, completed, total);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s MODEL_ROOT\n", argv[0]);
        return 2;
    }
    char path[2048];
    int path_length = snprintf(
        path, sizeof(path), "%s/diffusion_models/"
        "minimax_h3_fl2va_pruned_int8_convrot.safetensors", argv[1]);
    if (path_length < 0 || (size_t)path_length >= sizeof(path))
        fail("optimized DiT path is too long");

    uint16_t text_values[TEXT_WIDTH];
    for (size_t index = 0; index < TEXT_WIDTH; index++)
        text_values[index] = bf16(
            (float)((int)(index % 31u) - 15) / 64.0f);
    h3_text_embedding text = {
        .tokens = TEXT_TOKENS,
        .width = TEXT_WIDTH,
        .values = text_values
    };
    h3_layout_spec spec = {
        TEXT_TOKENS, LATENT_TIME, LATENT_HEIGHT, LATENT_WIDTH, AUDIO_TIME,
        5, NULL, 0, NULL, 0
    };
    h3_layout layout;
    h3_sigma_schedule sigmas;
    char error[512];
    if (!h3_layout_build(&spec, &layout, error, sizeof(error))) fail(error);
    if (!h3_serving_schedule_build(2, &sigmas))
        fail("cannot build two-step serving schedule");

    h3_dit *dit = h3_dit_load_t2va(
        path, "h3_shaders.metal", &text, &layout, &sigmas,
        DIT_LAYERS, 1, 0, 1, 1.0f,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        progress, NULL, error, sizeof(error));
    if (!dit) {
        fprintf(stderr, "FAIL: optimized DiT load failed: %s\n", error);
        return 1;
    }

    float video[VIDEO_ELEMENTS], audio[AUDIO_ELEMENTS];
    float video_velocity[VIDEO_ELEMENTS], audio_velocity[AUDIO_ELEMENTS];
    for (size_t index = 0; index < VIDEO_ELEMENTS; index++)
        video[index] = (float)((int)(index % 17u) - 8) / 16.0f;
    for (size_t index = 0; index < AUDIO_ELEMENTS; index++)
        audio[index] = (float)((int)(index % 11u) - 5) / 16.0f;
    if (!h3_dit_forward(dit, 0, video, audio,
                        video_velocity, audio_velocity,
                        error, sizeof(error))) {
        fprintf(stderr, "FAIL: optimized DiT forward failed: %s\n", error);
        return 1;
    }
    double square_sum = 0.0;
    for (size_t index = 0; index < VIDEO_ELEMENTS; index++) {
        if (!isfinite(video_velocity[index]))
            fail("optimized DiT video velocity is not finite");
        square_sum += (double)video_velocity[index] * video_velocity[index];
    }
    for (size_t index = 0; index < AUDIO_ELEMENTS; index++) {
        if (!isfinite(audio_velocity[index]))
            fail("optimized DiT audio velocity is not finite");
        square_sum += (double)audio_velocity[index] * audio_velocity[index];
    }
    if (!(square_sum > 0.0))
        fail("optimized DiT velocities are all zero");

    h3_gpu_stats stats;
    if (!h3_dit_get_gpu_stats(dit, &stats))
        fail("cannot read optimized DiT GPU statistics");
    if (stats.peak_live_bytes >= UINT64_C(4) * 1024 * 1024 * 1024)
        fail("optimized DiT exceeded the bounded two-layer stream");
    printf("ok: optimized 50-layer DiT forward, %.3f GiB peak Metal "
           "residency, %.3f GiB cumulative allocations\n",
           (double)stats.peak_live_bytes / (1024.0 * 1024.0 * 1024.0),
           (double)stats.allocated_bytes / (1024.0 * 1024.0 * 1024.0));

    h3_dit_free(dit);
    h3_layout_free(&layout);
    return 0;
}
