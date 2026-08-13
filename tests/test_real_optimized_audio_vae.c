#include "h3_audio_vae.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    LATENT_CHANNELS = 32,
    STEREO = 2,
    LATENT_LENGTH = 1,
    LATENT_ELEMENTS = LATENT_CHANNELS * STEREO * LATENT_LENGTH,
    EXPECTED_SAMPLES = 800
};

static void fail(const char *message) {
    fprintf(stderr, "FAIL tests/test_real_optimized_audio_vae.c: %s\n",
            message);
    exit(1);
}

static void progress(int completed, int total, void *opaque) {
    (void)opaque;
    fprintf(stderr, "optimized AudioVAE stream: %d/%d stages\n",
            completed, total);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s MODEL_ROOT\n", argv[0]);
        return 2;
    }
    char path[2048];
    int length = snprintf(
        path, sizeof(path), "%s/vae/minimax_h3_audio_vae_fp32.safetensors",
        argv[1]);
    if (length < 0 || (size_t)length >= sizeof(path))
        fail("optimized AudioVAE path is too long");

    float latent[LATENT_ELEMENTS];
    for (size_t index = 0; index < LATENT_ELEMENTS; index++)
        latent[index] = (float)((int)(index % 19u) - 9) / 32.0f;

    char error[512];
    h3_audio_waveform waveform;
    if (!h3_audio_vae_decode(
            path, "h3_shaders.metal", latent, LATENT_LENGTH,
            progress, NULL, &waveform, error, sizeof(error))) {
        fprintf(stderr, "FAIL: optimized AudioVAE decode failed: %s\n", error);
        return 1;
    }
    if (waveform.channels != STEREO ||
        waveform.samples != EXPECTED_SAMPLES ||
        waveform.sample_rate != 32000 || !waveform.pcm)
        fail("optimized AudioVAE returned the wrong shape");

    double square_sum = 0.0;
    size_t elements = (size_t)waveform.channels * (size_t)waveform.samples;
    for (size_t index = 0; index < elements; index++) {
        float value = waveform.pcm[index];
        if (!isfinite(value) || value < -1.0f || value > 1.0f)
            fail("optimized AudioVAE produced invalid PCM");
        square_sum += (double)value * (double)value;
    }
    if (!(square_sum > 0.0))
        fail("optimized AudioVAE produced silent all-zero PCM");
    if (waveform.gpu_stats.peak_live_bytes >=
        UINT64_C(1024) * 1024 * 1024)
        fail("optimized AudioVAE exceeded its bounded stage residency");

    printf("ok: optimized AudioVAE decoded %d samples, %.3f GiB peak "
           "Metal residency, %.3f GiB cumulative allocations\n",
           waveform.samples,
           (double)waveform.gpu_stats.peak_live_bytes /
               (1024.0 * 1024.0 * 1024.0),
           (double)waveform.gpu_stats.allocated_bytes /
               (1024.0 * 1024.0 * 1024.0));
    h3_audio_waveform_free(&waveform);
    return 0;
}
