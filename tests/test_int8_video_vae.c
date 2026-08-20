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

static void test_layer_major(const char *path, const float *latent) {
    h3_video_frames legacy, layer_major;
    memset(&legacy, 0, sizeof(legacy));
    memset(&layer_major, 0, sizeof(layer_major));

    /* Force both sides through the low-memory path, while retaining the
     * production auto-tile plan. The only difference between the runs is the
     * traversal order under test. */
    setenv("H3_VAE_RESIDENT", "0", 1);
    unsetenv("H3_VAE_TILE_PIXELS");
    setenv("H3_VAE_LAYER_MAJOR", "0", 1);
    double legacy_seconds = decode(path, latent, &legacy);
    double legacy_load = last_load_seconds;
    setenv("H3_VAE_LAYER_MAJOR", "1", 1);
    double layer_major_seconds = decode(path, latent, &layer_major);
    double layer_major_load = last_load_seconds;

    if (legacy.frames != layer_major.frames ||
        legacy.height != layer_major.height ||
        legacy.width != layer_major.width)
        fail("layer-major decoder returned a different shape");
    size_t elements = (size_t)legacy.frames * (size_t)legacy.height *
                      (size_t)legacy.width * 3;
    if (memcmp(legacy.rgb, layer_major.rgb, elements * sizeof(*legacy.rgb)))
        fail("layer-major decoder is not byte-identical to tile-major");

    printf("shape %dx%dx%d\n", legacy.frames, legacy.height, legacy.width);
    printf("tile-major  load %.2f s + decode %.2f s = %.2f s\n",
           legacy_load, legacy_seconds, legacy_load + legacy_seconds);
    printf("layer-major load %.2f s + decode %.2f s = %.2f s\n",
           layer_major_load, layer_major_seconds,
           layer_major_load + layer_major_seconds);
    printf("streaming decode speed %.2fx\n",
           layer_major_seconds > 0.0 ?
               legacy_seconds / layer_major_seconds : 0.0);
    h3_video_frames_free(&legacy);
    h3_video_frames_free(&layer_major);
    puts("ok: layer-major streaming is byte-identical to tile-major");
}

static void test_native_f16(const char *path, const float *latent) {
    h3_video_frames expanded, native;
    memset(&expanded, 0, sizeof(expanded));
    memset(&native, 0, sizeof(native));

    setenv("H3_VAE_RESIDENT", "1", 1);
    unsetenv("H3_VAE_TILE_PIXELS");
    setenv("H3_VAE_NATIVE_F16", "0", 1);
    double expanded_seconds = decode(path, latent, &expanded);
    double expanded_load = last_load_seconds;
    setenv("H3_VAE_NATIVE_F16", "1", 1);
    double native_seconds = decode(path, latent, &native);
    double native_load = last_load_seconds;

    if (expanded.frames != native.frames ||
        expanded.height != native.height || expanded.width != native.width)
        fail("native F16 decoder returned a different shape");
    size_t elements = (size_t)expanded.frames * (size_t)expanded.height *
                      (size_t)expanded.width * 3;
    if (memcmp(expanded.rgb, native.rgb, elements * sizeof(*expanded.rgb)))
        fail("native F16 decoder is not byte-identical to expanded F32");

    printf("shape %dx%dx%d\n", expanded.frames, expanded.height,
           expanded.width);
    printf("expanded F32 load %.2f s + decode %.2f s = %.2f s, "
           "peak %.3f GiB\n",
           expanded_load, expanded_seconds, expanded_load + expanded_seconds,
           (double)expanded.gpu_stats.peak_live_bytes /
               (1024.0 * 1024.0 * 1024.0));
    printf("native F16   load %.2f s + decode %.2f s = %.2f s, "
           "peak %.3f GiB\n",
           native_load, native_seconds, native_load + native_seconds,
           (double)native.gpu_stats.peak_live_bytes /
               (1024.0 * 1024.0 * 1024.0));
    h3_video_frames_free(&expanded);
    h3_video_frames_free(&native);
    puts("ok: native F16 video VAE is byte-identical to expanded F32");
}

int main(int argc, char **argv) {
    int layer_major_test = argc == 3 && !strcmp(argv[1], "--layer-major");
    int native_f16_test = argc == 3 && !strcmp(argv[1], "--native-f16");
    if ((!layer_major_test && !native_f16_test && argc != 3) ||
        ((layer_major_test || native_f16_test) && !argv[2][0])) {
        fprintf(stderr,
                "usage: %s FP16_VAE INT8_VAE\n"
                "       %s --layer-major VAE\n"
                "       %s --native-f16 VAE\n", argv[0], argv[0], argv[0]);
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

    if (layer_major_test) {
        test_layer_major(argv[2], latent);
        free(latent);
        return 0;
    }
    if (native_f16_test) {
        test_native_f16(argv[2], latent);
        free(latent);
        return 0;
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
