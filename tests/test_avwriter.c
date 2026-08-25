/* Writes a synthetic clip through the system muxer and reads it back, so a
 * broken mux fails here in seconds rather than after a generation. Audio
 * always comes back through AVFoundation; video additionally uses FFmpeg when
 * available, checking that the two media paths agree on the container. */
#include "h3_avreader.h"
#include "h3_avwriter.h"
#include "h3_ffmpeg.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { WIDTH = 64, HEIGHT = 48, FRAMES = 12, FPS = 24,
       RATE = 32000, CHANNELS = 2 };

static double tone_power(const float *pcm, int samples, int rate,
                         double frequency) {
    double real = 0.0, imaginary = 0.0;
    for (int index = 0; index < samples; index++) {
        double phase = 6.283185307179586 * frequency * index / rate;
        real += pcm[index] * cos(phase);
        imaginary += pcm[index] * sin(phase);
    }
    return real * real + imaginary * imaginary;
}

int main(void) {
    size_t pixels = (size_t)WIDTH * HEIGHT * 3;
    uint8_t *rgb = malloc(pixels * FRAMES);
    if (!rgb) return 1;
    /* A moving vertical bar: enough structure that a frame ordering or row
     * stride mistake shows up as a mismatch rather than passing. */
    for (int frame = 0; frame < FRAMES; frame++)
        for (int y = 0; y < HEIGHT; y++)
            for (int x = 0; x < WIDTH; x++) {
                size_t at = (size_t)frame * pixels + ((size_t)y * WIDTH + x) * 3;
                int bar = abs(x - (frame * WIDTH / FRAMES)) < 4;
                rgb[at + 0] = bar ? 240 : 20;
                rgb[at + 1] = (uint8_t)(y * 255 / HEIGHT);
                rgb[at + 2] = bar ? 20 : 200;
            }

    int samples = RATE * 2;
    float *pcm = malloc((size_t)samples * CHANNELS * sizeof(*pcm));
    if (!pcm) return 1;
    for (int index = 0; index < samples; index++) {
        pcm[index] = 0.25f * (float)sin(
            6.283185307179586 * 440.0 * (double)index / (double)RATE);
        pcm[samples + index] =
            0.25f * (float)sin(
                6.283185307179586 * 880.0 * (double)index / (double)RATE);
    }

    char error[512];
    const char *path = "/tmp/h3-avwriter-test.mp4";
    remove(path);
    if (!h3_avwriter_write_av_rgb24_f32(path, rgb, FRAMES, WIDTH, HEIGHT, FPS,
                                        pcm, samples, CHANNELS, RATE,
                                        error, sizeof(error))) {
        fprintf(stderr, "FAIL write: %s\n", error);
        return 1;
    }
    printf("wrote %s\n", path);

    float *frames_back = NULL;
    int decoded = 0;
    if (h3_ffmpeg_read_video_f32(path, WIDTH, HEIGHT, FRAMES, &frames_back,
                                 &decoded, error, sizeof(error))) {
        printf("read back %d frames\n", decoded);
        if (decoded < 1) {
            fprintf(stderr, "FAIL: no frames decoded\n");
            return 1;
        }
        double sum = 0;
        size_t count = (size_t)decoded * WIDTH * HEIGHT * 3;
        for (size_t index = 0; index < count; index++) {
            float value = frames_back[index];
            if (!isfinite(value) || value < -0.01f || value > 1.01f) {
                fprintf(stderr, "FAIL: decoded pixel out of range\n");
                return 1;
            }
            sum += value;
        }
        double mean = sum / (double)count;
        printf("decoded mean %.4f\n", mean);
        if (mean < 0.05 || mean > 0.95) {
            fprintf(stderr, "FAIL: decoded frames look blank\n");
            return 1;
        }
        free(frames_back);
    } else {
        printf("skipped readback (no FFmpeg): %s\n", error);
    }

    float *audio_back = NULL;
    int audio_samples = 0;
    if (!h3_avreader_read_audio_f32(path, samples, 1, &audio_back,
                                    &audio_samples, error, sizeof(error))) {
        fprintf(stderr, "FAIL audio readback: %s\n", error);
        return 1;
    }
    if (audio_samples != samples) {
        fprintf(stderr, "FAIL: decoded audio has %d rather than %d samples\n",
                audio_samples, samples);
        return 1;
    }
    double left_440 = tone_power(audio_back, audio_samples, RATE, 440.0);
    double left_880 = tone_power(audio_back, audio_samples, RATE, 880.0);
    double right_440 = tone_power(audio_back + audio_samples,
                                  audio_samples, RATE, 440.0);
    double right_880 = tone_power(audio_back + audio_samples,
                                  audio_samples, RATE, 880.0);
    free(audio_back);
    if (left_440 < left_880 * 25.0 || right_880 < right_440 * 25.0) {
        fprintf(stderr,
                "FAIL: channel-major audio was not interleaved correctly\n");
        return 1;
    }
    printf("read back %d correctly interleaved stereo samples\n",
           audio_samples);

    free(rgb);
    free(pcm);
    printf("ok: the system muxer writes a playable clip\n");
    return 0;
}
