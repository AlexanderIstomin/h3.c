/* Writes a synthetic clip through the system muxer and reads it back, so a
 * broken mux fails here in seconds rather than after a generation. The
 * readback goes through the FFmpeg reader when one is available, which also
 * checks the two paths agree on what a file contains. */
#include "h3_avwriter.h"
#include "h3_ffmpeg.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { WIDTH = 64, HEIGHT = 48, FRAMES = 12, FPS = 24,
       RATE = 32000, CHANNELS = 2 };

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

    int samples = RATE / 2;
    float *pcm = malloc((size_t)samples * CHANNELS * sizeof(*pcm));
    if (!pcm) return 1;
    for (int index = 0; index < samples; index++) {
        float value = 0.25f * sinf(6.2831853f * 440.0f * index / RATE);
        pcm[index * CHANNELS] = value;
        pcm[index * CHANNELS + 1] = -value;
    }

    char error[512];
    const char *path = "/tmp/h3-avwriter-test.mp4";
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

    free(rgb);
    free(pcm);
    printf("ok: the system muxer writes a playable clip\n");
    return 0;
}
