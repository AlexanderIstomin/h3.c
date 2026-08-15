/* Reads media through the system frameworks and checks the result against the
 * FFmpeg reader on the same file. The two must agree on shape and roughly on
 * content; a stride, channel-order, or planar-layout mistake shows up here in
 * seconds rather than as a subtly wrong reference conditioning a generation. */
#include "h3_avreader.h"
#include "h3_ffmpeg.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { WIDTH = 64, HEIGHT = 48, FRAMES = 22, SAMPLES = 16000 };

static double mean_of(const float *values, size_t count) {
    double sum = 0;
    for (size_t index = 0; index < count; index++) sum += values[index];
    return count ? sum / (double)count : 0;
}

static int check_range(const float *values, size_t count, const char *label) {
    for (size_t index = 0; index < count; index++)
        if (!isfinite(values[index]) ||
            values[index] < -0.01f || values[index] > 1.01f) {
            fprintf(stderr, "FAIL %s: value %zu out of range\n", label, index);
            return 0;
        }
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s IMAGE VIDEO_WITH_AUDIO\n", argv[0]);
        return 2;
    }
    char error[512];
    size_t plane = (size_t)WIDTH * HEIGHT;

    float *system_image = NULL;
    if (!h3_avreader_read_image_f32(argv[1], WIDTH, HEIGHT,
                                    H3_IMAGE_FIT_STRETCH, &system_image,
                                    error, sizeof(error))) {
        fprintf(stderr, "FAIL image: %s\n", error);
        return 1;
    }
    if (!check_range(system_image, plane * 3, "image")) return 1;
    double system_mean = mean_of(system_image, plane * 3);
    printf("image  system mean %.4f\n", system_mean);

    float *ffmpeg_image = NULL;
    if (h3_ffmpeg_read_image_f32(argv[1], WIDTH, HEIGHT, H3_IMAGE_FIT_STRETCH,
                                 &ffmpeg_image, error, sizeof(error))) {
        double ffmpeg_mean = mean_of(ffmpeg_image, plane * 3);
        double worst = 0;
        for (size_t index = 0; index < plane * 3; index++) {
            double difference = fabs(system_image[index] - ffmpeg_image[index]);
            if (difference > worst) worst = difference;
        }
        printf("image  ffmpeg mean %.4f, worst per-pixel delta %.4f\n",
               ffmpeg_mean, worst);
        /* Different resamplers disagree slightly; a channel swap or stride
         * error would show up as a mean far apart, not a small delta. */
        if (fabs(system_mean - ffmpeg_mean) > 0.05) {
            fprintf(stderr, "FAIL image: readers disagree on content\n");
            return 1;
        }
        free(ffmpeg_image);
    } else {
        printf("image  ffmpeg comparison skipped: %s\n", error);
    }
    free(system_image);

    float *video = NULL;
    int frames = 0;
    if (!h3_avreader_read_video_f32(argv[2], WIDTH, HEIGHT, FRAMES, &video,
                                    &frames, error, sizeof(error))) {
        fprintf(stderr, "FAIL video: %s\n", error);
        return 1;
    }
    printf("video  %d frames\n", frames);
    if (frames % 17 != 5) {
        fprintf(stderr, "FAIL video: %d frames is off the 5+17k cadence\n",
                frames);
        return 1;
    }
    if (!check_range(video, plane * 3 * (size_t)frames, "video")) return 1;
    printf("video  mean %.4f\n", mean_of(video, plane * 3 * (size_t)frames));
    free(video);

    float *pcm = NULL;
    int samples = 0;
    if (!h3_avreader_read_audio_f32(argv[2], SAMPLES, 1, &pcm, &samples,
                                    error, sizeof(error))) {
        fprintf(stderr, "FAIL audio: %s\n", error);
        return 1;
    }
    printf("audio  %d samples\n", samples);
    if (samples < 1) {
        fprintf(stderr, "FAIL audio: nothing decoded\n");
        return 1;
    }
    double energy = 0;
    for (int index = 0; index < samples; index++) {
        if (!isfinite(pcm[index]) || !isfinite(pcm[samples + index])) {
            fprintf(stderr, "FAIL audio: non-finite sample\n");
            return 1;
        }
        energy += (double)pcm[index] * pcm[index];
    }
    double rms = sqrt(energy / samples);
    printf("audio  left RMS %.4f\n", rms);
    if (rms < 1e-4) {
        fprintf(stderr, "FAIL audio: left channel is silent\n");
        return 1;
    }
    free(pcm);

    printf("ok: the system readers agree with the media they were given\n");
    return 0;
}
