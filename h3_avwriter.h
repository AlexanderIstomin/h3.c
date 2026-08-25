#ifndef H3_AVWRITER_H
#define H3_AVWRITER_H

#include <stddef.h>
#include <stdint.h>

/* Muxing through the system's own frameworks rather than an FFmpeg process,
 * so a fresh install can produce video with nothing else installed and the
 * video encoder runs on hardware. */

/* Always true on macOS; present so callers can express the choice. */
int h3_avwriter_available(void);

/* Same contract as h3_ffmpeg_write_av_rgb24_f32: tightly packed RGB24 frames
 * and channel-major F32 PCM to an MP4. Passing no audio writes video only. */
int h3_avwriter_write_av_rgb24_f32(const char *path, const uint8_t *frames,
                                   int frame_count, int width, int height,
                                   int fps, const float *pcm, int samples,
                                   int channels, int sample_rate,
                                   char *error, size_t error_size);

#endif
