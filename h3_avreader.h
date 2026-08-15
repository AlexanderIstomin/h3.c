#ifndef H3_AVREADER_H
#define H3_AVREADER_H

#include "h3_ffmpeg.h"

#include <stddef.h>

/* Reading media through the system's own frameworks rather than an FFmpeg
 * process. These mirror the h3_ffmpeg_read_* contracts exactly, so a caller
 * can try one and fall back to the other. The system frameworks cover the
 * containers a macOS user is likely to hand an app — MP4, MOV, M4A, WAV,
 * MP3, AAC, and every image format ImageIO knows — but decline some that
 * FFmpeg reads, such as Matroska, which is why the fallback stays. */

/* Channel-major F32 [3,height,width] RGB in [0,1]. */
int h3_avreader_read_image_f32(const char *path, int width, int height,
                               h3_image_fit fit, float **pixels,
                               char *error, size_t error_size);

/* Channel-major F32 [3,T,H,W] in [0,1], resampled to 24 fps and trimmed to
 * the released 5+17k cadence. */
int h3_avreader_read_video_f32(const char *path, int width, int height,
                               int max_frames, float **pixels, int *frames,
                               char *error, size_t error_size);

/* Channel-major stereo F32 at 32 kHz. */
int h3_avreader_read_audio_f32(const char *path, int max_samples,
                               int truncate_at_limit, float **pcm,
                               int *samples, char *error, size_t error_size);

#endif
