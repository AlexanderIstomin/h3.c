#ifndef H3_CHECKPOINT_H
#define H3_CHECKPOINT_H

#include <stddef.h>

/* A denoiser checkpoint is deliberately independent from model loading and
 * application state. The caller supplies a SHA-256 fingerprint of every
 * immutable generation input; a file with any other fingerprint is ignored. */
typedef struct {
    int total_steps;
    int next_step;
    int reuse_interval;
    int last_evaluated;
    int previous_evaluated;
    size_t video_count;
    size_t audio_count;
    float *video;
    float *audio;
    float *last_video_velocity;
    float *last_audio_velocity;
    float *previous_video_velocity;
    float *previous_audio_velocity;
} h3_checkpoint_state;

/* Returns 1 when a matching checkpoint was loaded and 0 when the path is
 * absent, stale, incomplete, or corrupt. A rejected file never changes
 * `state`; `detail` explains why for diagnostic logging. */
int h3_checkpoint_load(const char *path, const char *fingerprint,
                       int total_steps, int reuse_interval,
                       size_t video_count, size_t audio_count,
                       h3_checkpoint_state *state,
                       char *detail, size_t detail_size);

/* Writes through a mode-0600 temporary file, fsyncs it, then atomically
 * replaces `path`. The previous complete checkpoint survives interruption. */
int h3_checkpoint_save(const char *path, const char *fingerprint,
                       const h3_checkpoint_state *state,
                       char *detail, size_t detail_size);

void h3_checkpoint_state_free(h3_checkpoint_state *state);
void h3_checkpoint_remove(const char *path);

#endif
