#include "h3_checkpoint.h"

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define H3_CHECKPOINT_HEADER_SIZE 144u
#define H3_CHECKPOINT_VERSION 2u
#define H3_CHECKPOINT_VELOCITIES 1u

static const unsigned char h3_checkpoint_magic[8] = {
    'H', '3', 'C', 'K', 'P', 'T', '2', '\n'
};

static void explain(char *detail, size_t size, const char *message) {
    if (detail && size) snprintf(detail, size, "%s", message);
}

static void explain_errno(char *detail, size_t size, const char *operation) {
    if (detail && size)
        snprintf(detail, size, "%s: %s", operation, strerror(errno));
}

static void put_u32(unsigned char *at, uint32_t value) {
    at[0] = (unsigned char)value;
    at[1] = (unsigned char)(value >> 8);
    at[2] = (unsigned char)(value >> 16);
    at[3] = (unsigned char)(value >> 24);
}

static void put_u64(unsigned char *at, uint64_t value) {
    for (unsigned shift = 0; shift < 64; shift += 8)
        at[shift / 8] = (unsigned char)(value >> shift);
}

static uint32_t get_u32(const unsigned char *at) {
    return (uint32_t)at[0] | (uint32_t)at[1] << 8 |
           (uint32_t)at[2] << 16 | (uint32_t)at[3] << 24;
}

static uint64_t get_u64(const unsigned char *at) {
    uint64_t value = 0;
    for (unsigned shift = 0; shift < 64; shift += 8)
        value |= (uint64_t)at[shift / 8] << shift;
    return value;
}

static uint64_t checksum_update(uint64_t checksum, const void *bytes,
                                size_t count) {
    const unsigned char *data = bytes;
    for (size_t index = 0; index < count; index++) {
        checksum ^= data[index];
        checksum *= UINT64_C(1099511628211);
    }
    return checksum;
}

static int fingerprint_valid(const char *fingerprint) {
    if (!fingerprint || strlen(fingerprint) != 64) return 0;
    for (size_t index = 0; index < 64; index++) {
        char value = fingerprint[index];
        if (!((value >= '0' && value <= '9') ||
              (value >= 'a' && value <= 'f'))) return 0;
    }
    return 1;
}

static int count_bytes(size_t count, size_t *bytes) {
    if (count > SIZE_MAX / sizeof(float)) return 0;
    *bytes = count * sizeof(float);
    return 1;
}

static int write_all(int file, const void *bytes, size_t count) {
    const unsigned char *cursor = bytes;
    while (count) {
        ssize_t written = write(file, cursor, count);
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0) return 0;
        cursor += (size_t)written;
        count -= (size_t)written;
    }
    return 1;
}

static int read_all(int file, void *bytes, size_t count) {
    unsigned char *cursor = bytes;
    while (count) {
        ssize_t received = read(file, cursor, count);
        if (received < 0 && errno == EINTR) continue;
        if (received <= 0) return 0;
        cursor += (size_t)received;
        count -= (size_t)received;
    }
    return 1;
}

void h3_checkpoint_state_free(h3_checkpoint_state *state) {
    if (!state) return;
    free(state->video);
    free(state->audio);
    free(state->last_video_velocity);
    free(state->last_audio_velocity);
    free(state->previous_video_velocity);
    free(state->previous_audio_velocity);
    memset(state, 0, sizeof(*state));
}

void h3_checkpoint_remove(const char *path) {
    if (path && *path && unlink(path) && errno != ENOENT)
        fprintf(stderr, "h3: warning: cannot remove checkpoint %s: %s\n",
                path, strerror(errno));
}

static int state_valid(const h3_checkpoint_state *state,
                       size_t *video_bytes, size_t *audio_bytes,
                       uint32_t *flags, uint64_t *payload_bytes) {
    if (!state || state->total_steps < 1 || state->next_step < 1 ||
        state->next_step > state->total_steps || state->reuse_interval < 1 ||
        !state->video || !state->audio ||
        !count_bytes(state->video_count, video_bytes) ||
        !count_bytes(state->audio_count, audio_bytes)) return 0;
    *flags = 0;
    uint64_t multiplier = 1;
    if (state->reuse_interval > 1 && state->next_step < state->total_steps) {
        if (state->last_evaluated < 0 || !state->last_video_velocity ||
            !state->last_audio_velocity) return 0;
        *flags |= H3_CHECKPOINT_VELOCITIES;
        multiplier += state->previous_evaluated >= 0 ? 2 : 1;
        if (state->previous_evaluated >= 0 &&
            (!state->previous_video_velocity ||
             !state->previous_audio_velocity)) return 0;
    }
    uint64_t pair = (uint64_t)*video_bytes + (uint64_t)*audio_bytes;
    if (multiplier && pair > UINT64_MAX / multiplier) return 0;
    *payload_bytes = pair * multiplier;
    return 1;
}

static uint64_t state_checksum(const unsigned char *header,
                               const h3_checkpoint_state *state,
                               size_t video_bytes, size_t audio_bytes,
                               uint32_t flags) {
    uint64_t checksum = UINT64_C(14695981039346656037);
    checksum = checksum_update(checksum, header, H3_CHECKPOINT_HEADER_SIZE);
    checksum = checksum_update(checksum, state->video, video_bytes);
    checksum = checksum_update(checksum, state->audio, audio_bytes);
    if (flags & H3_CHECKPOINT_VELOCITIES) {
        checksum = checksum_update(
            checksum, state->last_video_velocity, video_bytes);
        checksum = checksum_update(
            checksum, state->last_audio_velocity, audio_bytes);
        if (state->previous_evaluated >= 0) {
            checksum = checksum_update(
                checksum, state->previous_video_velocity, video_bytes);
            checksum = checksum_update(
                checksum, state->previous_audio_velocity, audio_bytes);
        }
    }
    return checksum;
}

int h3_checkpoint_save(const char *path, const char *fingerprint,
                       const h3_checkpoint_state *state,
                       char *detail, size_t detail_size) {
    if (detail && detail_size) detail[0] = '\0';
    size_t video_bytes = 0, audio_bytes = 0;
    uint32_t flags = 0;
    uint64_t payload_bytes = 0;
    if (!path || !*path || !fingerprint_valid(fingerprint) ||
        !state_valid(state, &video_bytes, &audio_bytes, &flags,
                     &payload_bytes)) {
        explain(detail, detail_size, "invalid checkpoint state");
        return 0;
    }

    unsigned char header[H3_CHECKPOINT_HEADER_SIZE] = {0};
    memcpy(header, h3_checkpoint_magic, sizeof(h3_checkpoint_magic));
    put_u32(header + 8, H3_CHECKPOINT_VERSION);
    put_u32(header + 12, H3_CHECKPOINT_HEADER_SIZE);
    memcpy(header + 16, fingerprint, 64);
    put_u32(header + 80, (uint32_t)state->total_steps);
    put_u32(header + 84, (uint32_t)state->next_step);
    put_u32(header + 88, (uint32_t)state->reuse_interval);
    put_u32(header + 92, flags);
    put_u32(header + 96, (uint32_t)state->last_evaluated);
    put_u32(header + 100, (uint32_t)state->previous_evaluated);
    put_u64(header + 104, (uint64_t)state->video_count);
    put_u64(header + 112, (uint64_t)state->audio_count);
    put_u64(header + 120, payload_bytes);
    put_u64(header + 128, 0);
    put_u64(header + 128, state_checksum(
        header, state, video_bytes, audio_bytes, flags));

    size_t temporary_length = strlen(path) + 48;
    char *temporary = malloc(temporary_length);
    if (!temporary) {
        explain(detail, detail_size,
                "out of memory naming checkpoint temporary file");
        return 0;
    }
    snprintf(temporary, temporary_length, "%s.tmp.%ld", path, (long)getpid());
    int file = open(temporary, O_CREAT | O_TRUNC | O_WRONLY, 0600);
    if (file < 0) {
        explain_errno(detail, detail_size, "cannot create checkpoint");
        free(temporary);
        return 0;
    }
    (void)fchmod(file, 0600);
    int ok = write_all(file, header, sizeof(header)) &&
             write_all(file, state->video, video_bytes) &&
             write_all(file, state->audio, audio_bytes);
    if (ok && (flags & H3_CHECKPOINT_VELOCITIES)) {
        ok = write_all(file, state->last_video_velocity, video_bytes) &&
             write_all(file, state->last_audio_velocity, audio_bytes);
        if (ok && state->previous_evaluated >= 0)
            ok = write_all(file, state->previous_video_velocity, video_bytes) &&
                 write_all(file, state->previous_audio_velocity, audio_bytes);
    }
    if (ok) ok = fsync(file) == 0;
    int close_status = close(file);
    if (ok) ok = close_status == 0;
    if (ok) ok = rename(temporary, path) == 0;
    if (!ok) {
        explain_errno(detail, detail_size, "cannot commit checkpoint");
        (void)unlink(temporary);
    }
    free(temporary);
    return ok;
}

static float *allocate_values(size_t bytes) {
    return bytes ? malloc(bytes) : NULL;
}

int h3_checkpoint_load(const char *path, const char *fingerprint,
                       int total_steps, int reuse_interval,
                       size_t video_count, size_t audio_count,
                       h3_checkpoint_state *state,
                       char *detail, size_t detail_size) {
    if (detail && detail_size) detail[0] = '\0';
    if (!path || !*path || !fingerprint_valid(fingerprint) || !state ||
        total_steps < 1 || reuse_interval < 1) {
        explain(detail, detail_size, "invalid checkpoint request");
        return 0;
    }
    int file = open(path, O_RDONLY);
    if (file < 0) {
        if (errno != ENOENT)
            explain_errno(detail, detail_size, "cannot open checkpoint");
        return 0;
    }
    unsigned char header[H3_CHECKPOINT_HEADER_SIZE];
    int ok = read_all(file, header, sizeof(header));
    if (!ok || memcmp(header, h3_checkpoint_magic, 8) ||
        get_u32(header + 8) != H3_CHECKPOINT_VERSION ||
        get_u32(header + 12) != H3_CHECKPOINT_HEADER_SIZE) {
        explain(detail, detail_size, "checkpoint header is incomplete or unsupported");
        close(file);
        return 0;
    }
    int next_step = (int)get_u32(header + 84);
    int last_evaluated = (int32_t)get_u32(header + 96);
    int previous_evaluated = (int32_t)get_u32(header + 100);
    uint32_t flags = get_u32(header + 92);
    if (memcmp(header + 16, fingerprint, 64) ||
        get_u32(header + 80) != (uint32_t)total_steps ||
        get_u32(header + 88) != (uint32_t)reuse_interval ||
        get_u64(header + 104) != (uint64_t)video_count ||
        get_u64(header + 112) != (uint64_t)audio_count ||
        next_step < 1 || next_step > total_steps ||
        (flags & ~H3_CHECKPOINT_VELOCITIES)) {
        explain(detail, detail_size, "checkpoint does not match this generation");
        close(file);
        return 0;
    }
    size_t video_bytes = 0, audio_bytes = 0;
    if (!count_bytes(video_count, &video_bytes) ||
        !count_bytes(audio_count, &audio_bytes)) {
        explain(detail, detail_size, "checkpoint dimensions overflow");
        close(file);
        return 0;
    }
    uint64_t multiplier = 1;
    if (flags & H3_CHECKPOINT_VELOCITIES)
        multiplier += previous_evaluated >= 0 ? 2 : 1;
    uint64_t pair = (uint64_t)video_bytes + (uint64_t)audio_bytes;
    if (pair > UINT64_MAX / multiplier ||
        get_u64(header + 120) != pair * multiplier ||
        ((reuse_interval > 1 && next_step < total_steps) !=
         !!(flags & H3_CHECKPOINT_VELOCITIES)) ||
        ((flags & H3_CHECKPOINT_VELOCITIES) &&
         (last_evaluated < 0 || last_evaluated >= next_step ||
          previous_evaluated < -1 ||
          previous_evaluated >= last_evaluated))) {
        explain(detail, detail_size, "checkpoint sampler state is inconsistent");
        close(file);
        return 0;
    }

    h3_checkpoint_state loaded = {
        .total_steps = total_steps,
        .next_step = next_step,
        .reuse_interval = reuse_interval,
        .last_evaluated = last_evaluated,
        .previous_evaluated = previous_evaluated,
        .video_count = video_count,
        .audio_count = audio_count,
        .video = allocate_values(video_bytes),
        .audio = allocate_values(audio_bytes)
    };
    if (flags & H3_CHECKPOINT_VELOCITIES) {
        loaded.last_video_velocity = allocate_values(video_bytes);
        loaded.last_audio_velocity = allocate_values(audio_bytes);
        if (previous_evaluated >= 0) {
            loaded.previous_video_velocity = allocate_values(video_bytes);
            loaded.previous_audio_velocity = allocate_values(audio_bytes);
        }
    }
    if (!loaded.video || !loaded.audio ||
        ((flags & H3_CHECKPOINT_VELOCITIES) &&
         (!loaded.last_video_velocity || !loaded.last_audio_velocity)) ||
        (previous_evaluated >= 0 &&
         (!loaded.previous_video_velocity ||
          !loaded.previous_audio_velocity))) {
        explain(detail, detail_size, "out of memory loading checkpoint");
        h3_checkpoint_state_free(&loaded);
        close(file);
        return 0;
    }
    ok = read_all(file, loaded.video, video_bytes) &&
         read_all(file, loaded.audio, audio_bytes);
    if (ok && (flags & H3_CHECKPOINT_VELOCITIES)) {
        ok = read_all(file, loaded.last_video_velocity, video_bytes) &&
             read_all(file, loaded.last_audio_velocity, audio_bytes);
        if (ok && previous_evaluated >= 0)
            ok = read_all(file, loaded.previous_video_velocity, video_bytes) &&
                 read_all(file, loaded.previous_audio_velocity, audio_bytes);
    }
    unsigned char extra = 0;
    if (ok) {
        ssize_t trailing = read(file, &extra, 1);
        ok = trailing == 0;
    }
    close(file);
    uint64_t stored_checksum = get_u64(header + 128);
    put_u64(header + 128, 0);
    if (!ok || state_checksum(header, &loaded, video_bytes, audio_bytes, flags)
                   != stored_checksum) {
        explain(detail, detail_size, "checkpoint payload is incomplete or corrupt");
        h3_checkpoint_state_free(&loaded);
        return 0;
    }
    *state = loaded;
    return 1;
}
