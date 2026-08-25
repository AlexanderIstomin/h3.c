#include "h3_inpaint.h"

#include <stdarg.h>
#include <stdio.h>

enum {
    H3_INPAINT_CHANNELS = 3,
    H3_INPAINT_SPATIAL_RATIO = 16,
    H3_INPAINT_PATCH = 2,
    H3_INPAINT_CHUNK_FRAMES = 17,
    H3_INPAINT_CHUNK_LATENTS = 5
};

static const int h3_inpaint_frame_spans[H3_INPAINT_CHUNK_LATENTS] = {
    1, 4, 4, 4, 4
};

static const int h3_inpaint_frame_offsets[H3_INPAINT_CHUNK_LATENTS] = {
    0, 1, 5, 9, 13
};

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static float mask_value(const float *mask, int mask_frames,
                        int height, int width, int frame, int y, int x) {
    size_t area = (size_t)height * (size_t)width;
    size_t plane = area * (size_t)mask_frames;
    size_t pixel = (size_t)frame * area +
                   (size_t)y * (size_t)width + (size_t)x;
    float value = 0.0f;
    for (int channel = 0; channel < H3_INPAINT_CHANNELS; channel++)
        value += mask[(size_t)channel * plane + pixel];
    return value / (float)H3_INPAINT_CHANNELS;
}

int h3_inpaint_hard_mask_rows(const float *mask, int mask_frames,
                              int source_frames, int height, int width,
                              int latent_time, int latent_height,
                              int latent_width, uint8_t *rows,
                              size_t row_count,
                              char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    size_t wanted = latent_time > 0 && latent_height > 0 && latent_width > 0
        ? (size_t)latent_time * (size_t)(latent_height / H3_INPAINT_PATCH) *
          (size_t)(latent_width / H3_INPAINT_PATCH)
        : 0;
    if (!mask || !rows || source_frames < 1 ||
        (mask_frames != 1 && mask_frames != source_frames) ||
        height < 32 || width < 32 ||
        height != latent_height * H3_INPAINT_SPATIAL_RATIO ||
        width != latent_width * H3_INPAINT_SPATIAL_RATIO ||
        latent_height % H3_INPAINT_PATCH ||
        latent_width % H3_INPAINT_PATCH || !wanted || row_count != wanted) {
        fail(error, error_size, "invalid H3 inpainting mask geometry");
        return 0;
    }

    size_t at = 0;
    for (int latent = 0; latent < latent_time; latent++) {
        int slot = latent % H3_INPAINT_CHUNK_LATENTS;
        int first = (latent / H3_INPAINT_CHUNK_LATENTS) *
                        H3_INPAINT_CHUNK_FRAMES +
                    h3_inpaint_frame_offsets[slot];
        int stop = first + h3_inpaint_frame_spans[slot];
        if (first >= source_frames) first = source_frames - 1;
        if (stop > source_frames) stop = source_frames;
        if (latent == latent_time - 1) stop = source_frames;

        for (int patch_y = 0; patch_y < latent_height / H3_INPAINT_PATCH;
             patch_y++) {
            int first_y = patch_y * H3_INPAINT_PATCH *
                          H3_INPAINT_SPATIAL_RATIO;
            int stop_y = first_y + H3_INPAINT_PATCH *
                                   H3_INPAINT_SPATIAL_RATIO;
            for (int patch_x = 0;
                 patch_x < latent_width / H3_INPAINT_PATCH; patch_x++) {
                int first_x = patch_x * H3_INPAINT_PATCH *
                              H3_INPAINT_SPATIAL_RATIO;
                int stop_x = first_x + H3_INPAINT_PATCH *
                                       H3_INPAINT_SPATIAL_RATIO;
                uint8_t regenerate = 0;
                for (int frame = first; frame < stop && !regenerate; frame++) {
                    int mask_frame = mask_frames == 1 ? 0 : frame;
                    for (int y = first_y; y < stop_y && !regenerate; y++)
                        for (int x = first_x; x < stop_x; x++)
                            if (mask_value(mask, mask_frames, height, width,
                                           mask_frame, y, x) >= 0.5f) {
                                regenerate = 1;
                                break;
                            }
                }
                rows[at++] = regenerate;
            }
        }
    }
    return at == row_count;
}

int h3_inpaint_composite_rgb24(uint8_t *generated, int frames,
                               int height, int width,
                               const float *source,
                               const float *mask, int mask_frames,
                               char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!generated || !source || !mask || frames < 1 || height < 1 ||
        width < 1 || (mask_frames != 1 && mask_frames != frames)) {
        fail(error, error_size, "invalid H3 inpainting composite geometry");
        return 0;
    }
    size_t area = (size_t)height * (size_t)width;
    size_t source_plane = area * (size_t)frames;
    for (int frame = 0; frame < frames; frame++) {
        int mask_frame = mask_frames == 1 ? 0 : frame;
        for (size_t pixel = 0; pixel < area; pixel++) {
            int y = (int)(pixel / (size_t)width);
            int x = (int)(pixel % (size_t)width);
            if (mask_value(mask, mask_frames, height, width,
                           mask_frame, y, x) >= 0.5f) continue;
            for (int channel = 0; channel < H3_INPAINT_CHANNELS; channel++) {
                float value = source[(size_t)channel * source_plane +
                                     (size_t)frame * area + pixel] * 255.0f;
                if (value < 0.0f) value = 0.0f;
                if (value > 255.0f) value = 255.0f;
                generated[((size_t)frame * area + pixel) * 3 +
                          (size_t)channel] = (uint8_t)(value + 0.5f);
            }
        }
    }
    return 1;
}
