#ifndef H3_INPAINT_H
#define H3_INPAINT_H

#include <stddef.h>
#include <stdint.h>

/* Reduce a black/white RGB mask to one hard value per target video row.
 * `mask` is channel-major [3,mask_frames,height,width]. A still mask has one
 * frame; a video mask must have one frame per source frame. The output order
 * is the DiT's [latent_time,latent_height/2,latent_width/2] row order. */
int h3_inpaint_hard_mask_rows(const float *mask, int mask_frames,
                              int source_frames, int height, int width,
                              int latent_time, int latent_height,
                              int latent_width, uint8_t *rows,
                              size_t row_count,
                              char *error, size_t error_size);

/* Restore every preserved source pixel after VAE decode. Generated frames are
 * frame-major interleaved RGB8; source and mask keep the readers' channel-
 * major F32 layouts. The hard 0.5 boundary matches the row reduction. */
int h3_inpaint_composite_rgb24(uint8_t *generated, int frames,
                               int height, int width,
                               const float *source,
                               const float *mask, int mask_frames,
                               char *error, size_t error_size);

#endif
