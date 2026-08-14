#ifndef H3_TAE_H
#define H3_TAE_H

#include "h3_video_vae.h"

#include <stddef.h>

/* Tiny 2D approximation of the video VAE decoder, for denoising previews.
 * Decodes one latent time slice to one RGB frame in milliseconds where the
 * full decoder needs seconds per preview and gigabytes of residency. The
 * weights are Kijai's taeh3 (Apache-2.0, ~9 MB); output is soft but easily
 * recognizable, which is all a live preview needs. */
typedef struct h3_tae h3_tae;

h3_tae *h3_tae_load(const char *weight_path, const char *shader_source_path,
                    char *error, size_t error_size);
void h3_tae_free(h3_tae *tae);

/* Mirrors h3_video_vae_decoder_preview: takes the denoiser's normalized
 * [24, T, H, W] latent, picks the same representative frame the full preview
 * picks, and returns one decoded frame at 16x the latent size. */
/* Sizes the work buffers for a latent canvas; call once before decoding. */
int h3_tae_prepare(h3_tae *tae, int latent_h, int latent_w,
                   char *error, size_t error_size);

int h3_tae_decode_preview(h3_tae *tae, const float *normalized_latent,
                          int latent_time, h3_video_frames *output,
                          int *output_frame_index,
                          char *error, size_t error_size);

#endif
