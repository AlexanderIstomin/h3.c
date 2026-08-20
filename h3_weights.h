#ifndef H3_WEIGHTS_H
#define H3_WEIGHTS_H

#include "h3_gpu.h"
#include "h3_safetensors.h"

#include <stddef.h>
#include <stdint.h>

typedef struct h3_weight_store h3_weight_store;

/* Open a single safetensors file or every safetensors header in a component
 * directory without reading tensor payloads. */
h3_weight_store *h3_weight_store_open(const char *location,
                                      char *error, size_t error_size);
void h3_weight_store_free(h3_weight_store *store);
size_t h3_weight_store_shards(const h3_weight_store *store);

const h3_st_tensor *h3_weight_find(const h3_weight_store *store,
                                   const char *name,
                                   const h3_st_header **header);

/* Validate an exact dtype and shape, allocate a shared Metal buffer, and read
 * the payload directly into that buffer with no intermediate host allocation. */
h3_gpu_tensor *h3_weight_load_bf16(const h3_weight_store *store, h3_gpu *gpu,
                                   const char *name, int ndim,
                                   const uint64_t *shape,
                                   char *error, size_t error_size);
h3_gpu_tensor *h3_weight_load_i8(const h3_weight_store *store, h3_gpu *gpu,
                                 const char *name, int ndim,
                                 const uint64_t *shape,
                                 char *error, size_t error_size);
h3_gpu_tensor *h3_weight_load_f32(const h3_weight_store *store, h3_gpu *gpu,
                                  const char *name, int ndim,
                                  const uint64_t *shape,
                                  char *error, size_t error_size);
h3_gpu_tensor *h3_weight_load_f16(const h3_weight_store *store, h3_gpu *gpu,
                                  const char *name, int ndim,
                                  const uint64_t *shape,
                                  char *error, size_t error_size);

/* Load IEEE F16 checkpoint tensors into F32 GPU storage when the consumer
 * needs the portable all-F32 path. */
h3_gpu_tensor *h3_weight_load_f16_as_f32(
    const h3_weight_store *store, h3_gpu *gpu, const char *name, int ndim,
    const uint64_t *shape, char *error, size_t error_size);

/* Read a small rank-one F32, F16, or BF16 tensor into ordinary host floats.
 * This is intended for checkpoint-owned configuration vectors such as VAE
 * latent means and standard deviations. */
int h3_weight_read_f32_vector(const h3_weight_store *store, const char *name,
                              float *values, size_t elements,
                              char *error, size_t error_size);

/* Load a pre-quantized row-major linear matrix and its per-output-channel
 * F32 scales. Checkpoints in the wild encode the scales as either [rows] or
 * [rows, 1]; both schemas have identical runtime storage and are accepted. */
int h3_weight_load_i8_linear(const h3_weight_store *store, h3_gpu *gpu,
                             const char *weight_name, uint64_t rows,
                             uint64_t columns, h3_gpu_tensor **weight,
                             h3_gpu_tensor **scales,
                             char *error, size_t error_size);

/* Read the optional Comfy quantization marker next to a linear weight. A
 * result of zero means the stored matrix is not ConvRot; a positive result is
 * the normalized regular-Hadamard group size required on the activation. */
int h3_weight_i8_linear_convrot_group(const h3_weight_store *store,
                                      const char *weight_name,
                                      uint32_t *group_size,
                                      char *error, size_t error_size);

#endif
