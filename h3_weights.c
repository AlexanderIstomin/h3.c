#include "h3_weights.h"

#include <ctype.h>
#include <dirent.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

struct h3_weight_store {
    h3_st_header *headers;
    size_t count;
};

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static int safetensors_name(const char *name) {
    static const char suffix[] = ".safetensors";
    size_t length = strlen(name);
    return length > sizeof(suffix) - 1 &&
           strcmp(name + length - (sizeof(suffix) - 1), suffix) == 0;
}

static int compare_paths(const void *left, const void *right) {
    const char *const *a = left;
    const char *const *b = right;
    return strcmp(*a, *b);
}

static void free_paths(char **paths, size_t count) {
    if (!paths) return;
    for (size_t index = 0; index < count; index++) free(paths[index]);
    free(paths);
}

h3_weight_store *h3_weight_store_open(const char *location,
                                      char *error, size_t error_size) {
    if (!location || !*location) {
        fail(error, error_size, "weight file or directory is required");
        return NULL;
    }
    char **paths = NULL;
    size_t count = 0;
    size_t capacity = 0;
    struct stat status;
    if (stat(location, &status) != 0) {
        fail(error, error_size, "cannot inspect weight location: %s", location);
        return NULL;
    }
    if (S_ISREG(status.st_mode)) {
        if (!safetensors_name(location)) {
            fail(error, error_size, "weight file is not safetensors: %s", location);
            return NULL;
        }
        paths = calloc(1, sizeof(*paths));
        if (paths) paths[0] = strdup(location);
        if (!paths || !paths[0]) {
            fail(error, error_size, "out of memory resolving weight file");
            free_paths(paths, paths && paths[0] ? 1 : 0);
            return NULL;
        }
        count = 1;
    } else if (S_ISDIR(status.st_mode)) {
        DIR *stream = opendir(location);
        if (!stream) {
            fail(error, error_size, "cannot open weight directory: %s", location);
            return NULL;
        }
        struct dirent *entry;
        while ((entry = readdir(stream)) != NULL) {
            if (!safetensors_name(entry->d_name)) continue;
            if (count == capacity) {
                size_t next = capacity ? capacity * 2 : 8;
                char **grown = realloc(paths, next * sizeof(*grown));
                if (!grown) {
                    fail(error, error_size, "out of memory listing weight shards");
                    closedir(stream);
                    free_paths(paths, count);
                    return NULL;
                }
                paths = grown;
                capacity = next;
            }
            size_t length = strlen(location) + strlen(entry->d_name) + 2;
            paths[count] = malloc(length);
            if (!paths[count]) {
                fail(error, error_size, "out of memory resolving a weight shard");
                closedir(stream);
                free_paths(paths, count);
                return NULL;
            }
            snprintf(paths[count], length, "%s/%s", location, entry->d_name);
            count++;
        }
        closedir(stream);
    } else {
        fail(error, error_size, "weight location is not a file or directory: %s",
             location);
        return NULL;
    }
    if (!count) {
        fail(error, error_size, "no safetensors shards in %s", location);
        free(paths);
        return NULL;
    }
    qsort(paths, count, sizeof(*paths), compare_paths);
    h3_weight_store *store = calloc(1, sizeof(*store));
    if (!store) {
        fail(error, error_size, "out of memory creating weight store");
        free_paths(paths, count);
        return NULL;
    }
    store->headers = calloc(count, sizeof(*store->headers));
    if (!store->headers) {
        fail(error, error_size, "out of memory allocating weight headers");
        free(store);
        free_paths(paths, count);
        return NULL;
    }
    store->count = count;
    for (size_t index = 0; index < count; index++) {
        char detail[384];
        if (!h3_st_read_header(paths[index], &store->headers[index], detail,
                               sizeof(detail))) {
            fail(error, error_size, "%s", detail);
            free_paths(paths, count);
            h3_weight_store_free(store);
            return NULL;
        }
    }
    free_paths(paths, count);
    return store;
}

void h3_weight_store_free(h3_weight_store *store) {
    if (!store) return;
    for (size_t index = 0; index < store->count; index++) {
        h3_st_free_header(&store->headers[index]);
    }
    free(store->headers);
    free(store);
}

size_t h3_weight_store_shards(const h3_weight_store *store) {
    return store ? store->count : 0;
}

const h3_st_tensor *h3_weight_find(const h3_weight_store *store,
                                   const char *name,
                                   const h3_st_header **header) {
    if (header) *header = NULL;
    if (!store || !name) return NULL;
    for (size_t index = 0; index < store->count; index++) {
        const h3_st_tensor *tensor = h3_st_find(&store->headers[index], name);
        if (tensor) {
            if (header) *header = &store->headers[index];
            return tensor;
        }
    }
    return NULL;
}

static h3_gpu_tensor *load_tensor(const h3_weight_store *store, h3_gpu *gpu,
                                  const char *name, int ndim,
                                  const uint64_t *shape, h3_dtype dtype,
                                  char *error, size_t error_size) {
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(store, name, &header);
    if (!tensor) {
        fail(error, error_size, "required weight is absent: %s", name);
        return NULL;
    }
    if (tensor->dtype != dtype || tensor->ndim != ndim) {
        fail(error, error_size, "weight %s has dtype/rank %s/%d, expected %s/%d",
             name, h3_dtype_name(tensor->dtype), tensor->ndim,
             h3_dtype_name(dtype), ndim);
        return NULL;
    }
    uint64_t elements = 1;
    for (int dimension = 0; dimension < ndim; dimension++) {
        if (tensor->shape[dimension] != shape[dimension]) {
            fail(error, error_size, "weight %s shape mismatch at dimension %d",
                 name, dimension);
            return NULL;
        }
        if (shape[dimension] && elements > UINT64_MAX / shape[dimension]) {
            fail(error, error_size, "weight %s shape overflows", name);
            return NULL;
        }
        elements *= shape[dimension];
    }
    if (elements > SIZE_MAX) {
        fail(error, error_size, "weight %s is too large for this process", name);
        return NULL;
    }
    h3_gpu_tensor *result = NULL;
    switch (dtype) {
        case H3_DTYPE_I8:
            result = h3_gpu_tensor_load_i8(
                gpu, header->path, tensor->file_offset, (size_t)elements);
            break;
        case H3_DTYPE_BF16:
            result = h3_gpu_tensor_load_bf16(
                gpu, header->path, tensor->file_offset, (size_t)elements);
            break;
        case H3_DTYPE_F16:
            result = h3_gpu_tensor_load_f16(
                gpu, header->path, tensor->file_offset, (size_t)elements);
            break;
        case H3_DTYPE_F32:
            result = h3_gpu_tensor_load_f32(
                gpu, header->path, tensor->file_offset, (size_t)elements);
            break;
        default:
            fail(error, error_size, "unsupported GPU weight dtype %s",
                 h3_dtype_name(dtype));
            return NULL;
    }
    if (!result) {
        fail(error, error_size, "cannot load %s: %s", name, h3_gpu_error(gpu));
    }
    return result;
}

h3_gpu_tensor *h3_weight_load_bf16(const h3_weight_store *store, h3_gpu *gpu,
                                   const char *name, int ndim,
                                   const uint64_t *shape,
                                   char *error, size_t error_size) {
    return load_tensor(store, gpu, name, ndim, shape, H3_DTYPE_BF16,
                       error, error_size);
}

h3_gpu_tensor *h3_weight_load_i8(const h3_weight_store *store, h3_gpu *gpu,
                                 const char *name, int ndim,
                                 const uint64_t *shape,
                                 char *error, size_t error_size) {
    return load_tensor(store, gpu, name, ndim, shape, H3_DTYPE_I8,
                       error, error_size);
}

h3_gpu_tensor *h3_weight_load_f32(const h3_weight_store *store, h3_gpu *gpu,
                                  const char *name, int ndim,
                                  const uint64_t *shape,
                                  char *error, size_t error_size) {
    return load_tensor(store, gpu, name, ndim, shape, H3_DTYPE_F32,
                       error, error_size);
}

h3_gpu_tensor *h3_weight_load_f16(const h3_weight_store *store, h3_gpu *gpu,
                                  const char *name, int ndim,
                                  const uint64_t *shape,
                                  char *error, size_t error_size) {
    return load_tensor(store, gpu, name, ndim, shape, H3_DTYPE_F16,
                       error, error_size);
}

static float f16_to_f32(uint16_t value) {
    unsigned sign = value >> 15;
    unsigned exponent = (value >> 10) & 31u;
    unsigned fraction = value & 1023u;
    float result;
    if (!exponent) {
        result = ldexpf((float)fraction, -24);
    } else if (exponent == 31u) {
        result = fraction ? NAN : INFINITY;
    } else {
        result = ldexpf((float)(1024u + fraction), (int)exponent - 25);
    }
    return sign ? -result : result;
}

h3_gpu_tensor *h3_weight_load_f16_as_f32(
    const h3_weight_store *store, h3_gpu *gpu, const char *name, int ndim,
    const uint64_t *shape, char *error, size_t error_size) {
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(store, name, &header);
    if (!tensor) {
        fail(error, error_size, "required weight is absent: %s", name);
        return NULL;
    }
    if (!header || tensor->dtype != H3_DTYPE_F16 || tensor->ndim != ndim) {
        fail(error, error_size,
             "weight %s has dtype/rank %s/%d, expected F16/%d", name,
             h3_dtype_name(tensor->dtype), tensor->ndim, ndim);
        return NULL;
    }
    uint64_t elements = 1;
    for (int dimension = 0; dimension < ndim; dimension++) {
        if (tensor->shape[dimension] != shape[dimension]) {
            fail(error, error_size,
                 "weight %s shape mismatch at dimension %d", name,
                 dimension);
            return NULL;
        }
        if (shape[dimension] && elements > UINT64_MAX / shape[dimension]) {
            fail(error, error_size, "weight %s shape overflows", name);
            return NULL;
        }
        elements *= shape[dimension];
    }
    if (elements > SIZE_MAX / sizeof(float)) {
        fail(error, error_size, "weight %s is too large for this process",
             name);
        return NULL;
    }
    size_t count = (size_t)elements;
    h3_gpu_tensor *result = h3_gpu_tensor_new_f32(gpu, count);
    if (!result) {
        fail(error, error_size, "cannot load %s: %s", name,
             h3_gpu_error(gpu));
        return NULL;
    }
    if (!h3_gpu_tensor_stream_file_f16_as_f32(
            result, header->path, tensor->file_offset, count,
            error, error_size)) {
        h3_gpu_tensor_free(result);
        return NULL;
    }
    return result;
}

int h3_weight_read_f32_vector(const h3_weight_store *store, const char *name,
                              float *values, size_t elements,
                              char *error, size_t error_size) {
    if (!store || !name || !*name || !values || !elements) {
        fail(error, error_size, "invalid host weight-vector arguments");
        return 0;
    }
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(store, name, &header);
    if (!tensor) {
        fail(error, error_size, "required weight is absent: %s", name);
        return 0;
    }
    if (!header || tensor->ndim != 1 || tensor->shape[0] != elements ||
        (tensor->dtype != H3_DTYPE_F32 && tensor->dtype != H3_DTYPE_F16 &&
         tensor->dtype != H3_DTYPE_BF16)) {
        fail(error, error_size,
             "weight %s is not a supported %zu-element float vector",
             name, elements);
        return 0;
    }
    if (tensor->dtype == H3_DTYPE_F32)
        return h3_st_read_data(header, tensor, values,
                               elements * sizeof(*values),
                               error, error_size);
    if (elements > SIZE_MAX / sizeof(uint16_t)) {
        fail(error, error_size, "weight %s is too large for this process",
             name);
        return 0;
    }
    uint16_t *source = malloc(elements * sizeof(*source));
    if (!source) {
        fail(error, error_size, "out of memory reading weight %s", name);
        return 0;
    }
    int ok = h3_st_read_data(header, tensor, source,
                             elements * sizeof(*source), error, error_size);
    if (ok) for (size_t index = 0; index < elements; index++) {
        if (tensor->dtype == H3_DTYPE_F16) {
            values[index] = f16_to_f32(source[index]);
        } else {
            uint32_t bits = (uint32_t)source[index] << 16;
            memcpy(&values[index], &bits, sizeof(bits));
        }
    }
    free(source);
    return ok;
}

int h3_weight_load_i8_linear(const h3_weight_store *store, h3_gpu *gpu,
                             const char *weight_name, uint64_t rows,
                             uint64_t columns, h3_gpu_tensor **weight,
                             h3_gpu_tensor **scales,
                             char *error, size_t error_size) {
    if (weight) *weight = NULL;
    if (scales) *scales = NULL;
    if (!store || !gpu || !weight_name || !*weight_name || !rows ||
        !columns || !weight || !scales) {
        fail(error, error_size, "invalid pre-quantized linear arguments");
        return 0;
    }
    size_t name_size = strlen(weight_name) + sizeof("_scale");
    char *scale_name = malloc(name_size);
    if (!scale_name) {
        fail(error, error_size, "out of memory resolving linear scales");
        return 0;
    }
    snprintf(scale_name, name_size, "%s_scale", weight_name);
    const h3_st_tensor *scale_tensor = h3_weight_find(
        store, scale_name, NULL);
    if (!scale_tensor || scale_tensor->dtype != H3_DTYPE_F32 ||
        !((scale_tensor->ndim == 1 && scale_tensor->shape[0] == rows) ||
          (scale_tensor->ndim == 2 && scale_tensor->shape[0] == rows &&
           scale_tensor->shape[1] == 1))) {
        fail(error, error_size,
             "linear scale has the wrong schema: %s", scale_name);
        free(scale_name);
        return 0;
    }
    const uint64_t weight_shape[] = {rows, columns};
    const uint64_t scale_shape_1d[] = {rows};
    const uint64_t scale_shape_2d[] = {rows, 1};
    h3_gpu_tensor *loaded_weight = h3_weight_load_i8(
        store, gpu, weight_name, 2, weight_shape, error, error_size);
    h3_gpu_tensor *loaded_scales = NULL;
    if (loaded_weight) {
        loaded_scales = h3_weight_load_f32(
            store, gpu, scale_name, scale_tensor->ndim,
            scale_tensor->ndim == 1 ? scale_shape_1d : scale_shape_2d,
            error, error_size);
    }
    free(scale_name);
    if (!loaded_weight || !loaded_scales) {
        h3_gpu_tensor_free(loaded_weight);
        h3_gpu_tensor_free(loaded_scales);
        return 0;
    }
    *weight = loaded_weight;
    *scales = loaded_scales;
    return 1;
}

static const char *json_value(const char *json, const char *key) {
    const char *position = strstr(json, key);
    if (!position) return NULL;
    position = strchr(position + strlen(key), ':');
    if (!position) return NULL;
    position++;
    while (*position && isspace((unsigned char)*position)) position++;
    return position;
}

static int regular_hadamard_size(uint64_t size) {
    if (size < 4) return 0;
    while (size > 1 && size % 4 == 0) size /= 4;
    return size == 1;
}

int h3_weight_i8_linear_convrot_group(const h3_weight_store *store,
                                      const char *weight_name,
                                      uint32_t *group_size,
                                      char *error, size_t error_size) {
    if (group_size) *group_size = 0;
    if (!store || !weight_name || !*weight_name || !group_size) {
        fail(error, error_size, "invalid quantization metadata arguments");
        return 0;
    }
    static const char weight_suffix[] = ".weight";
    size_t name_length = strlen(weight_name);
    size_t suffix_length = sizeof(weight_suffix) - 1;
    if (name_length <= suffix_length ||
        strcmp(weight_name + name_length - suffix_length,
               weight_suffix) != 0) {
        fail(error, error_size,
             "linear weight name does not end in .weight: %s", weight_name);
        return 0;
    }
    static const char marker_suffix[] = ".comfy_quant";
    size_t marker_size = name_length - suffix_length + sizeof(marker_suffix);
    char *marker_name = malloc(marker_size);
    if (!marker_name) {
        fail(error, error_size, "out of memory resolving quantization marker");
        return 0;
    }
    memcpy(marker_name, weight_name, name_length - suffix_length);
    memcpy(marker_name + name_length - suffix_length, marker_suffix,
           sizeof(marker_suffix));
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(
        store, marker_name, &header);
    if (!tensor) {
        free(marker_name);
        return 1;
    }
    uint64_t bytes = h3_st_tensor_elements(tensor);
    if (!header || tensor->dtype != H3_DTYPE_U8 || tensor->ndim != 1 ||
        !bytes || bytes > 511) {
        fail(error, error_size,
             "quantization marker has the wrong schema: %s", marker_name);
        free(marker_name);
        return 0;
    }
    char marker[512];
    if (!h3_st_read_data(header, tensor, marker, (size_t)bytes,
                         error, error_size)) {
        free(marker_name);
        return 0;
    }
    marker[bytes] = '\0';
    if (!strstr(marker, "\"format\"") ||
        !strstr(marker, "\"int8_tensorwise\"")) {
        fail(error, error_size,
             "unsupported quantization format in %s", marker_name);
        free(marker_name);
        return 0;
    }
    const char *convrot = json_value(marker, "\"convrot\"");
    if (!convrot || !strncmp(convrot, "false", 5)) {
        free(marker_name);
        return 1;
    }
    if (strncmp(convrot, "true", 4) != 0) {
        fail(error, error_size, "invalid ConvRot flag in %s", marker_name);
        free(marker_name);
        return 0;
    }
    const char *group = json_value(marker, "\"convrot_groupsize\"");
    char *end = NULL;
    uint64_t parsed = group ? strtoull(group, &end, 10) : 0;
    if (!group || end == group || parsed > UINT32_MAX ||
        !regular_hadamard_size(parsed)) {
        fail(error, error_size, "invalid ConvRot group size in %s",
             marker_name);
        free(marker_name);
        return 0;
    }
    *group_size = (uint32_t)parsed;
    free(marker_name);
    return 1;
}
