#include "h3_dit.h"
#include "h3_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { HEADS = 1, HEAD_DIM = 128 };

static uint32_t random_state = 0x56334133u;

static float random_value(float scale) {
    random_state ^= random_state << 13;
    random_state ^= random_state >> 17;
    random_state ^= random_state << 5;
    return ((float)(random_state & 0xffffu) / 32767.5f - 1.0f) * scale;
}

static uint16_t to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

static float from_bf16(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static float dot(const float *left, const float *right) {
    float result = 0.0f;
    for (unsigned dimension = 0; dimension < HEAD_DIM; dimension++)
        result += left[dimension] * right[dimension];
    return result;
}

static int cpu_reference(const h3_vsa_geometry *geometry,
                         const float *query, const float *key,
                         const float *value, const float *gate,
                         float *output) {
    const size_t tiled_elements =
        (size_t)geometry->padded_rows * HEAD_DIM;
    const size_t pooled_elements = (size_t)geometry->tiles * HEAD_DIM;
    float *tiled_query = calloc(tiled_elements, sizeof(*tiled_query));
    float *tiled_key = calloc(tiled_elements, sizeof(*tiled_key));
    float *tiled_value = calloc(tiled_elements, sizeof(*tiled_value));
    float *pooled_query = calloc(pooled_elements, sizeof(*pooled_query));
    float *pooled_key = calloc(pooled_elements, sizeof(*pooled_key));
    float *pooled_value = calloc(pooled_elements, sizeof(*pooled_value));
    float *compressed = calloc(pooled_elements, sizeof(*compressed));
    float *scores = malloc((size_t)geometry->tiles * sizeof(*scores));
    uint32_t topk = (geometry->video_tiles + 9u) / 10u;
    uint32_t *selected = malloc(
        (size_t)geometry->video_tiles * topk * sizeof(*selected));
    if (!tiled_query || !tiled_key || !tiled_value || !pooled_query ||
        !pooled_key || !pooled_value || !compressed || !scores || !selected) {
        free(tiled_query); free(tiled_key); free(tiled_value);
        free(pooled_query); free(pooled_key); free(pooled_value);
        free(compressed); free(scores); free(selected);
        return 0;
    }
    for (uint32_t tiled_row = 0; tiled_row < geometry->padded_rows;
         tiled_row++) {
        uint32_t packed_row = geometry->tiled_to_packed[tiled_row];
        if (packed_row == UINT32_MAX) continue;
        size_t source = (size_t)packed_row * HEAD_DIM;
        size_t destination = (size_t)tiled_row * HEAD_DIM;
        memcpy(tiled_query + destination, query + source,
               HEAD_DIM * sizeof(float));
        memcpy(tiled_key + destination, key + source,
               HEAD_DIM * sizeof(float));
        memcpy(tiled_value + destination, value + source,
               HEAD_DIM * sizeof(float));
    }
    for (uint32_t tile = 0; tile < geometry->tiles; tile++) {
        float inverse = 1.0f / (float)geometry->block_sizes[tile];
        for (uint32_t row = 0; row < geometry->block_sizes[tile]; row++) {
            size_t source = ((size_t)tile * 64u + row) * HEAD_DIM;
            size_t destination = (size_t)tile * HEAD_DIM;
            for (unsigned dimension = 0; dimension < HEAD_DIM; dimension++) {
                pooled_query[destination + dimension] +=
                    tiled_query[source + dimension] * inverse;
                pooled_key[destination + dimension] +=
                    tiled_key[source + dimension] * inverse;
                pooled_value[destination + dimension] +=
                    tiled_value[source + dimension] * inverse;
            }
        }
    }
    const float scale = 1.0f / sqrtf((float)HEAD_DIM);
    for (uint32_t query_tile = 0; query_tile < geometry->tiles; query_tile++) {
        float maximum = -INFINITY;
        for (uint32_t key_tile = 0; key_tile < geometry->tiles; key_tile++) {
            scores[key_tile] = dot(
                pooled_query + (size_t)query_tile * HEAD_DIM,
                pooled_key + (size_t)key_tile * HEAD_DIM) * scale;
            if (scores[key_tile] > maximum) maximum = scores[key_tile];
        }
        float denominator = 0.0f;
        for (uint32_t key_tile = 0; key_tile < geometry->tiles; key_tile++)
            denominator += expf(scores[key_tile] - maximum);
        for (uint32_t key_tile = 0; key_tile < geometry->tiles; key_tile++) {
            float probability = expf(scores[key_tile] - maximum) / denominator;
            for (unsigned dimension = 0; dimension < HEAD_DIM; dimension++)
                compressed[(size_t)query_tile * HEAD_DIM + dimension] +=
                    probability * pooled_value[
                        (size_t)key_tile * HEAD_DIM + dimension];
        }
        if (query_tile >= geometry->prefix_tiles) {
            size_t destination =
                (size_t)(query_tile - geometry->prefix_tiles) * topk;
            for (uint32_t rank = 0; rank < topk; rank++) {
                float best = -INFINITY;
                uint32_t best_tile = geometry->prefix_tiles;
                for (uint32_t video_tile = 0;
                     video_tile < geometry->video_tiles; video_tile++) {
                    uint32_t absolute = geometry->prefix_tiles + video_tile;
                    if (scores[absolute] > best) {
                        best = scores[absolute];
                        best_tile = absolute;
                    }
                }
                selected[destination + rank] = best_tile;
                scores[best_tile] = -INFINITY;
            }
        }
    }
    for (uint32_t packed_query = 0; packed_query < geometry->sequence;
         packed_query++) {
        uint32_t query_tile = geometry->packed_to_tiled[packed_query] / 64u;
        uint32_t route_count = query_tile < geometry->prefix_tiles
            ? geometry->tiles : geometry->prefix_tiles + topk;
        const float *query_row = query + (size_t)packed_query * HEAD_DIM;
        float maximum = -INFINITY;
        for (uint32_t route = 0; route < route_count; route++) {
            uint32_t key_tile = route < geometry->prefix_tiles ||
                                query_tile < geometry->prefix_tiles
                ? route : selected[
                    (size_t)(query_tile - geometry->prefix_tiles) * topk +
                    route - geometry->prefix_tiles];
            for (uint32_t key_row = 0;
                 key_row < geometry->block_sizes[key_tile]; key_row++) {
                const float *key_values = tiled_key +
                    ((size_t)key_tile * 64u + key_row) * HEAD_DIM;
                float score = dot(query_row, key_values) * scale;
                if (score > maximum) maximum = score;
            }
        }
        float denominator = 0.0f;
        float exact[HEAD_DIM] = {0};
        for (uint32_t route = 0; route < route_count; route++) {
            uint32_t key_tile = route < geometry->prefix_tiles ||
                                query_tile < geometry->prefix_tiles
                ? route : selected[
                    (size_t)(query_tile - geometry->prefix_tiles) * topk +
                    route - geometry->prefix_tiles];
            for (uint32_t key_row = 0;
                 key_row < geometry->block_sizes[key_tile]; key_row++) {
                size_t tiled_row = (size_t)key_tile * 64u + key_row;
                const float *key_values = tiled_key + tiled_row * HEAD_DIM;
                const float *value_values = tiled_value + tiled_row * HEAD_DIM;
                float weight = expf(dot(query_row, key_values) * scale - maximum);
                denominator += weight;
                for (unsigned dimension = 0; dimension < HEAD_DIM; dimension++)
                    exact[dimension] += weight * value_values[dimension];
            }
        }
        size_t offset = (size_t)packed_query * HEAD_DIM;
        size_t compressed_offset = (size_t)query_tile * HEAD_DIM;
        for (unsigned dimension = 0; dimension < HEAD_DIM; dimension++)
            output[offset + dimension] = exact[dimension] / denominator +
                gate[offset + dimension] *
                compressed[compressed_offset + dimension];
    }
    free(tiled_query); free(tiled_key); free(tiled_value);
    free(pooled_query); free(pooled_key); free(pooled_value);
    free(compressed); free(scores); free(selected);
    return 1;
}

int main(int argc, char **argv) {
    const char *shader = argc > 1 ? argv[1] : "h3_shaders.metal";
    const uint32_t prefix[] = {3, 2, 4};
    h3_vsa_geometry geometry = {0};
    char error[8192] = {0};
    if (!h3_vsa_geometry_build(prefix, 3, 5, 5, 9, &geometry,
                               error, sizeof(error))) {
        fprintf(stderr, "FAIL VSA geometry: %s\n", error);
        return 1;
    }
    size_t count = (size_t)geometry.sequence * HEAD_DIM;
    uint16_t *query16 = malloc(count * sizeof(*query16));
    uint16_t *key16 = malloc(count * sizeof(*key16));
    uint16_t *value16 = malloc(count * sizeof(*value16));
    uint16_t *gate16 = malloc(count * sizeof(*gate16));
    uint16_t *actual16 = malloc(count * sizeof(*actual16));
    float *query = malloc(count * sizeof(*query));
    float *key = malloc(count * sizeof(*key));
    float *value = malloc(count * sizeof(*value));
    float *gate = malloc(count * sizeof(*gate));
    float *expected = malloc(count * sizeof(*expected));
    if (!query16 || !key16 || !value16 || !gate16 || !actual16 ||
        !query || !key || !value || !gate || !expected) return 1;
    for (size_t index = 0; index < count; index++) {
        query16[index] = to_bf16(random_value(0.35f));
        key16[index] = to_bf16(random_value(0.35f));
        value16[index] = to_bf16(random_value(0.5f));
        gate16[index] = to_bf16(random_value(0.2f));
        query[index] = from_bf16(query16[index]);
        key[index] = from_bf16(key16[index]);
        value[index] = from_bf16(value16[index]);
        gate[index] = from_bf16(gate16[index]);
    }
    if (!cpu_reference(&geometry, query, key, value, gate, expected)) return 1;

    h3_gpu *gpu = h3_gpu_create_with_sparse_attention(
        shader, error, sizeof(error));
    if (!gpu) {
        fprintf(stderr, "FAIL VSA Metal setup: %s\n", error);
        return 1;
    }
    size_t tiled = (size_t)geometry.padded_rows * HEAD_DIM;
    size_t summaries = (size_t)geometry.tiles * HEAD_DIM;
    size_t pooled = summaries * 3u + (summaries + 1u) / 2u;
    uint32_t topk = (geometry.video_tiles + 9u) / 10u;
    h3_gpu_tensor *q = h3_gpu_tensor_from_bf16(gpu, query16, count);
    h3_gpu_tensor *k = h3_gpu_tensor_from_bf16(gpu, key16, count);
    h3_gpu_tensor *v = h3_gpu_tensor_from_bf16(gpu, value16, count);
    h3_gpu_tensor *g = h3_gpu_tensor_from_bf16(gpu, gate16, count);
    h3_gpu_tensor *out = h3_gpu_tensor_new_bf16(gpu, count);
    h3_gpu_tensor *qkvg = h3_gpu_tensor_new_bf16(gpu, tiled * 4u);
    h3_gpu_tensor *pool = h3_gpu_tensor_new_f32(gpu, pooled);
    h3_gpu_tensor *selection = h3_gpu_tensor_new_u32(
        gpu, (size_t)geometry.video_tiles * topk);
    h3_gpu_tensor *sizes = h3_gpu_tensor_from_u32(
        gpu, geometry.block_sizes, geometry.tiles);
    h3_gpu_tensor *map = h3_gpu_tensor_from_u32(
        gpu, geometry.tiled_to_packed, geometry.padded_rows);
    int ok = q && k && v && g && out && qkvg && pool && selection && sizes && map &&
        h3_gpu_begin(gpu) &&
        h3_gpu_vsa_attention_bf16(
            gpu, out, q, k, v, g, qkvg, pool, selection, sizes, map,
            geometry.sequence, geometry.padded_rows, geometry.tiles,
            geometry.prefix_tiles, geometry.video_tiles, topk,
            HEADS, HEAD_DIM, 1.0f / sqrtf((float)HEAD_DIM)) &&
        h3_gpu_submit(gpu) &&
        h3_gpu_tensor_read_bf16(out, actual16, count);
    if (!ok) {
        fprintf(stderr, "FAIL VSA dispatch: %s\n", h3_gpu_error(gpu));
        return 1;
    }
    double squared_expected = 0.0, squared_actual = 0.0;
    double product = 0.0, absolute_error = 0.0, maximum_error = 0.0;
    for (size_t index = 0; index < count; index++) {
        double actual = from_bf16(actual16[index]);
        double delta = actual - expected[index];
        double magnitude = fabs(delta);
        squared_expected += (double)expected[index] * expected[index];
        squared_actual += actual * actual;
        product += actual * expected[index];
        absolute_error += magnitude;
        if (magnitude > maximum_error) maximum_error = magnitude;
    }
    double cosine = product / sqrt(squared_expected * squared_actual);
    double mae = absolute_error / (double)count;
    printf("VSA tiles=%u prefix=%u video=%u cosine=%.9f mae=%.9f max=%.9f\n",
           geometry.tiles, geometry.prefix_tiles, geometry.video_tiles,
           cosine, mae, maximum_error);
    ok = isfinite(cosine) && cosine > 0.9999 && mae < 0.002 &&
         maximum_error < 0.02;
    if (!ok) fprintf(stderr, "FAIL learned VSA-H3 parity\n");

    h3_gpu_tensor_free(q); h3_gpu_tensor_free(k); h3_gpu_tensor_free(v);
    h3_gpu_tensor_free(g); h3_gpu_tensor_free(out); h3_gpu_tensor_free(qkvg);
    h3_gpu_tensor_free(pool); h3_gpu_tensor_free(selection);
    h3_gpu_tensor_free(sizes); h3_gpu_tensor_free(map); h3_gpu_free(gpu);
    h3_vsa_geometry_free(&geometry);
    free(query16); free(key16); free(value16); free(gate16); free(actual16);
    free(query); free(key); free(value); free(gate); free(expected);
    return ok ? 0 : 1;
}
