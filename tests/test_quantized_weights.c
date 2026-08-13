#include "h3_gpu.h"
#include "h3_weights.h"

#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

enum {
    ROWS = 128,
    COLUMNS = 128,
    WEIGHT_ELEMENTS = ROWS * COLUMNS,
    ACTIVATION_ROWS = 17,
    INPUT_ELEMENTS = ACTIVATION_ROWS * COLUMNS,
    OUTPUT_ELEMENTS = ACTIVATION_ROWS * ROWS,
    CONVROT_GROUP = 256,
    CONVROT_ROWS = 3,
    CONVROT_ELEMENTS = CONVROT_GROUP * CONVROT_ROWS
};

static const uint16_t FIXTURE_F16[] = {
    0x0000u, 0x3c00u, 0xc000u, 0x7bffu, 0x0400u, 0x0001u
};

static void fail(const char *message) {
    fprintf(stderr, "FAIL: %s\n", message);
    exit(1);
}

static void require(int condition, const char *message) {
    if (!condition) fail(message);
}

static void write_all(int descriptor, const void *data, size_t bytes) {
    const unsigned char *cursor = data;
    while (bytes) {
        ssize_t written = write(descriptor, cursor, bytes);
        if (written <= 0) fail("cannot write quantized safetensors fixture");
        cursor += (size_t)written;
        bytes -= (size_t)written;
    }
}

static void write_quantized_fixture(const char *path, const int8_t *weights,
                                    const float *scales) {
    static const char quant_marker[] =
        "{\"format\": \"int8_tensorwise\", \"convrot\": true, "
        "\"convrot_groupsize\": 256}";
    char header[1024];
    size_t scale_bytes = sizeof(float) * ROWS;
    size_t weight_bytes = sizeof(int8_t) * WEIGHT_ELEMENTS;
    size_t second_scale = scale_bytes + weight_bytes;
    size_t second_weight = second_scale + scale_bytes;
    size_t marker_offset = second_weight + weight_bytes;
    size_t f16_offset = marker_offset + sizeof(quant_marker) - 1;
    int header_bytes = snprintf(
        header, sizeof(header),
        "{\"block.weight_scale\":{\"dtype\":\"F32\",\"shape\":[%d],"
        "\"data_offsets\":[0,%zu]},\"block.weight\":{\"dtype\":\"I8\","
        "\"shape\":[%d,%d],\"data_offsets\":[%zu,%zu]},"
        "\"block_2d.weight_scale\":{\"dtype\":\"F32\","
        "\"shape\":[%d,1],\"data_offsets\":[%zu,%zu]},"
        "\"block_2d.weight\":{\"dtype\":\"I8\","
        "\"shape\":[%d,%d],\"data_offsets\":[%zu,%zu]},"
        "\"block_2d.comfy_quant\":{\"dtype\":\"U8\"," 
        "\"shape\":[%zu],\"data_offsets\":[%zu,%zu]},"
        "\"compact.weight\":{\"dtype\":\"F16\",\"shape\":[%zu],"
        "\"data_offsets\":[%zu,%zu]}}",
        ROWS, scale_bytes, ROWS, COLUMNS, scale_bytes,
        scale_bytes + weight_bytes, ROWS, second_scale,
        second_scale + scale_bytes, ROWS, COLUMNS, second_weight,
        second_weight + weight_bytes, sizeof(quant_marker) - 1,
        marker_offset, marker_offset + sizeof(quant_marker) - 1,
        sizeof(FIXTURE_F16) / sizeof(*FIXTURE_F16), f16_offset,
        f16_offset + sizeof(FIXTURE_F16));
    require(header_bytes > 0 && (size_t)header_bytes < sizeof(header),
            "quantized safetensors header overflow");

    int descriptor = open(path, O_CREAT | O_EXCL | O_WRONLY, 0600);
    require(descriptor >= 0, "cannot create quantized safetensors fixture");
    uint64_t length = (uint64_t)header_bytes;
    unsigned char prefix[8];
    for (unsigned index = 0; index < sizeof(prefix); index++)
        prefix[index] = (unsigned char)(length >> (8u * index));
    write_all(descriptor, prefix, sizeof(prefix));
    write_all(descriptor, header, (size_t)header_bytes);
    write_all(descriptor, scales, scale_bytes);
    write_all(descriptor, weights, weight_bytes);
    write_all(descriptor, scales, scale_bytes);
    write_all(descriptor, weights, weight_bytes);
    write_all(descriptor, quant_marker, sizeof(quant_marker) - 1);
    write_all(descriptor, FIXTURE_F16, sizeof(FIXTURE_F16));
    require(close(descriptor) == 0, "cannot close quantized fixture");
}

static uint16_t bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += UINT32_C(0x7fff) + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

static float bf16_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static void convrot_reference(float values[CONVROT_GROUP]) {
    static const int h4[4][4] = {
        { 1,  1,  1, -1},
        { 1,  1, -1,  1},
        { 1, -1,  1,  1},
        {-1,  1,  1,  1}
    };
    float input[CONVROT_GROUP];
    memcpy(input, values, sizeof(input));
    for (size_t column = 0; column < CONVROT_GROUP; column++) {
        float sum = 0.0f;
        for (size_t row = 0; row < CONVROT_GROUP; row++) {
            size_t row_digits = row;
            size_t column_digits = column;
            int sign = 1;
            for (unsigned digit = 0; digit < 4; digit++) {
                sign *= h4[row_digits & 3u][column_digits & 3u];
                row_digits >>= 2;
                column_digits >>= 2;
            }
            sum += input[row] * (float)sign;
        }
        values[column] = sum * (1.0f / 16.0f);
    }
}

static void test_convrot_bf16(h3_gpu *gpu) {
    uint16_t input_values[CONVROT_ELEMENTS];
    uint16_t expected[CONVROT_ELEMENTS];
    for (size_t index = 0; index < CONVROT_ELEMENTS; index++) {
        int value = (int)(index % 43u) - 21;
        input_values[index] = bf16((float)value / 32.0f);
    }
    for (size_t row = 0; row < CONVROT_ROWS; row++) {
        float values[CONVROT_GROUP];
        for (size_t index = 0; index < CONVROT_GROUP; index++)
            values[index] = bf16_f32(
                input_values[row * CONVROT_GROUP + index]);
        convrot_reference(values);
        for (size_t index = 0; index < CONVROT_GROUP; index++)
            expected[row * CONVROT_GROUP + index] = bf16(values[index]);
    }
    h3_gpu_tensor *tensor = h3_gpu_tensor_from_bf16(
        gpu, input_values, CONVROT_ELEMENTS);
    require(tensor != NULL, "cannot allocate ConvRot test tensor");
    require(h3_gpu_begin(gpu), "cannot begin ConvRot test");
    require(h3_gpu_convrot_bf16(
                gpu, tensor, tensor, CONVROT_ROWS, CONVROT_GROUP,
                CONVROT_GROUP),
            "in-place ConvRot dispatch failed");
    require(h3_gpu_submit(gpu), "ConvRot submit failed");
    uint16_t actual[CONVROT_ELEMENTS];
    require(h3_gpu_tensor_read_bf16(tensor, actual, CONVROT_ELEMENTS),
            "cannot read ConvRot output");
    for (size_t index = 0; index < CONVROT_ELEMENTS; index++) {
        float wanted = bf16_f32(expected[index]);
        float observed = bf16_f32(actual[index]);
        require(fabsf(wanted - observed) <= 0.0078125f,
                "ConvRot output differs from CPU reference");
    }
    h3_gpu_tensor_free(tensor);
}

static double monotonic_seconds(void) {
    struct timespec value;
    require(clock_gettime(CLOCK_MONOTONIC, &value) == 0,
            "cannot read monotonic clock");
    return (double)value.tv_sec + (double)value.tv_nsec / 1000000000.0;
}

static int runtime_quantize(h3_gpu *gpu, int8_t *weights, float *scales) {
    if (!h3_gpu_has_int8_mlp(gpu)) return 0;
    uint16_t source[WEIGHT_ELEMENTS];
    for (size_t index = 0; index < WEIGHT_ELEMENTS; index++) {
        int value = (int)(index % 61u) - 30;
        source[index] = bf16((float)value / 16.0f);
    }
    h3_gpu_tensor *input = h3_gpu_tensor_from_bf16(
        gpu, source, WEIGHT_ELEMENTS);
    h3_gpu_tensor *output = h3_gpu_tensor_new_i8(gpu, WEIGHT_ELEMENTS);
    h3_gpu_tensor *scale = h3_gpu_tensor_new_f32(gpu, ROWS);
    int ok = input && output && scale && h3_gpu_begin(gpu) &&
        h3_gpu_quantize_weight_int8(gpu, output, scale, input, ROWS, COLUMNS) &&
        h3_gpu_submit(gpu) &&
        h3_gpu_tensor_read_i8(output, weights, WEIGHT_ELEMENTS) &&
        h3_gpu_tensor_read_f32(scale, scales, ROWS);
    h3_gpu_tensor_free(input);
    h3_gpu_tensor_free(output);
    h3_gpu_tensor_free(scale);
    require(ok, "M5 runtime weight quantization failed");
    return 1;
}

static void deterministic_quantized_values(int8_t *weights, float *scales) {
    for (size_t index = 0; index < WEIGHT_ELEMENTS; index++)
        weights[index] = (int8_t)((int)(index % 255u) - 127);
    for (size_t row = 0; row < ROWS; row++)
        scales[row] = (float)(row + 1) / 1024.0f;
}

static void test_portable_int8_linear(h3_gpu *gpu,
                                      const h3_gpu_tensor *weight,
                                      const h3_gpu_tensor *scales,
                                      const int8_t *weight_values,
                                      const float *scale_values) {
    uint16_t input_values[INPUT_ELEMENTS];
    for (size_t index = 0; index < INPUT_ELEMENTS; index++) {
        int value = (int)(index % 29u) - 14;
        input_values[index] = bf16((float)value / 32.0f);
    }
    h3_gpu_tensor *input = h3_gpu_tensor_from_bf16(
        gpu, input_values, INPUT_ELEMENTS);
    h3_gpu_tensor *output = h3_gpu_tensor_new_bf16(gpu, OUTPUT_ELEMENTS);
    require(input && output, "cannot allocate portable int8 linear tensors");
    require(h3_gpu_begin(gpu), "cannot begin portable int8 linear command");
    require(h3_gpu_linear_i8_weight_bf16(
                gpu, output, input, weight, scales, NULL, ACTIVATION_ROWS,
                COLUMNS, ROWS),
            "portable int8 linear dispatch failed");
    require(h3_gpu_submit(gpu), "portable int8 linear submit failed");

    uint16_t actual[OUTPUT_ELEMENTS];
    require(h3_gpu_tensor_read_bf16(output, actual, OUTPUT_ELEMENTS),
            "cannot read portable int8 linear output");
    for (size_t row = 0; row < ACTIVATION_ROWS; row++) {
        for (size_t column = 0; column < ROWS; column++) {
            float sum = 0.0f;
            for (size_t k = 0; k < COLUMNS; k++)
                sum = fmaf(bf16_f32(input_values[row * COLUMNS + k]),
                           (float)weight_values[column * COLUMNS + k], sum);
            float expected = bf16_f32(bf16(sum * scale_values[column]));
            float observed = bf16_f32(actual[row * ROWS + column]);
            float tolerance = fmaxf(0.015625f, fabsf(expected) * 0.015625f);
            if (fabsf(expected - observed) > tolerance)
                fail("portable int8 linear differs from CPU reference");
        }
    }
    h3_gpu_tensor_free(input);
    h3_gpu_tensor_free(output);
}

static void submit_portable_int8(h3_gpu *gpu, h3_gpu_tensor *output,
                                 const h3_gpu_tensor *input,
                                 const h3_gpu_tensor *weight,
                                 const h3_gpu_tensor *scales, uint32_t rows,
                                 uint32_t input_dim, uint32_t output_dim) {
    require(h3_gpu_begin(gpu), "cannot begin benchmark int8 command");
    require(h3_gpu_linear_i8_weight_bf16(
                gpu, output, input, weight, scales, NULL, rows, input_dim,
                output_dim),
            "benchmark int8 dispatch failed");
    require(h3_gpu_submit(gpu), "benchmark int8 submit failed");
}

static void submit_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                        const h3_gpu_tensor *input,
                        const h3_gpu_tensor *weight, uint32_t rows,
                        uint32_t input_dim, uint32_t output_dim) {
    require(h3_gpu_begin(gpu), "cannot begin benchmark BF16 command");
    require(h3_gpu_linear_bf16(gpu, output, input, weight, NULL, rows,
                               input_dim, output_dim),
            "benchmark BF16 dispatch failed");
    require(h3_gpu_submit(gpu), "benchmark BF16 submit failed");
}

static void benchmark_portable_int8(h3_gpu *gpu) {
    if (!getenv("H3_BENCH_PORTABLE_INT8")) return;
    enum {
        BENCH_ROWS = 128,
        BENCH_INPUT_DIM = 5376,
        BENCH_OUTPUT_DIM = 7168,
        BENCH_ITERATIONS = 3
    };
    size_t input_count = (size_t)BENCH_ROWS * BENCH_INPUT_DIM;
    size_t weight_count = (size_t)BENCH_OUTPUT_DIM * BENCH_INPUT_DIM;
    size_t output_count = (size_t)BENCH_ROWS * BENCH_OUTPUT_DIM;
    uint16_t *input_values = malloc(input_count * sizeof(*input_values));
    int8_t *weight_values = malloc(weight_count * sizeof(*weight_values));
    float *scale_values = malloc(
        (size_t)BENCH_OUTPUT_DIM * sizeof(*scale_values));
    uint16_t *bf16_weights = malloc(weight_count * sizeof(*bf16_weights));
    uint16_t *int8_result = malloc(output_count * sizeof(*int8_result));
    uint16_t *bf16_result = malloc(output_count * sizeof(*bf16_result));
    require(input_values && weight_values && scale_values && bf16_weights &&
                int8_result && bf16_result,
            "cannot allocate portable int8 benchmark values");
    for (size_t index = 0; index < input_count; index++) {
        int value = (int)(index % 29u) - 14;
        input_values[index] = bf16((float)value / 32.0f);
    }
    for (size_t column = 0; column < BENCH_OUTPUT_DIM; column++) {
        float scale = (float)(column % 31u + 1u) / 4096.0f;
        scale_values[column] = scale;
        for (size_t k = 0; k < BENCH_INPUT_DIM; k++) {
            size_t index = column * BENCH_INPUT_DIM + k;
            int8_t value = (int8_t)((int)(index % 255u) - 127);
            weight_values[index] = value;
            bf16_weights[index] = bf16((float)value * scale);
        }
    }
    h3_gpu_tensor *input = h3_gpu_tensor_from_bf16(
        gpu, input_values, input_count);
    h3_gpu_tensor *int8_weight = h3_gpu_tensor_from_i8(
        gpu, weight_values, weight_count);
    h3_gpu_tensor *scales = h3_gpu_tensor_from_f32(
        gpu, scale_values, BENCH_OUTPUT_DIM);
    h3_gpu_tensor *bf16_weight = h3_gpu_tensor_from_bf16(
        gpu, bf16_weights, weight_count);
    h3_gpu_tensor *int8_output = h3_gpu_tensor_new_bf16(gpu, output_count);
    h3_gpu_tensor *bf16_output = h3_gpu_tensor_new_bf16(gpu, output_count);
    require(input && int8_weight && scales && bf16_weight && int8_output &&
                bf16_output,
            "cannot allocate portable int8 benchmark tensors");
    free(input_values);
    free(weight_values);
    free(scale_values);
    free(bf16_weights);

    submit_portable_int8(gpu, int8_output, input, int8_weight, scales,
                         BENCH_ROWS, BENCH_INPUT_DIM, BENCH_OUTPUT_DIM);
    submit_bf16(gpu, bf16_output, input, bf16_weight, BENCH_ROWS,
                BENCH_INPUT_DIM, BENCH_OUTPUT_DIM);
    double int8_seconds = 0.0;
    double bf16_seconds = 0.0;
    for (unsigned iteration = 0; iteration < BENCH_ITERATIONS; iteration++) {
        double started = monotonic_seconds();
        if (iteration & 1u) {
            submit_bf16(gpu, bf16_output, input, bf16_weight, BENCH_ROWS,
                        BENCH_INPUT_DIM, BENCH_OUTPUT_DIM);
            bf16_seconds += monotonic_seconds() - started;
            started = monotonic_seconds();
            submit_portable_int8(gpu, int8_output, input, int8_weight, scales,
                                 BENCH_ROWS, BENCH_INPUT_DIM,
                                 BENCH_OUTPUT_DIM);
            int8_seconds += monotonic_seconds() - started;
        } else {
            submit_portable_int8(gpu, int8_output, input, int8_weight, scales,
                                 BENCH_ROWS, BENCH_INPUT_DIM,
                                 BENCH_OUTPUT_DIM);
            int8_seconds += monotonic_seconds() - started;
            started = monotonic_seconds();
            submit_bf16(gpu, bf16_output, input, bf16_weight, BENCH_ROWS,
                        BENCH_INPUT_DIM, BENCH_OUTPUT_DIM);
            bf16_seconds += monotonic_seconds() - started;
        }
    }
    require(h3_gpu_tensor_read_bf16(int8_output, int8_result, output_count) &&
                h3_gpu_tensor_read_bf16(bf16_output, bf16_result, output_count),
            "cannot read portable int8 benchmark results");
    double reference_squared = 0.0;
    double difference_squared = 0.0;
    for (size_t index = 0; index < output_count; index++) {
        double reference = (double)bf16_f32(bf16_result[index]);
        double difference = (double)bf16_f32(int8_result[index]) - reference;
        reference_squared += reference * reference;
        difference_squared += difference * difference;
    }
    double int8_ms = int8_seconds * 1000.0 / (double)BENCH_ITERATIONS;
    double bf16_ms = bf16_seconds * 1000.0 / (double)BENCH_ITERATIONS;
    double relative_l2 = reference_squared > 0.0 ?
        sqrt(difference_squared / reference_squared) : 0.0;
    printf("bench: H3 128x5376x7168 portable I8 %.3f ms, BF16 %.3f ms, "
           "ratio %.3fx, relL2 %.6g, weights %.2f/%.2f MiB\n",
           int8_ms, bf16_ms, int8_ms / bf16_ms, relative_l2,
           (double)weight_count / (1024.0 * 1024.0),
           (double)(weight_count * sizeof(uint16_t)) / (1024.0 * 1024.0));

    free(int8_result);
    free(bf16_result);
    h3_gpu_tensor_free(input);
    h3_gpu_tensor_free(int8_weight);
    h3_gpu_tensor_free(scales);
    h3_gpu_tensor_free(bf16_weight);
    h3_gpu_tensor_free(int8_output);
    h3_gpu_tensor_free(bf16_output);
}

int main(void) {
    char error[512];
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) {
        fprintf(stderr, "FAIL: cannot create Metal context: %s\n", error);
        return 1;
    }

    int8_t expected_weights[WEIGHT_ELEMENTS];
    float expected_scales[ROWS];
    int checked_runtime_quantization = runtime_quantize(
        gpu, expected_weights, expected_scales);
    if (!checked_runtime_quantization)
        deterministic_quantized_values(expected_weights, expected_scales);

    char root[] = "/tmp/h3_quantized_weights_XXXXXX";
    require(mkdtemp(root) != NULL, "cannot create quantized fixture directory");
    char directory[512];
    char path[640];
    int directory_length = snprintf(
        directory, sizeof(directory), "%s/transformer", root);
    require(directory_length > 0 &&
            (size_t)directory_length < sizeof(directory),
            "cannot build transformer fixture directory path");
    require(mkdir(directory, 0700) == 0,
            "cannot create transformer fixture directory");
    int path_length = snprintf(
        path, sizeof(path), "%s/weights.safetensors", directory);
    require(path_length > 0 && (size_t)path_length < sizeof(path),
            "cannot build quantized fixture path");
    write_quantized_fixture(path, expected_weights, expected_scales);

    h3_weight_store *store = h3_weight_store_open(directory, error,
                                                  sizeof(error));
    if (!store) {
        fprintf(stderr, "FAIL: cannot open quantized fixture: %s\n", error);
        return 1;
    }
    const uint64_t scale_shape[] = {ROWS};
    h3_gpu_tensor *loaded_weight = NULL;
    h3_gpu_tensor *loaded_scale = NULL;
    if (!h3_weight_load_i8_linear(
            store, gpu, "block.weight", ROWS, COLUMNS,
            &loaded_weight, &loaded_scale, error, sizeof(error))) {
        fprintf(stderr, "FAIL: cannot load serialized I8 linear: %s\n", error);
        return 1;
    }

    int8_t actual_weights[WEIGHT_ELEMENTS];
    float actual_scales[ROWS];
    require(h3_gpu_tensor_dtype(loaded_weight) == H3_GPU_I8,
            "serialized weight has the wrong GPU dtype");
    require(h3_gpu_tensor_elements(loaded_weight) == WEIGHT_ELEMENTS,
            "serialized weight has the wrong element count");
    require(h3_gpu_tensor_read_i8(
                loaded_weight, actual_weights, WEIGHT_ELEMENTS),
            "cannot read serialized I8 weight");
    require(h3_gpu_tensor_read_f32(loaded_scale, actual_scales, ROWS),
            "cannot read serialized F32 scales");
    require(memcmp(expected_weights, actual_weights,
                   sizeof(expected_weights)) == 0,
            "serialized I8 weight changed during reload");
    require(memcmp(expected_scales, actual_scales,
                   sizeof(expected_scales)) == 0,
            "serialized F32 scales changed during reload");

    h3_gpu_tensor *loaded_weight_2d = NULL;
    h3_gpu_tensor *loaded_scale_2d = NULL;
    require(h3_weight_load_i8_linear(
                store, gpu, "block_2d.weight", ROWS, COLUMNS,
                &loaded_weight_2d, &loaded_scale_2d, error, sizeof(error)),
            "linear loader rejected [output, 1] scales");
    require(h3_gpu_tensor_read_i8(
                loaded_weight_2d, actual_weights, WEIGHT_ELEMENTS) &&
                h3_gpu_tensor_read_f32(
                    loaded_scale_2d, actual_scales, ROWS),
            "cannot read [output, 1] quantized linear");
    require(memcmp(expected_weights, actual_weights,
                   sizeof(expected_weights)) == 0 &&
                memcmp(expected_scales, actual_scales,
                       sizeof(expected_scales)) == 0,
            "[output, 1] quantized linear changed during reload");
    h3_gpu_tensor_free(loaded_weight_2d);
    h3_gpu_tensor_free(loaded_scale_2d);
    uint32_t convrot_group = 0;
    require(h3_weight_i8_linear_convrot_group(
                store, "block_2d.weight", &convrot_group,
                error, sizeof(error)) && convrot_group == CONVROT_GROUP,
            "cannot read ConvRot metadata");

    const uint64_t f16_shape[] = {
        sizeof(FIXTURE_F16) / sizeof(*FIXTURE_F16)
    };
    h3_gpu_tensor *converted_f16 = h3_weight_load_f16_as_f32(
        store, gpu, "compact.weight", 1, f16_shape, error, sizeof(error));
    require(converted_f16 != NULL, "cannot load F16 weight as F32");
    float actual_f16[sizeof(FIXTURE_F16) / sizeof(*FIXTURE_F16)];
    const float expected_f16[] = {
        0.0f, 1.0f, -2.0f, 65504.0f, 0.00006103515625f,
        0.000000059604644775390625f
    };
    require(h3_gpu_tensor_read_f32(
                converted_f16, actual_f16,
                sizeof(actual_f16) / sizeof(*actual_f16)),
            "cannot read converted F16 weight");
    require(memcmp(actual_f16, expected_f16, sizeof(expected_f16)) == 0,
            "F16 weight changed during F32 conversion");
    h3_gpu_tensor_free(converted_f16);
    memset(actual_f16, 0, sizeof(actual_f16));
    require(h3_weight_read_f32_vector(
                store, "compact.weight", actual_f16,
                sizeof(actual_f16) / sizeof(*actual_f16),
                error, sizeof(error)),
            "cannot read F16 checkpoint vector on the host");
    require(memcmp(actual_f16, expected_f16, sizeof(expected_f16)) == 0,
            "host F16 vector conversion changed values");

    test_portable_int8_linear(gpu, loaded_weight, loaded_scale,
                              expected_weights, expected_scales);
    test_convrot_bf16(gpu);
    benchmark_portable_int8(gpu);

    h3_gpu_tensor *wrong_dtype = h3_weight_load_i8(
        store, gpu, "block.weight_scale", 1, scale_shape,
        error, sizeof(error));
    require(!wrong_dtype, "I8 loader accepted an F32 scale tensor");

    h3_gpu_tensor_free(loaded_weight);
    h3_gpu_tensor_free(loaded_scale);
    h3_weight_store_free(store);
    store = h3_weight_store_open(path, error, sizeof(error));
    require(store != NULL, "cannot open a single-file weight store");
    require(h3_weight_store_shards(store) == 1,
            "single-file weight store has the wrong shard count");
    require(h3_weight_find(store, "block.weight", NULL) != NULL,
            "single-file weight store cannot find its tensor");
    h3_weight_store_free(store);
    h3_gpu_free(gpu);
    require(unlink(path) == 0, "cannot remove quantized fixture");
    require(rmdir(directory) == 0, "cannot remove transformer fixture directory");
    require(rmdir(root) == 0, "cannot remove quantized fixture directory");

    if (checked_runtime_quantization)
        puts("ok: portable int8 linear and serialized I8 reload match M5 quantization");
    else
        puts("ok: portable int8 linear and serialized I8 reload (pre-M5 path)");
    return 0;
}
