#include "h3_dit_schedule.h"
#include "h3_gpu.h"
#include "h3_weights.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    HIDDEN = 5376,
    HEADS = 56,
    HEAD_DIM = 128,
    INNER = HEADS * HEAD_DIM,
    FFN = 14336,
    ROPE_HALF = 48,
    ADALN_BASIS = 8,
    ADALN_OUTPUT = 3 * 6 * HIDDEN,
    CHECKED_OUTPUTS = 16
};

static void fail(const char *message) {
    fprintf(stderr, "FAIL: %s\n", message);
    exit(1);
}

static void require(int condition, const char *message) {
    if (!condition) fail(message);
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

static void load_linear(h3_weight_store *store, h3_gpu *gpu,
                        const char *name, uint32_t output_dim,
                        uint32_t input_dim, h3_gpu_tensor **weight,
                        h3_gpu_tensor **scales, uint32_t *convrot_group,
                        char *error,
                        size_t error_size) {
    if (!h3_weight_load_i8_linear(
            store, gpu, name, output_dim, input_dim, weight, scales,
            error, error_size)) {
        fprintf(stderr, "FAIL: cannot load %s: %s\n", name, error);
        exit(1);
    }
    if (!h3_weight_i8_linear_convrot_group(
            store, name, convrot_group, error, error_size)) {
        fprintf(stderr, "FAIL: cannot read %s metadata: %s\n", name, error);
        exit(1);
    }
    if (*convrot_group != 256) {
        fprintf(stderr, "FAIL: %s does not use supported H256 ConvRot\n", name);
        exit(1);
    }
}

static void validate_linear(h3_gpu_tensor *input,
                            h3_gpu_tensor *weight,
                            h3_gpu_tensor *scales,
                            h3_gpu_tensor *output,
                            uint32_t input_dim, uint32_t output_dim,
                            const char *label) {
    uint16_t *input_values = malloc(
        (size_t)input_dim * sizeof(*input_values));
    int8_t *weight_values = malloc(
        (size_t)output_dim * input_dim * sizeof(*weight_values));
    float *scale_values = malloc(
        (size_t)output_dim * sizeof(*scale_values));
    uint16_t *output_values = malloc(
        (size_t)output_dim * sizeof(*output_values));
    require(input_values && weight_values && scale_values && output_values,
            "cannot allocate real linear validation buffers");
    require(h3_gpu_tensor_read_bf16(input, input_values, input_dim),
            "cannot read real linear input");
    require(h3_gpu_tensor_read_i8(
                weight, weight_values, (size_t)output_dim * input_dim),
            "cannot read real INT8 weight for CPU reference");
    require(h3_gpu_tensor_read_f32(scales, scale_values, output_dim),
            "cannot read real linear scales for CPU reference");
    require(h3_gpu_tensor_read_bf16(output, output_values, output_dim),
            "cannot read real linear output");

    uint32_t checked = output_dim < CHECKED_OUTPUTS ?
        output_dim : CHECKED_OUTPUTS;
    for (uint32_t row = 0; row < checked; row++) {
        float sum = 0.0f;
        for (uint32_t column = 0; column < input_dim; column++) {
            sum = fmaf(
                bf16_f32(input_values[column]),
                (float)weight_values[(size_t)row * input_dim + column],
                sum);
        }
        float expected = bf16_f32(bf16(sum * scale_values[row]));
        float actual = bf16_f32(output_values[row]);
        float tolerance = fmaxf(0.015625f, fabsf(expected) * 0.015625f);
        if (!isfinite(actual) || fabsf(actual - expected) > tolerance) {
            fprintf(stderr,
                    "FAIL: %s output %u differs: %.8g vs %.8g\n",
                    label, row, actual, expected);
            exit(1);
        }
    }
    free(input_values);
    free(weight_values);
    free(scale_values);
    free(output_values);
}

static void validate_compact_schedule(h3_weight_store *store, h3_gpu *gpu,
                                      char *error, size_t error_size) {
    h3_sigma_schedule sigmas;
    require(h3_schedule_build(2, &sigmas),
            "cannot build compact AdaLN test schedule");
    h3_dit_schedule *schedule = h3_dit_schedule_precompute(
        store, gpu, &sigmas, 0, 0, NULL, NULL, error, error_size);
    if (!schedule) {
        fprintf(stderr, "FAIL: compact AdaLN precompute failed: %s\n", error);
        exit(1);
    }
    require(h3_dit_schedule_time_rows(schedule) == 3,
            "compact AdaLN test has the wrong time-row count");
    require(h3_dit_schedule_final(schedule) != NULL,
            "compact final AdaLN projection is absent");

    const h3_st_header *table_header = NULL;
    const h3_st_tensor *table_tensor = h3_weight_find(
        store, "adaln_t_table", &table_header);
    require(table_tensor && table_header &&
                table_tensor->dtype == H3_DTYPE_F32 &&
                table_tensor->ndim == 2 &&
                table_tensor->shape[1] == ADALN_BASIS,
            "compact AdaLN table has the wrong schema");
    size_t table_count = (size_t)h3_st_tensor_elements(table_tensor);
    float *table = malloc(table_count * sizeof(*table));
    require(table != NULL, "cannot allocate compact AdaLN table reference");
    require(h3_st_read_data(table_header, table_tensor, table,
                            table_count * sizeof(*table), error, error_size),
            "cannot read compact AdaLN table reference");

    const uint64_t weight_shape[] = {ADALN_OUTPUT, ADALN_BASIS};
    const uint64_t bias_shape[] = {ADALN_OUTPUT};
    h3_gpu_tensor *weight = h3_weight_load_f16_as_f32(
        store, gpu, "blocks.0.adaln_proj.linear.weight", 2, weight_shape,
        error, error_size);
    h3_gpu_tensor *bias = h3_weight_load_f16_as_f32(
        store, gpu, "blocks.0.adaln_proj.linear.bias", 1, bias_shape,
        error, error_size);
    require(weight && bias, "cannot load compact block-0 AdaLN weights");
    float weight_values[CHECKED_OUTPUTS * ADALN_BASIS];
    float bias_values[CHECKED_OUTPUTS];
    uint32_t selected_row = h3_dit_schedule_video_row(schedule, 1);
    require(selected_row < h3_dit_schedule_time_rows(schedule),
            "compact AdaLN selected row is invalid");
    size_t actual_count = (size_t)selected_row * ADALN_OUTPUT +
        CHECKED_OUTPUTS;
    uint16_t *actual = malloc(actual_count * sizeof(*actual));
    require(actual != NULL, "cannot allocate compact AdaLN output reference");
    require(h3_gpu_tensor_read_f32(
                weight, weight_values,
                sizeof(weight_values) / sizeof(*weight_values)) &&
                h3_gpu_tensor_read_f32(
                    bias, bias_values,
                    sizeof(bias_values) / sizeof(*bias_values)) &&
                h3_gpu_tensor_read_bf16(
                    h3_dit_schedule_block(schedule, 0), actual,
                    actual_count),
            "cannot read compact block-0 AdaLN result");
    float time = 1.0f - sigmas.video[1];
    float position = time * (float)(table_tensor->shape[0] - 1);
    size_t lower = (size_t)floorf(position);
    if (lower >= table_tensor->shape[0] - 1)
        lower = (size_t)table_tensor->shape[0] - 2;
    float fraction = position - (float)lower;
    float curve[ADALN_BASIS];
    for (size_t column = 0; column < ADALN_BASIS; column++) {
        float left = table[lower * ADALN_BASIS + column];
        float right = table[(lower + 1) * ADALN_BASIS + column];
        curve[column] = left + fraction * (right - left);
    }
    for (size_t output = 0; output < CHECKED_OUTPUTS; output++) {
        float expected = bias_values[output];
        for (size_t column = 0; column < ADALN_BASIS; column++)
            expected = fmaf(curve[column],
                            weight_values[output * ADALN_BASIS + column],
                            expected);
        expected = bf16_f32(bf16(expected));
        float observed = bf16_f32(
            actual[(size_t)selected_row * ADALN_OUTPUT + output]);
        float tolerance = fmaxf(0.0009765625f,
                                fabsf(expected) * 0.0078125f);
        if (!isfinite(observed) || fabsf(observed - expected) > tolerance) {
            fprintf(stderr,
                    "FAIL: compact AdaLN output %zu differs: %.8g vs %.8g\n",
                    output, observed, expected);
            exit(1);
        }
    }
    free(actual);
    free(table);
    h3_gpu_tensor_free(weight);
    h3_gpu_tensor_free(bias);
    h3_dit_schedule_free(schedule);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s MODEL_ROOT\n", argv[0]);
        return 2;
    }
    size_t path_size = strlen(argv[1]) + 128;
    char *path = malloc(path_size);
    require(path != NULL, "cannot allocate optimized checkpoint path");
    int length = snprintf(
        path, path_size, "%s/diffusion_models/"
        "minimax_h3_fl2va_pruned_int8_convrot.safetensors", argv[1]);
    require(length > 0 && (size_t)length < path_size,
            "optimized checkpoint path is too long");

    char error[512];
    h3_weight_store *store = h3_weight_store_open(path, error, sizeof(error));
    if (!store) {
        fprintf(stderr, "FAIL: cannot open optimized checkpoint: %s\n", error);
        return 1;
    }
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) {
        fprintf(stderr, "FAIL: cannot create Metal context: %s\n", error);
        return 1;
    }

    uint16_t *input_values = malloc(HIDDEN * sizeof(*input_values));
    require(input_values != NULL, "cannot allocate optimized block input");
    for (size_t index = 0; index < HIDDEN; index++) {
        int value = (int)(index % 29u) - 14;
        input_values[index] = bf16((float)value / 64.0f);
    }
    h3_gpu_tensor *attention_input = h3_gpu_tensor_from_bf16(
        gpu, input_values, HIDDEN);
    h3_gpu_tensor *mlp_input = h3_gpu_tensor_from_bf16(
        gpu, input_values, HIDDEN);
    free(input_values);
    require(attention_input && mlp_input,
            "cannot allocate optimized block input tensors");

    h3_gpu_tensor *qkv_weight = NULL, *qkv_scales = NULL;
    h3_gpu_tensor *out_weight = NULL, *out_scales = NULL;
    uint32_t qkv_convrot = 0, out_convrot = 0;
    load_linear(store, gpu, "blocks.0.attn.qkv_proj.weight",
                INNER * 3, HIDDEN, &qkv_weight, &qkv_scales,
                &qkv_convrot, error, sizeof(error));
    load_linear(store, gpu, "blocks.0.attn.out_proj.weight",
                HIDDEN, INNER, &out_weight, &out_scales,
                &out_convrot, error, sizeof(error));
    const uint64_t norm_shape[] = {HEAD_DIM};
    h3_gpu_tensor *q_norm = h3_weight_load_bf16(
        store, gpu, "blocks.0.attn.q_norm.weight", 1, norm_shape,
        error, sizeof(error));
    h3_gpu_tensor *k_norm = h3_weight_load_bf16(
        store, gpu, "blocks.0.attn.k_norm.weight", 1, norm_shape,
        error, sizeof(error));
    require(q_norm && k_norm, "cannot load real Q/K normalization weights");

    uint16_t rope_cos_values[ROPE_HALF];
    uint16_t rope_sin_values[ROPE_HALF];
    for (size_t index = 0; index < ROPE_HALF; index++) {
        rope_cos_values[index] = bf16(1.0f);
        rope_sin_values[index] = bf16(0.0f);
    }
    h3_gpu_tensor *rope_cos = h3_gpu_tensor_from_bf16(
        gpu, rope_cos_values, ROPE_HALF);
    h3_gpu_tensor *rope_sin = h3_gpu_tensor_from_bf16(
        gpu, rope_sin_values, ROPE_HALF);
    h3_gpu_tensor *qkv = h3_gpu_tensor_new_bf16(gpu, INNER * 3);
    h3_gpu_tensor *query = h3_gpu_tensor_new_bf16(gpu, INNER);
    h3_gpu_tensor *key = h3_gpu_tensor_new_bf16(gpu, INNER);
    h3_gpu_tensor *value = h3_gpu_tensor_new_bf16(gpu, INNER);
    h3_gpu_tensor *heads = h3_gpu_tensor_new_bf16(gpu, INNER);
    h3_gpu_tensor *attention_output = h3_gpu_tensor_new_bf16(gpu, HIDDEN);
    require(rope_cos && rope_sin && qkv && query && key && value && heads &&
                attention_output,
            "cannot allocate real optimized attention tensors");
    require(h3_gpu_begin(gpu), "cannot begin optimized attention path");
    require(h3_gpu_convrot_bf16(
                gpu, attention_input, attention_input,
                1, HIDDEN, qkv_convrot),
            "optimized QKV ConvRot dispatch failed");
    require(h3_gpu_linear_i8_weight_bf16(
                gpu, qkv, attention_input, qkv_weight, qkv_scales, NULL,
                1, HIDDEN, INNER * 3),
            "optimized QKV projection dispatch failed");
    /* Comfy's optimized H3 tensor keeps Q, K, and V as three contiguous
     * projection ranges; only the released checkpoint uses grouped rows. */
    require(h3_gpu_qkv_rope_bf16(
                gpu, query, key, value, qkv, q_norm, k_norm,
                rope_cos, rope_sin, 1, HEADS, HEAD_DIM, ROPE_HALF, 1e-5f),
            "optimized QKV normalization/RoPE dispatch failed");
    require(h3_gpu_sdpa_bf16(
                gpu, heads, query, key, value, 1, HEADS, HEAD_DIM,
                1.0f / sqrtf((float)HEAD_DIM)),
            "optimized attention dispatch failed");
    require(h3_gpu_convrot_bf16(
                gpu, heads, heads, 1, INNER, out_convrot),
            "optimized attention-output ConvRot dispatch failed");
    require(h3_gpu_linear_i8_weight_bf16(
                gpu, attention_output, heads, out_weight, out_scales, NULL,
                1, INNER, HIDDEN),
            "optimized attention-output projection dispatch failed");
    require(h3_gpu_submit(gpu), "optimized attention path submit failed");
    validate_linear(attention_input, qkv_weight, qkv_scales, qkv,
                    HIDDEN, INNER * 3, "QKV");
    validate_linear(heads, out_weight, out_scales, attention_output,
                    INNER, HIDDEN, "attention output");

    h3_gpu_tensor_free(qkv_weight); h3_gpu_tensor_free(qkv_scales);
    h3_gpu_tensor_free(out_weight); h3_gpu_tensor_free(out_scales);
    h3_gpu_tensor_free(q_norm); h3_gpu_tensor_free(k_norm);
    h3_gpu_tensor_free(rope_cos); h3_gpu_tensor_free(rope_sin);
    h3_gpu_tensor_free(qkv); h3_gpu_tensor_free(query);
    h3_gpu_tensor_free(key); h3_gpu_tensor_free(value);
    h3_gpu_tensor_free(heads); h3_gpu_tensor_free(attention_output);

    h3_gpu_tensor *fc1_weight = NULL, *fc1_scales = NULL;
    h3_gpu_tensor *fc2_weight = NULL, *fc2_scales = NULL;
    uint32_t fc1_convrot = 0, fc2_convrot = 0;
    load_linear(store, gpu, "blocks.0.mlp.fc1.weight",
                FFN * 2, HIDDEN, &fc1_weight, &fc1_scales,
                &fc1_convrot, error, sizeof(error));
    load_linear(store, gpu, "blocks.0.mlp.fc2.weight",
                HIDDEN, FFN, &fc2_weight, &fc2_scales,
                &fc2_convrot, error, sizeof(error));
    h3_gpu_tensor *fc1 = h3_gpu_tensor_new_bf16(gpu, FFN * 2);
    h3_gpu_tensor *activated = h3_gpu_tensor_new_bf16(gpu, FFN);
    h3_gpu_tensor *mlp_output = h3_gpu_tensor_new_bf16(gpu, HIDDEN);
    require(fc1 && activated && mlp_output,
            "cannot allocate real optimized MLP tensors");
    require(h3_gpu_begin(gpu), "cannot begin optimized MLP path");
    require(h3_gpu_convrot_bf16(
                gpu, mlp_input, mlp_input, 1, HIDDEN, fc1_convrot),
            "optimized FC1 ConvRot dispatch failed");
    require(h3_gpu_linear_i8_weight_bf16(
                gpu, fc1, mlp_input, fc1_weight, fc1_scales, NULL,
                1, HIDDEN, FFN * 2),
            "optimized FC1 dispatch failed");
    require(h3_gpu_swiglu_bf16(gpu, activated, fc1, 1, FFN),
            "optimized SwiGLU dispatch failed");
    require(h3_gpu_convrot_bf16(
                gpu, activated, activated, 1, FFN, fc2_convrot),
            "optimized FC2 ConvRot dispatch failed");
    require(h3_gpu_linear_i8_weight_bf16(
                gpu, mlp_output, activated, fc2_weight, fc2_scales, NULL,
                1, FFN, HIDDEN),
            "optimized FC2 dispatch failed");
    require(h3_gpu_submit(gpu), "optimized MLP path submit failed");
    validate_linear(mlp_input, fc1_weight, fc1_scales, fc1,
                    HIDDEN, FFN * 2, "FC1");
    validate_linear(activated, fc2_weight, fc2_scales, mlp_output,
                    FFN, HIDDEN, "FC2");

    validate_compact_schedule(store, gpu, error, sizeof(error));

    printf("ok: real ConvRot INT8 block-0 attention and MLP paths run "
           "on this GPU, and compact AdaLN precomputes; "
           "%d CPU rows checked per projection\n",
           CHECKED_OUTPUTS);
    free(path);
    h3_gpu_tensor_free(attention_input);
    h3_gpu_tensor_free(mlp_input);
    h3_gpu_tensor_free(fc1_weight); h3_gpu_tensor_free(fc1_scales);
    h3_gpu_tensor_free(fc2_weight); h3_gpu_tensor_free(fc2_scales);
    h3_gpu_tensor_free(fc1); h3_gpu_tensor_free(activated);
    h3_gpu_tensor_free(mlp_output);
    h3_weight_store_free(store);
    h3_gpu_free(gpu);
    return 0;
}
