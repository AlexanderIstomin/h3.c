/* Run LTX-2.5's pre-quantized projections through h3.c's existing ConvRot
 * INT8 path.
 *
 * LTX-2.5 publishes both its 22B distilled transformer and its Gemma 4 12B
 * text tower in the same Comfy INT8 ConvRot encoding as the optimized
 * MiniMax H3 packages: row-major I8 matrices, one F32 scale per output
 * channel, and a `comfy_quant` marker whose 72 bytes are identical in both
 * checkpoints. This test proves the claim on real weights rather than on the
 * headers alone, by loading fixtures through h3_weight_load_i8_linear and
 * comparing the GPU product against a host reference.
 *
 * A fixture is a standalone safetensors file holding one projection from each
 * distinct shape family, extracted from a published checkpoint by byte range.
 * Pass its path as the only argument. Every projection the fixture contains
 * is checked and the rest are skipped, so one binary covers both files.
 *
 * Shapes come from each checkpoint's own metadata. In the DiT the video
 * stream is 4096 wide with a 16384 feed-forward, the audio stream is 2048
 * wide with 8192, and the two cross-modal projections bridge those widths.
 * The Gemma tower is 3840 wide across 48 layers, with 16 query heads and 8
 * key/value heads of dimension 256 -- except every sixth layer, which widens
 * to 16 heads of dimension 512 against a single key head and carries no
 * value projection at all. */

#include "h3_gpu.h"
#include "h3_weights.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    VIDEO_HIDDEN = 4096,
    AUDIO_HIDDEN = 2048,
    AUDIO_FFN = 8192,
    VIDEO_FFN = 16384,
    TEXT_HIDDEN = 3840,
    TEXT_FFN = 15360,
    TEXT_QUERY = 4096,      /* 16 heads x 256 */
    TEXT_KEY_VALUE = 2048,  /*  8 heads x 256 */
    TEXT_WIDE_QUERY = 8192, /* 16 heads x 512, every sixth layer */
    TEXT_WIDE_KEY = 512,    /*  1 head  x 512 */
    /* More than one row so the tiled kernel is exercised over its row loop
     * rather than only its degenerate single-row case. */
    ROWS = 3,
    CHECKED_OUTPUTS = 12,
    EXPECTED_CONVROT_GROUP = 256
};

typedef struct {
    const char *name;
    uint32_t output_dim;
    uint32_t input_dim;
    const char *role;
} ltx_projection;

/* One projection per distinct shape family across both checkpoints. Each LTX
 * block repeats the first four widths over its eight attentions and two
 * feed-forwards; the Gemma tower repeats the rest over its 48 layers.
 *
 * The DiT names carry their full path, which is what the released file uses.
 * They were written bare here first, against a block-0 extract that had the
 * prefix stripped, and matched nothing when the real checkpoint arrived. The
 * text encoder's names needed no change: its file already ships them whole. */
static const ltx_projection PROJECTIONS[] = {
    {"model.diffusion_model.transformer_blocks.0.attn1.to_q.weight",
     VIDEO_HIDDEN, VIDEO_HIDDEN,
     "DiT video self-attention query"},
    {"model.diffusion_model.transformer_blocks.0.audio_to_video_attn.to_q.weight",
     AUDIO_HIDDEN, VIDEO_HIDDEN,
     "DiT cross-modal query, video rows into audio width"},
    {"model.diffusion_model.transformer_blocks.0.audio_ff.net.2.weight",
     AUDIO_HIDDEN, AUDIO_FFN,
     "DiT audio feed-forward down"},
    {"model.diffusion_model.transformer_blocks.0.ff.net.0.proj.weight",
     VIDEO_FFN, VIDEO_HIDDEN,
     "DiT video feed-forward up"},
    {"model.layers.0.self_attn.q_proj.weight", TEXT_QUERY, TEXT_HIDDEN,
     "Gemma GQA query, 16 heads of 256"},
    {"model.layers.0.self_attn.k_proj.weight", TEXT_KEY_VALUE, TEXT_HIDDEN,
     "Gemma GQA key, 8 heads of 256"},
    {"model.layers.0.mlp.gate_proj.weight", TEXT_FFN, TEXT_HIDDEN,
     "Gemma SwiGLU gate"},
    {"model.layers.5.self_attn.q_proj.weight", TEXT_WIDE_QUERY, TEXT_HIDDEN,
     "Gemma wide-head query, 16 heads of 512"},
    {"model.layers.5.self_attn.k_proj.weight", TEXT_WIDE_KEY, TEXT_HIDDEN,
     "Gemma wide-head key, one head of 512, no value projection"}
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

/* Replace the trailing ".weight" of a projection name with another suffix. */
static void companion_name(const ltx_projection *projection,
                           const char *suffix, char *buffer, size_t size) {
    size_t length = strlen(projection->name);
    size_t stem = length - strlen(".weight");
    int written = snprintf(buffer, size, "%.*s%s", (int)stem,
                           projection->name, suffix);
    require(written > 0 && (size_t)written < size,
            "companion tensor name is too long");
}

/* Recompute the kernel's arithmetic on the host: the rotated BF16 activation
 * against the stored int8 matrix, scaled per output channel, then the
 * optional BF16 bias, then one BF16 rounding. This mirrors
 * h3_linear_i8_weight_bf16 exactly, including the order of the final two
 * operations. */
static void validate(const ltx_projection *projection,
                     h3_gpu_tensor *rotated_input, h3_gpu_tensor *weight,
                     h3_gpu_tensor *scales, h3_gpu_tensor *bias,
                     h3_gpu_tensor *output) {
    uint32_t input_dim = projection->input_dim;
    uint32_t output_dim = projection->output_dim;
    size_t weight_elements = (size_t)output_dim * input_dim;

    uint16_t *input_values = malloc((size_t)ROWS * input_dim *
                                    sizeof(*input_values));
    int8_t *weight_values = malloc(weight_elements * sizeof(*weight_values));
    float *scale_values = malloc((size_t)output_dim * sizeof(*scale_values));
    uint16_t *bias_values = bias ?
        malloc((size_t)output_dim * sizeof(*bias_values)) : NULL;
    uint16_t *output_values = malloc((size_t)ROWS * output_dim *
                                     sizeof(*output_values));
    require(input_values && weight_values && scale_values && output_values &&
                (!bias || bias_values),
            "cannot allocate LTX projection reference buffers");

    require(h3_gpu_tensor_read_bf16(rotated_input, input_values,
                                    (size_t)ROWS * input_dim),
            "cannot read the rotated LTX activation");
    require(h3_gpu_tensor_read_i8(weight, weight_values, weight_elements),
            "cannot read the LTX int8 matrix for the host reference");
    require(h3_gpu_tensor_read_f32(scales, scale_values, output_dim),
            "cannot read the LTX per-channel scales");
    require(!bias || h3_gpu_tensor_read_bf16(bias, bias_values, output_dim),
            "cannot read the LTX projection bias");
    require(h3_gpu_tensor_read_bf16(output, output_values,
                                    (size_t)ROWS * output_dim),
            "cannot read the LTX projection output");

    uint32_t checked = output_dim < CHECKED_OUTPUTS ?
        output_dim : CHECKED_OUTPUTS;
    for (uint32_t row = 0; row < ROWS; row++) {
        for (uint32_t column = 0; column < checked; column++) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < input_dim; k++)
                sum = fmaf(bf16_f32(input_values[(size_t)row * input_dim + k]),
                           (float)weight_values[(size_t)column * input_dim + k],
                           sum);
            sum *= scale_values[column];
            if (bias_values) sum += bf16_f32(bias_values[column]);
            float expected = bf16_f32(bf16(sum));
            float actual = bf16_f32(
                output_values[(size_t)row * output_dim + column]);
            float tolerance = fmaxf(0.015625f, fabsf(expected) * 0.015625f);
            if (!isfinite(actual) || fabsf(actual - expected) > tolerance) {
                fprintf(stderr,
                        "FAIL: %s row %u column %u differs: %.8g vs %.8g\n",
                        projection->name, row, column, actual, expected);
                exit(1);
            }
        }
    }

    free(input_values);
    free(weight_values);
    free(scale_values);
    free(bias_values);
    free(output_values);
}

/* Returns 1 when the fixture carries this projection and it checked out, and
 * 0 when the fixture is for the other checkpoint and does not contain it. */
static int run_projection(h3_weight_store *store, h3_gpu *gpu,
                          const ltx_projection *projection) {
    char error[512];
    char name[256];

    const h3_st_header *weight_header = NULL;
    if (!h3_weight_find(store, projection->name, &weight_header)) return 0;

    uint32_t convrot_group = 0;
    if (!h3_weight_i8_linear_convrot_group(store, projection->name,
                                          &convrot_group, error,
                                          sizeof(error))) {
        fprintf(stderr, "FAIL: cannot read %s quantization marker: %s\n",
                projection->name, error);
        exit(1);
    }
    if (convrot_group != EXPECTED_CONVROT_GROUP) {
        fprintf(stderr,
                "FAIL: %s reports ConvRot group %u, not the supported %d\n",
                projection->name, convrot_group, EXPECTED_CONVROT_GROUP);
        exit(1);
    }

    h3_gpu_tensor *weight = NULL, *scales = NULL;
    if (!h3_weight_load_i8_linear(store, gpu, projection->name,
                                  projection->output_dim,
                                  projection->input_dim, &weight, &scales,
                                  error, sizeof(error))) {
        fprintf(stderr, "FAIL: cannot load %s: %s\n",
                projection->name, error);
        exit(1);
    }

    /* Bias is present on the attention and audio feed-forward projections and
     * absent on the video feed-forward, matching the checkpoint's asymmetric
     * ff_bias. Load it when the checkpoint has it. */
    companion_name(projection, ".bias", name, sizeof(name));
    const uint64_t bias_shape[] = {projection->output_dim};
    h3_gpu_tensor *bias = NULL;
    const h3_st_header *bias_header = NULL;
    if (h3_weight_find(store, name, &bias_header)) {
        bias = h3_weight_load_bf16(store, gpu, name, 1, bias_shape,
                                   error, sizeof(error));
        if (!bias) {
            fprintf(stderr, "FAIL: cannot load %s: %s\n", name, error);
            exit(1);
        }
    }

    size_t input_elements = (size_t)ROWS * projection->input_dim;
    uint16_t *activation = malloc(input_elements * sizeof(*activation));
    require(activation != NULL, "cannot allocate the LTX activation");
    for (size_t index = 0; index < input_elements; index++) {
        int value = (int)(index % 29u) - 14;
        activation[index] = bf16((float)value / 64.0f);
    }
    h3_gpu_tensor *input = h3_gpu_tensor_from_bf16(gpu, activation,
                                                  input_elements);
    free(activation);
    h3_gpu_tensor *output = h3_gpu_tensor_new_bf16(
        gpu, (size_t)ROWS * projection->output_dim);
    require(input && output, "cannot allocate LTX projection tensors");

    require(h3_gpu_begin(gpu), "cannot begin the LTX projection path");
    /* The rotation is in place, exactly as the H3 DiT drives it, so the host
     * reference below reads the already-rotated activation. */
    require(h3_gpu_convrot_bf16(gpu, input, input, ROWS,
                                projection->input_dim, convrot_group),
            "LTX ConvRot dispatch failed");
    require(h3_gpu_linear_i8_weight_bf16(gpu, output, input, weight, scales,
                                         bias, ROWS, projection->input_dim,
                                         projection->output_dim),
            "LTX int8 projection dispatch failed");
    require(h3_gpu_submit(gpu), "LTX projection submit failed");

    validate(projection, input, weight, scales, bias, output);

    printf("  ok  %-38s %5u x %-5u  bias %-3s  %s\n",
           projection->name, projection->output_dim, projection->input_dim,
           bias ? "yes" : "no", projection->role);

    h3_gpu_tensor_free(input);
    h3_gpu_tensor_free(output);
    h3_gpu_tensor_free(weight);
    h3_gpu_tensor_free(scales);
    h3_gpu_tensor_free(bias);
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s LTX_BLOCK0_FIXTURE.safetensors\n", argv[0]);
        return 2;
    }

    char error[512];
    h3_weight_store *store = h3_weight_store_open(argv[1], error,
                                                  sizeof(error));
    if (!store) {
        fprintf(stderr, "FAIL: cannot open the LTX fixture: %s\n", error);
        return 1;
    }
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) {
        fprintf(stderr, "FAIL: cannot create Metal context: %s\n", error);
        return 1;
    }

    size_t count = sizeof(PROJECTIONS) / sizeof(*PROJECTIONS);
    size_t checked = 0;
    printf("LTX-2.5 projections through h3.c's ConvRot INT8 path:\n");
    for (size_t index = 0; index < count; index++)
        checked += (size_t)run_projection(store, gpu, &PROJECTIONS[index]);

    if (!checked)
        fail("the fixture contains none of the known LTX-2.5 projections");

    printf("ok: %zu real LTX-2.5 projections loaded through "
           "h3_weight_load_i8_linear and matched a host reference "
           "(%d rows x %d outputs each)\n",
           checked, ROWS, CHECKED_OUTPUTS);

    h3_weight_store_free(store);
    h3_gpu_free(gpu);
    return 0;
}
