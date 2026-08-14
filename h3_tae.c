#include "h3_tae.h"

#include "h3_gpu.h"
#include "h3_weights.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Kijai's taeh3: a TAESD-style 2D decoder for H3's 24-channel latent. The
 * network is a plain Sequential; weight names are the layer indices. Four
 * nearest 2x upsamples take one latent slice to the VAE's 16x canvas:
 *
 *   1: conv 24->96        3,4,5: blocks(96)      6: up   7: conv(96)
 *   8,9,10: blocks(96)   11: up  12: conv(96)   13: block 96->64 (1x1 skip)
 *   14,15: blocks(64)    16: up  17: conv(64)   18,19: blocks(64)
 *   20: up  21: conv(64) 22: block(64)          23: conv 64->3
 *
 * A block is conv-relu-conv-relu-conv plus the (possibly projected) input,
 * then relu. Activations stay NHWC, which is what the MPSGraph convolution
 * consumes and what interleaved RGB output wants. */

#define TAE_LATENT_CHANNELS 24
#define TAE_SPATIAL_RATIO 16

enum { TAE_WIDE = 96, TAE_NARROW = 64 };

typedef struct {
    h3_gpu_tensor *weight;
    h3_gpu_tensor *bias;
} tae_conv;

typedef struct {
    tae_conv conv0;
    tae_conv conv2;
    tae_conv conv4;
    h3_gpu_tensor *skip; /* 1x1 projection, NULL for identity */
    int input_channels;
    int output_channels;
} tae_block;

struct h3_tae {
    h3_gpu *gpu;
    h3_weight_store *store;
    tae_conv conv_in;                  /* "1": 24 -> 96 */
    tae_block blocks_a[3];             /* "3","4","5" */
    tae_conv conv_up1;                 /* "7" */
    tae_block blocks_b[3];             /* "8","9","10" */
    tae_conv conv_up2;                 /* "12" */
    tae_block blocks_c[3];             /* "13" (96->64), "14", "15" */
    tae_conv conv_up3;                 /* "17" */
    tae_block blocks_d[2];             /* "18","19" */
    tae_conv conv_up4;                 /* "21" */
    tae_block block_e;                 /* "22" */
    tae_conv conv_out;                 /* "23": 64 -> 3 */
    /* Which stage of the forward pass is running, for failure messages. */
    const char *stage;
    /* Work buffers sized for the largest stage at the current latent size. */
    h3_gpu_tensor *work[3];
    size_t work_elements;
    int latent_h;
    int latent_w;
};

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static h3_gpu_tensor *load_tensor(h3_tae *tae, const char *name, int ndim,
                                  const uint64_t *shape, char *error,
                                  size_t error_size) {
    return h3_weight_load_f32(tae->store, tae->gpu, name, ndim, shape,
                              error, error_size);
}

static int load_conv(h3_tae *tae, tae_conv *conv, const char *index,
                     int input_channels, int output_channels, int kernel,
                     int has_bias, char *error, size_t error_size) {
    char name[64];
    uint64_t shape[] = {(uint64_t)output_channels, (uint64_t)input_channels,
                        (uint64_t)kernel, (uint64_t)kernel};
    snprintf(name, sizeof(name), "%s.weight", index);
    conv->weight = load_tensor(tae, name, 4, shape, error, error_size);
    if (!conv->weight) return 0;
    conv->bias = NULL;
    if (has_bias) {
        uint64_t bias_shape[] = {(uint64_t)output_channels};
        snprintf(name, sizeof(name), "%s.bias", index);
        conv->bias = load_tensor(tae, name, 1, bias_shape, error, error_size);
        if (!conv->bias) return 0;
    }
    return 1;
}

static int load_block(h3_tae *tae, tae_block *block, int index,
                      int input_channels, int output_channels,
                      char *error, size_t error_size) {
    char name[64];
    block->input_channels = input_channels;
    block->output_channels = output_channels;
    snprintf(name, sizeof(name), "%d.conv.0", index);
    if (!load_conv(tae, &block->conv0, name, input_channels, output_channels,
                   3, 1, error, error_size)) return 0;
    snprintf(name, sizeof(name), "%d.conv.2", index);
    if (!load_conv(tae, &block->conv2, name, output_channels, output_channels,
                   3, 1, error, error_size)) return 0;
    snprintf(name, sizeof(name), "%d.conv.4", index);
    if (!load_conv(tae, &block->conv4, name, output_channels, output_channels,
                   3, 1, error, error_size)) return 0;
    block->skip = NULL;
    if (input_channels != output_channels) {
        uint64_t shape[] = {(uint64_t)output_channels,
                            (uint64_t)input_channels, 1, 1};
        snprintf(name, sizeof(name), "%d.skip.weight", index);
        block->skip = load_tensor(tae, name, 4, shape, error, error_size);
        if (!block->skip) return 0;
    }
    return 1;
}

h3_tae *h3_tae_load(const char *weight_path, const char *shader_source_path,
                    char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    h3_tae *tae = calloc(1, sizeof(*tae));
    if (!tae) {
        fail(error, error_size, "out of memory for the preview decoder");
        return NULL;
    }
    tae->gpu = h3_gpu_create(shader_source_path, error, error_size);
    if (tae->gpu) h3_gpu_profile_set_label(tae->gpu, "TAE preview decoder");
    if (tae->gpu)
        tae->store = h3_weight_store_open(weight_path, error, error_size);
    int ok = tae->gpu && tae->store &&
        load_conv(tae, &tae->conv_in, "1", TAE_LATENT_CHANNELS, TAE_WIDE, 3, 1,
                  error, error_size) &&
        load_block(tae, &tae->blocks_a[0], 3, TAE_WIDE, TAE_WIDE, error, error_size) &&
        load_block(tae, &tae->blocks_a[1], 4, TAE_WIDE, TAE_WIDE, error, error_size) &&
        load_block(tae, &tae->blocks_a[2], 5, TAE_WIDE, TAE_WIDE, error, error_size) &&
        load_conv(tae, &tae->conv_up1, "7", TAE_WIDE, TAE_WIDE, 3, 0,
                  error, error_size) &&
        load_block(tae, &tae->blocks_b[0], 8, TAE_WIDE, TAE_WIDE, error, error_size) &&
        load_block(tae, &tae->blocks_b[1], 9, TAE_WIDE, TAE_WIDE, error, error_size) &&
        load_block(tae, &tae->blocks_b[2], 10, TAE_WIDE, TAE_WIDE, error, error_size) &&
        load_conv(tae, &tae->conv_up2, "12", TAE_WIDE, TAE_WIDE, 3, 0,
                  error, error_size) &&
        load_block(tae, &tae->blocks_c[0], 13, TAE_WIDE, TAE_NARROW, error, error_size) &&
        load_block(tae, &tae->blocks_c[1], 14, TAE_NARROW, TAE_NARROW, error, error_size) &&
        load_block(tae, &tae->blocks_c[2], 15, TAE_NARROW, TAE_NARROW, error, error_size) &&
        load_conv(tae, &tae->conv_up3, "17", TAE_NARROW, TAE_NARROW, 3, 0,
                  error, error_size) &&
        load_block(tae, &tae->blocks_d[0], 18, TAE_NARROW, TAE_NARROW, error, error_size) &&
        load_block(tae, &tae->blocks_d[1], 19, TAE_NARROW, TAE_NARROW, error, error_size) &&
        load_conv(tae, &tae->conv_up4, "21", TAE_NARROW, TAE_NARROW, 3, 0,
                  error, error_size) &&
        load_block(tae, &tae->block_e, 22, TAE_NARROW, TAE_NARROW, error, error_size) &&
        load_conv(tae, &tae->conv_out, "23", TAE_NARROW, 3, 3, 1,
                  error, error_size);
    if (!ok) {
        h3_tae_free(tae);
        return NULL;
    }
    return tae;
}

void h3_tae_free(h3_tae *tae) {
    if (!tae) return;
    for (int index = 0; index < 3; index++)
        h3_gpu_tensor_free(tae->work[index]);
    h3_weight_store_free(tae->store);
    h3_gpu_free(tae->gpu);
    free(tae);
}

static int ensure_work(h3_tae *tae, int latent_h, int latent_w,
                       char *error, size_t error_size) {
    size_t canvas = (size_t)latent_h * TAE_SPATIAL_RATIO *
                    (size_t)latent_w * TAE_SPATIAL_RATIO;
    size_t elements = canvas * TAE_NARROW;
    /* The 96-wide stages run at 4x the latent size or less, so the widest
     * full-canvas stage (64 channels) bounds every buffer. */
    if (tae->work_elements >= elements && tae->latent_h == latent_h &&
        tae->latent_w == latent_w) return 1;
    for (int index = 0; index < 3; index++) {
        h3_gpu_tensor_free(tae->work[index]);
        tae->work[index] = h3_gpu_tensor_new_f32(tae->gpu, elements);
        if (!tae->work[index]) {
            fail(error, error_size, "out of memory for preview buffers");
            return 0;
        }
    }
    tae->work_elements = elements;
    tae->latent_h = latent_h;
    tae->latent_w = latent_w;
    return 1;
}

static int run_conv(h3_tae *tae, const tae_conv *conv, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input, int height, int width,
                    int input_channels, int output_channels, int kernel) {
    return h3_gpu_conv3d_same_f32(
        tae->gpu, output, input, conv->weight, conv->bias, 1, 1,
        (uint32_t)height, (uint32_t)width, (uint32_t)input_channels,
        (uint32_t)output_channels, 1, (uint32_t)kernel, (uint32_t)kernel,
        1, 1, 1);
}

/* Runs one residual block with the activation in work[a]; the result is left
 * in work[a] again so the caller never juggles buffer indices. */
static int run_block(h3_tae *tae, const tae_block *block, int a, int height,
                     int width) {
    h3_gpu_tensor *input = tae->work[a];
    h3_gpu_tensor *first = tae->work[(a + 1) % 3];
    h3_gpu_tensor *second = tae->work[(a + 2) % 3];
    uint32_t out_elements = (uint32_t)height * (uint32_t)width *
                            (uint32_t)block->output_channels;
    if (!run_conv(tae, &block->conv0, first, input, height, width,
                  block->input_channels, block->output_channels, 3) ||
        !h3_gpu_relu_f32(tae->gpu, first, first, out_elements) ||
        !run_conv(tae, &block->conv2, second, first, height, width,
                  block->output_channels, block->output_channels, 3) ||
        !h3_gpu_relu_f32(tae->gpu, second, second, out_elements) ||
        !run_conv(tae, &block->conv4, first, second, height, width,
                  block->output_channels, block->output_channels, 3))
        return 0;
    if (block->skip) {
        tae_conv projection = {block->skip, NULL};
        if (!run_conv(tae, &projection, second, input, height, width,
                      block->input_channels, block->output_channels, 1))
            return 0;
        input = second;
    }
    return h3_gpu_add_scaled_f32(tae->gpu, tae->work[a], first, input,
                                 1.0f, 1.0f, out_elements) &&
           h3_gpu_relu_f32(tae->gpu, tae->work[a], tae->work[a], out_elements);
}

static int run_upsample(h3_tae *tae, int *a, int *height, int *width,
                        int channels) {
    int next = (*a + 1) % 3;
    if (!h3_gpu_nearest2x_nhwc_f32(tae->gpu, tae->work[next], tae->work[*a],
                                   (uint32_t)*height, (uint32_t)*width,
                                   (uint32_t)channels)) return 0;
    *a = next;
    *height *= 2;
    *width *= 2;
    return 1;
}

int h3_tae_decode_preview(h3_tae *tae, const float *normalized_latent,
                          int latent_time, h3_video_frames *output,
                          int *output_frame_index,
                          char *error, size_t error_size) {
    if (output) memset(output, 0, sizeof(*output));
    if (error && error_size) error[0] = '\0';
    if (!tae || !normalized_latent || !output || !output_frame_index ||
        latent_time < 2 || (latent_time - 2) % 5 || !tae->latent_h) {
        fail(error, error_size, "invalid TAE preview arguments");
        return 0;
    }
    /* The same representative frame the full preview decoder picks. */
    int chunks = (latent_time - 2) / 5;
    int chunk = chunks / 2;
    int output_frames = chunks * 17 + 5;
    int global_frame = chunk * 17 + 8;
    if (global_frame >= output_frames) global_frame = output_frames - 1;
    int slice = chunk * 5 + 4;
    if (slice >= latent_time) slice = latent_time - 1;

    int height = tae->latent_h;
    int width = tae->latent_w;
    size_t plane = (size_t)height * (size_t)width;
    float *nhwc = malloc(plane * TAE_LATENT_CHANNELS * sizeof(*nhwc));
    if (!nhwc) {
        fail(error, error_size, "out of memory staging the preview latent");
        return 0;
    }
    /* Channel-major latent to NHWC, with TAESD's tanh soft clamp folded in. */
    for (int channel = 0; channel < TAE_LATENT_CHANNELS; channel++) {
        const float *source = normalized_latent +
            ((size_t)channel * latent_time + slice) * plane;
        for (size_t pixel = 0; pixel < plane; pixel++)
            nhwc[pixel * TAE_LATENT_CHANNELS + channel] =
                3.0f * tanhf(source[pixel] / 3.0f);
    }
    int a = 0;
    int ok = h3_gpu_tensor_write_f32(tae->work[0], nhwc,
                                     plane * TAE_LATENT_CHANNELS);
    free(nhwc);
    if (!ok) {
        fail(error, error_size, "cannot upload the preview latent");
        return 0;
    }

    if (!h3_gpu_begin(tae->gpu)) {
        fail(error, error_size, "cannot begin the TAE command buffer: %s",
             h3_gpu_error(tae->gpu));
        return 0;
    }
    int next = 1;
    tae->stage = "the input convolution";
    ok = run_conv(tae, &tae->conv_in, tae->work[next], tae->work[a], height,
                  width, TAE_LATENT_CHANNELS, TAE_WIDE, 3) &&
         h3_gpu_relu_f32(tae->gpu, tae->work[next], tae->work[next],
                         (uint32_t)(plane * TAE_WIDE));
    a = next;
    if (ok) tae->stage = "the first block group";
    for (int index = 0; ok && index < 3; index++)
        ok = run_block(tae, &tae->blocks_a[index], a, height, width);
    if (ok) tae->stage = "the first upsample";
    ok = ok && run_upsample(tae, &a, &height, &width, TAE_WIDE);
    if (ok) {
        next = (a + 1) % 3;
        ok = run_conv(tae, &tae->conv_up1, tae->work[next], tae->work[a],
                      height, width, TAE_WIDE, TAE_WIDE, 3);
        a = next;
    }
    if (ok) tae->stage = "the second block group";
    for (int index = 0; ok && index < 3; index++)
        ok = run_block(tae, &tae->blocks_b[index], a, height, width);
    ok = ok && run_upsample(tae, &a, &height, &width, TAE_WIDE);
    if (ok) {
        next = (a + 1) % 3;
        ok = run_conv(tae, &tae->conv_up2, tae->work[next], tae->work[a],
                      height, width, TAE_WIDE, TAE_WIDE, 3);
        a = next;
    }
    if (ok) tae->stage = "the third block group";
    for (int index = 0; ok && index < 3; index++)
        ok = run_block(tae, &tae->blocks_c[index], a, height, width);
    ok = ok && run_upsample(tae, &a, &height, &width, TAE_NARROW);
    if (ok) {
        next = (a + 1) % 3;
        ok = run_conv(tae, &tae->conv_up3, tae->work[next], tae->work[a],
                      height, width, TAE_NARROW, TAE_NARROW, 3);
        a = next;
    }
    if (ok) tae->stage = "the fourth block group";
    for (int index = 0; ok && index < 2; index++)
        ok = run_block(tae, &tae->blocks_d[index], a, height, width);
    ok = ok && run_upsample(tae, &a, &height, &width, TAE_NARROW);
    if (ok) {
        next = (a + 1) % 3;
        ok = run_conv(tae, &tae->conv_up4, tae->work[next], tae->work[a],
                      height, width, TAE_NARROW, TAE_NARROW, 3);
        a = next;
    }
    if (ok) tae->stage = "the final block";
    ok = ok && run_block(tae, &tae->block_e, a, height, width);
    if (ok) {
        tae->stage = "the output convolution";
        next = (a + 1) % 3;
        ok = run_conv(tae, &tae->conv_out, tae->work[next], tae->work[a],
                      height, width, TAE_NARROW, 3, 3);
        a = next;
    }
    ok = ok && h3_gpu_submit(tae->gpu);
    if (!ok) {
        const char *reason = h3_gpu_error(tae->gpu);
        fail(error, error_size, "TAE preview decode failed at %s: %s",
             tae->stage ? tae->stage : "an unlabelled stage",
             reason && *reason ? reason : "no GPU detail");
        return 0;
    }

    size_t pixels = (size_t)height * (size_t)width;
    float *rgb = malloc(pixels * 3 * sizeof(*rgb));
    if (!rgb) {
        fail(error, error_size, "out of memory for the preview frame");
        return 0;
    }
    if (!h3_gpu_tensor_read_f32(tae->work[a], rgb, pixels * 3)) {
        free(rgb);
        fail(error, error_size, "cannot read the preview frame back");
        return 0;
    }
    for (size_t index = 0; index < pixels * 3; index++) {
        float value = rgb[index];
        rgb[index] = value < 0.0f ? 0.0f : (value > 1.0f ? 1.0f : value);
    }
    output->frames = 1;
    output->height = height;
    output->width = width;
    output->rgb = rgb;
    *output_frame_index = global_frame;
    return 1;
}

int h3_tae_prepare(h3_tae *tae, int latent_h, int latent_w,
                   char *error, size_t error_size) {
    if (!tae || latent_h < 1 || latent_w < 1) {
        fail(error, error_size, "invalid TAE canvas");
        return 0;
    }
    return ensure_work(tae, latent_h, latent_w, error, error_size);
}
