/* LTX-2.5's video VAE decoder, on the released weights.
 *
 * 128 latent channels at 2x4x4 out to 3 channels at 9x128x128 -- the smallest
 * shape that exercises every stage, since the stack compresses 8x in time
 * (8*(k-1)+1 frames) and 32x in space.
 *
 * **The reference this is anchored against is patched.** The public
 * `CausalVideoAutoencoder` builds `compress_time` and `compress_space` without
 * passing their `multiplier` through, and LTX-2.5 was trained with all three
 * compress blocks honouring it; the names and the tensor count match either
 * way, so `load_state_dict(strict=False)` says nothing and only 45 of 84
 * shapes disagree. Two things say the patch is right, and the second is the
 * one that matters: all 84 shapes reconcile *and* the patched pair encodes and
 * decodes a structured image at 30.8 dB. Shapes agreeing only proves the parts
 * fit; a round trip proves they are wired in the right order.
 *
 * Conventions that are easy to get quietly wrong, all of them checked here:
 *
 *   - **temporal padding replicates, spatial padding zero-fills.** Every
 *     convolution is a `CausalConv3d`, but `causal_decoder` is *false*, so the
 *     temporal path pads with one copy of the first frame in front and one of
 *     the last behind, while height and width get `nn.Conv3d`'s zeros. Two
 *     different padding modes on the same convolution;
 *
 *   - **`compress_time` and `compress_all` drop their first output frame.**
 *     Unconditionally, whatever `causal` says -- which is why 2 latent frames
 *     become 3, then 5, then 9 rather than 4, 8, 16;
 *
 *   - **pixel norm is parameter-free and its epsilon is inside the root:**
 *     `x / sqrt(mean(x^2 over channels) + 1e-8)`. The checkpoint carries no
 *     norm tensors at all, which is the tell. That epsilon is the one
 *     convention here no mutation catches, because at these magnitudes it is
 *     inert -- so its margin is measured instead of asserted; and
 *
 *   - **the unpatchify channel order is (c, t, w, h), not (c, t, h, w).** The
 *     reference reads `(c p r q) -> (f p) (h q) (w r)`, so the fastest-varying
 *     channel index lands on *height* and the next one on width.
 *
 * The pixel shuffles run on the host. There are only four of them and the
 * largest moves 1.2M floats, so it costs little here -- but it is a readback
 * per shuffle, and a real decode wants a kernel. That is a performance gap,
 * not a correctness one, and it is the only piece of this the GPU cannot yet
 * do end to end.
 *
 * usage: h3_real_ltx_video_vae_test VAE.safetensors ANCHOR.safetensors */

#include "h3_gpu.h"
#include "h3_safetensors.h"
#include "h3_weights.h"

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum {
    LATENT_CHANNELS = 128,
    LATENT_FRAMES = 2, LATENT_HEIGHT = 4, LATENT_WIDTH = 4,
    PATCH = 4,
    IMAGE_CHANNELS = 3,
    UP_BLOCKS = 9,
    KERNEL = 3
};

#define PIXEL_NORM_EPSILON 1e-8f

/* Both sides compute in F32, so this is not a precision floor -- it is the
 * gap between MPS's convolution accumulation order and torch's. Measured
 * across all fourteen stages: 6.0e-07 at conv_in rising to 1.2e-05 at the
 * output, monotonically with depth. Set just above the worst of them. */
#define TOLERANCE 5e-5

static int failures = 0;
static h3_weight_store *store = NULL;
static h3_weight_store *anchor = NULL;
static h3_gpu *gpu = NULL;

static void fail(const char *format, ...)
    __attribute__((format(printf, 1, 2), noreturn));

static void fail(const char *format, ...) {
    va_list arguments;
    va_start(arguments, format);
    fprintf(stderr, "FAIL: ");
    vfprintf(stderr, format, arguments);
    fprintf(stderr, "\n");
    va_end(arguments);
    exit(1);
}

static void require(int condition, const char *message) {
    if (!condition) fail("%s", message);
}

#define GPU_OP(call, what) \
    do { if (!(call)) fail("%s: %s", (what), h3_gpu_error(gpu)); } while (0)

static double now(void) {
    struct timespec moment;
    clock_gettime(CLOCK_MONOTONIC, &moment);
    return (double)moment.tv_sec + (double)moment.tv_nsec * 1e-9;
}

/* ------------------------------------------------------------------ shapes */

/* Channels last: the convolution kernel takes NDHWC and OIDHW, and OIDHW is
 * exactly how torch stores a Conv3d weight, so weights need no rearranging and
 * activations do. */
typedef struct { uint32_t depth, height, width, channels; } shape;

static size_t volume(shape of) {
    return (size_t)of.depth * of.height * of.width * of.channels;
}

/* ------------------------------------------------------------------ loading */

static float from_bf16(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static float *read_tensor(const h3_weight_store *from, const char *name,
                          size_t expected) {
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(from, name, &header);
    if (!tensor) fail("no tensor %s", name);
    size_t elements = (size_t)h3_st_tensor_elements(tensor);
    if (expected && elements != expected)
        fail("%s has %zu elements, expected %zu", name, elements, expected);
    float *values = malloc(elements * sizeof(*values));
    require(values != NULL, "cannot allocate a tensor");
    char error[512];
    if (tensor->dtype == H3_DTYPE_F32) {
        require(h3_st_read_data(header, tensor, values,
                                elements * sizeof(*values), error, sizeof(error)),
                "cannot read an F32 tensor");
    } else if (tensor->dtype == H3_DTYPE_BF16) {
        uint16_t *raw = malloc(elements * sizeof(*raw));
        require(raw != NULL, "cannot allocate a staging buffer");
        require(h3_st_read_data(header, tensor, raw, elements * sizeof(*raw),
                                error, sizeof(error)),
                "cannot read a BF16 tensor");
        for (size_t index = 0; index < elements; index++)
            values[index] = from_bf16(raw[index]);
        free(raw);
    } else {
        fail("%s is %s", name, h3_dtype_name(tensor->dtype));
    }
    return values;
}

static float *golden(const char *name, size_t expected) {
    return read_tensor(anchor, name, expected);
}

static h3_gpu_tensor *upload(const float *values, size_t count) {
    h3_gpu_tensor *tensor = h3_gpu_tensor_from_f32(gpu, values, count);
    require(tensor != NULL, "cannot upload a tensor");
    return tensor;
}

static float *download(const h3_gpu_tensor *tensor, size_t count) {
    float *values = malloc(count * sizeof(*values));
    require(values != NULL, "cannot allocate a readback buffer");
    require(h3_gpu_tensor_read_f32(tensor, values, count),
            "cannot read a GPU tensor");
    return values;
}

typedef struct { h3_gpu_tensor *weight, *bias; } conv3d;

/* Uploading writes through a staging path that a *later* dispatch in an
 * already-open command buffer does not see: a convolution encoded after an
 * upload into the same buffer reads the allocation as zeros and emits its bias
 * alone. That is what this test did first, and the bias was recognisable in
 * the output. So every load happens before `h3_gpu_begin`, and each stage
 * opens its own buffer around compute only. */
static void load_conv(const char *name, uint32_t out_channels,
                      uint32_t in_channels, conv3d *into) {
    char full[256];
    snprintf(full, sizeof(full), "decoder.%s.conv.weight", name);
    const size_t taps = (size_t)KERNEL * KERNEL * KERNEL;
    float *weight = read_tensor(store, full, (size_t)out_channels *
                                in_channels * taps);
    into->weight = upload(weight, (size_t)out_channels * in_channels * taps);
    free(weight);
    snprintf(full, sizeof(full), "decoder.%s.conv.bias", name);
    float *bias = read_tensor(store, full, out_channels);
    into->bias = upload(bias, out_channels);
    free(bias);
}

static void free_conv(conv3d *which) {
    h3_gpu_tensor_free(which->weight);
    h3_gpu_tensor_free(which->bias);
    memset(which, 0, sizeof(*which));
}

/* ---------------------------------------------------------------- compare */

/* The golden tensors are torch's [C][D][H][W]; the engine's are [D][H][W][C].
 * Walking both here rather than permuting one keeps the convention in a single
 * place and applies it to every stage. */
static void compare(const char *label, const float *engine, const float *expect,
                    shape of, double tolerance) {
    double worst = 0.0, peak = 0.0;
    size_t worst_at = 0;
    const size_t spatial = (size_t)of.depth * of.height * of.width;
    for (size_t position = 0; position < spatial; position++)
        for (uint32_t channel = 0; channel < of.channels; channel++) {
            const double actual = engine[position * of.channels + channel];
            const double wanted = expect[(size_t)channel * spatial + position];
            if (!isfinite(actual)) {
                fprintf(stderr, "FAIL %-22s produced %f\n", label, actual);
                failures++;
                return;
            }
            const double delta = fabs(actual - wanted);
            if (delta > worst) { worst = delta; worst_at = position; }
            if (fabs(wanted) > peak) peak = fabs(wanted);
        }
    const double relative = peak > 0.0 ? worst / peak : worst;
    if (relative > tolerance) {
        fprintf(stderr, "FAIL %-22s %.3e = %.2e of peak %.4f, at %zu\n",
                label, worst, relative, peak, worst_at);
        failures++;
    } else {
        printf("  ok  %-22s [%4u,%3u,%3u,%3u] %.3e = %.2e of peak %.4f\n",
               label, of.channels, of.depth, of.height, of.width, worst,
               relative, peak);
    }
}

/* -------------------------------------------------------------- the stages */

/* One copy of the first frame in front and one of the last behind. Frames are
 * the outermost axis of a channels-last buffer, so each is a contiguous run
 * and three copies do the whole thing -- no kernel needed. */
static h3_gpu_tensor *pad_time(const h3_gpu_tensor *input, shape of) {
    const size_t frame = (size_t)of.height * of.width * of.channels;
    const size_t padded = (size_t)(of.depth + 2) * frame;
    h3_gpu_tensor *out = h3_gpu_tensor_new_f32(gpu, padded);
    require(out != NULL, "cannot allocate a padded frame stack");
    GPU_OP(h3_gpu_copy_f32(gpu, out, frame, input, 0,
                           (size_t)of.depth * frame), "pad interior");
    GPU_OP(h3_gpu_copy_f32(gpu, out, 0, input, 0, frame), "pad front");
    GPU_OP(h3_gpu_copy_f32(gpu, out, (size_t)(of.depth + 1) * frame, input,
                           (size_t)(of.depth - 1) * frame, frame), "pad back");
    return out;
}

/* Replicate in time, zeros in height and width -- the conv kernel same-pads
 * the spatial axes and leaves depth alone, which is exactly the split the
 * reference's `nn.Conv3d(padding=(0, 1, 1))` after an explicit temporal
 * concatenation produces. */
static h3_gpu_tensor *run_conv(const conv3d *kernel, const h3_gpu_tensor *input,
                               shape of, uint32_t out_channels) {
    h3_gpu_tensor *padded = pad_time(input, of);
    shape result = {of.depth, of.height, of.width, out_channels};
    h3_gpu_tensor *out = h3_gpu_tensor_new_f32(gpu, volume(result));
    require(out != NULL, "cannot allocate a convolution output");
    GPU_OP(h3_gpu_conv3d_same_f32(gpu, out, padded, kernel->weight,
                                  kernel->bias, 1, of.depth + 2, of.height,
                                  of.width, of.channels, out_channels,
                                  KERNEL, KERNEL, KERNEL, 1, 1, 1),
           "convolution");
    h3_gpu_tensor_free(padded);
    return out;
}

/* x / sqrt(mean(x^2 over channels) + eps), which is the RMS norm kernel with a
 * weight of ones -- the epsilon sits inside the root in both. */
static void pixel_norm(h3_gpu_tensor *out, const h3_gpu_tensor *input,
                       const h3_gpu_tensor *ones, shape of) {
    GPU_OP(h3_gpu_rms_norm_f32(gpu, out, input, ones,
                               (uint32_t)((size_t)of.depth * of.height * of.width),
                               of.channels, PIXEL_NORM_EPSILON), "pixel norm");
}

/* `b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)`, channels last, on the
 * host. The channel index decomposes major-to-minor as (c, p1, p2, p3), so the
 * depth factor is the *slowest* of the three sub-indices. */
static void pixel_shuffle(const float *input, float *out, shape of,
                          uint32_t depth_up, uint32_t height_up,
                          uint32_t width_up) {
    const uint32_t factor = depth_up * height_up * width_up;
    const uint32_t out_channels = of.channels / factor;
    const uint32_t out_height = of.height * height_up;
    const uint32_t out_width = of.width * width_up;
    for (uint32_t d = 0; d < of.depth; d++)
        for (uint32_t h = 0; h < of.height; h++)
            for (uint32_t w = 0; w < of.width; w++) {
                const float *source = input +
                    (((size_t)d * of.height + h) * of.width + w) * of.channels;
                for (uint32_t c = 0; c < out_channels; c++)
                    for (uint32_t i = 0; i < depth_up; i++)
                        for (uint32_t j = 0; j < height_up; j++)
                            for (uint32_t k = 0; k < width_up; k++) {
                                const uint32_t from =
                                    ((c * depth_up + i) * height_up + j) *
                                    width_up + k;
                                const size_t to =
                                    (((size_t)(d * depth_up + i) * out_height +
                                      h * height_up + j) * out_width +
                                     w * width_up + k) * out_channels + c;
                                out[to] = source[from];
                            }
            }
}

/* --------------------------------------------------------------- the blocks */

typedef struct {
    const char *kind;
    int layers;          /* res_x only */
    int multiplier;      /* compress only */
    uint32_t depth_up, height_up, width_up;
} up_block;

/* `decoder_blocks` from the checkpoint config, reversed -- the decoder walks
 * it backwards. Written out rather than derived so the table can be read
 * against the config by eye. */
static const up_block PLAN[UP_BLOCKS] = {
    {"res_x",          2, 0, 0, 0, 0},
    {"compress_all",   0, 2, 2, 2, 2},
    {"res_x",          2, 0, 0, 0, 0},
    {"compress_all",   0, 1, 2, 2, 2},
    {"res_x",          4, 0, 0, 0, 0},
    {"compress_time",  0, 2, 2, 1, 1},
    {"res_x",          6, 0, 0, 0, 0},
    {"compress_space", 0, 2, 1, 2, 2},
    {"res_x",          4, 0, 0, 0, 0}
};

/* pixel norm, SiLU, conv, again, then add the input back. Every res_x block
 * keeps its channel count, so the reference's shortcut projection and its
 * third norm are both Identity -- which is why the checkpoint carries neither. */
static h3_gpu_tensor *resnet(int block, int layer, h3_gpu_tensor *x, shape of,
                             const h3_gpu_tensor *ones) {
    char name[128];
    conv3d first, second;
    snprintf(name, sizeof(name), "up_blocks.%d.res_blocks.%d.conv1", block, layer);
    load_conv(name, of.channels, of.channels, &first);
    snprintf(name, sizeof(name), "up_blocks.%d.res_blocks.%d.conv2", block, layer);
    load_conv(name, of.channels, of.channels, &second);
    h3_gpu_tensor *scratch = h3_gpu_tensor_new_f32(gpu, volume(of));
    require(scratch != NULL, "cannot allocate resnet scratch");

    GPU_OP(h3_gpu_begin(gpu), "begin resnet");
    pixel_norm(scratch, x, ones, of);
    GPU_OP(h3_gpu_silu_f32(gpu, scratch, scratch, (uint32_t)volume(of)),
           "resnet activation");
    h3_gpu_tensor *hidden = run_conv(&first, scratch, of, of.channels);
    pixel_norm(scratch, hidden, ones, of);
    GPU_OP(h3_gpu_silu_f32(gpu, scratch, scratch, (uint32_t)volume(of)),
           "resnet activation");
    h3_gpu_tensor_free(hidden);
    hidden = run_conv(&second, scratch, of, of.channels);
    GPU_OP(h3_gpu_add_scaled_f32(gpu, x, x, hidden, 1.0f, 1.0f,
                                 (uint32_t)volume(of)), "resnet residual");
    GPU_OP(h3_gpu_submit(gpu), "submit resnet");
    h3_gpu_tensor_free(hidden);
    h3_gpu_tensor_free(scratch);
    free_conv(&first); free_conv(&second);
    return x;
}

/* Convolve up to `prod(stride) * channels / multiplier`, shuffle those channels
 * out into space and time, then -- when the temporal factor is 2 -- discard the
 * first output frame. The discard is unconditional in the reference, so 2
 * latent frames become 3 rather than 4. */
static h3_gpu_tensor *compress(int block, const up_block *description,
                               h3_gpu_tensor *x, shape *of) {
    char name[128];
    const uint32_t factor = description->depth_up * description->height_up *
                            description->width_up;
    const uint32_t wide = of->channels * factor /
                          (uint32_t)description->multiplier;
    snprintf(name, sizeof(name), "up_blocks.%d.conv", block);
    conv3d kernel;
    load_conv(name, wide, of->channels, &kernel);
    GPU_OP(h3_gpu_begin(gpu), "begin compress");
    h3_gpu_tensor *convolved = run_conv(&kernel, x, *of, wide);
    GPU_OP(h3_gpu_submit(gpu), "submit compress");
    free_conv(&kernel);
    h3_gpu_tensor_free(x);

    shape wide_shape = {of->depth, of->height, of->width, wide};
    float *host = download(convolved, volume(wide_shape));
    h3_gpu_tensor_free(convolved);
    shape shuffled = {of->depth * description->depth_up,
                      of->height * description->height_up,
                      of->width * description->width_up, wide / factor};
    float *expanded = malloc(volume(shuffled) * sizeof(*expanded));
    require(expanded != NULL, "cannot allocate a shuffled stack");
    pixel_shuffle(host, expanded, wide_shape, description->depth_up,
                  description->height_up, description->width_up);
    free(host);

    const float *keep = expanded;
    if (description->depth_up == 2) {
        const size_t frame = (size_t)shuffled.height * shuffled.width *
                             shuffled.channels;
        keep = expanded + frame;
        shuffled.depth -= 1;
    }
    h3_gpu_tensor *out = upload(keep, volume(shuffled));
    free(expanded);
    *of = shuffled;
    return out;
}

/* ------------------------------------------------------------------- main */

/* `b (c p r q) f h w -> b c (f p) (h q) (w r)` with p = 1. The sub-index that
 * varies fastest in the channel packing is `q`, and it lands on *height*; `r`
 * is the slower one and lands on width. Reading that pair the other way round
 * transposes every 4x4 patch, which looks like a plausible image. */
static void unpatchify(const float *input, float *out, shape of) {
    const uint32_t channels = of.channels / (PATCH * PATCH);
    const uint32_t height = of.height * PATCH, width = of.width * PATCH;
    const size_t plane = (size_t)of.depth * height * width;
    for (uint32_t d = 0; d < of.depth; d++)
        for (uint32_t h = 0; h < of.height; h++)
            for (uint32_t w = 0; w < of.width; w++) {
                const float *source = input +
                    (((size_t)d * of.height + h) * of.width + w) * of.channels;
                for (uint32_t c = 0; c < channels; c++)
                    for (uint32_t r = 0; r < PATCH; r++)
                        for (uint32_t q = 0; q < PATCH; q++)
                            out[(size_t)c * plane +
                                ((size_t)d * height + h * PATCH + q) * width +
                                w * PATCH + r] =
                                source[(c * PATCH + r) * PATCH + q];
            }
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s VAE.safetensors ANCHOR.safetensors\n",
                argv[0]);
        return 2;
    }
    char error[512];
    store = h3_weight_store_open(argv[1], error, sizeof(error));
    if (!store) fail("cannot open the VAE: %s", error);
    anchor = h3_weight_store_open(argv[2], error, sizeof(error));
    if (!anchor) fail("cannot open the anchor: %s", error);
    gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) fail("cannot create the Metal context: %s", error);

    shape of = {LATENT_FRAMES, LATENT_HEIGHT, LATENT_WIDTH, LATENT_CHANNELS};
    printf("LTX-2.5 video VAE decoder: [%u, %u, %u, %u] latent\n",
           of.channels, of.depth, of.height, of.width);
    const double began = now();

    /* The latent arrives as torch's [C][D][H][W] and the kernels want
     * [D][H][W][C]. */
    float *latent = golden("latent", volume(of));
    float *staged = malloc(volume(of) * sizeof(*staged));
    require(staged != NULL, "cannot allocate the staged latent");
    {
        const size_t spatial = (size_t)of.depth * of.height * of.width;
        for (size_t position = 0; position < spatial; position++)
            for (uint32_t channel = 0; channel < of.channels; channel++)
                staged[position * of.channels + channel] =
                    latent[(size_t)channel * spatial + position];
    }
    h3_gpu_tensor *x = upload(staged, volume(of));
    free(latent); free(staged);

    /* One ones vector, wide enough for every channel count the stack uses. */
    float *ones_host = malloc(1024 * sizeof(*ones_host));
    require(ones_host != NULL, "cannot allocate a norm weight");
    for (int index = 0; index < 1024; index++) ones_host[index] = 1.0f;
    h3_gpu_tensor *ones = upload(ones_host, 1024);
    free(ones_host);

    /* ----------------------------------------------------------- conv_in */

    {
        conv3d kernel;
        load_conv("conv_in", 1024, LATENT_CHANNELS, &kernel);
        GPU_OP(h3_gpu_begin(gpu), "begin conv_in");
        h3_gpu_tensor *out = run_conv(&kernel, x, of, 1024);
        GPU_OP(h3_gpu_submit(gpu), "submit conv_in");
        free_conv(&kernel);
        h3_gpu_tensor_free(x);
        x = out;
        of.channels = 1024;
    }
    {
        float *have = download(x, volume(of));
        float *want = golden("conv_in", volume(of));
        compare("conv_in", have, want, of, TOLERANCE);
        free(have); free(want);
    }

    /* ---------------------------------------------------------- up blocks */

    for (int index = 0; index < UP_BLOCKS; index++) {
        const up_block *description = &PLAN[index];
        if (description->layers) {
            for (int layer = 0; layer < description->layers; layer++)
                x = resnet(index, layer, x, of, ones);
        } else {
            x = compress(index, description, x, &of);
        }
        char label[64];
        snprintf(label, sizeof(label), "block%d_%s", index, description->kind);
        float *have = download(x, volume(of));
        float *want = golden(label, volume(of));
        compare(label, have, want, of, TOLERANCE);
        free(have); free(want);
    }

    /* ------------------------------------------------------------- the tail */

    /* Pixel norm's epsilon is the one convention here that no mutation of this
     * file can catch: setting it to zero changes nothing measurable, and the
     * reason is worth stating rather than leaving as a silent gap. It only
     * bites when a position's channel vector is small enough that its mean
     * square approaches 1e-8. So measure how close anything actually gets --
     * that turns "untested" into "shown not to apply here", and the day a
     * latent does produce a near-empty position this line says so. */
    {
        float *have = download(x, volume(of));
        const size_t positions = (size_t)of.depth * of.height * of.width;
        double smallest = INFINITY;
        for (size_t position = 0; position < positions; position++) {
            double sum = 0.0;
            for (uint32_t channel = 0; channel < of.channels; channel++) {
                const double value = have[position * of.channels + channel];
                sum += value * value;
            }
            const double mean = sum / (double)of.channels;
            if (mean < smallest) smallest = mean;
        }
        printf("  --  %-22s smallest mean square %.3e, %.0fx the %.0e epsilon\n",
               "epsilon is inert here", smallest,
               smallest / (double)PIXEL_NORM_EPSILON, (double)PIXEL_NORM_EPSILON);
        require(smallest > (double)PIXEL_NORM_EPSILON * 1e3,
                "a position is small enough that the epsilon matters, so it "
                "can no longer be treated as inert");
        free(have);
    }

    GPU_OP(h3_gpu_begin(gpu), "begin tail");
    h3_gpu_tensor *normed = h3_gpu_tensor_new_f32(gpu, volume(of));
    require(normed != NULL, "cannot allocate the normalized stack");
    pixel_norm(normed, x, ones, of);
    GPU_OP(h3_gpu_submit(gpu), "submit norm");
    {
        float *have = download(normed, volume(of));
        float *want = golden("norm_out", volume(of));
        compare("norm_out", have, want, of, TOLERANCE);
        free(have); free(want);
    }
    conv3d tail;
    load_conv("conv_out", IMAGE_CHANNELS * PATCH * PATCH, of.channels, &tail);
    GPU_OP(h3_gpu_begin(gpu), "begin head");
    GPU_OP(h3_gpu_silu_f32(gpu, normed, normed, (uint32_t)volume(of)),
           "output activation");
    h3_gpu_tensor *out = run_conv(&tail, normed, of, IMAGE_CHANNELS * PATCH * PATCH);
    /* Submit before releasing anything the encoded work still reads. */
    GPU_OP(h3_gpu_submit(gpu), "submit head");
    free_conv(&tail);
    h3_gpu_tensor_free(normed);
    h3_gpu_tensor_free(x);
    shape patched = {of.depth, of.height, of.width, IMAGE_CHANNELS * PATCH * PATCH};
    {
        float *have = download(out, volume(patched));
        float *want = golden("conv_out", volume(patched));
        compare("conv_out", have, want, patched, TOLERANCE);
        free(have); free(want);
    }

    /* --------------------------------------------------------- unpatchify */

    shape image = {patched.depth, patched.height * PATCH, patched.width * PATCH,
                   IMAGE_CHANNELS};
    float *packed = download(out, volume(patched));
    h3_gpu_tensor_free(out);
    float *pixels = malloc(volume(image) * sizeof(*pixels));
    require(pixels != NULL, "cannot allocate the frames");
    unpatchify(packed, pixels, patched);
    free(packed);
    {
        /* Already [C][D][H][W], so this one comparison is direct. */
        float *want = golden("restored", volume(image));
        double worst = 0.0, peak = 0.0;
        for (size_t index = 0; index < volume(image); index++) {
            require(isfinite(pixels[index]), "a pixel is not finite");
            const double delta = fabs((double)pixels[index] - (double)want[index]);
            if (delta > worst) worst = delta;
            if (fabs((double)want[index]) > peak) peak = fabs((double)want[index]);
        }
        if (worst / peak > TOLERANCE) {
            fprintf(stderr, "FAIL %-22s %.3e = %.2e of peak\n", "frames",
                    worst, worst / peak);
            failures++;
        } else {
            printf("  ok  %-22s [%4u,%3u,%3u,%3u] %.3e = %.2e of peak %.4f\n",
                   "frames", image.channels, image.depth, image.height,
                   image.width, worst, worst / peak, peak);
        }
        /* The decode has to have *done* something: an all-constant output
         * would agree with a reference that was also broken the same way. */
        double spread = 0.0;
        for (size_t index = 1; index < volume(image); index++)
            if (fabs((double)pixels[index] - (double)pixels[0]) > spread)
                spread = fabs((double)pixels[index] - (double)pixels[0]);
        require(spread > 0.1, "the decoded frames are nearly constant");
        free(want);
    }

    printf("\ndecoded %u frames of %ux%u in %.2f s\n", image.depth,
           image.height, image.width, now() - began);
    free(pixels);
    h3_gpu_tensor_free(ones);
    h3_gpu_free(gpu);
    h3_weight_store_free(store);
    h3_weight_store_free(anchor);
    if (failures) {
        fprintf(stderr, "\n%d comparisons failed\n", failures);
        return 1;
    }
    printf("ok: the video VAE decoder runs end to end -- 25 residual blocks, "
           "four shuffles and the unpatchify, on the released weights\n");
    return 0;
}
