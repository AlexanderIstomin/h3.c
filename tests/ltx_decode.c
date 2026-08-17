/* Decode an LTX-2.5 video latent to frames.
 *
 * The driver half of `test_real_ltx_video_vae.c`: the same decode, at whatever
 * geometry the latent carries, with the stage comparisons removed and the
 * per-channel statistics read from the checkpoint rather than a fixture. What
 * it writes is raw pixels for something else to turn into an image.
 *
 * Original header follows, because every convention it records still applies.
 *
 * LTX-2.5's video VAE, both halves, on the released weights.
 *
 * 3 channels at 9x128x128 in to 128 latent channels at 2x4x4 and back -- the
 * smallest shape that exercises every stage, since the stack compresses 8x in
 * time (8*(k-1)+1 frames) and 32x in space.
 *
 * Anchored against `ltx_core`, the package that ships with the model -- the
 * same one the DiT and the text tower use. An earlier version of this test
 * used `ltx_video`'s `CausalVideoAutoencoder` with a hand-written patch,
 * because that looked like the only implementation available. It was not, and
 * the reference that came with the model needs no patch at all.
 *
 * That mattered, because the two disagree about something the patch could not
 * have supplied. **The latent the DiT works in is normalized, and the VAE is
 * what maps in and out of that space**: the encoder ends with
 * `(x - mean) / std` and the decoder *begins* with `x * std + mean`, from the
 * per-channel statistics stored beside the weights. `ltx_video` has neither.
 * The scales are not small -- std spans 0.074 to 0.914 and mean -0.55 to
 * +0.43 -- so a decoder without them is wrong by up to 13x per channel plus
 * an offset, and would turn any real DiT latent into noise.
 *
 * **A round trip cannot catch that, and this file used to rely on one.** Both
 * omissions sit at opposite ends and are exact inverses, so encode-then-decode
 * closes at precisely the right 30.80 dB with both steps missing. Forcing the
 * statistics to an identity leaves every stage comparison here off by 3x to
 * 12x of peak while the round-trip line still reports 30.80 dB. A round trip
 * validates the composition; only the boundary tensors validate the boundary.
 *
 * The two halves are here together because their conventions only make sense
 * side by side -- they are mirror images that disagree in three places:
 *
 *   - **the encoder is causal and the decoder is not.** `Encoder.forward`
 *     passes no `causal` argument at all, so every convolution takes the
 *     default: two copies of the first frame in front, none behind. The
 *     decoder was built with `causal_decoder: False` and pads one at each end.
 *     Same class, opposite convention, and nothing in either file says so;
 *
 *   - **the encoder's downsample carries a skip the decoder's upsample has
 *     no counterpart for**: space-to-depth of its own input, averaged over
 *     consecutive channel groups. The group size is implied, not stored --
 *     `in_channels * prod(stride) / out_channels`;
 *
 *   - **frame handling is asymmetric.** The encoder *prepends* a duplicate
 *     first frame before a temporal downsample; the decoder *discards* its
 *     first output frame after a temporal upsample, unconditionally. That
 *     pair is what keeps the stack on 8*(k-1)+1 frames in both directions.
 *
 * Shared conventions, all mutation-checked:
 *
 *   - **temporal padding replicates, spatial padding zero-fills**, on the same
 *     convolution -- the temporal path copies edge frames while height and
 *     width get `nn.Conv3d`'s zeros;
 *
 *   - **pixel norm is parameter-free and its epsilon is inside the root:**
 *     `x / sqrt(mean(x^2 over channels) + 1e-8)`. The checkpoint carries no
 *     norm tensors at all, which is the tell. That epsilon is the one
 *     convention here no mutation catches, because at these magnitudes it is
 *     inert -- so its margin is measured instead of asserted; and
 *
 *   - **the patch channel order is (c, t, w, h), not (c, t, h, w).** The
 *     reference reads `(c p r q) -> (f p) (h q) (w r)`, so the fastest-varying
 *     channel index lands on *height* and the next one on width. Getting that
 *     backwards transposes every 4x4 patch, which still looks like an image.
 *
 * F32 throughout is not a convenience. The encoder's activations reach a peak
 * of 195458 two blocks before the end, and the final pixel norm is what brings
 * that back to single digits; BF16 would not survive the trip.
 *
 * The space-to-depth and depth-to-space steps run on the host -- eight of
 * them, one readback each, the largest moving 1.2M floats. That is a
 * performance gap rather than a correctness one, and the only piece of either
 * half the GPU cannot yet do end to end.
 *
 * usage: h3_ltx_decode VAE.safetensors LATENT.bin OUT.bin */

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
    IMAGE_FRAMES = 9, IMAGE_SIZE = 128,
    PATCH = 4,
    IMAGE_CHANNELS = 3,
    PACKED_CHANNELS = IMAGE_CHANNELS * PATCH * PATCH,
    /* `latent_log_var: uniform` gives the encoder one extra channel, which is
     * broadcast into a log variance the sampler never uses here. */
    MOMENT_CHANNELS = LATENT_CHANNELS + 1,
    WIDEST_CHANNELS = 1024,
    BLOCKS = 9,
    KERNEL = 3
};

#define PIXEL_NORM_EPSILON 1e-8f

/* Both sides compute in F32, so this bounds the gap between MPS's convolution
 * accumulation order and torch's, not a difference in precision. Measured
 * across every stage of both halves, that gap is 6.0e-07 to 1.95e-05. */
#define TOLERANCE 5e-5

/* Except at two stages, where F32 *itself* cannot do better than this bound.
 *
 * `gen_ltx_vae_anchor.py --floor` runs the reference twice, F32 and F64, and
 * the spread is not uniform across the stack: it sits near 5e-06 almost
 * everywhere but spikes to 1.65e-04 at the encoder's first downsample and is
 * still 7.1e-05 at the block after it. That is the hardest arithmetic in the
 * model -- `compress_space_res` builds a peak of 85 out of inputs near 19,
 * summing a 64-channel convolution against a group-averaged skip, so it
 * cancels hard -- and the engine's own deviation there, 9.9e-05, is *smaller*
 * than what F32 costs the reference.
 *
 * A single flat bound across a 50-block stack is a guess. These two entries
 * are the measurement, and the bound each gets is twice its floor. Anything
 * structurally wrong is off by a factor of ten thousand, not two. */
static const struct { const char *label; double floor; } MEASURED_FLOOR[] = {
    {"enc_block1_compress_space_res", 1.65e-04},
    {"enc_block2_res_x",              7.12e-05}
};

static double tolerance_for(const char *label) {
    for (size_t index = 0; index < sizeof(MEASURED_FLOOR) /
                                   sizeof(*MEASURED_FLOOR); index++)
        if (!strcmp(label, MEASURED_FLOOR[index].label)) {
            const double bound = 2.0 * MEASURED_FLOOR[index].floor;
            return bound > TOLERANCE ? bound : TOLERANCE;
        }
    return TOLERANCE;
}

static int failures = 0;
static h3_weight_store *store = NULL;
static h3_weight_store *anchor = NULL;
static h3_gpu *gpu = NULL;
static h3_gpu_tensor *ones = NULL;
/* Per-channel latent statistics. The DiT works in the *normalized* latent
 * space and the VAE is what maps in and out of it: the encoder ends with
 * `(x - mean) / std` and the decoder begins with `x * std + mean`. Neither is
 * cosmetic -- std spans 0.074 to 0.914 and mean -0.55 to +0.43, so a decoder
 * without them is wrong by up to 13x per channel plus an offset. */
static float *latent_mean = NULL;
static float *latent_std = NULL;

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

static size_t positions(shape of) {
    return (size_t)of.depth * of.height * of.width;
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
 * opens its own buffer around compute only. The mirror of the same rule
 * applies at the end: submit before freeing anything encoded work still
 * reads. */
static void load_conv(const char *half, const char *name, uint32_t out_channels,
                      uint32_t in_channels, conv3d *into) {
    char full[256];
    snprintf(full, sizeof(full), "%s.%s.conv.weight", half, name);
    const size_t taps = (size_t)KERNEL * KERNEL * KERNEL;
    float *weight = read_tensor(store, full, (size_t)out_channels *
                                in_channels * taps);
    into->weight = upload(weight, (size_t)out_channels * in_channels * taps);
    free(weight);
    snprintf(full, sizeof(full), "%s.%s.conv.bias", half, name);
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
    const size_t spatial = positions(of);
    for (size_t position = 0; position < spatial; position++)
        for (uint32_t channel = 0; channel < of.channels; channel++) {
            const double actual = engine[position * of.channels + channel];
            const double wanted = expect[(size_t)channel * spatial + position];
            if (!isfinite(actual)) {
                fprintf(stderr, "FAIL %-28s produced %f\n", label, actual);
                failures++;
                return;
            }
            const double delta = fabs(actual - wanted);
            if (delta > worst) { worst = delta; worst_at = position; }
            if (fabs(wanted) > peak) peak = fabs(wanted);
        }
    const double relative = peak > 0.0 ? worst / peak : worst;
    if (relative > tolerance) {
        fprintf(stderr, "FAIL %-28s %.3e = %.2e of peak %.4f, at %zu\n",
                label, worst, relative, peak, worst_at);
        failures++;
    } else {
        printf("  ok  %-28s [%4u,%2u,%3u,%3u] %.3e = %.2e of peak %.4g\n",
               label, of.channels, of.depth, of.height, of.width, worst,
               relative, peak);
    }
}

/* Compare a stage against the anchor and release both buffers. */
static void check(const char *label, const h3_gpu_tensor *tensor, shape of) {
    float *have = download(tensor, volume(of));
    float *want = golden(label, volume(of));
    compare(label, have, want, of, tolerance_for(label));
    free(have); free(want);
}

/* ----------------------------------------------------------- the padding */

/* Frames are the outermost axis of a channels-last buffer, so each is a
 * contiguous run and a handful of copies do the whole job -- no kernel needed.
 *
 * `causal` decides the shape of the replication, and this is the one place the
 * two halves differ at the level of a single convolution: the encoder pads two
 * frames in front and none behind, the decoder one at each end. Both widen the
 * stack by the same two frames, so the convolution below is identical. */
static h3_gpu_tensor *pad_time(const h3_gpu_tensor *input, shape of,
                               int causal) {
    const size_t frame = (size_t)of.height * of.width * of.channels;
    h3_gpu_tensor *out = h3_gpu_tensor_new_f32(gpu, (size_t)(of.depth + 2) * frame);
    require(out != NULL, "cannot allocate a padded frame stack");
    if (causal) {
        GPU_OP(h3_gpu_copy_f32(gpu, out, 2 * frame, input, 0,
                               (size_t)of.depth * frame), "pad interior");
        GPU_OP(h3_gpu_copy_f32(gpu, out, 0, input, 0, frame), "pad first");
        GPU_OP(h3_gpu_copy_f32(gpu, out, frame, input, 0, frame), "pad second");
    } else {
        GPU_OP(h3_gpu_copy_f32(gpu, out, frame, input, 0,
                               (size_t)of.depth * frame), "pad interior");
        GPU_OP(h3_gpu_copy_f32(gpu, out, 0, input, 0, frame), "pad front");
        GPU_OP(h3_gpu_copy_f32(gpu, out, (size_t)(of.depth + 1) * frame, input,
                               (size_t)(of.depth - 1) * frame, frame), "pad back");
    }
    return out;
}

/* Replicate in time, zeros in height and width -- the conv kernel same-pads
 * the spatial axes and leaves depth alone, which is exactly the split the
 * reference's `nn.Conv3d(padding=(0, 1, 1))` after an explicit temporal
 * concatenation produces. */
static h3_gpu_tensor *run_conv(const conv3d *kernel, const h3_gpu_tensor *input,
                               shape of, uint32_t out_channels, int causal) {
    h3_gpu_tensor *padded = pad_time(input, of, causal);
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
                       shape of) {
    GPU_OP(h3_gpu_rms_norm_f32(gpu, out, input, ones,
                               (uint32_t)positions(of), of.channels,
                               PIXEL_NORM_EPSILON), "pixel norm");
}

/* ------------------------------------------------------- packing on the host */

/* `b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)`, channels last. The
 * channel index decomposes major-to-minor as (c, p1, p2, p3), so the depth
 * factor is the *slowest* of the three sub-indices. */
static void depth_to_space(const float *input, float *out, shape of,
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

/* The exact inverse: `b c (d p1) (h p2) (w p3) -> b (c p1 p2 p3) d h w`, so
 * the same (c, p1, p2, p3) decomposition read the other way. */
static void space_to_depth(const float *input, float *out, shape of,
                           uint32_t depth_down, uint32_t height_down,
                           uint32_t width_down) {
    const uint32_t factor = depth_down * height_down * width_down;
    const uint32_t out_depth = of.depth / depth_down;
    const uint32_t out_height = of.height / height_down;
    const uint32_t out_width = of.width / width_down;
    for (uint32_t d = 0; d < out_depth; d++)
        for (uint32_t h = 0; h < out_height; h++)
            for (uint32_t w = 0; w < out_width; w++) {
                float *target = out +
                    (((size_t)d * out_height + h) * out_width + w) *
                    of.channels * factor;
                for (uint32_t c = 0; c < of.channels; c++)
                    for (uint32_t i = 0; i < depth_down; i++)
                        for (uint32_t j = 0; j < height_down; j++)
                            for (uint32_t k = 0; k < width_down; k++) {
                                const uint32_t to =
                                    ((c * depth_down + i) * height_down + j) *
                                    width_down + k;
                                const size_t from =
                                    (((size_t)(d * depth_down + i) * of.height +
                                      h * height_down + j) * of.width +
                                     w * width_down + k) * of.channels + c;
                                target[to] = input[from];
                            }
            }
}

/* Average consecutive runs of `group` channels: `b (c g) d h w -> b c d h w`.
 * The group size is nowhere in the checkpoint -- it follows from the widths,
 * `in_channels * prod(stride) / out_channels`, which is why it is computed
 * from the block description rather than read. */
static void group_mean(const float *input, float *out, size_t count,
                       uint32_t channels, uint32_t group) {
    const uint32_t narrow = channels / group;
    for (size_t position = 0; position < count; position++)
        for (uint32_t c = 0; c < narrow; c++) {
            double sum = 0.0;
            for (uint32_t member = 0; member < group; member++)
                sum += input[position * channels + c * group + member];
            out[position * narrow + c] = (float)(sum / (double)group);
        }
}

/* `b c (f p) (h q) (w r) -> b (c p r q) f h w` with p = 1, from torch's
 * [C][D][H][W] straight into the engine's [D][H][W][C]. `q` varies fastest in
 * the channel packing and indexes *height*; `r` is the slower one and indexes
 * width. */
static void patchify(const float *image, float *out, shape packed) {
    const uint32_t height = packed.height * PATCH, width = packed.width * PATCH;
    const size_t plane = (size_t)packed.depth * height * width;
    for (uint32_t d = 0; d < packed.depth; d++)
        for (uint32_t h = 0; h < packed.height; h++)
            for (uint32_t w = 0; w < packed.width; w++) {
                float *target = out +
                    (((size_t)d * packed.height + h) * packed.width + w) *
                    packed.channels;
                for (uint32_t c = 0; c < IMAGE_CHANNELS; c++)
                    for (uint32_t r = 0; r < PATCH; r++)
                        for (uint32_t q = 0; q < PATCH; q++)
                            target[(c * PATCH + r) * PATCH + q] =
                                image[(size_t)c * plane +
                                      ((size_t)d * height + h * PATCH + q) *
                                      width + w * PATCH + r];
            }
}

/* The exact inverse, back into torch's [C][D][H][W] so the final comparison
 * needs no permutation of its own. */
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

/* --------------------------------------------------------------- the blocks */

typedef struct {
    const char *kind;
    int layers;                 /* res_x only */
    int multiplier;             /* compress only */
    uint32_t depth_step, height_step, width_step;
} vae_block;

/* `decoder_blocks` from the checkpoint config, reversed -- the decoder walks it
 * backwards -- and `encoder_blocks` in order. Written out rather than derived
 * so both can be read against the config by eye. */
static const vae_block DECODE[BLOCKS] = {
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

static const vae_block ENCODE[BLOCKS] = {
    {"res_x",              4, 0, 0, 0, 0},
    {"compress_space_res", 0, 2, 1, 2, 2},
    {"res_x",              6, 0, 0, 0, 0},
    {"compress_time_res",  0, 2, 2, 1, 1},
    {"res_x",              4, 0, 0, 0, 0},
    {"compress_all_res",   0, 2, 2, 2, 2},
    {"res_x",              2, 0, 0, 0, 0},
    {"compress_all_res",   0, 1, 2, 2, 2},
    {"res_x",              2, 0, 0, 0, 0}
};

/* pixel norm, SiLU, conv, again, then add the input back. Every res_x block
 * keeps its channel count, so the reference's shortcut projection and its
 * third norm are both Identity -- which is why the checkpoint carries neither. */
static void resnet(const char *half, int block, int layer, h3_gpu_tensor *x,
                   shape of, int causal) {
    char name[128];
    conv3d first, second;
    snprintf(name, sizeof(name), "%s_blocks.%d.res_blocks.%d.conv1",
             causal ? "down" : "up", block, layer);
    load_conv(half, name, of.channels, of.channels, &first);
    snprintf(name, sizeof(name), "%s_blocks.%d.res_blocks.%d.conv2",
             causal ? "down" : "up", block, layer);
    load_conv(half, name, of.channels, of.channels, &second);
    h3_gpu_tensor *scratch = h3_gpu_tensor_new_f32(gpu, volume(of));
    require(scratch != NULL, "cannot allocate resnet scratch");

    GPU_OP(h3_gpu_begin(gpu), "begin resnet");
    pixel_norm(scratch, x, of);
    GPU_OP(h3_gpu_silu_f32(gpu, scratch, scratch, (uint32_t)volume(of)),
           "resnet activation");
    h3_gpu_tensor *hidden = run_conv(&first, scratch, of, of.channels, causal);
    pixel_norm(scratch, hidden, of);
    GPU_OP(h3_gpu_silu_f32(gpu, scratch, scratch, (uint32_t)volume(of)),
           "resnet activation");
    h3_gpu_tensor *second_out = run_conv(&second, scratch, of, of.channels,
                                         causal);
    GPU_OP(h3_gpu_add_scaled_f32(gpu, x, x, second_out, 1.0f, 1.0f,
                                 (uint32_t)volume(of)), "resnet residual");
    GPU_OP(h3_gpu_submit(gpu), "submit resnet");
    h3_gpu_tensor_free(hidden);
    h3_gpu_tensor_free(second_out);
    h3_gpu_tensor_free(scratch);
    free_conv(&first); free_conv(&second);
}

/* Convolve up to `prod(stride) * channels / multiplier`, spread those channels
 * out into space and time, then -- when the temporal factor is 2 -- discard the
 * first output frame. The discard is unconditional in the reference, so 2
 * latent frames become 3 rather than 4. */
static h3_gpu_tensor *upsample(int block, const vae_block *plan,
                               h3_gpu_tensor *x, shape *of) {
    char name[128];
    const uint32_t factor = plan->depth_step * plan->height_step *
                            plan->width_step;
    const uint32_t wide = of->channels * factor / (uint32_t)plan->multiplier;
    snprintf(name, sizeof(name), "up_blocks.%d.conv", block);
    conv3d kernel;
    load_conv("decoder", name, wide, of->channels, &kernel);
    GPU_OP(h3_gpu_begin(gpu), "begin upsample");
    h3_gpu_tensor *convolved = run_conv(&kernel, x, *of, wide, 0);
    GPU_OP(h3_gpu_submit(gpu), "submit upsample");
    free_conv(&kernel);
    h3_gpu_tensor_free(x);

    shape wide_shape = {of->depth, of->height, of->width, wide};
    float *host = download(convolved, volume(wide_shape));
    h3_gpu_tensor_free(convolved);
    shape spread = {of->depth * plan->depth_step,
                    of->height * plan->height_step,
                    of->width * plan->width_step, wide / factor};
    float *expanded = malloc(volume(spread) * sizeof(*expanded));
    require(expanded != NULL, "cannot allocate an expanded stack");
    depth_to_space(host, expanded, wide_shape, plan->depth_step,
                   plan->height_step, plan->width_step);
    free(host);

    const float *keep = expanded;
    if (plan->depth_step == 2) {
        keep = expanded + (size_t)spread.height * spread.width * spread.channels;
        spread.depth -= 1;
    }
    h3_gpu_tensor *out = upload(keep, volume(spread));
    free(expanded);
    *of = spread;
    return out;
}

/* The encoder's mirror, with the extra term the decoder has no counterpart
 * for: a skip that is the space-to-depth of the block's own input, averaged
 * over consecutive channel groups. The duplicated leading frame goes in
 * *before* both branches, so the skip sees it too. */
static h3_gpu_tensor *downsample(int block, const vae_block *plan,
                                 h3_gpu_tensor *x, shape *of) {
    char name[128];
    const uint32_t factor = plan->depth_step * plan->height_step *
                            plan->width_step;
    const uint32_t out_channels = of->channels * (uint32_t)plan->multiplier;
    const uint32_t group = of->channels * factor / out_channels;
    const uint32_t narrow = out_channels / factor;

    shape extended = *of;
    h3_gpu_tensor *source = x;
    if (plan->depth_step == 2) {
        const size_t frame = (size_t)of->height * of->width * of->channels;
        extended.depth += 1;
        source = h3_gpu_tensor_new_f32(gpu, (size_t)extended.depth * frame);
        require(source != NULL, "cannot allocate an extended stack");
        GPU_OP(h3_gpu_begin(gpu), "begin frame duplication");
        GPU_OP(h3_gpu_copy_f32(gpu, source, frame, x, 0,
                               (size_t)of->depth * frame), "extend interior");
        GPU_OP(h3_gpu_copy_f32(gpu, source, 0, x, 0, frame), "extend front");
        GPU_OP(h3_gpu_submit(gpu), "submit frame duplication");
        h3_gpu_tensor_free(x);
    }

    snprintf(name, sizeof(name), "down_blocks.%d.conv", block);
    conv3d kernel;
    load_conv("encoder", name, narrow, extended.channels, &kernel);
    GPU_OP(h3_gpu_begin(gpu), "begin downsample");
    h3_gpu_tensor *convolved = run_conv(&kernel, source, extended, narrow, 1);
    GPU_OP(h3_gpu_submit(gpu), "submit downsample");
    free_conv(&kernel);

    shape packed = {extended.depth / plan->depth_step,
                    extended.height / plan->height_step,
                    extended.width / plan->width_step, out_channels};
    float *skip = malloc(volume(packed) * sizeof(*skip));
    float *branch = malloc(volume(packed) * sizeof(*branch));
    require(skip && branch, "cannot allocate a downsample stack");
    {
        float *host = download(source, volume(extended));
        float *wide = malloc(positions(packed) * extended.channels * factor *
                             sizeof(*wide));
        require(wide != NULL, "cannot allocate the skip stack");
        space_to_depth(host, wide, extended, plan->depth_step,
                       plan->height_step, plan->width_step);
        group_mean(wide, skip, positions(packed),
                   extended.channels * factor, group);
        free(wide); free(host);
    }
    {
        shape narrow_shape = {extended.depth, extended.height, extended.width,
                              narrow};
        float *host = download(convolved, volume(narrow_shape));
        space_to_depth(host, branch, narrow_shape, plan->depth_step,
                       plan->height_step, plan->width_step);
        free(host);
    }
    h3_gpu_tensor_free(source);
    h3_gpu_tensor_free(convolved);
    for (size_t index = 0; index < volume(packed); index++)
        branch[index] += skip[index];
    h3_gpu_tensor *out = upload(branch, volume(packed));
    free(skip); free(branch);
    *of = packed;
    return out;
}

/* ------------------------------------------------------------------- halves */

/* Image in [C][D][H][W] to moments in [D][H][W][C]. `check` compares every
 * stage; the second pass through skips that so the end-to-end round trip does
 * not report the same numbers twice. */
static float *encode(const float *image, shape *of, int check_stages) {
    shape packed = {IMAGE_FRAMES, IMAGE_SIZE / PATCH, IMAGE_SIZE / PATCH,
                    PACKED_CHANNELS};
    float *staged = malloc(volume(packed) * sizeof(*staged));
    require(staged != NULL, "cannot allocate the patchified image");
    patchify(image, staged, packed);
    h3_gpu_tensor *x = upload(staged, volume(packed));
    free(staged);
    if (check_stages) check("enc_patchify", x, packed);

    shape at = packed;
    {
        conv3d kernel;
        load_conv("encoder", "conv_in", 128, PACKED_CHANNELS, &kernel);
        GPU_OP(h3_gpu_begin(gpu), "begin enc conv_in");
        h3_gpu_tensor *out = run_conv(&kernel, x, at, 128, 1);
        GPU_OP(h3_gpu_submit(gpu), "submit enc conv_in");
        free_conv(&kernel);
        h3_gpu_tensor_free(x);
        x = out;
        at.channels = 128;
    }
    if (check_stages) check("enc_conv_in", x, at);

    for (int index = 0; index < BLOCKS; index++) {
        const vae_block *plan = &ENCODE[index];
        if (plan->layers)
            for (int layer = 0; layer < plan->layers; layer++)
                resnet("encoder", index, layer, x, at, 1);
        else
            x = downsample(index, plan, x, &at);
        if (check_stages) {
            char label[64];
            snprintf(label, sizeof(label), "enc_block%d_%s", index, plan->kind);
            check(label, x, at);
        }
    }

    h3_gpu_tensor *normed = h3_gpu_tensor_new_f32(gpu, volume(at));
    require(normed != NULL, "cannot allocate the normalized stack");
    GPU_OP(h3_gpu_begin(gpu), "begin enc tail");
    pixel_norm(normed, x, at);
    GPU_OP(h3_gpu_submit(gpu), "submit enc norm");
    if (check_stages) check("enc_norm_out", normed, at);
    h3_gpu_tensor_free(x);

    conv3d tail;
    load_conv("encoder", "conv_out", MOMENT_CHANNELS, at.channels, &tail);
    GPU_OP(h3_gpu_begin(gpu), "begin enc head");
    GPU_OP(h3_gpu_silu_f32(gpu, normed, normed, (uint32_t)volume(at)),
           "encoder output activation");
    h3_gpu_tensor *moments = run_conv(&tail, normed, at, MOMENT_CHANNELS, 1);
    GPU_OP(h3_gpu_submit(gpu), "submit enc head");
    free_conv(&tail);
    h3_gpu_tensor_free(normed);
    at.channels = MOMENT_CHANNELS;
    if (check_stages) check("enc_conv_out", moments, at);

    /* The leading 128 of the 129 channels are the means; the last is a shared
     * log variance the deterministic path never reads. Normalizing them is
     * what produces a latent in the space the DiT works in -- and it is
     * invisible to a round trip, because the decoder's inverse step cancels
     * it exactly. Only this comparison catches it. */
    shape latent = {at.depth, at.height, at.width, LATENT_CHANNELS};
    float *host = download(moments, volume(at));
    h3_gpu_tensor_free(moments);
    float *normalized = malloc(volume(latent) * sizeof(*normalized));
    require(normalized != NULL, "cannot allocate the normalized latent");
    for (size_t position = 0; position < positions(latent); position++)
        for (uint32_t channel = 0; channel < LATENT_CHANNELS; channel++)
            normalized[position * LATENT_CHANNELS + channel] =
                (host[position * MOMENT_CHANNELS + channel] -
                 latent_mean[channel]) / latent_std[channel];
    free(host);
    if (check_stages) {
        float *want = golden("enc_normalized", volume(latent));
        compare("enc_normalized", normalized, want, latent,
                tolerance_for("enc_normalized"));
        free(want);
    }
    *of = latent;
    return normalized;
}

/* Latent in [D][H][W][C] to frames in [C][D][H][W]. Takes ownership of its
 * input. */
static float *decode(const float *latent, shape at, int check_stages,
                     shape *image_shape) {
    /* The inverse of the encoder's last step, and the decoder's first. */
    float *denormalized = malloc(volume(at) * sizeof(*denormalized));
    require(denormalized != NULL, "cannot allocate the denormalized latent");
    for (size_t position = 0; position < positions(at); position++)
        for (uint32_t channel = 0; channel < LATENT_CHANNELS; channel++)
            denormalized[position * LATENT_CHANNELS + channel] =
                latent[position * LATENT_CHANNELS + channel] *
                latent_std[channel] + latent_mean[channel];
    if (check_stages) {
        float *want = golden("dec_denormalized", volume(at));
        compare("dec_denormalized", denormalized, want, at,
                tolerance_for("dec_denormalized"));
        free(want);
    }
    h3_gpu_tensor *x = upload(denormalized, volume(at));
    free(denormalized);
    {
        conv3d kernel;
        load_conv("decoder", "conv_in", WIDEST_CHANNELS, at.channels, &kernel);
        GPU_OP(h3_gpu_begin(gpu), "begin conv_in");
        h3_gpu_tensor *out = run_conv(&kernel, x, at, WIDEST_CHANNELS, 0);
        GPU_OP(h3_gpu_submit(gpu), "submit conv_in");
        free_conv(&kernel);
        h3_gpu_tensor_free(x);
        x = out;
        at.channels = WIDEST_CHANNELS;
    }
    if (check_stages) check("conv_in", x, at);

    for (int index = 0; index < BLOCKS; index++) {
        const vae_block *plan = &DECODE[index];
        if (plan->layers)
            for (int layer = 0; layer < plan->layers; layer++)
                resnet("decoder", index, layer, x, at, 0);
        else
            x = upsample(index, plan, x, &at);
        if (check_stages) {
            char label[64];
            snprintf(label, sizeof(label), "block%d_%s", index, plan->kind);
            check(label, x, at);
        }
    }

    /* Pixel norm's epsilon is the one convention here no mutation of this file
     * can catch: setting it to zero changes nothing measurable, because it only
     * bites when a position's channel vector is small enough that its mean
     * square approaches 1e-8. So measure how close anything actually gets --
     * that turns "untested" into "shown not to apply here", and the day a
     * latent does produce a near-empty position this line says so. */
    if (check_stages) {
        float *have = download(x, volume(at));
        double smallest = INFINITY;
        for (size_t position = 0; position < positions(at); position++) {
            double sum = 0.0;
            for (uint32_t channel = 0; channel < at.channels; channel++) {
                const double value = have[position * at.channels + channel];
                sum += value * value;
            }
            const double mean = sum / (double)at.channels;
            if (mean < smallest) smallest = mean;
        }
        printf("  --  %-28s smallest mean square %.3e, %.0fx the %.0e epsilon\n",
               "epsilon is inert here", smallest,
               smallest / (double)PIXEL_NORM_EPSILON, (double)PIXEL_NORM_EPSILON);
        require(smallest > (double)PIXEL_NORM_EPSILON * 1e3,
                "a position is small enough that the epsilon matters, so it "
                "can no longer be treated as inert");
        free(have);
    }

    h3_gpu_tensor *normed = h3_gpu_tensor_new_f32(gpu, volume(at));
    require(normed != NULL, "cannot allocate the normalized stack");
    GPU_OP(h3_gpu_begin(gpu), "begin tail");
    pixel_norm(normed, x, at);
    GPU_OP(h3_gpu_submit(gpu), "submit norm");
    if (check_stages) check("norm_out", normed, at);

    conv3d tail;
    load_conv("decoder", "conv_out", PACKED_CHANNELS, at.channels, &tail);
    GPU_OP(h3_gpu_begin(gpu), "begin head");
    GPU_OP(h3_gpu_silu_f32(gpu, normed, normed, (uint32_t)volume(at)),
           "output activation");
    h3_gpu_tensor *out = run_conv(&tail, normed, at, PACKED_CHANNELS, 0);
    GPU_OP(h3_gpu_submit(gpu), "submit head");
    free_conv(&tail);
    h3_gpu_tensor_free(normed);
    h3_gpu_tensor_free(x);

    shape packed = {at.depth, at.height, at.width, PACKED_CHANNELS};
    if (check_stages) check("conv_out", out, packed);

    shape image = {packed.depth, packed.height * PATCH, packed.width * PATCH,
                   IMAGE_CHANNELS};
    float *host = download(out, volume(packed));
    h3_gpu_tensor_free(out);
    float *pixels = malloc(volume(image) * sizeof(*pixels));
    require(pixels != NULL, "cannot allocate the frames");
    unpatchify(host, pixels, packed);
    free(host);
    *image_shape = image;
    return pixels;
}

/* ------------------------------------------------------------------- main */

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s VAE.safetensors LATENT.bin OUT.bin\n",
                argv[0]);
        return 2;
    }
    char error[512];
    store = h3_weight_store_open(argv[1], error, sizeof(error));
    if (!store) fail("cannot open the VAE: %s", error);
    gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) fail("cannot create the Metal context: %s", error);

    float *helper = malloc(WIDEST_CHANNELS * sizeof(*helper));
    require(helper != NULL, "cannot allocate a norm weight");
    for (int index = 0; index < WIDEST_CHANNELS; index++) helper[index] = 1.0f;
    ones = upload(helper, WIDEST_CHANNELS);
    free(helper);

    /* From the checkpoint, not a fixture: the statistics ship beside the
     * weights and the decoder is meaningless without them. */
    latent_mean = read_tensor(store, "per_channel_statistics.mean-of-means",
                              LATENT_CHANNELS);
    latent_std = read_tensor(store, "per_channel_statistics.std-of-means",
                             LATENT_CHANNELS);

    /* ---------------------------------------------------------- the latent */

    shape at;
    float *latent = NULL;
    {
        FILE *file = fopen(argv[2], "rb");
        if (!file) fail("cannot open %s", argv[2]);
        uint32_t header[8];
        require(fread(header, sizeof(header), 1, file) == 1,
                "cannot read the latent header");
        if (header[0] != UINT32_C(0x4C545847)) fail("%s is not a latent", argv[2]);
        at.depth = header[1]; at.height = header[2]; at.width = header[3];
        at.channels = header[4];
        if (at.channels != LATENT_CHANNELS)
            fail("the latent has %u channels, expected %d", at.channels,
                 LATENT_CHANNELS);
        latent = malloc(volume(at) * sizeof(*latent));
        require(latent != NULL, "cannot allocate the latent");
        require(fread(latent, sizeof(float), volume(at), file) == volume(at),
                "cannot read the video latent");
        fclose(file);
        printf("decoding [%u, %u, %u, %u] -> [%d, %u, %u, %u]\n",
               at.channels, at.depth, at.height, at.width, IMAGE_CHANNELS,
               (at.depth - 1) * 8 + 1, at.height * 32, at.width * 32);
    }

    const double began = now();
    shape image;
    /* The generator writes tokens in frame, row, column order, which is
     * exactly the channels-last layout the decoder wants. */
    float *pixels = decode(latent, at, 0, &image);
    printf("decoded in %.2f s\n", now() - began);

    double lowest = INFINITY, highest = -INFINITY, mean = 0.0;
    for (size_t index = 0; index < volume(image); index++) {
        require(isfinite(pixels[index]), "a pixel is not finite");
        if (pixels[index] < lowest) lowest = pixels[index];
        if (pixels[index] > highest) highest = pixels[index];
        mean += (double)pixels[index];
    }
    mean /= (double)volume(image);
    printf("pixels: %.3f .. %.3f, mean %.3f (the VAE works in [-1, 1])\n",
           lowest, highest, mean);
    require(highest - lowest > 0.05,
            "the frames are nearly constant, so nothing was generated");

    {
        FILE *out = fopen(argv[3], "wb");
        if (!out) fail("cannot open %s for writing", argv[3]);
        const uint32_t header[4] = {UINT32_C(0x4C545846), image.channels,
                                    image.depth, image.height};
        require(fwrite(header, sizeof(header), 1, out) == 1,
                "cannot write the frame header");
        require(fwrite(&image.width, sizeof(uint32_t), 1, out) == 1,
                "cannot write the frame width");
        require(fwrite(pixels, sizeof(float), volume(image), out) ==
                    volume(image), "cannot write the frames");
        require(fclose(out) == 0, "cannot close the output");
        printf("wrote %s: %u frames of %ux%u, %d channels\n", argv[3],
               image.depth, image.height, image.width, IMAGE_CHANNELS);
    }

    free(pixels); free(latent);
    free(latent_mean); free(latent_std);
    h3_gpu_tensor_free(ones);
    h3_gpu_free(gpu);
    h3_weight_store_free(store);
    return 0;
}
