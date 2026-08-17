/* LTX-2.5's audio VAE decoder, on the released weights.
 *
 * A latent of [8, 8, 16] out to a stereo log-mel of [2, 29, 64] -- the piece
 * that turns the DiT's audio stream into something a vocoder can speak.
 *
 * Anchored against `ltx_core.model.audio_vae`, which ships with the model and
 * matches the checkpoint exactly and unpatched: 56 decoder tensors, nothing
 * missing, unexpected or misshapen.
 *
 * This is a **2D** VAE over log-mel spectrograms, and its layout is
 * `(channels, frames, mel_bins)` -- so the axis a 2D model would call "height"
 * is *time*. Four conventions follow, and every one of them is a place to be
 * quietly wrong:
 *
 *   - **the convolutions are causal along frames, with ZEROS.** `causality_axis:
 *     height` makes `CausalConv2d` pad `(1, 1, 2, 0)`: one column of mel either
 *     side, two frames in front, none behind. Note what that is *not* -- the
 *     video VAE replicated its edge frames, and this one zero-fills. Same
 *     shape of asymmetry, opposite fill;
 *
 *   - **the upsample discards its first frame, not its last.** Nearest-2x then
 *     a causal convolution then `x[:, :, 1:, :]`, so 8 frames become 15 rather
 *     than 16. The reference explains why in a comment worth reading: after
 *     interpolation the first two outputs both depend only on the first input,
 *     so dropping the leading one is what undoes the encoder's padding and
 *     keeps the length at 1 + 2n;
 *
 *   - **the latent is denormalized through the patchifier.** The statistics
 *     are 128 wide while the latent is 8 channels, because the normalization
 *     lives in the *patchified* space the DiT works in: `[8, T, 16]` reads as
 *     `[T, 128]` with channel index `c * 16 + f`. A port that takes the shapes
 *     at face value and tries to scale 8 channels by a 128-vector does not
 *     even fit. As on the video side, no round trip can check this -- the
 *     encoder's inverse cancels it exactly -- so the boundary tensor is
 *     compared directly; and
 *
 *   - **`mid_block_add_attention` is false**, so the mid block's `attn_1` is an
 *     Identity and the block is just two residuals. The checkpoint carrying no
 *     attention tensors at all is the tell.
 *
 * The 3x3 convolutions map onto the 3D kernel by putting frames on the depth
 * axis and mel on width, with height a degenerate 1. Depth is left unpadded,
 * so the causal frame padding is explicit and the mel axis is same-padded by
 * the kernel -- exactly the split this VAE wants, with no kernel work.
 *
 * usage: h3_real_ltx_audio_vae_test VAE.safetensors ANCHOR.safetensors */

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
    Z_CHANNELS = 8,
    LATENT_FRAMES = 8, LATENT_MELS = 16,
    MEL_FRAMES = 29, MEL_BINS = 64,
    STEREO = 2,
    BASE_CHANNELS = 128,
    WIDEST_CHANNELS = 512,
    /* The patchified width the statistics are indexed by: z-channels times
     * latent mel bins. This is the DiT's audio token width. */
    PATCHED = Z_CHANNELS * LATENT_MELS,
    LEVELS = 3,
    BLOCKS_PER_LEVEL = 3,
    KERNEL = 3
};

#define PIXEL_NORM_EPSILON 1e-8f

/* Both sides compute in F32, so this bounds MPS's convolution accumulation
 * order against torch's rather than a difference in precision.
 *
 * `gen_ltx_audio_vae.py --floor` runs the reference at F32 and F64: unlike the
 * video VAE's stack, whose spread spikes by two orders at one stage, this one
 * is flat at 1.0e-06 to 2.9e-06 the whole way down. The engine sits inside
 * that -- 3.0e-08 at the latent boundary to 2.6e-06 at the deepest upsample --
 * so a single bound is defensible here, and it is set just above both. */
#define TOLERANCE 1e-5

static int failures = 0;
static h3_weight_store *store = NULL;
static h3_weight_store *anchor = NULL;
static h3_gpu *gpu = NULL;
static h3_gpu_tensor *ones = NULL;
static h3_gpu_tensor *zeros = NULL;
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

/* Channels last, with mel where a 2D model would put width and frames where it
 * would put height -- which is time. */
typedef struct { uint32_t frames, mels, channels; } shape;

static size_t volume(shape of) {
    return (size_t)of.frames * of.mels * of.channels;
}

static size_t positions(shape of) {
    return (size_t)of.frames * of.mels;
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

typedef struct { h3_gpu_tensor *weight, *bias; uint32_t kernel; } conv2d;

/* Every load happens before `h3_gpu_begin`: an upload into an already-open
 * command buffer is not seen by work encoded after it, and the convolution
 * quietly reads its input as zeros. */
static void load_conv(const char *name, uint32_t out_channels,
                      uint32_t in_channels, uint32_t kernel, conv2d *into) {
    char full[256];
    snprintf(full, sizeof(full), "audio_vae.decoder.%s.conv.weight", name);
    const size_t taps = (size_t)kernel * kernel;
    float *weight = read_tensor(store, full,
                                (size_t)out_channels * in_channels * taps);
    into->weight = upload(weight, (size_t)out_channels * in_channels * taps);
    free(weight);
    snprintf(full, sizeof(full), "audio_vae.decoder.%s.conv.bias", name);
    float *bias = read_tensor(store, full, out_channels);
    into->bias = upload(bias, out_channels);
    free(bias);
    into->kernel = kernel;
}

static void free_conv(conv2d *which) {
    h3_gpu_tensor_free(which->weight);
    h3_gpu_tensor_free(which->bias);
    memset(which, 0, sizeof(*which));
}

/* ---------------------------------------------------------------- compare */

/* The golden tensors are torch's [C][frames][mels]; the engine's are
 * [frames][mels][C]. Walking both keeps the convention in one place. */
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
        fprintf(stderr, "FAIL %-22s %.3e = %.2e of peak %.4g, at %zu\n",
                label, worst, relative, peak, worst_at);
        failures++;
    } else {
        printf("  ok  %-22s [%4u,%3u,%3u] %.3e = %.2e of peak %.4g\n",
               label, of.channels, of.frames, of.mels, worst, relative, peak);
    }
}

static void check(const char *label, const h3_gpu_tensor *tensor, shape of) {
    float *have = download(tensor, volume(of));
    float *want = golden(label, volume(of));
    compare(label, have, want, of, TOLERANCE);
    free(have); free(want);
}

/* ------------------------------------------------------------- the stages */

/* Two zero frames in front and none behind -- `CausalConv2d`'s
 * `(pad_w/2, pad_w-pad_w/2, pad_h, 0)` under `causality_axis: height`, where
 * pad_h is `kernel - 1`. Frames are the outermost axis of a channels-last
 * buffer, so this is one copy plus one zero fill.
 *
 * The fill is *zeros*, not replicated edge frames. The video VAE replicates;
 * this one does not, and nothing but the reference says so. */
static h3_gpu_tensor *pad_frames(const h3_gpu_tensor *input, shape of,
                                 uint32_t kernel) {
    const uint32_t front = kernel - 1;
    const size_t frame = (size_t)of.mels * of.channels;
    h3_gpu_tensor *out =
        h3_gpu_tensor_new_f32(gpu, (size_t)(of.frames + front) * frame);
    require(out != NULL, "cannot allocate a padded frame stack");
    if (front)
        GPU_OP(h3_gpu_copy_f32(gpu, out, 0, zeros, 0, (size_t)front * frame),
               "zero the causal pad");
    GPU_OP(h3_gpu_copy_f32(gpu, out, (size_t)front * frame, input, 0,
                           (size_t)of.frames * frame), "pad interior");
    return out;
}

/* Causal zeros along frames, symmetric zeros along mel. The convolution kernel
 * same-pads height and width and leaves depth alone, so mapping frames onto
 * depth, mel onto width and a degenerate 1 onto height gives exactly that --
 * `kernel_height` of 1 pads nothing, `kernel_width` of 3 pads one each side. */
static h3_gpu_tensor *run_conv(const conv2d *kernel, const h3_gpu_tensor *input,
                               shape of, uint32_t out_channels) {
    h3_gpu_tensor *padded = pad_frames(input, of, kernel->kernel);
    shape result = {of.frames, of.mels, out_channels};
    h3_gpu_tensor *out = h3_gpu_tensor_new_f32(gpu, volume(result));
    require(out != NULL, "cannot allocate a convolution output");
    GPU_OP(h3_gpu_conv3d_same_f32(gpu, out, padded, kernel->weight,
                                  kernel->bias, 1,
                                  of.frames + kernel->kernel - 1, 1, of.mels,
                                  of.channels, out_channels,
                                  kernel->kernel, 1, kernel->kernel, 1, 1, 1),
           "convolution");
    h3_gpu_tensor_free(padded);
    return out;
}

/* x / sqrt(mean(x^2 over channels) + eps) -- the RMS norm kernel with a weight
 * of ones, epsilon inside the root in both. */
static void pixel_norm(h3_gpu_tensor *out, const h3_gpu_tensor *input,
                       shape of) {
    GPU_OP(h3_gpu_rms_norm_f32(gpu, out, input, ones,
                               (uint32_t)positions(of), of.channels,
                               PIXEL_NORM_EPSILON), "pixel norm");
}

/* norm, activate, convolve, twice, then add the input back -- through a 1x1
 * projection when the channel count changes, which for this decoder is the
 * first block of each of the two narrowing levels. */
static h3_gpu_tensor *resnet(const char *prefix, h3_gpu_tensor *x, shape of,
                             uint32_t out_channels) {
    char name[160];
    conv2d first, second, shortcut;
    memset(&shortcut, 0, sizeof(shortcut));
    snprintf(name, sizeof(name), "%s.conv1", prefix);
    load_conv(name, out_channels, of.channels, KERNEL, &first);
    snprintf(name, sizeof(name), "%s.conv2", prefix);
    load_conv(name, out_channels, out_channels, KERNEL, &second);
    if (of.channels != out_channels) {
        snprintf(name, sizeof(name), "%s.nin_shortcut", prefix);
        load_conv(name, out_channels, of.channels, 1, &shortcut);
    }
    shape wide = {of.frames, of.mels, out_channels};
    h3_gpu_tensor *scratch = h3_gpu_tensor_new_f32(gpu, volume(of));
    h3_gpu_tensor *inner = h3_gpu_tensor_new_f32(gpu, volume(wide));
    require(scratch && inner, "cannot allocate resnet scratch");

    GPU_OP(h3_gpu_begin(gpu), "begin resnet");
    pixel_norm(scratch, x, of);
    GPU_OP(h3_gpu_silu_f32(gpu, scratch, scratch, (uint32_t)volume(of)),
           "resnet activation");
    h3_gpu_tensor *hidden = run_conv(&first, scratch, of, out_channels);
    pixel_norm(inner, hidden, wide);
    GPU_OP(h3_gpu_silu_f32(gpu, inner, inner, (uint32_t)volume(wide)),
           "resnet activation");
    h3_gpu_tensor *branch = run_conv(&second, inner, wide, out_channels);
    /* The residual takes the block's *input*, projected when the width
     * changes -- not the normalized copy above. */
    h3_gpu_tensor *residual = x;
    if (of.channels != out_channels)
        residual = run_conv(&shortcut, x, of, out_channels);
    GPU_OP(h3_gpu_add_scaled_f32(gpu, branch, residual, branch, 1.0f, 1.0f,
                                 (uint32_t)volume(wide)), "resnet residual");
    GPU_OP(h3_gpu_submit(gpu), "submit resnet");

    if (residual != x) h3_gpu_tensor_free(residual);
    h3_gpu_tensor_free(hidden);
    h3_gpu_tensor_free(inner);
    h3_gpu_tensor_free(scratch);
    h3_gpu_tensor_free(x);
    free_conv(&first); free_conv(&second);
    if (shortcut.weight) free_conv(&shortcut);
    return branch;
}

/* Nearest-2x on both axes, a causal convolution, then discard the *first*
 * frame. That last step is what keeps the length at 1 + 2n: after
 * interpolation the first two outputs both depend only on the first input, so
 * dropping the leading one undoes the encoder's padding. Dropping the trailing
 * one instead would keep the same shape and be wrong. */
static h3_gpu_tensor *upsample(int level, h3_gpu_tensor *x, shape *of) {
    char name[160];
    conv2d kernel;
    snprintf(name, sizeof(name), "up.%d.upsample.conv", level);
    load_conv(name, of->channels, of->channels, KERNEL, &kernel);

    shape doubled = {of->frames * 2, of->mels * 2, of->channels};
    h3_gpu_tensor *big = h3_gpu_tensor_new_f32(gpu, volume(doubled));
    require(big != NULL, "cannot allocate an upsampled stack");
    GPU_OP(h3_gpu_begin(gpu), "begin upsample");
    GPU_OP(h3_gpu_nearest2x_nhwc_f32(gpu, big, x, of->frames, of->mels,
                                     of->channels), "nearest upsample");
    h3_gpu_tensor *convolved = run_conv(&kernel, big, doubled, of->channels);
    GPU_OP(h3_gpu_submit(gpu), "submit upsample");
    free_conv(&kernel);
    h3_gpu_tensor_free(big);
    h3_gpu_tensor_free(x);

    shape kept = {doubled.frames - 1, doubled.mels, doubled.channels};
    h3_gpu_tensor *out = h3_gpu_tensor_new_f32(gpu, volume(kept));
    require(out != NULL, "cannot allocate the trimmed stack");
    GPU_OP(h3_gpu_begin(gpu), "begin trim");
    GPU_OP(h3_gpu_copy_f32(gpu, out, 0, convolved,
                           (size_t)doubled.mels * doubled.channels,
                           volume(kept)), "drop the first frame");
    GPU_OP(h3_gpu_submit(gpu), "submit trim");
    h3_gpu_tensor_free(convolved);
    *of = kept;
    return out;
}

/* ------------------------------------------------------------------- main */

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

    float *helper = malloc(WIDEST_CHANNELS * sizeof(*helper));
    require(helper != NULL, "cannot allocate a norm weight");
    for (int index = 0; index < WIDEST_CHANNELS; index++) helper[index] = 1.0f;
    ones = upload(helper, WIDEST_CHANNELS);
    /* Two frames of the widest stage, for the causal pad. The widest is not a
     * level input but an upsample's *output*: nearest-2x doubles the mel axis
     * while leaving the channel count alone, so 512 channels meet 32 mel bins
     * and 256 meet 64 -- both 16384, twice any level input. */
    const size_t widest_frame = (size_t)MEL_BINS * WIDEST_CHANNELS;
    float *blank = calloc(2 * widest_frame, sizeof(*blank));
    require(blank != NULL, "cannot allocate the causal pad");
    zeros = upload(blank, 2 * widest_frame);
    free(helper); free(blank);

    latent_mean = golden("mean_of_means", PATCHED);
    latent_std = golden("std_of_means", PATCHED);

    printf("LTX-2.5 audio VAE decoder: [%d, %d, %d] latent -> "
           "[%d, %d, %d] log-mel\n", Z_CHANNELS, LATENT_FRAMES, LATENT_MELS,
           STEREO, MEL_FRAMES, MEL_BINS);
    const double began = now();

    /* ------------------------------------------------- the latent boundary */

    shape at = {LATENT_FRAMES, LATENT_MELS, Z_CHANNELS};
    float *want_latent = golden("latent", volume(at));
    float *staged = malloc(volume(at) * sizeof(*staged));
    require(staged != NULL, "cannot allocate the staged latent");
    /* Golden is [C][frames][mels]; the engine wants [frames][mels][C]. */
    for (size_t position = 0; position < positions(at); position++)
        for (uint32_t channel = 0; channel < Z_CHANNELS; channel++)
            staged[position * Z_CHANNELS + channel] =
                want_latent[(size_t)channel * positions(at) + position];
    free(want_latent);

    /* `x * std + mean`, indexed in the *patchified* space: the statistics are
     * 128 wide and the latent is 8 channels, because the patchifier reads
     * `[8, T, 16]` as `[T, 128]` with channel `c * 16 + f`. Getting this wrong
     * does not fit rather than merely disagreeing, which is a small mercy. */
    for (uint32_t frame = 0; frame < at.frames; frame++)
        for (uint32_t mel = 0; mel < at.mels; mel++)
            for (uint32_t channel = 0; channel < Z_CHANNELS; channel++) {
                const size_t element =
                    ((size_t)frame * at.mels + mel) * Z_CHANNELS + channel;
                const uint32_t patched = channel * LATENT_MELS + mel;
                staged[element] = staged[element] * latent_std[patched] +
                                  latent_mean[patched];
            }
    {
        float *want = golden("dec_denormalized", volume(at));
        compare("dec_denormalized", staged, want, at, TOLERANCE);
        free(want);
    }
    h3_gpu_tensor *x = upload(staged, volume(at));
    free(staged);

    /* ------------------------------------------------------------ conv_in */

    {
        conv2d kernel;
        load_conv("conv_in", WIDEST_CHANNELS, Z_CHANNELS, KERNEL, &kernel);
        GPU_OP(h3_gpu_begin(gpu), "begin conv_in");
        h3_gpu_tensor *out = run_conv(&kernel, x, at, WIDEST_CHANNELS);
        GPU_OP(h3_gpu_submit(gpu), "submit conv_in");
        free_conv(&kernel);
        h3_gpu_tensor_free(x);
        x = out;
        at.channels = WIDEST_CHANNELS;
    }
    check("dec_conv_in", x, at);

    /* --------------------------------------------------------- the mid block */

    /* Two residuals and nothing between them: `mid_block_add_attention` is
     * false, so `attn_1` is an Identity and the checkpoint has no attention
     * tensors to load. */
    x = resnet("mid.block_1", x, at, at.channels);
    x = resnet("mid.block_2", x, at, at.channels);
    check("dec_mid", x, at);

    /* ------------------------------------------------------- the up levels */

    /* `ch_mult` is [1, 2, 4] against a base of 128, walked in reverse: three
     * residuals at 512, then 256, then 128, with an upsample after every level
     * but the last. */
    static const uint32_t LEVEL_CHANNELS[LEVELS] = {
        BASE_CHANNELS, BASE_CHANNELS * 2, BASE_CHANNELS * 4
    };
    for (int level = LEVELS - 1; level >= 0; level--) {
        char prefix[160], label[64];
        for (int block = 0; block < BLOCKS_PER_LEVEL; block++) {
            snprintf(prefix, sizeof(prefix), "up.%d.block.%d", level, block);
            x = resnet(prefix, x, at, LEVEL_CHANNELS[level]);
            at.channels = LEVEL_CHANNELS[level];
        }
        snprintf(label, sizeof(label), "dec_up%d", level);
        check(label, x, at);
        if (level != 0) {
            x = upsample(level, x, &at);
            snprintf(label, sizeof(label), "dec_up%d_sampled", level);
            check(label, x, at);
        }
    }
    require(at.frames == MEL_FRAMES && at.mels == MEL_BINS,
            "the up path did not reach the mel geometry");

    /* ------------------------------------------------------------ the tail */

    /* Pixel norm's epsilon is the one convention here no mutation catches:
     * setting it to zero changes nothing measurable, because it only bites
     * when a position's channel vector is small enough that its mean square
     * approaches 1e-8. Measure how close anything gets, so this reads as
     * "shown not to apply" rather than "untested" -- and so the day a latent
     * does produce a near-empty position, this line says so. */
    {
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
        printf("  --  %-22s smallest mean square %.3e, %.0fx the %.0e epsilon\n",
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
    check("dec_norm_out", normed, at);
    h3_gpu_tensor_free(x);

    conv2d tail;
    load_conv("conv_out", STEREO, at.channels, KERNEL, &tail);
    GPU_OP(h3_gpu_begin(gpu), "begin head");
    GPU_OP(h3_gpu_silu_f32(gpu, normed, normed, (uint32_t)volume(at)),
           "output activation");
    h3_gpu_tensor *out = run_conv(&tail, normed, at, STEREO);
    GPU_OP(h3_gpu_submit(gpu), "submit head");
    free_conv(&tail);
    h3_gpu_tensor_free(normed);
    at.channels = STEREO;
    check("dec_conv_out", out, at);

    /* `_adjust_output_shape` crops or pads to `4 * latent_frames - 3` frames
     * and the configured mel bins. At this geometry the stack already lands
     * there exactly, so the general path is unexercised -- asserted rather
     * than implemented, so a shape that would need it fails loudly. */
    require(at.frames == MEL_FRAMES && at.mels == MEL_BINS,
            "the decode needs the crop-and-pad path, which is not implemented");
    {
        float *have = download(out, volume(at));
        float *want = golden("restored", volume(at));
        compare("mel", have, want, at, TOLERANCE);
        /* The mel has to carry structure, not just agree: a constant output
         * would match a reference that was broken the same way. */
        double spread = 0.0;
        for (size_t index = 1; index < volume(at); index++)
            if (fabs((double)have[index] - (double)have[0]) > spread)
                spread = fabs((double)have[index] - (double)have[0]);
        require(spread > 1.0, "the decoded mel is nearly constant");
        printf("  --  %-22s spans %.2f in log-mel\n", "the mel has structure",
               spread);
        free(have); free(want);
    }

    printf("\ndecoded %d frames of %d mel bins in %.2f s\n",
           MEL_FRAMES, MEL_BINS, now() - began);
    h3_gpu_tensor_free(out);
    h3_gpu_tensor_free(ones);
    h3_gpu_tensor_free(zeros);
    free(latent_mean); free(latent_std);
    h3_gpu_free(gpu);
    h3_weight_store_free(store);
    h3_weight_store_free(anchor);
    if (failures) {
        fprintf(stderr, "\n%d comparisons failed\n", failures);
        return 1;
    }
    printf("ok: the audio VAE decoder runs end to end -- 11 residual blocks, "
           "two upsamples and the patchified latent boundary\n");
    return 0;
}
