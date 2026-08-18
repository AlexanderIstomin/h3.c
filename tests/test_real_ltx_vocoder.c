/* LTX-2.5's vocoder, on the released weights.
 *
 * A stereo log-mel of [2, 29, 64] out to 4640 samples of 16 kHz stereo -- the
 * piece that turns what the audio VAE decoder produced into something you can
 * hear. 29 mel frames at hop 160 is exactly 0.29 s.
 *
 * Anchored against `ltx_core.model.audio_vae.vocoder`, which ships with the
 * model and matches the checkpoint exactly and unpatched: all 1227 tensors.
 *
 * This is BigVGAN, and `h3_audio_vae.c` already drives one for H3 -- same
 * resblock kernels [3, 7, 11] and dilations [1, 3, 5], same alias-free
 * snakebeta, channels halving per stage. What differs is worth stating,
 * because the shapes alone do not say any of it:
 *
 *   - **the 128 input channels are stereo x 64 mel bins**, not a latent width.
 *     `forward` transposes to `(b, c, mel, t)` and folds stereo into channels,
 *     so `conv_pre` sees 2 x 64 and stereo stays folded the whole way down,
 *     reappearing only as `conv_post`'s two outputs. H3 runs stereo as the
 *     *batch*; this runs a batch of one;
 *
 *   - **the three resblocks in a stage are averaged, not summed.** Each sees
 *     the same input. Summing instead is a factor of three on every stage and
 *     still produces a plausible-looking waveform;
 *
 *   - **AMP1 applies no activation before the upsample.** The `leaky_relu` in
 *     the generic BigVGAN path is guarded by `is_amp`, which is true here;
 *
 *   - **the output is clamped to [-1, 1].** `use_tanh_at_final` is false and
 *     `apply_final_activation` is left at its default, so `forward` ends in a
 *     clamp -- not a tanh, and not nothing. Only the bandwidth extender turns
 *     it off. This sample peaks at 0.23, so the clamp never bites and the
 *     anchor cannot tell it from its absence; it is here from the config, and
 *     `voc_waveform` is checked so that a sample which *does* reach the limit
 *     one day is not silently wrong. See the headroom line at the end; and
 *
 *   - **weight norm is already folded.** No `weight_g`/`weight_v` pair exists
 *     anywhere in this checkpoint, so H3's whole normalization step -- and
 *     `h3_gpu_weight_norm_f32` with it -- does not apply.
 *
 * Every kernel it needs already exists: `h3_gpu_conv_transpose1d_f32` for the
 * upsamples, `h3_gpu_conv1d_f32` for the dilated pairs,
 * `h3_gpu_alias_free_snake_f32` for the 109 activations. That last one is an
 * exact match for LTX's convention and not merely a similar one -- 12 taps,
 * ratio 2, replicate padding on both the source and the doubled signal, and
 * `x + sin^2(x * alpha) / (beta + 1e-9)` between them.
 *
 * usage: h3_real_ltx_vocoder_test VAE.safetensors ANCHOR.safetensors [OUT.f32] */

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
    STEREO = 2,
    MEL_FRAMES = 29, MEL_BINS = 64,
    /* Stereo folded into channels: what `conv_pre` actually sees. */
    INPUT_CHANNELS = STEREO * MEL_BINS,
    INITIAL_CHANNELS = 1536,
    STAGES = 6,
    RESBLOCKS = 3,
    RESIDUAL_PAIRS = 3,
    HEAD_KERNEL = 7,
    FILTER_TAPS = 12,
    SAMPLE_RATE = 16000,
    HOP_LENGTH = 160
};

static const uint32_t upsample_rates[STAGES] = {5, 2, 2, 2, 2, 2};
static const uint32_t upsample_kernels[STAGES] = {11, 4, 4, 4, 4, 4};
static const uint32_t residual_kernels[RESBLOCKS] = {3, 7, 11};
static const uint32_t residual_dilations[RESIDUAL_PAIRS] = {1, 3, 5};

/* Both sides compute in F32, so these bound MPS's accumulation order against
 * torch's rather than a difference in precision.
 *
 * `gen_ltx_vocoder.py --floor` runs the reference at F32 and F64. Unlike the
 * audio VAE's stack, which is flat at 1e-06 the whole way down, this one
 * climbs by a factor of thirty: six upsamples turn 29 frames into 4640
 * samples, and the arithmetic per output grows with them. A single bound here
 * would be either too loose to catch anything early or too tight to pass late,
 * so each stage carries what F32 itself costs at that depth. */
#define TOLERANCE 5e-6

static const struct { const char *label; double floor; } MEASURED_FLOOR[] = {
    {"voc_conv_pre",  1.17e-06}, {"voc_up0",  6.42e-07}, {"voc_res0", 1.60e-06},
    {"voc_up1",       1.83e-06}, {"voc_res1", 3.55e-06},
    {"voc_up2",       2.09e-06}, {"voc_res2", 4.96e-06},
    {"voc_up3",       4.76e-06}, {"voc_res3", 6.54e-06},
    {"voc_up4",       9.04e-06}, {"voc_res4", 3.50e-05},
    {"voc_up5",       3.79e-05}, {"voc_res5", 2.61e-05},
    {"voc_act_post",  2.96e-05},
    {"voc_conv_post", 2.50e-05}, {"voc_waveform", 2.50e-05}
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

/* Channels last: [length][channels], which is what every conv1d kernel in
 * h3_gpu wants. The golden tensors are torch's [channels][length]. */
typedef struct { uint32_t length, channels; } shape;

static size_t volume(shape of) {
    return (size_t)of.length * of.channels;
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

static h3_gpu_tensor *allocate(size_t count) {
    h3_gpu_tensor *tensor = h3_gpu_tensor_new_f32(gpu, count);
    require(tensor != NULL, "cannot allocate a GPU tensor");
    return tensor;
}

static float *download(const h3_gpu_tensor *tensor, size_t count) {
    float *values = malloc(count * sizeof(*values));
    require(values != NULL, "cannot allocate a readback buffer");
    require(h3_gpu_tensor_read_f32(tensor, values, count),
            "cannot read a GPU tensor");
    return values;
}

/* ------------------------------------------------------------------- convs */

typedef struct {
    h3_gpu_tensor *weight, *bias;
    uint32_t in, out, kernel, padding, dilation, stride;
    int transpose;
} conv1d;

/* Every load happens before `h3_gpu_begin`: an upload into an already-open
 * command buffer is not seen by work encoded after it, and the convolution
 * quietly reads its input as zeros. The video VAE learned this the hard way. */
static void load_conv(const char *name, conv1d *into, uint32_t in, uint32_t out,
                      uint32_t kernel, uint32_t padding, uint32_t dilation,
                      uint32_t stride, int transpose, int biased) {
    char full[256];
    /* Conv1d stores [out][in][k]; ConvTranspose1d stores [in][out][k]. Both
     * are the layouts h3_gpu's kernels already expect, so neither is
     * transposed here -- but the element count differs by which is which only
     * in what it asserts, which is the point of asserting it. */
    const size_t taps = (size_t)in * out * kernel;
    snprintf(full, sizeof(full), "vocoder.vocoder.%s.weight", name);
    float *weight = read_tensor(store, full, taps);
    into->weight = upload(weight, taps);
    free(weight);
    if (biased) {
        snprintf(full, sizeof(full), "vocoder.vocoder.%s.bias", name);
        float *bias = read_tensor(store, full, out);
        into->bias = upload(bias, out);
        free(bias);
    } else {
        /* `use_bias_at_final` is false, so `conv_post` has no bias tensor at
         * all. The kernel still wants one; zeros are the same convolution. */
        float *zero = calloc(out, sizeof(*zero));
        require(zero != NULL, "cannot allocate a zero bias");
        into->bias = upload(zero, out);
        free(zero);
    }
    into->in = in; into->out = out; into->kernel = kernel;
    into->padding = padding; into->dilation = dilation; into->stride = stride;
    into->transpose = transpose;
}

static void free_conv(conv1d *which) {
    h3_gpu_tensor_free(which->weight);
    h3_gpu_tensor_free(which->bias);
    memset(which, 0, sizeof(*which));
}

static uint32_t conv_output_length(const conv1d *conv, uint32_t length) {
    if (!conv->transpose) return length;
    /* PyTorch's ConvTranspose1d with output_padding 0. The rate table and the
     * kernel table together make this exactly x160 over the six stages, which
     * is the hop -- so the sample count is the mel frame count times the hop,
     * and the assertion at the end of main is not a coincidence. */
    return (length - 1) * conv->stride + conv->kernel - 2 * conv->padding;
}

static void run_conv(h3_gpu_tensor *out, const h3_gpu_tensor *in,
                     const conv1d *conv, uint32_t length) {
    if (conv->transpose)
        GPU_OP(h3_gpu_conv_transpose1d_f32(gpu, out, in, conv->weight,
                                           conv->bias, 1, length, conv->in,
                                           conv->out, conv->kernel,
                                           conv->stride, conv->padding),
               "transposed convolution");
    else
        GPU_OP(h3_gpu_conv1d_f32(gpu, out, in, conv->weight, conv->bias, 1,
                                 length, conv->in, conv->out, conv->kernel,
                                 conv->padding, conv->dilation),
               "convolution");
}

/* ------------------------------------------------------------- activations */

/* The 2x upsample, snakebeta and 2x low-pass are fused into one kernel at the
 * original rate, so no doubled buffer is ever materialized. */
typedef struct { h3_gpu_tensor *alpha, *beta; } activation;

static h3_gpu_tensor *upsample_filter = NULL;
static h3_gpu_tensor *downsample_filter = NULL;

static void load_activation(const char *name, activation *into,
                            uint32_t channels) {
    char full[256];
    snprintf(full, sizeof(full), "vocoder.vocoder.%s.act.alpha", name);
    float *alpha = read_tensor(store, full, channels);
    into->alpha = upload(alpha, channels);
    free(alpha);
    snprintf(full, sizeof(full), "vocoder.vocoder.%s.act.beta", name);
    float *beta = read_tensor(store, full, channels);
    into->beta = upload(beta, channels);
    free(beta);
}

static void free_activation(activation *which) {
    h3_gpu_tensor_free(which->alpha);
    h3_gpu_tensor_free(which->beta);
    memset(which, 0, sizeof(*which));
}

static void run_activation(h3_gpu_tensor *out, const h3_gpu_tensor *in,
                           const activation *act, shape of) {
    GPU_OP(h3_gpu_alias_free_snake_f32(gpu, out, in, act->alpha, act->beta,
                                       upsample_filter, downsample_filter, 1,
                                       of.length, of.channels),
           "alias-free SnakeBeta");
}

/* One filter pair is loaded and shared by all 109 activations. H3 does the
 * same, but H3's checkpoint is not this one -- so check rather than inherit
 * the assumption. Every `kaiser_sinc_filter1d` here is built from the same
 * cutoff and width, which is *why* they agree; if a checkpoint ever ships one
 * that does not, this says so instead of quietly using the wrong taps. */
static void verify_filters(const char *name, const float *up, const float *down) {
    char full[256];
    snprintf(full, sizeof(full), "vocoder.vocoder.%s.upsample.filter", name);
    float *mine = read_tensor(store, full, FILTER_TAPS);
    for (int tap = 0; tap < FILTER_TAPS; tap++)
        if (mine[tap] != up[tap])
            fail("%s.upsample.filter differs from the shared one at tap %d: "
                 "%.9g vs %.9g", name, tap, mine[tap], up[tap]);
    free(mine);
    snprintf(full, sizeof(full), "vocoder.vocoder.%s.downsample.lowpass.filter",
             name);
    mine = read_tensor(store, full, FILTER_TAPS);
    for (int tap = 0; tap < FILTER_TAPS; tap++)
        if (mine[tap] != down[tap])
            fail("%s.downsample.lowpass.filter differs from the shared one at "
                 "tap %d: %.9g vs %.9g", name, tap, mine[tap], down[tap]);
    free(mine);
}

static int load_filters(void) {
    float *up = read_tensor(store, "vocoder.vocoder.act_post.upsample.filter",
                            FILTER_TAPS);
    float *down = read_tensor(
        store, "vocoder.vocoder.act_post.downsample.lowpass.filter",
        FILTER_TAPS);
    int checked = 1;                                    /* act_post itself */
    char name[160];
    for (int global = 0; global < STAGES * RESBLOCKS; global++)
        for (int pair = 0; pair < RESIDUAL_PAIRS; pair++)
            for (int which = 1; which <= 2; which++) {
                snprintf(name, sizeof(name), "resblocks.%d.acts%d.%d", global,
                         which, pair);
                verify_filters(name, up, down);
                checked++;
            }
    upsample_filter = upload(up, FILTER_TAPS);
    downsample_filter = upload(down, FILTER_TAPS);
    free(up); free(down);
    return checked;
}

/* ---------------------------------------------------------------- compare */

static void compare(const char *label, const float *engine, const float *expect,
                    shape of, double tolerance) {
    double worst = 0.0, peak = 0.0;
    size_t worst_at = 0;
    for (uint32_t time = 0; time < of.length; time++)
        for (uint32_t channel = 0; channel < of.channels; channel++) {
            const double actual = engine[(size_t)time * of.channels + channel];
            const double wanted = expect[(size_t)channel * of.length + time];
            if (!isfinite(actual)) {
                fprintf(stderr, "FAIL %-18s produced %f\n", label, actual);
                failures++;
                return;
            }
            const double delta = fabs(actual - wanted);
            if (delta > worst) { worst = delta; worst_at = time; }
            if (fabs(wanted) > peak) peak = fabs(wanted);
        }
    const double relative = peak > 0.0 ? worst / peak : worst;
    if (relative > tolerance) {
        fprintf(stderr, "FAIL %-18s %.3e = %.2e of peak %.4g (bound %.1e), "
                "at sample %zu\n", label, worst, relative, peak, tolerance,
                worst_at);
        failures++;
    } else {
        printf("  ok  %-18s [%5u,%6u] %.3e = %.2e of peak %.4g\n", label,
               of.channels, of.length, worst, relative, peak);
    }
}

static void check(const char *label, const h3_gpu_tensor *tensor, shape of) {
    float *have = download(tensor, volume(of));
    float *want = golden(label, volume(of));
    compare(label, have, want, of, tolerance_for(label));
    free(have); free(want);
}

/* ------------------------------------------------------------- the stages */

typedef struct {
    activation acts1[RESIDUAL_PAIRS], acts2[RESIDUAL_PAIRS];
    conv1d convs1[RESIDUAL_PAIRS], convs2[RESIDUAL_PAIRS];
} resblock;

typedef struct { conv1d up; resblock blocks[RESBLOCKS]; } stage;

/* `get_padding(k, d) = d * (k - 1) / 2` -- same padding at whatever dilation,
 * so the length never moves inside a resblock. convs2 is always dilation 1;
 * only convs1 walks [1, 3, 5]. */
static void load_stage(stage *into, int index) {
    const uint32_t in = INITIAL_CHANNELS >> index;
    const uint32_t out = INITIAL_CHANNELS >> (index + 1);
    const uint32_t rate = upsample_rates[index];
    const uint32_t kernel = upsample_kernels[index];
    char name[160];
    snprintf(name, sizeof(name), "ups.%d", index);
    load_conv(name, &into->up, in, out, kernel, (kernel - rate) / 2, 1, rate,
              1, 1);

    for (int block = 0; block < RESBLOCKS; block++) {
        const int global = index * RESBLOCKS + block;
        const uint32_t residual = residual_kernels[block];
        for (int pair = 0; pair < RESIDUAL_PAIRS; pair++) {
            const uint32_t dilation = residual_dilations[pair];
            snprintf(name, sizeof(name), "resblocks.%d.acts1.%d", global, pair);
            load_activation(name, &into->blocks[block].acts1[pair], out);
            snprintf(name, sizeof(name), "resblocks.%d.acts2.%d", global, pair);
            load_activation(name, &into->blocks[block].acts2[pair], out);
            snprintf(name, sizeof(name), "resblocks.%d.convs1.%d", global, pair);
            load_conv(name, &into->blocks[block].convs1[pair], out, out,
                      residual, dilation * (residual - 1) / 2, dilation, 1,
                      0, 1);
            snprintf(name, sizeof(name), "resblocks.%d.convs2.%d", global, pair);
            load_conv(name, &into->blocks[block].convs2[pair], out, out,
                      residual, (residual - 1) / 2, 1, 1, 0, 1);
        }
    }
}

static void free_stage(stage *which) {
    free_conv(&which->up);
    for (int block = 0; block < RESBLOCKS; block++)
        for (int pair = 0; pair < RESIDUAL_PAIRS; pair++) {
            free_activation(&which->blocks[block].acts1[pair]);
            free_activation(&which->blocks[block].acts2[pair]);
            free_conv(&which->blocks[block].convs1[pair]);
            free_conv(&which->blocks[block].convs2[pair]);
        }
    memset(which, 0, sizeof(*which));
}

/* activate, convolve, activate, convolve, add the input back -- three times,
 * each pair at its own dilation. */
static void run_resblock(h3_gpu_tensor *out, const h3_gpu_tensor *in,
                         const resblock *block, shape of,
                         h3_gpu_tensor *activated, h3_gpu_tensor *branch) {
    GPU_OP(h3_gpu_copy_f32(gpu, out, 0, in, 0, volume(of)),
           "seed the residual");
    for (int pair = 0; pair < RESIDUAL_PAIRS; pair++) {
        run_activation(activated, out, &block->acts1[pair], of);
        run_conv(branch, activated, &block->convs1[pair], of.length);
        run_activation(activated, branch, &block->acts2[pair], of);
        run_conv(branch, activated, &block->convs2[pair], of.length);
        GPU_OP(h3_gpu_add_scaled_f32(gpu, out, out, branch, 1.0f, 1.0f,
                                     (uint32_t)volume(of)), "residual add");
    }
}

/* ------------------------------------------------------------------- main */

int main(int argc, char **argv) {
    if (argc < 3 || argc > 4) {
        fprintf(stderr,
                "usage: %s VAE.safetensors ANCHOR.safetensors [OUT.f32]\n",
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
    const double began = now();

    const int filters = load_filters();
    printf("  --  %-18s one pair shared by %d activations, all identical\n",
           "resampling filters", filters);

    /* ---------------------------------------------------------- the input */

    /* `(batch, stereo, frames, mels)` becomes `(batch, stereo * mels, frames)`:
     * a transpose and a fold, and the reason `conv_pre` takes 128 channels
     * rather than 64. Reading those 128 as anything else -- a latent width, or
     * mel-major -- fits and runs. */
    shape at = {MEL_FRAMES, INPUT_CHANNELS};
    float *mel = golden("mel", (size_t)STEREO * MEL_FRAMES * MEL_BINS);
    float *staged = malloc(volume(at) * sizeof(*staged));
    require(staged != NULL, "cannot allocate the folded mel");
    for (uint32_t side = 0; side < STEREO; side++)
        for (uint32_t frame = 0; frame < MEL_FRAMES; frame++)
            for (uint32_t bin = 0; bin < MEL_BINS; bin++)
                staged[(size_t)frame * INPUT_CHANNELS + side * MEL_BINS + bin] =
                    mel[((size_t)side * MEL_FRAMES + frame) * MEL_BINS + bin];
    free(mel);
    {
        float *want = golden("voc_input", volume(at));
        compare("voc_input", staged, want, at, TOLERANCE);
        free(want);
    }
    h3_gpu_tensor *x = upload(staged, volume(at));
    free(staged);

    /* --------------------------------------------------------- conv_pre */

    {
        conv1d pre;
        load_conv("conv_pre", &pre, INPUT_CHANNELS, INITIAL_CHANNELS,
                  HEAD_KERNEL, HEAD_KERNEL / 2, 1, 1, 0, 1);
        shape wide = {at.length, INITIAL_CHANNELS};
        h3_gpu_tensor *out = allocate(volume(wide));
        GPU_OP(h3_gpu_begin(gpu), "begin conv_pre");
        run_conv(out, x, &pre, at.length);
        GPU_OP(h3_gpu_submit(gpu), "submit conv_pre");
        free_conv(&pre);
        h3_gpu_tensor_free(x);
        x = out;
        at = wide;
    }
    check("voc_conv_pre", x, at);

    /* ------------------------------------------------------- the six stages */

    for (int index = 0; index < STAGES; index++) {
        stage weights;
        memset(&weights, 0, sizeof(weights));
        load_stage(&weights, index);

        shape wider = {conv_output_length(&weights.up, at.length),
                       INITIAL_CHANNELS >> (index + 1)};
        h3_gpu_tensor *upsampled = allocate(volume(wider));
        h3_gpu_tensor *accumulated = allocate(volume(wider));
        h3_gpu_tensor *work = allocate(volume(wider));
        h3_gpu_tensor *activated = allocate(volume(wider));
        h3_gpu_tensor *branch = allocate(volume(wider));

        GPU_OP(h3_gpu_begin(gpu), "begin upsample");
        run_conv(upsampled, x, &weights.up, at.length);
        GPU_OP(h3_gpu_submit(gpu), "submit upsample");
        h3_gpu_tensor_free(x);
        at = wider;
        char label[32];
        snprintf(label, sizeof(label), "voc_up%d", index);
        check(label, upsampled, at);

        /* Every block reads the same input and the stage takes their *mean*.
         * Summing is a factor of three and still sounds like something. */
        GPU_OP(h3_gpu_begin(gpu), "begin resblocks");
        for (int block = 0; block < RESBLOCKS; block++) {
            h3_gpu_tensor *target = block == 0 ? accumulated : work;
            run_resblock(target, upsampled, &weights.blocks[block], at,
                         activated, branch);
            if (block == 0) continue;
            const float scale =
                block == RESBLOCKS - 1 ? 1.0f / (float)RESBLOCKS : 1.0f;
            GPU_OP(h3_gpu_add_scaled_f32(gpu, accumulated, accumulated, work,
                                         scale, scale, (uint32_t)volume(at)),
                   "resblock mean");
        }
        GPU_OP(h3_gpu_submit(gpu), "submit resblocks");
        snprintf(label, sizeof(label), "voc_res%d", index);
        check(label, accumulated, at);

        x = accumulated;
        h3_gpu_tensor_free(upsampled);
        h3_gpu_tensor_free(work);
        h3_gpu_tensor_free(activated);
        h3_gpu_tensor_free(branch);
        free_stage(&weights);
    }

    /* -------------------------------------------------------------- the tail */

    {
        activation post;
        load_activation("act_post", &post, at.channels);
        h3_gpu_tensor *out = allocate(volume(at));
        GPU_OP(h3_gpu_begin(gpu), "begin act_post");
        run_activation(out, x, &post, at);
        GPU_OP(h3_gpu_submit(gpu), "submit act_post");
        free_activation(&post);
        h3_gpu_tensor_free(x);
        x = out;
    }
    check("voc_act_post", x, at);

    {
        conv1d post;
        load_conv("conv_post", &post, at.channels, STEREO, HEAD_KERNEL,
                  HEAD_KERNEL / 2, 1, 1, 0, 0);
        shape narrow = {at.length, STEREO};
        h3_gpu_tensor *out = allocate(volume(narrow));
        GPU_OP(h3_gpu_begin(gpu), "begin conv_post");
        run_conv(out, x, &post, at.length);
        GPU_OP(h3_gpu_submit(gpu), "submit conv_post");
        free_conv(&post);
        h3_gpu_tensor_free(x);
        x = out;
        at = narrow;
    }
    check("voc_conv_post", x, at);

    require(at.length == MEL_FRAMES * HOP_LENGTH,
            "the six stages did not multiply the frame count by the hop");

    /* The clamp. `use_tanh_at_final` is false and `apply_final_activation` is
     * default-true, so this is what `forward` ends in. */
    float *samples = download(x, volume(at));
    double peak = 0.0;
    size_t clipped = 0;
    for (size_t index = 0; index < volume(at); index++) {
        if (fabs((double)samples[index]) > peak) peak = fabs(samples[index]);
        if (samples[index] > 1.0f) { samples[index] = 1.0f; clipped++; }
        if (samples[index] < -1.0f) { samples[index] = -1.0f; clipped++; }
    }
    {
        float *want = golden("voc_waveform", volume(at));
        compare("voc_waveform", samples, want, at, tolerance_for("voc_waveform"));
        free(want);
    }
    /* Say plainly what this run does and does not establish. The clamp is read
     * off the config, and on a sample that never reaches +-1 the anchor agrees
     * with a port that omits it entirely. */
    printf("  --  %-18s peak %.4f, %.1fx inside the clamp -- %zu samples "
           "clipped, so the clamp is %s here\n", "headroom", peak,
           peak > 0.0 ? 1.0 / peak : INFINITY, clipped,
           clipped ? "exercised" : "inert and asserted from the config");

    /* A waveform has to carry structure, not merely agree: a silent buffer
     * would match a reference that was broken the same way. */
    double energy = 0.0;
    for (size_t index = 0; index < volume(at); index++)
        energy += (double)samples[index] * samples[index];
    const double rms = sqrt(energy / (double)volume(at));
    require(rms > 1e-3, "the waveform is essentially silent");
    printf("  --  %-18s rms %.4f, peak %.4f over %.3f s\n", "it makes a sound",
           rms, peak, (double)at.length / (double)SAMPLE_RATE);

    if (argc == 4) {
        FILE *handle = fopen(argv[3], "wb");
        if (!handle) fail("cannot write %s", argv[3]);
        /* Interleaved stereo F32, which is what the buffer already is. */
        require(fwrite(samples, sizeof(*samples), volume(at), handle) ==
                volume(at), "cannot write the waveform");
        fclose(handle);
        printf("  --  %-18s %s, %u interleaved stereo F32 frames at %d Hz\n",
               "wrote", argv[3], at.length, SAMPLE_RATE);
    }

    printf("\nspoke %u samples of %d Hz stereo from %d mel frames in %.2f s\n",
           at.length, SAMPLE_RATE, MEL_FRAMES, now() - began);
    free(samples);
    h3_gpu_tensor_free(x);
    h3_gpu_tensor_free(upsample_filter);
    h3_gpu_tensor_free(downsample_filter);
    h3_gpu_free(gpu);
    h3_weight_store_free(store);
    h3_weight_store_free(anchor);
    if (failures) {
        fprintf(stderr, "\n%d comparisons failed\n", failures);
        return 1;
    }
    printf("ok: the vocoder runs end to end -- six upsamples, 18 resblocks "
           "and %d alias-free activations\n", filters);
    return 0;
}
