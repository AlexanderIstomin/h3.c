/* Turn an LTX-2.5 audio latent into a sound.
 *
 * The seam. `h3_ltx_generate` denoises both streams and writes both latents;
 * until now nothing has ever read the audio one. Three pieces that each work
 * on their own -- the patchified latent boundary, the audio VAE decoder and
 * the vocoder -- with nothing joining them, which is exactly where this
 * project's failures live: every component of the video path passed its own
 * anchor while the rope was fed the wrong coordinates, and only a generated
 * image showed it.
 *
 * So this writes a WAV. Listen to it.
 *
 *   latent [R, 128] -> unpatchify [8, R, 16] -> denormalize
 *                   -> audio VAE decoder -> log-mel [2, 4R-3, 64]
 *                   -> vocoder -> 16 kHz stereo, (4R-3) * 160 samples
 *
 * The driver half of `test_real_ltx_audio_vae.c` and
 * `test_real_ltx_vocoder.c` together, at whatever length the latent carries,
 * with the stage comparisons removed and the per-channel statistics read from
 * the checkpoint rather than a fixture. Every convention those two files
 * record still applies; the ones that bite hardest here are repeated where
 * they are used.
 *
 * usage: h3_ltx_audio_decode AUDIO_VAE.safetensors LATENT.bin OUT.wav */

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
    /* ---- the VAE half ---- */
    Z_CHANNELS = 8, LATENT_MELS = 16, MEL_BINS = 64,
    /* The DiT's audio token width: z-channels times latent mel bins. */
    PATCHED = Z_CHANNELS * LATENT_MELS,
    BASE_CHANNELS = 128, WIDEST_CHANNELS = 512,
    LEVELS = 3, BLOCKS_PER_LEVEL = 3, KERNEL = 3,
    /* ---- the vocoder half ---- */
    VOICE_CHANNELS = STEREO * MEL_BINS,
    INITIAL_CHANNELS = 1536,
    STAGES = 6, RESBLOCKS = 3, RESIDUAL_PAIRS = 3,
    HEAD_KERNEL = 7, FILTER_TAPS = 12,
    SAMPLE_RATE = 16000, HOP_LENGTH = 160
};

#define PIXEL_NORM_EPSILON 1e-8f

static const uint32_t upsample_rates[STAGES] = {5, 2, 2, 2, 2, 2};
static const uint32_t upsample_kernels[STAGES] = {11, 4, 4, 4, 4, 4};
static const uint32_t residual_kernels[RESBLOCKS] = {3, 7, 11};
static const uint32_t residual_dilations[RESIDUAL_PAIRS] = {1, 3, 5};

static h3_weight_store *store = NULL;
static h3_gpu *gpu = NULL;
static h3_gpu_tensor *ones = NULL;
static h3_gpu_tensor *zeros = NULL;
static h3_gpu_tensor *upsample_filter = NULL;
static h3_gpu_tensor *downsample_filter = NULL;

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

/* The VAE works in (frames, mels, channels); the vocoder in (length,
 * channels). Both are channels-last, which is what every kernel here wants. */
typedef struct { uint32_t frames, mels, channels; } shape;
typedef struct { uint32_t length, channels; } span;

static size_t volume(shape of) {
    return (size_t)of.frames * of.mels * of.channels;
}

static size_t positions(shape of) {
    return (size_t)of.frames * of.mels;
}

static size_t extent(span of) {
    return (size_t)of.length * of.channels;
}

/* ----------------------------------------------------------------- loading */

static float from_bf16(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static float *read_tensor(const char *name, size_t expected) {
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(store, name, &header);
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

/* ------------------------------------------------------- the VAE decoder */

typedef struct { h3_gpu_tensor *weight, *bias; uint32_t kernel; } conv2d;

/* Every load happens before `h3_gpu_begin`: an upload into an already-open
 * command buffer is not seen by work encoded after it, and the convolution
 * quietly reads its input as zeros. */
static void load_vae_conv(const char *name, uint32_t out_channels,
                          uint32_t in_channels, uint32_t kernel, conv2d *into) {
    char full[256];
    snprintf(full, sizeof(full), "audio_vae.decoder.%s.conv.weight", name);
    const size_t taps = (size_t)kernel * kernel;
    float *weight = read_tensor(full, (size_t)out_channels * in_channels * taps);
    into->weight = upload(weight, (size_t)out_channels * in_channels * taps);
    free(weight);
    snprintf(full, sizeof(full), "audio_vae.decoder.%s.conv.bias", name);
    float *bias = read_tensor(full, out_channels);
    into->bias = upload(bias, out_channels);
    free(bias);
    into->kernel = kernel;
}

static void free_vae_conv(conv2d *which) {
    h3_gpu_tensor_free(which->weight);
    h3_gpu_tensor_free(which->bias);
    memset(which, 0, sizeof(*which));
}

/* Two zero frames in front and none behind. The fill is *zeros*, not
 * replicated edge frames -- the video VAE replicates and this one does not. */
static h3_gpu_tensor *pad_frames(const h3_gpu_tensor *input, shape of,
                                 uint32_t kernel) {
    const uint32_t front = kernel - 1;
    const size_t frame = (size_t)of.mels * of.channels;
    h3_gpu_tensor *out = allocate((size_t)(of.frames + front) * frame);
    if (front)
        GPU_OP(h3_gpu_copy_f32(gpu, out, 0, zeros, 0, (size_t)front * frame),
               "zero the causal pad");
    GPU_OP(h3_gpu_copy_f32(gpu, out, (size_t)front * frame, input, 0,
                           (size_t)of.frames * frame), "pad interior");
    return out;
}

/* Frames onto depth, mel onto width, a degenerate 1 onto height: depth is left
 * unpadded for the explicit causal pad while mel is same-padded by the kernel. */
static h3_gpu_tensor *run_vae_conv(const conv2d *kernel,
                                   const h3_gpu_tensor *input, shape of,
                                   uint32_t out_channels) {
    h3_gpu_tensor *padded = pad_frames(input, of, kernel->kernel);
    shape result = {of.frames, of.mels, out_channels};
    h3_gpu_tensor *out = allocate(volume(result));
    GPU_OP(h3_gpu_conv3d_same_f32(gpu, out, padded, kernel->weight,
                                  kernel->bias, 1,
                                  of.frames + kernel->kernel - 1, 1, of.mels,
                                  of.channels, out_channels,
                                  kernel->kernel, 1, kernel->kernel, 1, 1, 1),
           "convolution");
    h3_gpu_tensor_free(padded);
    return out;
}

static void pixel_norm(h3_gpu_tensor *out, const h3_gpu_tensor *input,
                       shape of) {
    GPU_OP(h3_gpu_rms_norm_f32(gpu, out, input, ones,
                               (uint32_t)positions(of), of.channels,
                               PIXEL_NORM_EPSILON), "pixel norm");
}

static h3_gpu_tensor *resnet(const char *prefix, h3_gpu_tensor *x, shape of,
                             uint32_t out_channels) {
    char name[160];
    conv2d first, second, shortcut;
    memset(&shortcut, 0, sizeof(shortcut));
    snprintf(name, sizeof(name), "%s.conv1", prefix);
    load_vae_conv(name, out_channels, of.channels, KERNEL, &first);
    snprintf(name, sizeof(name), "%s.conv2", prefix);
    load_vae_conv(name, out_channels, out_channels, KERNEL, &second);
    if (of.channels != out_channels) {
        snprintf(name, sizeof(name), "%s.nin_shortcut", prefix);
        load_vae_conv(name, out_channels, of.channels, 1, &shortcut);
    }
    shape wide = {of.frames, of.mels, out_channels};
    h3_gpu_tensor *scratch = allocate(volume(of));
    h3_gpu_tensor *inner = allocate(volume(wide));

    GPU_OP(h3_gpu_begin(gpu), "begin resnet");
    pixel_norm(scratch, x, of);
    GPU_OP(h3_gpu_silu_f32(gpu, scratch, scratch, (uint32_t)volume(of)),
           "resnet activation");
    h3_gpu_tensor *hidden = run_vae_conv(&first, scratch, of, out_channels);
    pixel_norm(inner, hidden, wide);
    GPU_OP(h3_gpu_silu_f32(gpu, inner, inner, (uint32_t)volume(wide)),
           "resnet activation");
    h3_gpu_tensor *branch = run_vae_conv(&second, inner, wide, out_channels);
    /* The residual takes the block's *input*, projected when the width
     * changes -- not the normalized copy above. */
    h3_gpu_tensor *residual = x;
    if (of.channels != out_channels)
        residual = run_vae_conv(&shortcut, x, of, out_channels);
    GPU_OP(h3_gpu_add_scaled_f32(gpu, branch, residual, branch, 1.0f, 1.0f,
                                 (uint32_t)volume(wide)), "resnet residual");
    GPU_OP(h3_gpu_submit(gpu), "submit resnet");

    if (residual != x) h3_gpu_tensor_free(residual);
    h3_gpu_tensor_free(hidden);
    h3_gpu_tensor_free(inner);
    h3_gpu_tensor_free(scratch);
    h3_gpu_tensor_free(x);
    free_vae_conv(&first); free_vae_conv(&second);
    if (shortcut.weight) free_vae_conv(&shortcut);
    return branch;
}

/* Nearest-2x on both axes, a causal convolution, then discard the *first*
 * frame -- which is what keeps the length at 1 + 2n. */
static h3_gpu_tensor *upsample(int level, h3_gpu_tensor *x, shape *of) {
    char name[160];
    conv2d kernel;
    snprintf(name, sizeof(name), "up.%d.upsample.conv", level);
    load_vae_conv(name, of->channels, of->channels, KERNEL, &kernel);

    shape doubled = {of->frames * 2, of->mels * 2, of->channels};
    h3_gpu_tensor *big = allocate(volume(doubled));
    GPU_OP(h3_gpu_begin(gpu), "begin upsample");
    GPU_OP(h3_gpu_nearest2x_nhwc_f32(gpu, big, x, of->frames, of->mels,
                                     of->channels), "nearest upsample");
    h3_gpu_tensor *convolved = run_vae_conv(&kernel, big, doubled,
                                            of->channels);
    GPU_OP(h3_gpu_submit(gpu), "submit upsample");
    free_vae_conv(&kernel);
    h3_gpu_tensor_free(big);
    h3_gpu_tensor_free(x);

    shape kept = {doubled.frames - 1, doubled.mels, doubled.channels};
    h3_gpu_tensor *out = allocate(volume(kept));
    GPU_OP(h3_gpu_begin(gpu), "begin trim");
    GPU_OP(h3_gpu_copy_f32(gpu, out, 0, convolved,
                           (size_t)doubled.mels * doubled.channels,
                           volume(kept)), "drop the first frame");
    GPU_OP(h3_gpu_submit(gpu), "submit trim");
    h3_gpu_tensor_free(convolved);
    *of = kept;
    return out;
}

/* ---------------------------------------------------------- the vocoder */

typedef struct {
    h3_gpu_tensor *weight, *bias;
    uint32_t in, out, kernel, padding, dilation, stride;
    int transpose;
} conv1d;

typedef struct { h3_gpu_tensor *alpha, *beta; } activation;

typedef struct {
    activation acts1[RESIDUAL_PAIRS], acts2[RESIDUAL_PAIRS];
    conv1d convs1[RESIDUAL_PAIRS], convs2[RESIDUAL_PAIRS];
} resblock;

typedef struct { conv1d up; resblock blocks[RESBLOCKS]; } stage;

static void load_voc_conv(const char *name, conv1d *into, uint32_t in,
                          uint32_t out, uint32_t kernel, uint32_t padding,
                          uint32_t dilation, uint32_t stride, int transpose,
                          int biased) {
    char full[256];
    const size_t taps = (size_t)in * out * kernel;
    snprintf(full, sizeof(full), "vocoder.vocoder.%s.weight", name);
    float *weight = read_tensor(full, taps);
    into->weight = upload(weight, taps);
    free(weight);
    if (biased) {
        snprintf(full, sizeof(full), "vocoder.vocoder.%s.bias", name);
        float *bias = read_tensor(full, out);
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

static void free_voc_conv(conv1d *which) {
    h3_gpu_tensor_free(which->weight);
    h3_gpu_tensor_free(which->bias);
    memset(which, 0, sizeof(*which));
}

static void load_activation(const char *name, activation *into,
                            uint32_t channels) {
    char full[256];
    snprintf(full, sizeof(full), "vocoder.vocoder.%s.act.alpha", name);
    float *alpha = read_tensor(full, channels);
    into->alpha = upload(alpha, channels);
    free(alpha);
    snprintf(full, sizeof(full), "vocoder.vocoder.%s.act.beta", name);
    float *beta = read_tensor(full, channels);
    into->beta = upload(beta, channels);
    free(beta);
}

static void free_activation(activation *which) {
    h3_gpu_tensor_free(which->alpha);
    h3_gpu_tensor_free(which->beta);
    memset(which, 0, sizeof(*which));
}

static void run_activation(h3_gpu_tensor *out, const h3_gpu_tensor *in,
                           const activation *act, span of) {
    GPU_OP(h3_gpu_alias_free_snake_f32(gpu, out, in, act->alpha, act->beta,
                                       upsample_filter, downsample_filter, 1,
                                       of.length, of.channels),
           "alias-free SnakeBeta");
}

static void run_voc_conv(h3_gpu_tensor *out, const h3_gpu_tensor *in,
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

/* All 109 activations share one resampling filter pair -- checked in
 * `test_real_ltx_vocoder.c`, assumed here. */
static void load_filters(void) {
    float *up = read_tensor("vocoder.vocoder.act_post.upsample.filter",
                            FILTER_TAPS);
    float *down = read_tensor(
        "vocoder.vocoder.act_post.downsample.lowpass.filter", FILTER_TAPS);
    upsample_filter = upload(up, FILTER_TAPS);
    downsample_filter = upload(down, FILTER_TAPS);
    free(up); free(down);
}

/* `get_padding(k, d) = d * (k - 1) / 2`; convs2 is always dilation 1. */
static void load_stage(stage *into, int index) {
    const uint32_t in = INITIAL_CHANNELS >> index;
    const uint32_t out = INITIAL_CHANNELS >> (index + 1);
    const uint32_t rate = upsample_rates[index];
    const uint32_t kernel = upsample_kernels[index];
    char name[160];
    snprintf(name, sizeof(name), "ups.%d", index);
    load_voc_conv(name, &into->up, in, out, kernel, (kernel - rate) / 2, 1,
                  rate, 1, 1);
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
            load_voc_conv(name, &into->blocks[block].convs1[pair], out, out,
                          residual, dilation * (residual - 1) / 2, dilation, 1,
                          0, 1);
            snprintf(name, sizeof(name), "resblocks.%d.convs2.%d", global, pair);
            load_voc_conv(name, &into->blocks[block].convs2[pair], out, out,
                          residual, (residual - 1) / 2, 1, 1, 0, 1);
        }
    }
}

static void free_stage(stage *which) {
    free_voc_conv(&which->up);
    for (int block = 0; block < RESBLOCKS; block++)
        for (int pair = 0; pair < RESIDUAL_PAIRS; pair++) {
            free_activation(&which->blocks[block].acts1[pair]);
            free_activation(&which->blocks[block].acts2[pair]);
            free_voc_conv(&which->blocks[block].convs1[pair]);
            free_voc_conv(&which->blocks[block].convs2[pair]);
        }
    memset(which, 0, sizeof(*which));
}

static void run_resblock(h3_gpu_tensor *out, const h3_gpu_tensor *in,
                         const resblock *block, span of,
                         h3_gpu_tensor *activated, h3_gpu_tensor *branch) {
    GPU_OP(h3_gpu_copy_f32(gpu, out, 0, in, 0, extent(of)),
           "seed the residual");
    for (int pair = 0; pair < RESIDUAL_PAIRS; pair++) {
        run_activation(activated, out, &block->acts1[pair], of);
        run_voc_conv(branch, activated, &block->convs1[pair], of.length);
        run_activation(activated, branch, &block->acts2[pair], of);
        run_voc_conv(branch, activated, &block->convs2[pair], of.length);
        GPU_OP(h3_gpu_add_scaled_f32(gpu, out, out, branch, 1.0f, 1.0f,
                                     (uint32_t)extent(of)), "residual add");
    }
}

/* The three blocks read the same input and the stage takes their *mean*. */
static void encode_blocks(h3_gpu_tensor *out, const h3_gpu_tensor *in,
                          const stage *weights, span of, h3_gpu_tensor *work,
                          h3_gpu_tensor *activated, h3_gpu_tensor *branch) {
    for (int block = 0; block < RESBLOCKS; block++) {
        h3_gpu_tensor *target = block == 0 ? out : work;
        run_resblock(target, in, &weights->blocks[block], of, activated,
                     branch);
        if (block == 0) continue;
        const float scale =
            block == RESBLOCKS - 1 ? 1.0f / (float)RESBLOCKS : 1.0f;
        GPU_OP(h3_gpu_add_scaled_f32(gpu, out, out, work, scale, scale,
                                     (uint32_t)extent(of)), "resblock mean");
    }
}

/* --------------------------------------------------------------- the WAV */

static void put32(FILE *out, uint32_t value) {
    uint8_t bytes[4] = {(uint8_t)value, (uint8_t)(value >> 8),
                        (uint8_t)(value >> 16), (uint8_t)(value >> 24)};
    require(fwrite(bytes, 1, 4, out) == 4, "cannot write the WAV header");
}

static void put16(FILE *out, uint16_t value) {
    uint8_t bytes[2] = {(uint8_t)value, (uint8_t)(value >> 8)};
    require(fwrite(bytes, 1, 2, out) == 2, "cannot write the WAV header");
}

static void write_wav(const char *path, const float *samples, uint32_t frames) {
    FILE *out = fopen(path, "wb");
    if (!out) fail("cannot open %s for writing", path);
    const uint32_t data_bytes = frames * STEREO * 2;
    require(fwrite("RIFF", 1, 4, out) == 4, "cannot write the WAV header");
    put32(out, 36 + data_bytes);
    require(fwrite("WAVEfmt ", 1, 8, out) == 8, "cannot write the WAV header");
    put32(out, 16);                       /* PCM chunk size */
    put16(out, 1);                        /* PCM */
    put16(out, STEREO);
    put32(out, SAMPLE_RATE);
    put32(out, SAMPLE_RATE * STEREO * 2); /* byte rate */
    put16(out, STEREO * 2);               /* block align */
    put16(out, 16);                       /* bits */
    require(fwrite("data", 1, 4, out) == 4, "cannot write the WAV header");
    put32(out, data_bytes);
    for (size_t index = 0; index < (size_t)frames * STEREO; index++) {
        double value = samples[index];
        if (value > 1.0) value = 1.0;
        if (value < -1.0) value = -1.0;
        put16(out, (uint16_t)(int16_t)lrint(value * 32767.0));
    }
    require(fclose(out) == 0, "cannot close the WAV");
}

/* ------------------------------------------------------------------- main */

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr,
                "usage: %s AUDIO_VAE.safetensors LATENT.bin OUT.wav\n", argv[0]);
        return 2;
    }
    char error[512];
    store = h3_weight_store_open(argv[1], error, sizeof(error));
    if (!store) fail("cannot open the audio VAE: %s", error);
    gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) fail("cannot create the Metal context: %s", error);
    const double began = now();

    /* ------------------------------------------------- read the latent */

    uint32_t rows = 0;
    float *tokens = NULL;
    {
        FILE *file = fopen(argv[2], "rb");
        if (!file) fail("cannot open %s", argv[2]);
        uint32_t header[8];
        require(fread(header, sizeof(header), 1, file) == 1,
                "cannot read the latent header");
        if (header[0] != UINT32_C(0x4C545847)) fail("%s is not a latent", argv[2]);
        const uint32_t frames = header[1], height = header[2], width = header[3];
        const uint32_t latent = header[4];
        rows = header[5];
        if (latent != PATCHED)
            fail("the latent is %u wide, expected %d", latent, PATCHED);
        require(rows > 0, "the latent carries no audio rows");
        /* The video latent comes first and is not wanted here. */
        const size_t skip = (size_t)frames * height * width * latent;
        require(fseek(file, (long)(skip * sizeof(float)), SEEK_CUR) == 0,
                "cannot skip the video latent");
        tokens = malloc((size_t)rows * PATCHED * sizeof(*tokens));
        require(tokens != NULL, "cannot allocate the audio latent");
        require(fread(tokens, sizeof(*tokens), (size_t)rows * PATCHED, file) ==
                (size_t)rows * PATCHED, "cannot read the audio latent");
        fclose(file);
        printf("audio latent [%u, %d] from %s (video was %u x %u x %u)\n",
               rows, PATCHED, argv[2], frames, height, width);
    }

    /* ------------------------------------- unpatchify and denormalize */

    /* The statistics are 128 wide while the latent is 8 channels, because the
     * normalization lives in the *patchified* space the DiT works in:
     * `[8, T, 16]` reads as `[T, 128]` with channel index `c * 16 + f`. A port
     * that takes the shapes at face value and scales 8 channels by a
     * 128-vector does not even fit. */
    float *latent_mean = read_tensor(
        "audio_vae.per_channel_statistics.mean-of-means", PATCHED);
    float *latent_std = read_tensor(
        "audio_vae.per_channel_statistics.std-of-means", PATCHED);

    shape at = {rows, LATENT_MELS, Z_CHANNELS};
    float *staged = malloc(volume(at) * sizeof(*staged));
    require(staged != NULL, "cannot allocate the staged latent");
    for (uint32_t frame = 0; frame < rows; frame++)
        for (uint32_t mel = 0; mel < LATENT_MELS; mel++)
            for (uint32_t channel = 0; channel < Z_CHANNELS; channel++) {
                const uint32_t patched = channel * LATENT_MELS + mel;
                staged[((size_t)frame * LATENT_MELS + mel) * Z_CHANNELS +
                       channel] =
                    tokens[(size_t)frame * PATCHED + patched] *
                    latent_std[patched] + latent_mean[patched];
            }
    free(tokens); free(latent_mean); free(latent_std);

    /* Two frames of the widest stage, for the causal pad. */
    {
        float *helper = calloc((size_t)MEL_BINS * WIDEST_CHANNELS,
                               sizeof(*helper));
        require(helper != NULL, "cannot allocate a pad buffer");
        zeros = upload(helper, (size_t)MEL_BINS * WIDEST_CHANNELS);
        for (int index = 0; index < WIDEST_CHANNELS; index++) helper[index] = 1.0f;
        ones = upload(helper, WIDEST_CHANNELS);
        free(helper);
    }

    h3_gpu_tensor *x = upload(staged, volume(at));
    free(staged);

    /* -------------------------------------------------- the VAE decoder */

    {
        conv2d kernel;
        load_vae_conv("conv_in", WIDEST_CHANNELS, Z_CHANNELS, KERNEL, &kernel);
        GPU_OP(h3_gpu_begin(gpu), "begin conv_in");
        h3_gpu_tensor *out = run_vae_conv(&kernel, x, at, WIDEST_CHANNELS);
        GPU_OP(h3_gpu_submit(gpu), "submit conv_in");
        free_vae_conv(&kernel);
        h3_gpu_tensor_free(x);
        x = out;
        at.channels = WIDEST_CHANNELS;
    }
    /* `mid_block_add_attention` is false, so the mid block is two residuals. */
    x = resnet("mid.block_1", x, at, at.channels);
    x = resnet("mid.block_2", x, at, at.channels);

    static const uint32_t LEVEL_CHANNELS[LEVELS] = {
        BASE_CHANNELS, BASE_CHANNELS * 2, BASE_CHANNELS * 4
    };
    for (int level = LEVELS - 1; level >= 0; level--) {
        char prefix[160];
        for (int block = 0; block < BLOCKS_PER_LEVEL; block++) {
            snprintf(prefix, sizeof(prefix), "up.%d.block.%d", level, block);
            x = resnet(prefix, x, at, LEVEL_CHANNELS[level]);
            at.channels = LEVEL_CHANNELS[level];
        }
        if (level != 0) x = upsample(level, x, &at);
    }

    h3_gpu_tensor *normed = allocate(volume(at));
    GPU_OP(h3_gpu_begin(gpu), "begin tail");
    pixel_norm(normed, x, at);
    GPU_OP(h3_gpu_submit(gpu), "submit norm");
    h3_gpu_tensor_free(x);

    {
        conv2d tail;
        load_vae_conv("conv_out", STEREO, at.channels, KERNEL, &tail);
        GPU_OP(h3_gpu_begin(gpu), "begin head");
        GPU_OP(h3_gpu_silu_f32(gpu, normed, normed, (uint32_t)volume(at)),
               "output activation");
        x = run_vae_conv(&tail, normed, at, STEREO);
        GPU_OP(h3_gpu_submit(gpu), "submit head");
        free_vae_conv(&tail);
        h3_gpu_tensor_free(normed);
        at.channels = STEREO;
    }
    require(at.frames == rows * 4 - 3 && at.mels == MEL_BINS,
            "the decode did not reach the mel geometry");
    printf("  mel [%d, %u, %u] in %.2f s\n", STEREO, at.frames, at.mels,
           now() - began);

    /* ------------------------------------------------------ the vocoder */

    /* `(stereo, frames, mels)` folds to `(stereo * mels, frames)`: the reason
     * `conv_pre` takes 128 channels rather than 64. The VAE left it
     * channels-last, so stereo is the fastest axis on the way in. */
    float *mel = download(x, volume(at));
    h3_gpu_tensor_free(x);
    span voice = {at.frames, VOICE_CHANNELS};
    float *folded = malloc(extent(voice) * sizeof(*folded));
    require(folded != NULL, "cannot allocate the folded mel");
    for (uint32_t frame = 0; frame < at.frames; frame++)
        for (uint32_t side = 0; side < STEREO; side++)
            for (uint32_t bin = 0; bin < MEL_BINS; bin++)
                folded[(size_t)frame * VOICE_CHANNELS + side * MEL_BINS + bin] =
                    mel[((size_t)frame * MEL_BINS + bin) * STEREO + side];
    free(mel);
    x = upload(folded, extent(voice));
    free(folded);

    load_filters();
    {
        conv1d pre;
        load_voc_conv("conv_pre", &pre, VOICE_CHANNELS, INITIAL_CHANNELS,
                      HEAD_KERNEL, HEAD_KERNEL / 2, 1, 1, 0, 1);
        span wide = {voice.length, INITIAL_CHANNELS};
        h3_gpu_tensor *out = allocate(extent(wide));
        GPU_OP(h3_gpu_begin(gpu), "begin conv_pre");
        run_voc_conv(out, x, &pre, voice.length);
        GPU_OP(h3_gpu_submit(gpu), "submit conv_pre");
        free_voc_conv(&pre);
        h3_gpu_tensor_free(x);
        x = out;
        voice = wide;
    }

    for (int index = 0; index < STAGES; index++) {
        stage weights;
        memset(&weights, 0, sizeof(weights));
        load_stage(&weights, index);
        const uint32_t stride = upsample_rates[index];
        const uint32_t kernel = upsample_kernels[index];
        const uint32_t padding = (kernel - stride) / 2;
        span wider = {(voice.length - 1) * stride + kernel - 2 * padding,
                      INITIAL_CHANNELS >> (index + 1)};
        h3_gpu_tensor *upsampled = allocate(extent(wider));
        h3_gpu_tensor *accumulated = allocate(extent(wider));
        h3_gpu_tensor *work = allocate(extent(wider));
        h3_gpu_tensor *activated = allocate(extent(wider));
        h3_gpu_tensor *branch = allocate(extent(wider));

        GPU_OP(h3_gpu_begin(gpu), "begin stage");
        run_voc_conv(upsampled, x, &weights.up, voice.length);
        encode_blocks(accumulated, upsampled, &weights, wider, work, activated,
                      branch);
        GPU_OP(h3_gpu_submit(gpu), "submit stage");

        h3_gpu_tensor_free(x);
        x = accumulated;
        voice = wider;
        h3_gpu_tensor_free(upsampled);
        h3_gpu_tensor_free(work);
        h3_gpu_tensor_free(activated);
        h3_gpu_tensor_free(branch);
        free_stage(&weights);
    }

    {
        activation post;
        load_activation("act_post", &post, voice.channels);
        h3_gpu_tensor *out = allocate(extent(voice));
        GPU_OP(h3_gpu_begin(gpu), "begin act_post");
        run_activation(out, x, &post, voice);
        GPU_OP(h3_gpu_submit(gpu), "submit act_post");
        free_activation(&post);
        h3_gpu_tensor_free(x);
        x = out;
    }
    {
        conv1d post;
        load_voc_conv("conv_post", &post, voice.channels, STEREO, HEAD_KERNEL,
                      HEAD_KERNEL / 2, 1, 1, 0, 0);
        span narrow = {voice.length, STEREO};
        h3_gpu_tensor *out = allocate(extent(narrow));
        GPU_OP(h3_gpu_begin(gpu), "begin conv_post");
        run_voc_conv(out, x, &post, voice.length);
        GPU_OP(h3_gpu_submit(gpu), "submit conv_post");
        free_voc_conv(&post);
        h3_gpu_tensor_free(x);
        x = out;
        voice = narrow;
    }

    /* --------------------------------------------------------- the sound */

    float *samples = download(x, extent(voice));
    h3_gpu_tensor_free(x);
    double peak = 0.0, energy = 0.0;
    for (size_t index = 0; index < extent(voice); index++) {
        const double value = samples[index];
        if (fabs(value) > peak) peak = fabs(value);
        energy += value * value;
    }
    const double rms = sqrt(energy / (double)extent(voice));
    write_wav(argv[3], samples, voice.length);
    printf("  wrote %s: %u samples, %.3f s at %d Hz, peak %.4f, rms %.4f\n",
           argv[3], voice.length, (double)voice.length / (double)SAMPLE_RATE,
           SAMPLE_RATE, peak, rms);
    if (rms < 1e-4)
        fprintf(stderr, "warning: the track is essentially silent\n");
    printf("decoded %u audio rows to sound in %.2f s\n", rows, now() - began);

    free(samples);
    h3_gpu_tensor_free(ones);
    h3_gpu_tensor_free(zeros);
    h3_gpu_tensor_free(upsample_filter);
    h3_gpu_tensor_free(downsample_filter);
    h3_gpu_free(gpu);
    h3_weight_store_free(store);
    return 0;
}
