/* LTX-2.5's DiT top level on the released weights: everything around the 48
 * blocks.
 *
 * Patchify and its keyframe marker, the eight AdaLN-single modules that turn
 * one sigma into six modulation tensors and two embedded timesteps, the 3D
 * rotary tables at released width, and the output head. All dense BF16 in the
 * checkpoint, so this is the half of the driver that needs no dequantizing --
 * and the half where a mistake is a wrong constant rather than a wrong matrix.
 *
 * Four things here are not guessable from the block's shape:
 *
 *   - the sinusoid runs at sigma * 1000, so its argument reaches 732 radians
 *     and it has to be built in F32 whatever the rest of the model does. The
 *     reference builds it in F32 and only then casts to the model dtype, which
 *     is why the MLP after it can be BF16 and this cannot;
 *   - the cross-modal modules carry their *own* multiplier. The released
 *     config sets it to 1000 like the others, where the library default of 1
 *     would put their gate at the raw sigma;
 *   - the output head is a LayerNorm -- the only norm in this model that
 *     subtracts a mean -- and the same embedded timestep goes into both its
 *     shift and its scale, so those differ only by the two-row table; and
 *   - the 3D rotary splits 682 frequencies an axis across 32 heads and pads
 *     the last two of 2048 with an identity rotation. The audio side is one
 *     axis, 1024 frequencies and no padding at all.
 *
 * usage: h3_real_ltx_dit_top_test DIT.safetensors ANCHOR.safetensors */

#include "h3_gpu.h"
#include "h3_safetensors.h"
#include "h3_weights.h"

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    VIDEO_DIM = 4096,
    AUDIO_DIM = 2048,
    LATENT = 128,
    HEADS = 32,
    TIMESTEP_FEATURES = 256,
    VIDEO_ROWS = 48,
    AUDIO_ROWS = 24,
    VIDEO_AXES = 3,
    AUDIO_AXES = 1,
    HEAD_SLOTS = 2
};

#define HEAD_EPSILON 1e-6f
#define TIMESTEP_MULTIPLIER 1000.0
#define MAX_PERIOD 10000.0

static const double VIDEO_MAX_POS[VIDEO_AXES] = {20.0, 2048.0, 2048.0};
static const double AUDIO_MAX_POS[AUDIO_AXES] = {20.0};
static const double ROPE_THETA = 10000.0;

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

static float from_bf16(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

/* ------------------------------------------------------------------ loading */

static float *read_tensor(const h3_weight_store *from, const char *name,
                          size_t expected) {
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(from, name, &header);
    if (!tensor) fail("no tensor %s", name);
    size_t elements = (size_t)h3_st_tensor_elements(tensor);
    if (elements != expected)
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
        require(raw != NULL, "cannot allocate a BF16 staging buffer");
        require(h3_st_read_data(header, tensor, raw, elements * sizeof(*raw),
                                error, sizeof(error)),
                "cannot read a BF16 tensor");
        for (size_t index = 0; index < elements; index++)
            values[index] = from_bf16(raw[index]);
        free(raw);
    } else {
        fail("%s is %s, expected F32 or BF16", name,
             h3_dtype_name(tensor->dtype));
    }
    return values;
}

static float *weight_of(const char *name, size_t expected) {
    char full[192];
    snprintf(full, sizeof(full), "model.diffusion_model.%s", name);
    return read_tensor(store, full, expected);
}

static float *golden(const char *name, size_t expected) {
    return read_tensor(anchor, name, expected);
}

/* The engine against the F32 reference, with the reference's own BF16 pass
 * beside it, as everywhere else in this port. */
static void compare(const char *label, const float *actual,
                    const float *expected, const float *floor, size_t count,
                    double tolerance) {
    double worst = 0.0, spread = 0.0, peak = 0.0;
    size_t worst_at = 0;
    for (size_t index = 0; index < count; index++) {
        if (!isfinite(actual[index])) {
            fprintf(stderr, "FAIL %-28s produced %f at %zu\n",
                    label, actual[index], index);
            failures++;
            return;
        }
        double delta = fabs((double)actual[index] - (double)expected[index]);
        if (delta > worst) { worst = delta; worst_at = index; }
        if (floor) {
            double reference = fabs((double)floor[index] - (double)expected[index]);
            if (reference > spread) spread = reference;
        }
        double magnitude = fabs((double)expected[index]);
        if (magnitude > peak) peak = magnitude;
    }
    const double relative = worst / peak;
    if (relative > tolerance) {
        fprintf(stderr, "FAIL %-28s %.3e = %.2e of peak, floor %.2e, at %zu "
                        "(%.6f vs %.6f)\n", label, worst, relative,
                spread / peak, worst_at, actual[worst_at], expected[worst_at]);
        failures++;
    } else if (floor) {
        printf("  ok  %-28s %.3e = %.2e of peak %.4f, reference's own BF16 "
               "floor %.2e\n", label, worst, relative, peak, spread / peak);
    } else {
        printf("  ok  %-28s %.3e = %.2e of peak %.4f\n",
               label, worst, relative, peak);
    }
}

static void compare_pair(const char *label, const char *name,
                         const float *actual, size_t count, double tolerance) {
    char coarse[128];
    snprintf(coarse, sizeof(coarse), "%s_bf16", name);
    float *expected = golden(name, count);
    float *floor = golden(coarse, count);
    compare(label, actual, expected, floor, count, tolerance);
    free(expected);
    free(floor);
}

/* -------------------------------------------------------- the timestep path */

/* The sinusoid, in double. `flip_sin_to_cos` puts cosine first, and the
 * exponent divides by half_dim rather than half_dim - 1 because
 * `downscale_freq_shift` is zero here -- both are one-character mistakes that
 * produce a perfectly smooth embedding of the wrong thing. */
static void sinusoid(float *out, double timestep) {
    const int half = TIMESTEP_FEATURES / 2;
    for (int index = 0; index < half; index++) {
        const double exponent = -log(MAX_PERIOD) * (double)index / (double)half;
        const double angle = timestep * exp(exponent);
        out[index] = (float)cos(angle);
        out[half + index] = (float)sin(angle);
    }
}

typedef struct {
    const char *name;
    uint32_t width;
    uint32_t slots;
} adaln_module;

/* Eight separate networks. Their slot counts differ and so do their widths,
 * and crossing two of them yields a well-formed modulation of the wrong size
 * only when the widths happen to differ -- when they do not, nothing complains
 * at all. */
static const adaln_module MODULES[] = {
    {"adaln_single", VIDEO_DIM, 9},
    {"audio_adaln_single", AUDIO_DIM, 9},
    {"prompt_adaln_single", VIDEO_DIM, 2},
    {"audio_prompt_adaln_single", AUDIO_DIM, 2},
    {"av_ca_video_scale_shift_adaln_single", VIDEO_DIM, 4},
    {"av_ca_audio_scale_shift_adaln_single", AUDIO_DIM, 4},
    {"av_ca_a2v_gate_adaln_single", VIDEO_DIM, 1},
    {"av_ca_v2a_gate_adaln_single", AUDIO_DIM, 1}
};

enum { MODULE_COUNT = (int)(sizeof(MODULES) / sizeof(MODULES[0])) };

static float silu(float value) {
    return value / (1.0f + expf(-value));
}

/* out = weight @ in + bias, with the weight row-major [out_dim, in_dim].
 * Small enough at these shapes that the host is the right place for it: the
 * largest is 36864 x 4096 and it runs once per denoising step, not per token. */
static void linear(float *out, const float *in, const float *weight,
                   const float *bias, uint32_t in_dim, uint32_t out_dim) {
    for (uint32_t row = 0; row < out_dim; row++) {
        float sum = bias ? bias[row] : 0.0f;
        const float *w = weight + (size_t)row * in_dim;
        for (uint32_t index = 0; index < in_dim; index++)
            sum = fmaf(in[index], w[index], sum);
        out[row] = sum;
    }
}

/* sigma -> one module's modulation and its embedded timestep. */
static void run_adaln(const adaln_module *module, double sigma,
                      float **modulation, float **embedded) {
    char name[192];
    float features[TIMESTEP_FEATURES];
    sinusoid(features, sigma * TIMESTEP_MULTIPLIER);

    snprintf(name, sizeof(name), "%s.emb.timestep_embedder.linear_1.weight",
             module->name);
    float *first = weight_of(name, (size_t)module->width * TIMESTEP_FEATURES);
    snprintf(name, sizeof(name), "%s.emb.timestep_embedder.linear_1.bias",
             module->name);
    float *first_bias = weight_of(name, module->width);
    float *hidden = malloc(module->width * sizeof(*hidden));
    require(hidden != NULL, "cannot allocate a timestep hidden state");
    linear(hidden, features, first, first_bias, TIMESTEP_FEATURES, module->width);
    for (uint32_t index = 0; index < module->width; index++)
        hidden[index] = silu(hidden[index]);
    free(first); free(first_bias);

    snprintf(name, sizeof(name), "%s.emb.timestep_embedder.linear_2.weight",
             module->name);
    float *second = weight_of(name, (size_t)module->width * module->width);
    snprintf(name, sizeof(name), "%s.emb.timestep_embedder.linear_2.bias",
             module->name);
    float *second_bias = weight_of(name, module->width);
    *embedded = malloc(module->width * sizeof(**embedded));
    require(*embedded != NULL, "cannot allocate an embedded timestep");
    linear(*embedded, hidden, second, second_bias, module->width, module->width);
    free(second); free(second_bias); free(hidden);

    /* A second SiLU before the projection, on the embedded timestep that the
     * output head also consumes unactivated. */
    float *activated = malloc(module->width * sizeof(*activated));
    require(activated != NULL, "cannot allocate the activated timestep");
    for (uint32_t index = 0; index < module->width; index++)
        activated[index] = silu((*embedded)[index]);
    snprintf(name, sizeof(name), "%s.linear.weight", module->name);
    float *projection = weight_of(
        name, (size_t)module->slots * module->width * module->width);
    snprintf(name, sizeof(name), "%s.linear.bias", module->name);
    float *projection_bias = weight_of(name,
                                       (size_t)module->slots * module->width);
    *modulation = malloc((size_t)module->slots * module->width *
                         sizeof(**modulation));
    require(*modulation != NULL, "cannot allocate a modulation");
    linear(*modulation, activated, projection, projection_bias, module->width,
           module->slots * module->width);
    free(projection); free(projection_bias); free(activated);
}

/* ------------------------------------------------------------------- rope */

/* LTX's split rope over `axes` position axes. The frequencies are spread
 * geometrically from 1 to theta over dim / (2 * axes) entries, scaled by pi/2,
 * and each axis contributes its own copy; positions are the midpoint of each
 * token's start and end, mapped onto [-1, 1] through max_pos.
 *
 * dim/2 is not generally a multiple of the axis count, so the leading slots
 * are padded with an identity rotation -- two of 2048 for video. The padding
 * goes in *front*, which is the detail that decides which head sees which
 * frequency once the result is split across 32 of them.
 *
 * Two things here were wrong first time and neither announced itself:
 *
 *   - the axes interleave. The reference builds [row, axis, frequency] and
 *     then transposes to [row, frequency, axis] before flattening, so
 *     consecutive slots step through t, x, y at one frequency rather than
 *     through the frequencies of one axis. Laying it out axis-major gives a
 *     table of the right shape, filled with the right numbers, in the wrong
 *     order -- and it flipped cosines from +1 to -1;
 *   - the frequencies are rounded to F32 before use, even though they are
 *     *computed* in float64 as the config asks. `generate_freq_grid_np` builds
 *     them in double and hands back a float32 tensor, and the angle is a
 *     product of two F32 values. Keeping the full double through the multiply
 *     is more accurate and disagrees, by 2.7e-03 where the angle lands near a
 *     zero crossing. Matching the reference means rounding on purpose. */
static void rope_tables(float *cosine, float *sine, const float *grid,
                        uint32_t rows, uint32_t dim, int axes,
                        const double *max_pos) {
    const uint32_t half = dim / 2;
    const uint32_t per_axis = dim / (2 * (uint32_t)axes);
    const uint32_t pad = half - per_axis * (uint32_t)axes;
    for (uint32_t row = 0; row < rows; row++) {
        for (uint32_t slot = 0; slot < pad; slot++) {
            cosine[(size_t)row * half + slot] = 1.0f;
            sine[(size_t)row * half + slot] = 0.0f;
        }
        for (int axis = 0; axis < axes; axis++) {
            /* [axis][row][start, end] -- the grid is axis-major. */
            const float *pair = grid + ((size_t)axis * rows + row) * 2;
            const float middle = (pair[0] + pair[1]) / 2.0f;
            const float fractional = middle / (float)max_pos[axis];
            const float position = fractional * 2.0f - 1.0f;
            for (uint32_t index = 0; index < per_axis; index++) {
                const double exponent = per_axis > 1 ?
                    (double)index / (double)(per_axis - 1) : 0.0;
                const float frequency =
                    (float)(pow(ROPE_THETA, exponent) * M_PI / 2.0);
                const float angle = frequency * position;
                const size_t at = (size_t)row * half + pad +
                                  (size_t)index * (uint32_t)axes + (uint32_t)axis;
                cosine[at] = cosf(angle);
                sine[at] = sinf(angle);
            }
        }
    }
}

/* ------------------------------------------------------------------- main */

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s DIT.safetensors ANCHOR.safetensors\n",
                argv[0]);
        return 2;
    }
    char error[512];
    store = h3_weight_store_open(argv[1], error, sizeof(error));
    if (!store) fail("cannot open the DiT checkpoint: %s", error);
    anchor = h3_weight_store_open(argv[2], error, sizeof(error));
    if (!anchor) fail("cannot open the anchor: %s", error);
    gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) fail("cannot create the Metal context: %s", error);

    float *sigma_values = golden("sigma", 1);
    const double sigma = sigma_values[0];
    free(sigma_values);
    printf("LTX-2.5 DiT top level on the released weights, sigma %.4f:\n", sigma);

    /* ----------------------------------------- the eight AdaLN-single modules */

    for (int index = 0; index < MODULE_COUNT; index++) {
        const adaln_module *module = &MODULES[index];
        float *modulation = NULL, *embedded = NULL;
        run_adaln(module, sigma, &modulation, &embedded);
        char name[192], label[96];
        snprintf(name, sizeof(name), "%s.modulation", module->name);
        snprintf(label, sizeof(label), "%.24s x%u", module->name, module->slots);
        compare_pair(label, name, modulation,
                     (size_t)module->slots * module->width, 6e-3);
        snprintf(name, sizeof(name), "%s.embedded", module->name);
        snprintf(label, sizeof(label), "%.24s emb", module->name);
        compare_pair(label, name, embedded, module->width, 6e-3);
        free(modulation);
        free(embedded);
    }

    /* -------------------------------------------------- patchify and keyframes */

    {
        float *latent = golden("video_latent", (size_t)VIDEO_ROWS * LATENT);
        float *weight = weight_of("patchify_proj.weight",
                                  (size_t)VIDEO_DIM * LATENT);
        float *bias = weight_of("patchify_proj.bias", VIDEO_DIM);
        float *out = malloc((size_t)VIDEO_ROWS * VIDEO_DIM * sizeof(*out));
        require(out != NULL, "cannot allocate the patchified video");
        for (int row = 0; row < VIDEO_ROWS; row++)
            linear(out + (size_t)row * VIDEO_DIM, latent + (size_t)row * LATENT,
                   weight, bias, LATENT, VIDEO_DIM);
        float *expected = golden("video_patchified",
                                 (size_t)VIDEO_ROWS * VIDEO_DIM);
        compare("video patchify", out, expected, NULL,
                (size_t)VIDEO_ROWS * VIDEO_DIM, 2e-6);
        free(expected);
        /* The keyframe marker. LTX's own comment calls it a no-op forever; the
         * released tensor has a maximum of 0.0034, so it is trained. */
        float *marker = weight_of("keyframes_abs_pos_embedding", VIDEO_DIM);
        float *mask = golden("keyframes_mask", VIDEO_ROWS);
        for (int row = 0; row < VIDEO_ROWS; row++)
            if (mask[row] > 0.0f)
                for (int index = 0; index < VIDEO_DIM; index++)
                    out[(size_t)row * VIDEO_DIM + index] += marker[index];
        expected = golden("video_keyframed", (size_t)VIDEO_ROWS * VIDEO_DIM);
        compare("video keyframe marker", out, expected, NULL,
                (size_t)VIDEO_ROWS * VIDEO_DIM, 2e-6);
        free(expected); free(marker); free(mask);
        free(latent); free(weight); free(bias); free(out);
    }
    {
        float *latent = golden("audio_latent", (size_t)AUDIO_ROWS * LATENT);
        float *weight = weight_of("audio_patchify_proj.weight",
                                  (size_t)AUDIO_DIM * LATENT);
        float *bias = weight_of("audio_patchify_proj.bias", AUDIO_DIM);
        float *out = malloc((size_t)AUDIO_ROWS * AUDIO_DIM * sizeof(*out));
        require(out != NULL, "cannot allocate the patchified audio");
        for (int row = 0; row < AUDIO_ROWS; row++)
            linear(out + (size_t)row * AUDIO_DIM, latent + (size_t)row * LATENT,
                   weight, bias, LATENT, AUDIO_DIM);
        float *expected = golden("audio_patchified",
                                 (size_t)AUDIO_ROWS * AUDIO_DIM);
        compare("audio patchify", out, expected, NULL,
                (size_t)AUDIO_ROWS * AUDIO_DIM, 2e-6);
        free(expected); free(latent); free(weight); free(bias); free(out);
    }

    /* ------------------------------------------------------------- the rope */

    {
        const uint32_t half = VIDEO_DIM / 2;
        float *grid = golden("video_grid", (size_t)VIDEO_AXES * VIDEO_ROWS * 2);
        float *cosine = malloc((size_t)VIDEO_ROWS * half * sizeof(*cosine));
        float *sine = malloc((size_t)VIDEO_ROWS * half * sizeof(*sine));
        require(cosine && sine, "cannot allocate the video rotary tables");
        rope_tables(cosine, sine, grid, VIDEO_ROWS, VIDEO_DIM, VIDEO_AXES,
                    VIDEO_MAX_POS);
        /* The reference emits [head, row, half/heads]; this builds [row, half]
         * and the two agree because the split is a reshape. */
        float *expected = golden("video_rope_cos", (size_t)VIDEO_ROWS * half);
        float *expected_sin = golden("video_rope_sin", (size_t)VIDEO_ROWS * half);
        float *reordered = malloc((size_t)VIDEO_ROWS * half * sizeof(*reordered));
        float *reordered_sin = malloc((size_t)VIDEO_ROWS * half * sizeof(*reordered_sin));
        require(reordered && reordered_sin, "cannot allocate a rotary probe");
        const uint32_t per_head = half / HEADS;
        for (uint32_t head = 0; head < HEADS; head++)
            for (uint32_t row = 0; row < VIDEO_ROWS; row++)
                for (uint32_t index = 0; index < per_head; index++) {
                    const size_t from = ((size_t)head * VIDEO_ROWS + row) *
                                        per_head + index;
                    const size_t to = (size_t)row * half + head * per_head + index;
                    reordered[to] = expected[from];
                    reordered_sin[to] = expected_sin[from];
                }
        compare("video rope cos", cosine, reordered, NULL,
                (size_t)VIDEO_ROWS * half, 1e-5);
        compare("video rope sin", sine, reordered_sin, NULL,
                (size_t)VIDEO_ROWS * half, 1e-5);
        free(expected); free(expected_sin); free(reordered); free(reordered_sin);
        free(cosine); free(sine); free(grid);
    }
    {
        const uint32_t half = AUDIO_DIM / 2;
        float *grid = golden("audio_grid", (size_t)AUDIO_AXES * AUDIO_ROWS * 2);
        float *cosine = malloc((size_t)AUDIO_ROWS * half * sizeof(*cosine));
        float *sine = malloc((size_t)AUDIO_ROWS * half * sizeof(*sine));
        require(cosine && sine, "cannot allocate the audio rotary tables");
        rope_tables(cosine, sine, grid, AUDIO_ROWS, AUDIO_DIM, AUDIO_AXES,
                    AUDIO_MAX_POS);
        float *expected = golden("audio_rope_cos", (size_t)AUDIO_ROWS * half);
        float *reordered = malloc((size_t)AUDIO_ROWS * half * sizeof(*reordered));
        require(reordered != NULL, "cannot allocate a rotary probe");
        const uint32_t per_head = half / HEADS;
        for (uint32_t head = 0; head < HEADS; head++)
            for (uint32_t row = 0; row < AUDIO_ROWS; row++)
                for (uint32_t index = 0; index < per_head; index++)
                    reordered[(size_t)row * half + head * per_head + index] =
                        expected[((size_t)head * AUDIO_ROWS + row) * per_head + index];
        compare("audio rope cos", cosine, reordered, NULL,
                (size_t)AUDIO_ROWS * half, 1e-5);
        free(expected); free(reordered);
        free(cosine); free(sine); free(grid);
    }

    /* -------------------------------------------------------- the output head */

    for (int stream = 0; stream < 2; stream++) {
        const uint32_t rows = stream ? AUDIO_ROWS : VIDEO_ROWS;
        const uint32_t dim = stream ? AUDIO_DIM : VIDEO_DIM;
        const char *tag = stream ? "audio" : "video";
        char name[128];
        snprintf(name, sizeof(name), "%s_head_input", tag);
        float *x = golden(name, (size_t)rows * dim);
        snprintf(name, sizeof(name), "%s%s", stream ? "audio_" : "",
                 "scale_shift_table");
        float *table = weight_of(name, (size_t)HEAD_SLOTS * dim);
        snprintf(name, sizeof(name), "%s.embedded",
                 stream ? "audio_adaln_single" : "adaln_single");
        float *embedded = golden(name, dim);
        snprintf(name, sizeof(name), "%s%s", stream ? "audio_" : "", "proj_out");
        char weight_name[160], bias_name[160];
        snprintf(weight_name, sizeof(weight_name), "%s.weight", name);
        snprintf(bias_name, sizeof(bias_name), "%s.bias", name);
        float *weight = weight_of(weight_name, (size_t)LATENT * dim);
        float *bias = weight_of(bias_name, LATENT);
        float *out = malloc((size_t)rows * LATENT * sizeof(*out));
        float *scratch = malloc(dim * sizeof(*scratch));
        require(out && scratch, "cannot allocate the head output");
        for (uint32_t row = 0; row < rows; row++) {
            const float *source = x + (size_t)row * dim;
            /* LayerNorm: the mean is subtracted, unlike every other norm in
             * this model, and there is no learnable weight. */
            double mean = 0.0;
            for (uint32_t index = 0; index < dim; index++) mean += source[index];
            mean /= (double)dim;
            double variance = 0.0;
            for (uint32_t index = 0; index < dim; index++) {
                const double centred = (double)source[index] - mean;
                variance += centred * centred;
            }
            variance /= (double)dim;
            const double inverse = 1.0 / sqrt(variance + HEAD_EPSILON);
            for (uint32_t index = 0; index < dim; index++) {
                /* The same embedded vector feeds both rows, so shift and scale
                 * differ only by the table. */
                const float shift = table[index] + embedded[index];
                const float scale = table[dim + index] + embedded[index];
                scratch[index] = (float)(((double)source[index] - mean) * inverse)
                                 * (1.0f + scale) + shift;
            }
            linear(out + (size_t)row * LATENT, scratch, weight, bias, dim, LATENT);
        }
        snprintf(name, sizeof(name), "%s_head_output", tag);
        float *expected = golden(name, (size_t)rows * LATENT);
        char label[64];
        snprintf(label, sizeof(label), "%s output head", tag);
        compare(label, out, expected, NULL, (size_t)rows * LATENT, 1e-5);
        free(expected); free(x); free(table); free(embedded);
        free(weight); free(bias); free(out); free(scratch);
    }

    h3_gpu_free(gpu);
    h3_weight_store_free(store);
    h3_weight_store_free(anchor);
    if (failures) {
        fprintf(stderr, "\n%d comparisons failed\n", failures);
        return 1;
    }
    printf("\nok: the DiT top level reproduces LTX's own -- %d AdaLN-single "
           "modules, patchify with its keyframe marker, the 3D and 1D rotary "
           "tables, and both output heads\n", MODULE_COUNT);
    return 0;
}
