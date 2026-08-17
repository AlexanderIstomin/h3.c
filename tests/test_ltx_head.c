/* LTX-2.5's patchify projection and output head, against its own code.
 *
 * The two ends of the DiT, on either side of the forty-eight blocks. Patchify
 * is a plain linear from the 128 latent channels to the model width and holds
 * no surprises. The head has three, each of which yields a plausible latent
 * rather than an error:
 *
 *   - it normalises with a *LayerNorm*, which subtracts the mean, where every
 *     other norm in this model is an RMS norm, which does not. The two agree
 *     only on centred input, and nothing here guarantees that;
 *   - the modulation adds the *embedded* timestep -- the intermediate from
 *     inside AdaLN-single, not its nine-slot output -- and adds the same
 *     vector to both rows of the table. Every other modulation in the model
 *     gives each slot its own slice, so this one reads as if a slice were
 *     missing when it is not; and
 *   - the table's rows are shift then scale, which is the block's order for
 *     its self-attention and feed-forward slices but the reverse of its
 *     cross-modal ones.
 *
 * The head is also where `embedded_timestep` is finally used, which is why
 * AdaLN-single returns it alongside the modulation it is derived from.
 *
 * The reference is called unbound in the generator rather than reimplemented
 * there, since _process_output reads nothing from self.
 *
 * usage: h3_ltx_head_test FIXTURE.safetensors */

#include "h3_safetensors.h"
#include "h3_weights.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    LATENT = 128,
    VIDEO_DIM = 32, AUDIO_DIM = 16,
    VIDEO_TOKENS = 6, AUDIO_TOKENS = 3
};

#define NORM_EPS 1e-6f

static int failures = 0;
static h3_weight_store *store = NULL;

static void fail(const char *message) {
    fprintf(stderr, "FAIL: %s\n", message);
    exit(1);
}

static void require(int condition, const char *message) {
    if (!condition) fail(message);
}

static float *load_f32(const char *name, size_t expected) {
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(store, name, &header);
    if (!tensor || !header) {
        fprintf(stderr, "FAIL: fixture has no tensor %s\n", name);
        exit(1);
    }
    size_t elements = (size_t)h3_st_tensor_elements(tensor);
    if (elements != expected) {
        fprintf(stderr, "FAIL: %s has %zu elements, expected %zu\n",
                name, elements, expected);
        exit(1);
    }
    float *values = malloc(elements * sizeof(*values));
    char error[256];
    require(values != NULL, "cannot allocate a fixture tensor");
    require(h3_st_read_data(header, tensor, values,
                            elements * sizeof(*values), error, sizeof(error)),
            "cannot read a fixture tensor");
    return values;
}

static void compare(const char *label, const float *actual,
                    const float *expected, size_t count, float tolerance) {
    double worst = 0.0;
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
    }
    if (worst > tolerance) {
        fprintf(stderr, "FAIL %-28s worst |delta| %.3e at %zu (%.8f vs %.8f)\n",
                label, worst, worst_at, actual[worst_at], expected[worst_at]);
        failures++;
    } else {
        printf("  ok  %-28s worst |delta| %.2e\n", label, worst);
    }
}

static void linear_bias(float *out, const float *in, const float *weight,
                        const float *bias, size_t rows, size_t input_dim,
                        size_t output_dim) {
    for (size_t row = 0; row < rows; row++) {
        for (size_t column = 0; column < output_dim; column++) {
            float sum = bias[column];
            const float *w = weight + column * input_dim;
            const float *x = in + row * input_dim;
            for (size_t k = 0; k < input_dim; k++) sum = fmaf(x[k], w[k], sum);
            out[row * output_dim + column] = sum;
        }
    }
}

/* A LayerNorm, so the mean comes out before the variance divides. An RMS norm
 * would skip both, and on uncentred rows the two do not agree. Parameter-free:
 * elementwise_affine is off, and the affine that follows comes from the
 * modulation instead. */
static void layer_norm_rows(float *out, const float *in, size_t rows,
                            size_t width) {
    for (size_t row = 0; row < rows; row++) {
        const float *x = in + row * width;
        float mean = 0.0f;
        for (size_t index = 0; index < width; index++) mean += x[index];
        mean /= (float)width;
        float variance = 0.0f;
        for (size_t index = 0; index < width; index++) {
            float centred = x[index] - mean;
            variance = fmaf(centred, centred, variance);
        }
        variance /= (float)width;
        float inverse = 1.0f / sqrtf(variance + NORM_EPS);
        for (size_t index = 0; index < width; index++)
            out[row * width + index] = (x[index] - mean) * inverse;
    }
}

/* norm, then modulate by the table plus the embedded timestep, then project
 * back to the latent channels. The embedded timestep is added to *both* rows
 * of the table, so shift and scale share it. */
static void process_output(float *out, const float *hidden,
                           const float *embedded, const float *table,
                           const float *weight, const float *bias,
                           size_t rows, size_t width) {
    float *normed = malloc(rows * width * sizeof(*normed));
    require(normed != NULL, "cannot allocate the head's norm buffer");
    layer_norm_rows(normed, hidden, rows, width);
    for (size_t row = 0; row < rows; row++) {
        for (size_t index = 0; index < width; index++) {
            float shift = table[index] + embedded[row * width + index];
            float scale = table[width + index] + embedded[row * width + index];
            size_t at = row * width + index;
            normed[at] = normed[at] * (1.0f + scale) + shift;
        }
    }
    linear_bias(out, normed, weight, bias, rows, width, LATENT);
    free(normed);
}

static void run_stream(const char *tag, size_t dim, size_t tokens) {
    char name[160];
    float *latent, *hidden, *embedded, *table;
    float *patch_w, *patch_b, *proj_w, *proj_b;

    snprintf(name, sizeof(name), "input.%s_latent", tag);
    latent = load_f32(name, tokens * LATENT);
    snprintf(name, sizeof(name), "input.%s_hidden", tag);
    hidden = load_f32(name, tokens * dim);
    snprintf(name, sizeof(name), "input.%s_embedded", tag);
    embedded = load_f32(name, tokens * dim);
    snprintf(name, sizeof(name), "%s.scale_shift_table", tag);
    table = load_f32(name, 2 * dim);
    snprintf(name, sizeof(name), "%s.patchify.weight", tag);
    patch_w = load_f32(name, dim * LATENT);
    snprintf(name, sizeof(name), "%s.patchify.bias", tag);
    patch_b = load_f32(name, dim);
    snprintf(name, sizeof(name), "%s.proj_out.weight", tag);
    proj_w = load_f32(name, (size_t)LATENT * dim);
    snprintf(name, sizeof(name), "%s.proj_out.bias", tag);
    proj_b = load_f32(name, LATENT);

    float *patched = malloc(tokens * dim * sizeof(*patched));
    float *out = malloc(tokens * LATENT * sizeof(*out));
    require(patched && out, "cannot allocate the head's buffers");

    linear_bias(patched, latent, patch_w, patch_b, tokens, LATENT, dim);
    process_output(out, hidden, embedded, table, proj_w, proj_b, tokens, dim);

    snprintf(name, sizeof(name), "reference.%s_patched", tag);
    float *expected_patched = load_f32(name, tokens * dim);
    snprintf(name, sizeof(name), "reference.%s_out", tag);
    float *expected_out = load_f32(name, tokens * LATENT);

    snprintf(name, sizeof(name), "%s patchify", tag);
    compare(name, patched, expected_patched, tokens * dim, 2e-6f);
    snprintf(name, sizeof(name), "%s output head", tag);
    compare(name, out, expected_out, tokens * LATENT, 2e-6f);

    free(latent); free(hidden); free(embedded); free(table);
    free(patch_w); free(patch_b); free(proj_w); free(proj_b);
    free(patched); free(out);
    free(expected_patched); free(expected_out);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s FIXTURE.safetensors\n", argv[0]);
        return 2;
    }
    char error[512];
    store = h3_weight_store_open(argv[1], error, sizeof(error));
    if (!store) {
        fprintf(stderr, "FAIL: cannot open the fixture: %s\n", error);
        return 1;
    }

    printf("LTX-2.5 patchify and output head against its own implementation:\n");
    run_stream("video", VIDEO_DIM, VIDEO_TOKENS);
    run_stream("audio", AUDIO_DIM, AUDIO_TOKENS);
    h3_weight_store_free(store);

    if (failures) {
        fprintf(stderr, "\n%d head comparisons failed\n", failures);
        return 1;
    }
    printf("ok: both ends of the DiT reproduce LTX-2.5's own output, "
           "LayerNorm and shared embedded timestep included\n");
    return 0;
}
