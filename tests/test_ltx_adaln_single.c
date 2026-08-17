/* LTX-2.5's AdaLN-single family, against its own implementation.
 *
 * Everything the DiT block consumes as modulation arrives through this one
 * module at four widths: the main nine-slot table, the two-slot prompt one,
 * and the cross-modal four-slot scale and shift with its one-slot gate. The
 * arithmetic is the same each time -- a sinusoidal embedding, two linears
 * around a SiLU, then a SiLU and one more linear -- so what differs between
 * them is only the output width and, crucially, the argument.
 *
 * Four details here are worth a fixture, because each produces a plausible
 * modulation rather than an error:
 *
 *   - the sinusoidal embedding is cosine first. `flip_sin_to_cos` is set, so
 *     the natural [sin, cos] order is reversed;
 *   - its frequencies divide by half_dim, not half_dim - 1. The commoner
 *     convention subtracts one, and `downscale_freq_shift` here is zero;
 *   - the argument is sigma * 1000, not sigma. Every path scales by
 *     `timestep_scale_multiplier` first; and
 *   - the cross-modal gate is the exception. Its argument is multiplied by
 *     `av_ca_factor`, which is 1 / 1000, so the 1000 divides straight back out
 *     and the gate alone is embedded at the raw sigma. Reusing the scaled
 *     argument there gives a gate from an utterly different frequency band.
 *
 * Two structural facts follow from the reference and matter to the driver
 * rather than to this file. The prompt modulation is driven by the modality's
 * scalar sigma, so it changes at every denoising step and is *not* cacheable
 * across them. And the cross-modal gate is driven by the *other* modality's
 * sigma, so what scales audio's contribution into video is how noisy the audio
 * still is.
 *
 * usage: h3_ltx_adaln_single_test FIXTURE.safetensors */

#include "h3_safetensors.h"
#include "h3_weights.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    DIM = 32,
    TOKENS = 5,
    /* The sinusoidal embedding is 256 wide whatever the model's width is. */
    PROJECTION = 256,
    HALF = PROJECTION / 2
};

#define TIMESTEP_SCALE 1000.0f
#define AV_CA_SCALE 1.0f
#define MAX_PERIOD 10000.0

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
        /* A non-finite result must fail loudly. Comparing it by magnitude
         * would not: every ordering test against NaN is false. */
        if (!isfinite(actual[index])) {
            fprintf(stderr, "FAIL %-26s produced %f at %zu\n",
                    label, actual[index], index);
            failures++;
            return;
        }
        double delta = fabs((double)actual[index] - (double)expected[index]);
        if (delta > worst) { worst = delta; worst_at = index; }
    }
    if (worst > tolerance) {
        fprintf(stderr, "FAIL %-26s worst |delta| %.3e at %zu (%.8f vs %.8f)\n",
                label, worst, worst_at, actual[worst_at], expected[worst_at]);
        failures++;
    } else {
        printf("  ok  %-26s worst |delta| %.2e\n", label, worst);
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

static float silu(float value) { return value / (1.0f + expf(-value)); }

/* Cosine first, then sine, and the exponent divides by half_dim rather than
 * half_dim - 1. Both are the non-obvious choice, and both are silent.
 *
 * Deliberately F32 rather than double, to match what the reference and the
 * engine both compute. It matters more than it looks: at sigma near one the
 * argument is a thousand, so the lowest-frequency channels evaluate cosine
 * around a thousand radians, where an F32 argument carries about 6e-05 of
 * absolute error on its own. That is the floor this whole file is measured
 * against, and it is also why this stage cannot be moved to BF16 -- eight
 * mantissa bits put the nearest representable neighbour of 700 several units
 * away, which is a different phase entirely. */
static void sinusoidal(float *out, const float *arguments, size_t count) {
    for (size_t row = 0; row < count; row++) {
        for (size_t index = 0; index < HALF; index++) {
            float exponent = -logf((float)MAX_PERIOD) * (float)index / (float)HALF;
            float angle = arguments[row] * expf(exponent);
            out[row * PROJECTION + index] = cosf(angle);
            out[row * PROJECTION + HALF + index] = sinf(angle);
        }
    }
}

/* The whole module: sinusoidal, then linear-SiLU-linear to the model width,
 * then SiLU and one more linear out to `slots * width`. The intermediate is
 * the `embedded_timestep` the final layer of the DiT reads, so it is returned
 * as well as consumed. */
static void adaln_single(const char *prefix, const float *arguments,
                         size_t count, size_t slots, float *out,
                         float *embedded) {
    char name[160];
    float projected[TOKENS * PROJECTION];
    float first[TOKENS * DIM];
    require(count <= TOKENS, "reference assumes a short batch");
    sinusoidal(projected, arguments, count);

    snprintf(name, sizeof(name), "%s.emb.timestep_embedder.linear_1.weight", prefix);
    float *w1 = load_f32(name, (size_t)DIM * PROJECTION);
    snprintf(name, sizeof(name), "%s.emb.timestep_embedder.linear_1.bias", prefix);
    float *b1 = load_f32(name, DIM);
    snprintf(name, sizeof(name), "%s.emb.timestep_embedder.linear_2.weight", prefix);
    float *w2 = load_f32(name, (size_t)DIM * DIM);
    snprintf(name, sizeof(name), "%s.emb.timestep_embedder.linear_2.bias", prefix);
    float *b2 = load_f32(name, DIM);
    snprintf(name, sizeof(name), "%s.linear.weight", prefix);
    float *w3 = load_f32(name, slots * DIM * DIM);
    snprintf(name, sizeof(name), "%s.linear.bias", prefix);
    float *b3 = load_f32(name, slots * DIM);

    linear_bias(first, projected, w1, b1, count, PROJECTION, DIM);
    for (size_t index = 0; index < count * DIM; index++)
        first[index] = silu(first[index]);
    linear_bias(embedded, first, w2, b2, count, DIM, DIM);
    /* A second SiLU before the output projection, on top of the one inside
     * the embedder. */
    for (size_t index = 0; index < count * DIM; index++)
        first[index] = silu(embedded[index]);
    linear_bias(out, first, w3, b3, count, DIM, slots * DIM);

    free(w1); free(b1); free(w2); free(b2); free(w3); free(b3);
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

    printf("LTX-2.5 AdaLN-single against its own implementation:\n");

    float *sigmas = load_f32("input.sigmas", TOKENS);
    float *scalar = load_f32("input.scalar_sigma", 1);
    float *cross = load_f32("input.cross_sigma", 1);

    float scaled[TOKENS], embedded[TOKENS * DIM];
    float main_out[TOKENS * 9 * DIM], cross_ss[TOKENS * 4 * DIM];
    float prompt_out[2 * DIM], gate_out[DIM];

    for (size_t index = 0; index < TOKENS; index++)
        scaled[index] = sigmas[index] * TIMESTEP_SCALE;
    adaln_single("main", scaled, TOKENS, 9, main_out, embedded);
    adaln_single("cross_ss", scaled, TOKENS, 4, cross_ss, embedded);

    float prompt_argument = scalar[0] * TIMESTEP_SCALE;
    float prompt_embedded[DIM];
    adaln_single("prompt", &prompt_argument, 1, 2, prompt_out, prompt_embedded);

    /* The gate's factor divides the scale straight back out, so this is the
     * raw sigma where every other path is a thousand times it. */
    float gate_argument = cross[0] * TIMESTEP_SCALE * (AV_CA_SCALE / TIMESTEP_SCALE);
    float gate_embedded[DIM];
    adaln_single("cross_gate", &gate_argument, 1, 1, gate_out, gate_embedded);

    /* Recompute the intermediate for the main path, which the loop above
     * overwrote with the cross one. */
    adaln_single("main", scaled, TOKENS, 9, main_out, embedded);

    float *expected_main = load_f32("reference.main", TOKENS * 9 * DIM);
    float *expected_embedded = load_f32("reference.embedded", TOKENS * DIM);
    float *expected_prompt = load_f32("reference.prompt", 2 * DIM);
    float *expected_ss = load_f32("reference.cross_ss", TOKENS * 4 * DIM);
    float *expected_gate = load_f32("reference.cross_gate", DIM);

    /* 5e-05 is the F32 floor described above, not a fitted bound: the trig at
     * a thousand radians sets it, and a 256-wide dot product on top. The gate
     * lands three orders below because its argument is the raw sigma, which
     * is the clearest evidence that the floor is the argument's size. Every
     * structural mistake checked in the commit message clears this by at
     * least three orders. */
    compare("nine-slot modulation", main_out, expected_main,
            TOKENS * 9 * DIM, 5e-5f);
    compare("embedded timestep", embedded, expected_embedded,
            TOKENS * DIM, 5e-5f);
    compare("prompt modulation", prompt_out, expected_prompt, 2 * DIM, 5e-5f);
    compare("cross scale and shift", cross_ss, expected_ss,
            TOKENS * 4 * DIM, 5e-5f);
    compare("cross gate", gate_out, expected_gate, DIM, 5e-5f);

    free(sigmas); free(scalar); free(cross);
    free(expected_main); free(expected_embedded); free(expected_prompt);
    free(expected_ss); free(expected_gate);
    h3_weight_store_free(store);

    if (failures) {
        fprintf(stderr, "\n%d modulation comparisons failed\n", failures);
        return 1;
    }
    printf("ok: all four AdaLN-single widths reproduce LTX-2.5's own output, "
           "including the gate's unscaled argument\n");
    return 0;
}
