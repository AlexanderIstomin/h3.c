/* LTX-2.5's sigma schedule and diffusion steps, against its own components.
 *
 * Pure arithmetic, so this is the one stage of the port checkable without any
 * weights at all. Four things here decide what the sampler actually does:
 *
 *   - the shift is *token-dependent*. It interpolates a line anchored at 1024
 *     and 4096 tokens, so the schedule for a 768x512 clip is not the schedule
 *     for a 256x256 one, and a table baked at one resolution is quietly wrong
 *     at every other. Nothing clamps it, so counts past 4096 extrapolate;
 *   - the stretch rescales so the last *non-zero* sigma lands exactly on the
 *     terminal value, 0.1 by default. Without it that sigma is 0.27, and the
 *     final step would leave a quarter of the noise behind. Excluding the
 *     trailing zero from *this* is what matters; excluding it from the shift
 *     above only looks like it does, because 1/0 sends that expression to
 *     infinity and the result to zero unaided. The guard there is defensive,
 *     and mutating it away changes nothing; and
 *   - both steps land on the denoised prediction at the end, and by the same
 *     arithmetic rather than by two conventions. Euler's velocity times dt
 *     cancels the whole residual; the ancestral form short-circuits on a zero
 *     next sigma and returns it directly. Replacing that short-circuit with
 *     the Euler step is measurably a no-op, so it is a numerical nicety and
 *     not a branch the driver has to reproduce.
 *
 * One boundary is a genuine defect upstream rather than a convention, and the
 * driver has to refuse it rather than pass it through: **a one-step schedule
 * is NaN**. With steps at one the only non-zero sigma is 1.0, so the stretch's
 * scale factor is (1 - 1) / 0.9 = 0, and the division that follows is 0 / 0.
 * This file pins that, so if upstream ever fixes it the test says so.
 *
 * usage: h3_ltx_sampler_test FIXTURE.safetensors */

#include "h3_safetensors.h"
#include "h3_weights.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { SIZE = 12, MAX_STEPS = 32 };

#define BASE_SHIFT_ANCHOR 1024.0
#define MAX_SHIFT_ANCHOR 4096.0
#define BASE_SHIFT 0.95
#define MAX_SHIFT 2.05
#define TERMINAL 0.1

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
            fprintf(stderr, "FAIL %-30s produced %f at %zu\n",
                    label, actual[index], index);
            failures++;
            return;
        }
        double delta = fabs((double)actual[index] - (double)expected[index]);
        if (delta > worst) { worst = delta; worst_at = index; }
    }
    if (worst > tolerance) {
        fprintf(stderr, "FAIL %-30s worst |delta| %.3e at %zu (%.8f vs %.8f)\n",
                label, worst, worst_at, actual[worst_at], expected[worst_at]);
        failures++;
    } else {
        printf("  ok  %-30s worst |delta| %.2e\n", label, worst);
    }
}

/* `steps + 1` sigmas from one down to zero, shifted by the token count and
 * then stretched so the last non-zero one lands on the terminal value. */
static void schedule(float *out, int steps, double tokens, int stretch) {
    double slope = (MAX_SHIFT - BASE_SHIFT) / (MAX_SHIFT_ANCHOR - BASE_SHIFT_ANCHOR);
    double intercept = BASE_SHIFT - slope * BASE_SHIFT_ANCHOR;
    double shift = exp(tokens * slope + intercept);

    for (int index = 0; index <= steps; index++) {
        double linear = 1.0 - (double)index / (double)steps;
        /* The trailing zero is a terminator: it is left alone here and
         * excluded from the stretch below. */
        out[index] = linear != 0.0
            ? (float)(shift / (shift + (1.0 / linear - 1.0))) : 0.0f;
    }
    if (!stretch) return;

    int last = -1;
    for (int index = 0; index <= steps; index++)
        if (out[index] != 0.0f) last = index;
    if (last < 0) return;
    double scale = (1.0 - (double)out[last]) / (1.0 - TERMINAL);
    for (int index = 0; index <= steps; index++)
        if (out[index] != 0.0f)
            out[index] = (float)(1.0 - (1.0 - (double)out[index]) / scale);
}

/* x + (x - denoised) / sigma * (sigma_next - sigma), which collapses to the
 * denoised prediction when the next sigma is zero. */
static void euler_step(float *out, const float *sample, const float *denoised,
                       const float *sigmas, int index) {
    float sigma = sigmas[index], next = sigmas[index + 1];
    float dt = next - sigma;
    for (size_t at = 0; at < SIZE; at++) {
        float velocity = (sample[at] - denoised[at]) / sigma;
        out[at] = sample[at] + velocity * dt;
    }
}

/* Deterministic to an intermediate sigma, then renoised back up, rescaling the
 * signal so the transition preserves variance. eta zero is a plain Euler step
 * and needs no noise; eta one is fully ancestral. */
static void ancestral_step(float *out, const float *sample,
                           const float *denoised, const float *noise,
                           const float *sigmas, int index, float eta) {
    float sigma = sigmas[index], next = sigmas[index + 1];
    if (next == 0.0f) {
        memcpy(out, denoised, SIZE * sizeof(*out));
        return;
    }
    float down = next * (1.0f + (next / sigma - 1.0f) * eta);
    float ratio = down / sigma;
    for (size_t at = 0; at < SIZE; at++)
        out[at] = ratio * sample[at] + (1.0f - ratio) * denoised[at];
    if (eta <= 0.0f) return;

    float alpha_next = 1.0f - next, alpha_down = 1.0f - down;
    float inner = next * next -
                  down * down * alpha_next * alpha_next / (alpha_down * alpha_down);
    float renoise = sqrtf(inner < 0.0f ? 0.0f : inner);
    for (size_t at = 0; at < SIZE; at++)
        out[at] = (alpha_next / alpha_down) * out[at] + noise[at] * renoise;
}

static void check_schedule(const char *tag, int steps, double tokens,
                           int stretch) {
    char name[128];
    float mine[MAX_STEPS + 1];
    snprintf(name, sizeof(name), "reference.sigmas_%s", tag);
    float *expected = load_f32(name, (size_t)steps + 1);
    schedule(mine, steps, tokens, stretch);
    snprintf(name, sizeof(name), "schedule, %s", tag);
    compare(name, mine, expected, (size_t)steps + 1, 2e-6f);
    free(expected);
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

    printf("LTX-2.5 sampler against its own implementation:\n");

    check_schedule("small", 8, 1024.0, 1);
    check_schedule("anchor", 20, 4096.0, 1);
    /* Past the upper anchor, where the line extrapolates rather than clamps. */
    check_schedule("large", 20, 13376.0, 1);
    check_schedule("unstretched", 8, 1024.0, 0);

    /* The stretch is what puts the last non-zero sigma on the terminal value.
     * Asserting the gap keeps this from silently becoming a no-op. */
    float stretched[16], plain[16];
    schedule(stretched, 8, 1024.0, 1);
    schedule(plain, 8, 1024.0, 0);
    if (fabsf(stretched[7] - (float)TERMINAL) > 1e-6f) {
        fprintf(stderr, "FAIL the stretch does not reach the terminal value: "
                "%.6f\n", stretched[7]);
        failures++;
    } else if (fabsf(plain[7] - stretched[7]) < 1e-3f) {
        fprintf(stderr, "FAIL stretched and unstretched schedules agree, so "
                "the stretch is not being exercised\n");
        failures++;
    } else {
        printf("  ok  %-30s terminal %.4f, unstretched %.4f\n",
               "the stretch matters", (double)stretched[7], (double)plain[7]);
    }

    /* The one-step schedule is NaN upstream, and reproducing it faithfully
     * means reproducing that too. Pinned so a future fix is visible here. */
    float single[2];
    schedule(single, 1, 1024.0, 1);
    float *reference_single = load_f32("reference.sigmas_tiny", 2);
    if (isfinite(single[0]) != isfinite(reference_single[0])) {
        fprintf(stderr, "FAIL the one-step schedule disagrees with the "
                "reference on being finite\n");
        failures++;
    } else if (isfinite(single[0])) {
        printf("  --  %-30s upstream now yields %.4f; the guard below can go\n",
               "one step is no longer NaN", (double)single[0]);
    } else {
        printf("  ok  %-30s NaN here and upstream, so the driver must "
               "refuse it\n", "one step");
    }
    free(reference_single);

    /* The steps, over a four-step schedule at 2048 tokens. */
    float *sample = load_f32("input.sample", SIZE);
    float *denoised = load_f32("input.denoised", SIZE);
    float *noise = load_f32("input.noise", SIZE);
    float *sigmas = load_f32("input.step_sigmas", 5);
    float out[SIZE];
    char name[128];

    for (int index = 0; index < 4; index++) {
        euler_step(out, sample, denoised, sigmas, index);
        snprintf(name, sizeof(name), "reference.euler_%d", index);
        float *expected = load_f32(name, SIZE);
        snprintf(name, sizeof(name), "Euler step %d", index);
        compare(name, out, expected, SIZE, 2e-6f);
        free(expected);
    }

    const struct { const char *tag; float eta; } etas[] = {
        {"eta0", 0.0f}, {"eta1", 1.0f}, {"eta05", 0.5f}
    };
    for (size_t which = 0; which < 3; which++) {
        for (int index = 0; index < 4; index++) {
            ancestral_step(out, sample, denoised, noise, sigmas, index,
                           etas[which].eta);
            snprintf(name, sizeof(name), "reference.ancestral_%s_%d",
                     etas[which].tag, index);
            float *expected = load_f32(name, SIZE);
            snprintf(name, sizeof(name), "ancestral %s, step %d",
                     etas[which].tag, index);
            compare(name, out, expected, SIZE, 2e-6f);
            free(expected);
        }
    }

    /* eta zero must reduce to the plain Euler step, which is the claim the
     * ancestral form makes about itself. */
    float euler_out[SIZE], ancestral_out[SIZE];
    euler_step(euler_out, sample, denoised, sigmas, 1);
    ancestral_step(ancestral_out, sample, denoised, noise, sigmas, 1, 0.0f);
    compare("eta zero is a plain Euler step", ancestral_out, euler_out,
            SIZE, 2e-6f);

    free(sample); free(denoised); free(noise); free(sigmas);
    h3_weight_store_free(store);

    if (failures) {
        fprintf(stderr, "\n%d sampler comparisons failed\n", failures);
        return 1;
    }
    printf("ok: the schedule and both step forms reproduce LTX-2.5's own, "
           "token-dependent shift and terminal stretch included\n");
    return 0;
}
