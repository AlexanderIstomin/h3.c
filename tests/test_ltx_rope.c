/* LTX-2.5's rotary tables, against its own precompute_freqs_cis.
 *
 * The video stream's positions are three-dimensional -- frame, row, column --
 * so the frequency grid is divided three ways and the leftovers are padded.
 * The connector's tables are the same construction at one dimension, already
 * checked in test_ltx_conditioning.c; this is the shape the DiT needs.
 *
 * Five things here are not what a reader would assume, and every one of them
 * yields a well-formed table rather than an error:
 *
 *   - positions are patch *midpoints*. `use_middle_indices_grid` defaults to
 *     true, and the grid carries a start and an end per token, because a
 *     latent covers eight frames and thirty-two pixels after compression.
 *     Taking the start instead shifts every token by half a patch;
 *   - the grid is axis-major, [dimension, token, extent], not token-major;
 *   - frequencies run geometrically from 1 to theta and are scaled by pi/2,
 *     rather than being the usual inverse powers;
 *   - positions are mapped onto [-1, 1] by max_pos before use, not left as
 *     indices; and
 *   - the flattened frequency axis is frequency-major with the three position
 *     dimensions innermost -- slot `f * 3 + axis` -- and what does not divide
 *     evenly is padded at the *front* with an identity rotation, cosine one
 *     and sine zero. At the released width that is two channels of head zero.
 *
 * The fixture also carries the float64 frequency grid, which
 * `double_precision_rope` selects and which the released configuration leaves
 * off, so the cost of the choice is visible rather than assumed.
 *
 * usage: h3_ltx_rope_test FIXTURE.safetensors */

#include "h3_safetensors.h"
#include "h3_weights.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    DIM = 64, HEADS = 4,
    HEAD_DIM = DIM / HEADS,
    HALF = DIM / 2,                 /* channels a token, across all heads */
    HEAD_HALF = HEAD_DIM / 2,
    AXES = 3,
    TOKENS = 12,
    /* One frequency per axis per group; what is left over is padded. */
    FREQUENCIES = DIM / (2 * AXES),
    PAD = HALF - AXES * FREQUENCIES
};

#define THETA 10000.0

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

/* Geometric from 1 to theta over `FREQUENCIES` points, times pi/2. In F32 to
 * match the released configuration; `double_precision_rope` would compute the
 * same expression in F64, and the fixture carries both so the gap is checked
 * rather than guessed. */
static void frequencies(float *out) {
    for (size_t index = 0; index < FREQUENCIES; index++) {
        float span = FREQUENCIES > 1
            ? (float)index / (float)(FREQUENCIES - 1) : 0.0f;
        out[index] = powf((float)THETA, span) * (float)M_PI / 2.0f;
    }
}

static void frequencies_f64(float *out) {
    for (size_t index = 0; index < FREQUENCIES; index++) {
        double span = FREQUENCIES > 1
            ? (double)index / (double)(FREQUENCIES - 1) : 0.0;
        out[index] = (float)(pow(THETA, span) * M_PI / 2.0);
    }
}

/* `grid` is [axis][token][start, end]. The tables come out as
 * [head][token][head_dim / 2], which is the layout the engine's rotary kernel
 * reads with a per-head stride. */
static void rope_tables(float *cos_table, float *sin_table, const float *grid,
                        const float *freqs, const int *max_pos, int middle) {
    for (size_t token = 0; token < TOKENS; token++) {
        float angles[HALF];
        /* The reference pads with a literal cosine one and sine zero, which is
         * the same thing as a zero angle, so the padded slots are left here at
         * zero and fall through the ordinary path below. Stating it twice --
         * a zero angle and a branch on `slot < PAD` -- would be redundant, and
         * measurably so: the two forms are indistinguishable. */
        for (size_t pad = 0; pad < PAD; pad++) angles[pad] = 0.0f;
        for (size_t axis = 0; axis < AXES; axis++) {
            const float *extent = grid + (axis * TOKENS + token) * 2;
            float position = middle ? (extent[0] + extent[1]) * 0.5f : extent[0];
            float fractional = position / (float)max_pos[axis];
            for (size_t index = 0; index < FREQUENCIES; index++)
                /* Frequency-major, the three axes innermost, after the pad. */
                angles[PAD + index * AXES + axis] =
                    freqs[index] * (fractional * 2.0f - 1.0f);
        }
        for (size_t slot = 0; slot < HALF; slot++) {
            size_t head = slot / HEAD_HALF;
            size_t within = slot % HEAD_HALF;
            size_t at = (head * TOKENS + token) * HEAD_HALF + within;
            cos_table[at] = cosf(angles[slot]);
            sin_table[at] = sinf(angles[slot]);
        }
    }
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

    printf("LTX-2.5 rotary tables against its own implementation "
           "(%d frequencies an axis, %d padded):\n", FREQUENCIES, PAD);

    const int max_pos[AXES] = {20, 2048, 2048};
    float *grid = load_f32("input.grid", (size_t)AXES * TOKENS * 2);

    float freqs[FREQUENCIES];
    frequencies(freqs);
    float *expected_freqs = load_f32("reference.indices", FREQUENCIES);
    compare("frequency grid", freqs, expected_freqs, FREQUENCIES, 1e-4f);

    size_t table = (size_t)HEADS * TOKENS * HEAD_HALF;
    float *cos_table = malloc(table * sizeof(*cos_table));
    float *sin_table = malloc(table * sizeof(*sin_table));
    require(cos_table && sin_table, "cannot allocate the rotary tables");

    rope_tables(cos_table, sin_table, grid, freqs, max_pos, 1);
    float *expected_cos = load_f32("reference.cos_f32", table);
    float *expected_sin = load_f32("reference.sin_f32", table);
    compare("cosine table", cos_table, expected_cos, table, 1e-6f);
    compare("sine table", sin_table, expected_sin, table, 1e-6f);

    /* The corner variant is not what the model runs; it is here so the
     * midpoint claim is measured rather than asserted. */
    float *corner_cos = load_f32("reference.cos_corner", table);
    double drift = 0.0;
    for (size_t index = 0; index < table; index++) {
        double delta = fabs((double)expected_cos[index] - (double)corner_cos[index]);
        if (delta > drift) drift = delta;
    }
    if (drift < 1e-3) {
        fprintf(stderr, "FAIL midpoint and corner positions agree to %.2e, "
                "so this fixture cannot tell them apart\n", drift);
        failures++;
    } else {
        printf("  ok  %-30s midpoints differ from corners by %.2e\n",
               "the midpoint matters", drift);
    }

    /* What double_precision_rope would buy. It is off in the released
     * configuration, so this records the cost of that rather than gating. */
    float f64_freqs[FREQUENCIES];
    frequencies_f64(f64_freqs);
    rope_tables(cos_table, sin_table, grid, f64_freqs, max_pos, 1);
    float *expected_f64 = load_f32("reference.cos_f64", table);
    compare("cosine at F64 frequencies", cos_table, expected_f64, table, 1e-6f);
    double gap = 0.0;
    for (size_t index = 0; index < table; index++) {
        double delta = fabs((double)expected_cos[index] - (double)expected_f64[index]);
        if (delta > gap) gap = delta;
    }
    printf("  --  %-30s F32 and F64 grids differ by %.2e\n",
           "double_precision_rope", gap);

    free(grid); free(expected_freqs); free(cos_table); free(sin_table);
    free(expected_cos); free(expected_sin); free(corner_cos); free(expected_f64);
    h3_weight_store_free(store);

    if (failures) {
        fprintf(stderr, "\n%d rotary comparisons failed\n", failures);
        return 1;
    }
    printf("ok: the three-dimensional rotary tables reproduce LTX-2.5's own, "
           "padding and midpoints included\n");
    return 0;
}
