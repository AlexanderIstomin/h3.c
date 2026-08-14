/* Encodes several prompts through the text tower and compares the resulting
 * embeddings. If a rain prompt and a talking-head prompt land in nearly the
 * same place, the conditioning cannot be steering generation and the fault is
 * upstream of the DiT; if they separate cleanly, text encoding works and a
 * wrong scene comes from somewhere later. Encoding is seconds, so this answers
 * the question without spending a generation on it. */
#include "h3_text_encoder.h"
#include "h3_tokenizer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float bf16_to_f32(uint16_t value) {
    union { uint32_t bits; float f; } cast;
    cast.bits = (uint32_t)value << 16;
    return cast.f;
}

/* One vector per prompt: mean over token rows, so prompts of different
 * lengths stay comparable. */
static double *mean_row(const h3_text_embedding *text) {
    double *mean = calloc(text->width, sizeof(*mean));
    if (!mean) return NULL;
    for (size_t token = 0; token < text->tokens; token++)
        for (size_t channel = 0; channel < text->width; channel++)
            mean[channel] +=
                (double)bf16_to_f32(text->values[token * text->width + channel]);
    for (size_t channel = 0; channel < text->width; channel++)
        mean[channel] /= (double)text->tokens;
    return mean;
}

static double cosine(const double *a, const double *b, size_t width) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t index = 0; index < width; index++) {
        dot += a[index] * b[index];
        na += a[index] * a[index];
        nb += b[index] * b[index];
    }
    if (na <= 0.0 || nb <= 0.0) return 0.0;
    return dot / (sqrt(na) * sqrt(nb));
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s TOKENIZER_JSON TEXT_ENCODER\n", argv[0]);
        return 2;
    }
    const char *prompts[] = {
        "A dense pine forest at night during a heavy rainstorm, rain streaming "
        "off the branches and thunder rolling in the distance.",
        "A man in a suit sits in front of a bookshelf and talks directly to "
        "the camera in a warmly lit room.",
        "A fluffy cat sits on a windowsill and turns its head slowly.",
    };
    const char *labels[] = {"rain-forest", "talking-head", "cat"};
    enum { PROMPTS = 3 };

    char error[512];
    h3_tokenizer *tokenizer = h3_tokenizer_load(argv[1], error, sizeof(error));
    if (!tokenizer) {
        fprintf(stderr, "tokenizer load failed: %s\n", error);
        return 1;
    }

    double *means[PROMPTS];
    size_t width = 0;
    for (int index = 0; index < PROMPTS; index++) {
        uint32_t *ids = NULL;
        size_t count = 0;
        if (!h3_tokenizer_encode(tokenizer, prompts[index], 1, &ids, &count,
                                 error, sizeof(error))) {
            fprintf(stderr, "tokenize failed: %s\n", error);
            return 1;
        }
        h3_text_embedding text;
        memset(&text, 0, sizeof(text));
        if (!h3_text_encode_bf16(argv[2], "h3_shaders.metal", ids, count,
                                 NULL, NULL, &text, error, sizeof(error))) {
            fprintf(stderr, "encode failed: %s\n", error);
            return 1;
        }
        printf("%-13s %zu tokens x %zu\n", labels[index], text.tokens,
               text.width);
        width = text.width;
        means[index] = mean_row(&text);
        h3_text_embedding_free(&text);
        h3_tokenizer_ids_free(ids);
        if (!means[index]) return 1;
    }
    h3_tokenizer_free(tokenizer);

    /* Raw cosine on LLM hidden states runs high for any two inputs because
     * the representation space is anisotropic, so it cannot answer this on its
     * own. Relative L2 distance and cosine after removing the direction all
     * prompts share are what show whether the prompt actually moves the
     * embedding. */
    double *centre = calloc(width, sizeof(*centre));
    if (!centre) return 1;
    for (int index = 0; index < PROMPTS; index++)
        for (size_t channel = 0; channel < width; channel++)
            centre[channel] += means[index][channel] / (double)PROMPTS;
    double *centred[PROMPTS];
    for (int index = 0; index < PROMPTS; index++) {
        centred[index] = malloc(width * sizeof(**centred));
        if (!centred[index]) return 1;
        for (size_t channel = 0; channel < width; channel++)
            centred[index][channel] = means[index][channel] - centre[channel];
    }

    printf("\n%-13s %-13s %8s %10s %10s\n", "prompt A", "prompt B", "cosine",
           "rel L2", "centred");
    for (int a = 0; a < PROMPTS; a++)
        for (int b = a + 1; b < PROMPTS; b++) {
            double norm_a = 0.0, norm_b = 0.0, diff = 0.0;
            for (size_t channel = 0; channel < width; channel++) {
                double delta = means[a][channel] - means[b][channel];
                diff += delta * delta;
                norm_a += means[a][channel] * means[a][channel];
                norm_b += means[b][channel] * means[b][channel];
            }
            double relative = sqrt(diff) / (0.5 * (sqrt(norm_a) + sqrt(norm_b)));
            printf("%-13s %-13s %8.4f %9.1f%% %10.4f\n", labels[a], labels[b],
                   cosine(means[a], means[b], width), relative * 100.0,
                   cosine(centred[a], centred[b], width));
        }
    for (int index = 0; index < PROMPTS; index++) {
        free(means[index]);
        free(centred[index]);
    }
    free(centre);
    printf("\nA tower returning near-constant output would show a relative L2"
           " near zero.\n");
    return 0;
}
