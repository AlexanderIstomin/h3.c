/* Resolve LTX-2.5's released DiT: every tensor, every ConvRot group, and what
 * a block actually costs.
 *
 * This is the step that goes before the driver, for the reason the text path
 * proved out: a wrong ConvRot group is silent and no shape check catches it,
 * and 48 blocks of dual-stream wiring is far too much to debug from a bad
 * result. Resolving the whole checkpoint without reading a payload costs
 * milliseconds and turns every structural assumption into an assertion.
 *
 * It was worth doing before writing anything. Reading the checkpoint against
 * the notes we had corrected three things:
 *
 *   - a block has **six** attention modules, not eight. attn1 and attn2 per
 *     stream, plus one cross-modal attention each way. The cost model this
 *     port was approved on counted eight, so a block is cheaper than planned;
 *   - the cross-modal modulation table is **five** rows, not four plus a
 *     separate one. Rows 0-3 are the two scale/shift pairs and row 4 is the
 *     gate, but their timestep halves come from two different AdaLN modules
 *     with four and one slots. Table width and timestep width disagree on
 *     purpose, which is exactly the sort of thing that indexes cleanly and
 *     silently off the end; and
 *   - the video feed-forward carries **no bias** while the audio one does.
 *     `ff_bias: false` in the config reads as global and is not.
 *
 * One more trap, in the names rather than the shapes: `scale_shift_table` at
 * the top level is the output head's [2, 4096], while `scale_shift_table`
 * inside a block is the modulation's [9, 4096]. Same name, different tensor,
 * distinguished only by prefix.
 *
 * usage: h3_real_ltx_dit_test DIT.safetensors [VIDEO_ROWS AUDIO_ROWS] */

#include "h3_safetensors.h"
#include "h3_weights.h"

#include <inttypes.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    VIDEO_DIM = 4096,
    AUDIO_DIM = 2048,
    VIDEO_FF = VIDEO_DIM * 4,
    AUDIO_FF = AUDIO_DIM * 4,
    HEADS = 32,
    VIDEO_HEAD = VIDEO_DIM / HEADS,
    AUDIO_HEAD = AUDIO_DIM / HEADS,
    BLOCKS = 48,
    LATENT = 128,
    TIMESTEP_FEATURES = 256,
    ADA_SLOTS = 9,
    PROMPT_SLOTS = 2,
    CROSS_SLOTS = 4,
    CROSS_TABLE_SLOTS = 5,
    GATE_SLOTS = 1,
    HEAD_SLOTS = 2,
    CONVROT_GROUP = 256,
    TEXT_SPAN = 128
};

static h3_weight_store *store = NULL;
static uint64_t quantized_parameters = 0;
static uint64_t dense_parameters = 0;

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

/* --------------------------------------------------------------- resolving */

static const h3_st_tensor *expect(const char *name, h3_dtype dtype,
                                  int ndim, uint64_t first, uint64_t second) {
    const h3_st_tensor *tensor = h3_weight_find(store, name, NULL);
    if (!tensor) fail("the checkpoint has no tensor %s", name);
    if (tensor->dtype != dtype)
        fail("%s is %s, expected %s", name, h3_dtype_name(tensor->dtype),
             h3_dtype_name(dtype));
    if (tensor->ndim != ndim)
        fail("%s has %d dimensions, expected %d", name, tensor->ndim, ndim);
    if (tensor->shape[0] != first || (ndim > 1 && tensor->shape[1] != second))
        fail("%s is [%" PRIu64 ", %" PRIu64 "], expected [%" PRIu64 ", %"
             PRIu64 "]", name, tensor->shape[0],
             ndim > 1 ? tensor->shape[1] : 0, first, second);
    if (dtype != H3_DTYPE_I8)
        dense_parameters += (uint64_t)h3_st_tensor_elements(tensor);
    return tensor;
}

static int present(const char *name) {
    return h3_weight_find(store, name, NULL) != NULL;
}

/* A quantized projection: the matrix, its per-output-channel scales, its
 * marker, and optionally a bias. Returns the ConvRot group so callers can
 * insist that projections sharing an activation agree on it. */
static uint32_t projection(const char *prefix, const char *suffix,
                           uint64_t out_dim, uint64_t in_dim, int bias) {
    char name[256];
    snprintf(name, sizeof(name), "%s%s.weight", prefix, suffix);
    const h3_st_tensor *tensor = expect(name, H3_DTYPE_I8, 2, out_dim, in_dim);
    quantized_parameters += (uint64_t)h3_st_tensor_elements(tensor);
    char scales[256];
    snprintf(scales, sizeof(scales), "%s%s.weight_scale", prefix, suffix);
    expect(scales, H3_DTYPE_F32, 2, out_dim, 1);
    uint32_t group = 0;
    char error[512];
    if (!h3_weight_i8_linear_convrot_group(store, name, &group,
                                           error, sizeof(error)))
        fail("cannot read the ConvRot marker for %s: %s", name, error);
    if (group != CONVROT_GROUP)
        fail("%s has ConvRot group %u, expected %d", name, group, CONVROT_GROUP);
    char bias_name[256];
    snprintf(bias_name, sizeof(bias_name), "%s%s.bias", prefix, suffix);
    if (bias) expect(bias_name, H3_DTYPE_BF16, 1, out_dim, 0);
    else if (present(bias_name)) fail("%s has a bias and should not", bias_name);
    return group;
}

/* -------------------------------------------------------------- attentions */

typedef struct {
    const char *name;
    uint32_t query_in;   /* what to_q reads */
    uint32_t kv_in;      /* what to_k and to_v read */
    uint32_t inner;      /* the attention's working width */
    uint32_t out;        /* what to_out writes */
    uint32_t gate_in;    /* what to_gate_logits reads: the attention's input */
} attention_shape;

/* Six, and the two cross-modal ones are not symmetric with the rest: each
 * takes its query from one stream and its keys from the other, so its input
 * width, working width and output width are three different numbers. */
static const attention_shape ATTENTIONS[] = {
    {"attn1",       VIDEO_DIM, VIDEO_DIM, VIDEO_DIM, VIDEO_DIM, VIDEO_DIM},
    {"attn2",       VIDEO_DIM, VIDEO_DIM, VIDEO_DIM, VIDEO_DIM, VIDEO_DIM},
    {"audio_attn1", AUDIO_DIM, AUDIO_DIM, AUDIO_DIM, AUDIO_DIM, AUDIO_DIM},
    {"audio_attn2", AUDIO_DIM, AUDIO_DIM, AUDIO_DIM, AUDIO_DIM, AUDIO_DIM},
    {"audio_to_video_attn",
                    VIDEO_DIM, AUDIO_DIM, AUDIO_DIM, VIDEO_DIM, VIDEO_DIM},
    {"video_to_audio_attn",
                    AUDIO_DIM, VIDEO_DIM, AUDIO_DIM, AUDIO_DIM, AUDIO_DIM}
};

enum { ATTENTION_COUNT = (int)(sizeof(ATTENTIONS) / sizeof(ATTENTIONS[0])) };

/* attn2 and the cross-modal pair read their keys from somewhere other than
 * their queries, so only k and v have to agree there. attn1 and audio_attn1
 * are true self-attention and all three share one activation. */
static int is_self_attention(int index) { return index == 0 || index == 2; }

static void resolve_attention(const char *block, const attention_shape *shape) {
    char prefix[256];
    snprintf(prefix, sizeof(prefix), "%s%s.", block, shape->name);
    const uint32_t q = projection(prefix, "to_q", shape->inner,
                                  shape->query_in, 1);
    const uint32_t k = projection(prefix, "to_k", shape->inner, shape->kv_in, 1);
    const uint32_t v = projection(prefix, "to_v", shape->inner, shape->kv_in, 1);
    projection(prefix, "to_out.0", shape->out, shape->inner, 1);
    if (k != v || (is_self_attention((int)(shape - ATTENTIONS)) && q != k)) {
        fail("%s ConvRot groups disagree across a shared activation: "
             "q %u, k %u, v %u", prefix, q, k, v);
    }
    char name[256];
    snprintf(name, sizeof(name), "%sq_norm.weight", prefix);
    expect(name, H3_DTYPE_BF16, 1, shape->inner, 0);
    snprintf(name, sizeof(name), "%sk_norm.weight", prefix);
    expect(name, H3_DTYPE_BF16, 1, shape->inner, 0);
    /* One logit per head from the attention's own input, dense so it never
     * sees a rotated activation. */
    snprintf(name, sizeof(name), "%sto_gate_logits.weight", prefix);
    expect(name, H3_DTYPE_BF16, 2, HEADS, shape->gate_in);
    snprintf(name, sizeof(name), "%sto_gate_logits.bias", prefix);
    expect(name, H3_DTYPE_BF16, 1, HEADS, 0);
}

/* ------------------------------------------------------------------ AdaLN */

typedef struct {
    const char *name;
    uint32_t width;
    uint32_t slots;
} adaln_shape;

/* Eight separate modules, each carrying its own two-layer timestep embedder
 * from 256 sinusoidal features. They are not interchangeable and their slot
 * counts differ, which is the whole reason to list them rather than derive
 * them. */
static const adaln_shape ADALN[] = {
    {"adaln_single",                         VIDEO_DIM, ADA_SLOTS},
    {"audio_adaln_single",                   AUDIO_DIM, ADA_SLOTS},
    {"prompt_adaln_single",                  VIDEO_DIM, PROMPT_SLOTS},
    {"audio_prompt_adaln_single",            AUDIO_DIM, PROMPT_SLOTS},
    {"av_ca_video_scale_shift_adaln_single", VIDEO_DIM, CROSS_SLOTS},
    {"av_ca_audio_scale_shift_adaln_single", AUDIO_DIM, CROSS_SLOTS},
    {"av_ca_a2v_gate_adaln_single",          VIDEO_DIM, GATE_SLOTS},
    {"av_ca_v2a_gate_adaln_single",          AUDIO_DIM, GATE_SLOTS}
};

enum { ADALN_COUNT = (int)(sizeof(ADALN) / sizeof(ADALN[0])) };

static void resolve_adaln(const adaln_shape *shape) {
    char name[256];
    const char *root = "model.diffusion_model.";
    snprintf(name, sizeof(name),
             "%s%s.emb.timestep_embedder.linear_1.weight", root, shape->name);
    expect(name, H3_DTYPE_BF16, 2, shape->width, TIMESTEP_FEATURES);
    snprintf(name, sizeof(name),
             "%s%s.emb.timestep_embedder.linear_1.bias", root, shape->name);
    expect(name, H3_DTYPE_BF16, 1, shape->width, 0);
    snprintf(name, sizeof(name),
             "%s%s.emb.timestep_embedder.linear_2.weight", root, shape->name);
    expect(name, H3_DTYPE_BF16, 2, shape->width, shape->width);
    snprintf(name, sizeof(name),
             "%s%s.emb.timestep_embedder.linear_2.bias", root, shape->name);
    expect(name, H3_DTYPE_BF16, 1, shape->width, 0);
    snprintf(name, sizeof(name), "%s%s.linear.weight", root, shape->name);
    expect(name, H3_DTYPE_BF16, 2,
           (uint64_t)shape->slots * shape->width, shape->width);
    snprintf(name, sizeof(name), "%s%s.linear.bias", root, shape->name);
    expect(name, H3_DTYPE_BF16, 1, (uint64_t)shape->slots * shape->width, 0);
}

/* ------------------------------------------------------------------- cost */

/* Every GEMM a block runs, so the cost model rests on the checkpoint rather
 * than on a count of modules. Returns multiply-accumulates. */
static double block_macs(double video, double audio, double text) {
    const double VIDEO_DIM = 4096.0, AUDIO_DIM = 2048.0;
    const double VIDEO_FF = VIDEO_DIM * 4.0, AUDIO_FF = AUDIO_DIM * 4.0;
    double total = 0.0;
    /* Video self-attention: four projections at video rows. */
    total += 4.0 * video * VIDEO_DIM * VIDEO_DIM;
    /* Video cross-attention to the text context: the query and the output run
     * at video rows, the keys and values at the context's own length. */
    total += 2.0 * video * VIDEO_DIM * VIDEO_DIM;
    total += 2.0 * text * VIDEO_DIM * VIDEO_DIM;
    /* Video feed-forward, plain GELU at 4x. */
    total += 2.0 * video * VIDEO_DIM * VIDEO_FF;
    /* The same three for audio. */
    total += 4.0 * audio * AUDIO_DIM * AUDIO_DIM;
    total += 2.0 * audio * AUDIO_DIM * AUDIO_DIM;
    total += 2.0 * text * AUDIO_DIM * AUDIO_DIM;
    total += 2.0 * audio * AUDIO_DIM * AUDIO_FF;
    /* Audio into video: the query and output are at video rows and bridge the
     * two widths; the keys and values are at audio rows and stay narrow. */
    total += video * VIDEO_DIM * AUDIO_DIM;
    total += 2.0 * audio * AUDIO_DIM * AUDIO_DIM;
    total += video * AUDIO_DIM * VIDEO_DIM;
    /* Video into audio, the mirror: keys and values read the video stream at
     * video rows, which is the expensive half of this direction. */
    total += audio * AUDIO_DIM * AUDIO_DIM;
    total += 2.0 * video * VIDEO_DIM * AUDIO_DIM;
    total += audio * AUDIO_DIM * AUDIO_DIM;
    return total;
}

/* ------------------------------------------------------------------- main */

int main(int argc, char **argv) {
    if (argc < 2 || argc > 4) {
        fprintf(stderr, "usage: %s DIT.safetensors [VIDEO_ROWS AUDIO_ROWS]\n",
                argv[0]);
        return 2;
    }
    char error[512];
    store = h3_weight_store_open(argv[1], error, sizeof(error));
    if (!store) fail("cannot open the DiT checkpoint: %s", error);

    printf("Resolving LTX-2.5's released DiT:\n");

    /* ------------------------------------------------------- the top level */

    const char *root = "model.diffusion_model.";
    char name[256];
    snprintf(name, sizeof(name), "%spatchify_proj.weight", root);
    expect(name, H3_DTYPE_BF16, 2, VIDEO_DIM, LATENT);
    snprintf(name, sizeof(name), "%spatchify_proj.bias", root);
    expect(name, H3_DTYPE_BF16, 1, VIDEO_DIM, 0);
    snprintf(name, sizeof(name), "%saudio_patchify_proj.weight", root);
    expect(name, H3_DTYPE_BF16, 2, AUDIO_DIM, LATENT);
    snprintf(name, sizeof(name), "%saudio_patchify_proj.bias", root);
    expect(name, H3_DTYPE_BF16, 1, AUDIO_DIM, 0);
    snprintf(name, sizeof(name), "%sproj_out.weight", root);
    expect(name, H3_DTYPE_BF16, 2, LATENT, VIDEO_DIM);
    snprintf(name, sizeof(name), "%sproj_out.bias", root);
    expect(name, H3_DTYPE_BF16, 1, LATENT, 0);
    snprintf(name, sizeof(name), "%saudio_proj_out.weight", root);
    expect(name, H3_DTYPE_BF16, 2, LATENT, AUDIO_DIM);
    snprintf(name, sizeof(name), "%saudio_proj_out.bias", root);
    expect(name, H3_DTYPE_BF16, 1, LATENT, 0);
    /* The output head's tables, which share a name with the blocks' nine-row
     * modulation tables and are a different thing. */
    snprintf(name, sizeof(name), "%sscale_shift_table", root);
    expect(name, H3_DTYPE_F32, 2, HEAD_SLOTS, VIDEO_DIM);
    snprintf(name, sizeof(name), "%saudio_scale_shift_table", root);
    expect(name, H3_DTYPE_F32, 2, HEAD_SLOTS, AUDIO_DIM);
    /* LTX's own comment calls this "zero-initialized, a no-op forever". The
     * released tensor has a maximum absolute value of 0.0034, so it is
     * trained and the comment is stale. */
    snprintf(name, sizeof(name), "%skeyframes_abs_pos_embedding", root);
    expect(name, H3_DTYPE_BF16, 2, 1, VIDEO_DIM);
    printf("  ok  patchify, output head and keyframe embedding\n");

    for (int index = 0; index < ADALN_COUNT; index++) resolve_adaln(&ADALN[index]);
    printf("  ok  %d AdaLN-single modules, slots", ADALN_COUNT);
    for (int index = 0; index < ADALN_COUNT; index++)
        printf("%s %u", index ? "," : "", ADALN[index].slots);
    printf(", each from %d sinusoidal features\n", TIMESTEP_FEATURES);

    /* ----------------------------------------------------------- the blocks */

    for (int index = 0; index < BLOCKS; index++) {
        char block[128];
        snprintf(block, sizeof(block), "%stransformer_blocks.%d.", root, index);
        for (int which = 0; which < ATTENTION_COUNT; which++)
            resolve_attention(block, &ATTENTIONS[which]);
        /* ff_bias is false for video and true for audio, which reads as a
         * single global switch in the config and is not. */
        projection(block, "ff.net.0.proj", VIDEO_FF, VIDEO_DIM, 0);
        projection(block, "ff.net.2", VIDEO_DIM, VIDEO_FF, 0);
        projection(block, "audio_ff.net.0.proj", AUDIO_FF, AUDIO_DIM, 1);
        projection(block, "audio_ff.net.2", AUDIO_DIM, AUDIO_FF, 1);
        char table[256];
        snprintf(table, sizeof(table), "%sscale_shift_table", block);
        expect(table, H3_DTYPE_F32, 2, ADA_SLOTS, VIDEO_DIM);
        snprintf(table, sizeof(table), "%saudio_scale_shift_table", block);
        expect(table, H3_DTYPE_F32, 2, ADA_SLOTS, AUDIO_DIM);
        snprintf(table, sizeof(table), "%sprompt_scale_shift_table", block);
        expect(table, H3_DTYPE_F32, 2, PROMPT_SLOTS, VIDEO_DIM);
        snprintf(table, sizeof(table), "%saudio_prompt_scale_shift_table", block);
        expect(table, H3_DTYPE_F32, 2, PROMPT_SLOTS, AUDIO_DIM);
        /* Five rows, where the timestep halves arrive as four and one from two
         * different modules. */
        snprintf(table, sizeof(table), "%sscale_shift_table_a2v_ca_video", block);
        expect(table, H3_DTYPE_F32, 2, CROSS_TABLE_SLOTS, VIDEO_DIM);
        snprintf(table, sizeof(table), "%sscale_shift_table_a2v_ca_audio", block);
        expect(table, H3_DTYPE_F32, 2, CROSS_TABLE_SLOTS, AUDIO_DIM);
    }
    printf("  ok  %d blocks x (%d attentions + 2 feed-forwards + 6 modulation "
           "tables), every projection ConvRot at group %d\n",
           BLOCKS, ATTENTION_COUNT, CONVROT_GROUP);

    /* Blocks and top level only. The connector's 450 tensors live in this
     * same file and are resolved by test_real_ltx_connector.c instead, which
     * is why this lands near 19B rather than the name's 22B. */
    const uint64_t total = quantized_parameters + dense_parameters;
    printf("  ok  %.2fB parameters outside the connector: %.2fB quantized, "
           "%.2fB dense\n", (double)total / 1e9,
           (double)quantized_parameters / 1e9, (double)dense_parameters / 1e9);

    /* -------------------------------------------------------------- the cost */

    const double video = argc > 2 ? atof(argv[2]) : 4096.0;
    const double audio = argc > 3 ? atof(argv[3]) : 256.0;
    const double macs = block_macs(video, audio, (double)TEXT_SPAN);
    printf("\nAt %.0f video rows, %.0f audio rows and a %d-token context:\n",
           video, audio, TEXT_SPAN);
    printf("  a block is %.1f GMAC, the tower %.1f TMAC per denoising pass\n",
           macs / 1e9, macs * (double)BLOCKS / 1e12);
    /* 3.4-3.6 TFLOP/s is what the retuned 64x40 tile measured on these exact
     * shapes; the shipped 8x8 one runs about 7x slower. Both are quoted
     * because which one is in the build decides whether the lane is viable. */
    printf("  at 3.5 TFLOP/s that is %.1f s a pass, at the shipped tile's "
           "0.5 TFLOP/s it is %.0f s\n",
           2.0 * macs * (double)BLOCKS / 3.5e12,
           2.0 * macs * (double)BLOCKS / 0.5e12);

    h3_weight_store_free(store);
    printf("\nok: the released DiT resolves completely -- %d blocks, %d "
           "attentions each, one ConvRot group throughout\n",
           BLOCKS, ATTENTION_COUNT);
    return 0;
}
