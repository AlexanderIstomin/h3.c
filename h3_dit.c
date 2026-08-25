#include "h3_dit.h"

#include "h3_dit_schedule.h"
#include "h3_weights.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sysctl.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

/* Casting three operands up and the result back costs time linear in the
 * rows; attention itself is quadratic in them, so the trade turns over as the
 * sequence grows. Measured at H3's 56 heads of 128: f32 wins by 1.87x at 1625
 * rows, 1.76x at 5095 and 1.72x at 8192. The threshold is where Z-Image put
 * it, below which the casts stop paying for themselves. */
#define H3_DIT_F32_ATTENTION_ROWS 512u
#define H3_DIT_SOL_ATTENTION_ROWS 4096u

enum {
    TEXT_DIM = 5120,
    HIDDEN = 5376,
    HEADS = 56,
    HEAD_DIM = 128,
    INNER = HEADS * HEAD_DIM,
    FFN = 14336,
    VIDEO_CHANNELS = 24,
    VIDEO_PATCH = 96,
    AUDIO_CHANNELS = 32,
    AUDIO_STREAMS = 2,
    ROPE_FREQS = 16,
    ROPE_HALF = 48,
    SLOTS = 6,
    FINAL_SLOTS = 2
};

typedef struct {
    h3_gpu_tensor *norm1;
    h3_gpu_tensor *norm2;
    h3_gpu_tensor *qkv;
    h3_gpu_tensor *qkv_int8;
    h3_gpu_tensor *qkv_scales;
    uint32_t qkv_convrot_group;
    h3_gpu_tensor *q_norm;
    h3_gpu_tensor *k_norm;
    h3_gpu_tensor *out;
    h3_gpu_tensor *out_int8;
    h3_gpu_tensor *out_scales;
    uint32_t out_convrot_group;
    h3_gpu_tensor *fc1;
    h3_gpu_tensor *fc2;
    h3_gpu_tensor *fc1_int8;
    h3_gpu_tensor *fc1_scales;
    uint32_t fc1_convrot_group;
    h3_gpu_tensor *fc2_int8;
    h3_gpu_tensor *fc2_scales;
    uint32_t fc2_convrot_group;
    /* Optional low-rank adapters. A is pre-rotated into the ConvRot
     * activation basis and B carries the requested strength, so the forward
     * pass is two plain BF16 products and an add. */
    h3_gpu_tensor *qkv_lora_a, *qkv_lora_b;
    h3_gpu_tensor *out_lora_a, *out_lora_b;
    h3_gpu_tensor *fc1_lora_a, *fc1_lora_b;
    h3_gpu_tensor *fc2_lora_a, *fc2_lora_b;
    uint32_t qkv_lora_rank;
    uint32_t out_lora_rank;
    uint32_t fc1_lora_rank;
    uint32_t fc2_lora_rank;
} h3_dit_block;

enum {
    STREAM_QKV,
    STREAM_QKV_SCALES,
    STREAM_OUT,
    STREAM_OUT_SCALES,
    STREAM_FC1,
    STREAM_FC1_SCALES,
    STREAM_FC2,
    STREAM_FC2_SCALES,
    STREAM_TENSORS
};

typedef struct {
    const char *path;
    uint64_t file_offset;
    size_t elements;
    unsigned field;
    h3_dtype dtype;
} h3_dit_stream_source;

typedef struct {
    h3_dit_stream_source sources[STREAM_TENSORS];
    unsigned count;
} h3_dit_stream_layer;

struct h3_dit {
    h3_gpu *gpu;
    h3_weight_store *weights;
    h3_weight_store *late_adaln_overlay;
    h3_dit_schedule *schedule;
    int fused_mlp;
    int nax_mlp;
    int int8_mlp;
    int int8_qkv;
    int int8_attention_out;
    int prequantized_int8;
    int keep_bf16_qkv;
    int keep_bf16_attention_out;
    int use_slower_row_major_attention_output;
    int use_slower_unfused_int8_inputs;
    int use_slower_unfused_qkv_rope;
    int use_slower_scalar_qkv_rms;
    int use_slower_uncached_int8_scales;
    int use_slower_dynamic_fc1_k;
    int use_slower_grouped_quantizer;
    int use_int8_row_fc2;
    int sol_attention;
    float sol_attention_tau;
    int ssd_streaming;
    int input_major_transformer;
    int conventional_core_qkv;
    int hybrid_adaln;
    int keep_bf16_mlp;
    int activation_aliases;
    int fused_patch_projection;
    int fused_patch_pack;
    int token_reduction;
    int token_reduction_active;
    unsigned token_reduction_begin;
    unsigned token_reduction_end;
    unsigned token_reduction_early_steps;
    unsigned token_reduction_early_end;
    float token_reduction_scale;
    float spatial_rope_scale;
    int bf16_final;
    unsigned core_reuse_interval;
    unsigned core_forward_count;
    int core_residual_ready;
    /* Block cache: gated steps run only a warm prefix of the blocks and
     * replay the cached tail residual from the last full step. */
    int block_cache;
    int cache_plan_ready;
    uint8_t cache_full_step[H3_MAX_STEPS];
    unsigned cache_warm_last;
    int cache_residual_ready;
    int cache_forward_is_cached;
    int cache_forward_captures;
    unsigned active_block_count;
    uint8_t block_active[H3_DIT_BLOCKS];
    h3_layout layout;
    h3_sigma_schedule sigmas;
    int latent_t;
    int latent_h;
    int latent_w;
    int audio_t;
    uint32_t text_rows;
    uint32_t video_condition_rows;
    uint32_t audio_condition_rows;
    uint32_t audio_rows;
    uint32_t video_rows;
    uint32_t video_total_rows;
    uint32_t audio_total_rows;
    uint32_t audio_target_start;
    uint32_t video_target_start;
    uint32_t sequence;
    uint32_t reduced_sequence;
    uint32_t reduced_video_rows;
    uint32_t token_baseline_rows;
    float *inpaint_video_source;
    size_t inpaint_video_source_elements;
    uint8_t *inpaint_video_generate_rows;
    size_t inpaint_video_generate_count;
    float *inpaint_audio_source;
    size_t inpaint_audio_source_elements;
    uint8_t *inpaint_audio_generate_rows;
    size_t inpaint_audio_generate_count;
    h3_gpu_tensor *refined_text;
    h3_gpu_tensor *rope_cos;
    h3_gpu_tensor *rope_sin;
    h3_gpu_tensor *reduced_rope_cos;
    h3_gpu_tensor *reduced_rope_sin;
    h3_gpu_tensor **row_maps;
    h3_gpu_tensor **reduced_row_maps;
    h3_gpu_tensor **final_audio_maps;
    h3_gpu_tensor **final_video_maps;
    h3_gpu_tensor *video_patch_w;
    h3_gpu_tensor *video_patch_b;
    h3_gpu_tensor *audio_patch_w;
    h3_gpu_tensor *audio_patch_b;
    h3_dit_block blocks[H3_DIT_BLOCKS];
    h3_dit_block stream_slots[2];
    h3_dit_stream_layer stream_layers[H3_DIT_BLOCKS];
    unsigned stream_ready_layer;
    unsigned stream_ready_slot;
    uint64_t stream_bytes;
    double stream_read_seconds;
    double stream_wait_seconds;
    h3_gpu_tensor *final_norm;
    h3_gpu_tensor *final_video_w;
    h3_gpu_tensor *final_video_b;
    h3_gpu_tensor *final_audio_w;
    h3_gpu_tensor *final_audio_b;
    h3_gpu_tensor *video_input;
    h3_gpu_tensor *audio_input;
    h3_gpu_tensor *video_projected_f32;
    h3_gpu_tensor *audio_projected_f32;
    h3_gpu_tensor *video_projected;
    h3_gpu_tensor *audio_projected;
    h3_gpu_tensor *video_projection_map;
    h3_gpu_tensor *audio_projection_map;
    h3_gpu_tensor *hidden;
    h3_gpu_tensor *core_input;
    h3_gpu_tensor *core_residual;
    h3_gpu_tensor *cache_snapshot;
    h3_gpu_tensor *cache_residual;
    h3_gpu_tensor *mod_attention;
    h3_gpu_tensor *qkv;
    h3_gpu_tensor *query;
    h3_gpu_tensor *key;
    h3_gpu_tensor *value;
    h3_gpu_tensor *attention_heads;
    /* Attention runs f32 above H3_DIT_F32_ATTENTION_ROWS; these hold the cast
     * operands. NULL below that size, where the casts cost more than they
     * save and nothing allocates them. */
    h3_gpu_tensor *query32;
    h3_gpu_tensor *key32;
    h3_gpu_tensor *value32;
    h3_gpu_tensor *heads32;
    h3_gpu_tensor *sol_query_centroids;
    h3_gpu_tensor *sol_key_centroids;
    h3_gpu_tensor *sol_value_sums;
    h3_gpu_tensor *sol_key_means;
    h3_gpu_tensor *sol_key_variances;
    h3_gpu_tensor *sol_thresholds;
    h3_gpu_tensor *sol_routes;
    size_t sol_route_count;
    int sol_routes_reported;
    h3_gpu_tensor *attention_output;
    h3_gpu_tensor *token_pool_pairs;
    h3_gpu_tensor *token_baseline_indices;
    h3_gpu_tensor *token_expand_parents;
    h3_gpu_tensor *token_original;
    int token_original_in_qkv;
    size_t token_original_offset;
    size_t token_baseline_offset;
    h3_gpu_tensor *mod_mlp;
    h3_gpu_tensor *fc1;
    h3_gpu_tensor *activated;
    h3_gpu_tensor *mlp_output;
    h3_gpu_tensor *int8_activation;
    h3_gpu_tensor *int8_activation_scales;
    const char *lora_path;
    float lora_strength;
    uint32_t lora_capacity;
    h3_gpu_tensor *lora_hidden;
    h3_gpu_tensor *lora_delta;
    h3_gpu_tensor *final_audio_input;
    h3_gpu_tensor *final_video_input;
    h3_gpu_tensor *final_audio_inverse;
    h3_gpu_tensor *final_video_inverse;
    h3_gpu_tensor *final_audio_norm;
    h3_gpu_tensor *final_video_norm;
    h3_gpu_tensor *final_audio_f32;
    h3_gpu_tensor *final_video_f32;
    h3_gpu_tensor *audio_output;
    h3_gpu_tensor *video_output;
    h3_gpu_tensor *audio_output_bf16;
    h3_gpu_tensor *video_output_bf16;
    h3_gpu_tensor *previous_audio_velocity;
    h3_gpu_tensor *previous_video_velocity;
};

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

/* Which int8 tile the DiT's projections use.
 *
 * The 8x8 tile the engine shipped with re-reads the weight matrix rows/8
 * times; the 64x40 one re-reads it rows/64 times, which on these shapes is
 * measured at 6.1x. Both read the weight [output][input], so this is a
 * kernel swap and nothing else — no re-quantized package, no loader change.
 *
 * H3_DIT_TILE=8x8 restores the old kernel, which is how the two were compared
 * end to end in one build.
 */
/* How much of this machine the f32 attention scratch may take.
 *
 * A sixteenth of physical memory, which is 2 GB on the 32 GB machine this was
 * measured on. That admits it at the sequences where it was shown to pay —
 * 0.58 GB at 5095 rows — and refuses it at the 2.5 GB a 768-canvas
 * five-second clip wants, where 21 GB of weights already has the machine in
 * swap. That configuration completed only with this scratch turned off, and
 * finishing is worth more than the 18% the trade buys. */
static uint64_t h3_dit_memory_budget(void) {
    uint64_t bytes = 0;
    size_t size = sizeof(bytes);
    if (sysctlbyname("hw.memsize", &bytes, &size, NULL, 0) != 0) return 0;
    return bytes / 16u;
}

static int use_retuned_tile(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *value = getenv("H3_DIT_TILE");
        cached = !(value && !strcmp(value, "8x8"));
    }
    return cached;
}

static int dit_int8_linear(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *input,
                           const h3_gpu_tensor *weight,
                           const h3_gpu_tensor *scales,
                           const h3_gpu_tensor *bias, uint32_t rows,
                           uint32_t input_dim, uint32_t output_dim,
                           int input_major) {
    if (input_major)
        return h3_gpu_linear_i8_weight_bf16_square(
            gpu, output, input, weight, scales, bias, rows, input_dim,
            output_dim);
    return use_retuned_tile()
        ? h3_gpu_linear_i8_weight_bf16_square_output_major(
              gpu, output, input, weight, scales, bias, rows, input_dim,
              output_dim)
        : h3_gpu_linear_i8_weight_bf16(gpu, output, input, weight, scales,
                                       bias, rows, input_dim, output_dim);
}

static unsigned command_block_interval(const h3_dit *dit) {
    const char *value = getenv("H3_DIT_COMMAND_BLOCKS");
    if (value && *value) {
        char *end = NULL;
        long parsed = strtol(value, &end, 10);
        return end != value && !*end && parsed >= 0 &&
               parsed <= H3_DIT_BLOCKS ? (unsigned)parsed : 0;
    }
    if (h3_gpu_is_m5(dit->gpu))
        return dit->active_block_count * 3 / 5;
    return dit->active_block_count == H3_DIT_BLOCKS ? 30u : 0u;
}

static int gpu_op(h3_dit *dit, int ok, char *error, size_t error_size,
                  const char *operation) {
    if (ok) return 1;
    fail(error, error_size, "%s: %s", operation, h3_gpu_error(dit->gpu));
    return 0;
}

static void report(h3_dit_progress progress, void *opaque, const char *phase,
                   int completed, int total) {
    if (progress) progress(phase, completed, total, opaque);
}

static void free_tensor(h3_gpu_tensor **tensor) {
    h3_gpu_tensor_free(*tensor);
    *tensor = NULL;
}

static h3_gpu_tensor *bf1(h3_dit *dit, const char *name, uint64_t width,
                          char *error, size_t error_size) {
    uint64_t shape[] = {width};
    return h3_weight_load_bf16(dit->weights, dit->gpu, name, 1, shape,
                               error, error_size);
}

static h3_gpu_tensor *bf2(h3_dit *dit, const char *name, uint64_t rows,
                          uint64_t columns, char *error, size_t error_size) {
    uint64_t shape[] = {rows, columns};
    return h3_weight_load_bf16(dit->weights, dit->gpu, name, 2, shape,
                               error, error_size);
}

static h3_gpu_tensor *f1(h3_dit *dit, const char *name, uint64_t width,
                         char *error, size_t error_size) {
    uint64_t shape[] = {width};
    return h3_weight_load_f32(dit->weights, dit->gpu, name, 1, shape,
                              error, error_size);
}

static h3_gpu_tensor *f2(h3_dit *dit, const char *name, uint64_t rows,
                         uint64_t columns, char *error, size_t error_size) {
    uint64_t shape[] = {rows, columns};
    return h3_weight_load_f32(dit->weights, dit->gpu, name, 2, shape,
                              error, error_size);
}

static int copy_layout(h3_dit *dit, const h3_layout *layout,
                       char *error, size_t error_size) {
    dit->layout = *layout;
    dit->layout.segments = NULL;
    dit->layout.positions = NULL;
    if (layout->segment_count) {
        dit->layout.segments = malloc(layout->segment_count *
                                      sizeof(*layout->segments));
        if (!dit->layout.segments) goto oom;
        memcpy(dit->layout.segments, layout->segments,
               layout->segment_count * sizeof(*layout->segments));
    }
    if (layout->seq_len) {
        dit->layout.positions = malloc(layout->seq_len *
                                       sizeof(*layout->positions));
        if (!dit->layout.positions) goto oom;
        memcpy(dit->layout.positions, layout->positions,
               layout->seq_len * sizeof(*layout->positions));
    }
    return 1;
oom:
    fail(error, error_size, "out of memory copying packed H3 layout");
    h3_layout_free(&dit->layout);
    return 0;
}

static int validate_layout(h3_dit *dit, const h3_text_embedding *text,
                           char *error, size_t error_size) {
    const h3_layout *layout = &dit->layout;
    if (!text || !text->values || text->width != TEXT_DIM || !text->tokens ||
        layout->signature[0] != (int)text->tokens ||
        !layout->segments || layout->segment_count < 3 ||
        layout->segments[0].kind != H3_SEG_TEXT ||
        layout->segments[layout->segment_count - 1].kind != H3_SEG_VIDEO ||
        layout->signature[1] < 1 || layout->signature[2] < 2 ||
        layout->signature[3] < 2 || layout->signature[4] < 1 ||
        layout->signature[2] % 2 || layout->signature[3] % 2 ||
        layout->seq_len > UINT32_MAX || text->tokens > UINT32_MAX ||
        layout->img_cond_rows > UINT32_MAX ||
        layout->audio_cond_rows > UINT32_MAX ||
        layout->audio_target_rows > UINT32_MAX ||
        layout->img_target_rows > UINT32_MAX) {
        fail(error, error_size,
             "DiT requires a valid contiguous H3 packed layout");
        return 0;
    }
    size_t cursor = 0, text_rows = 0, video_condition = 0;
    size_t audio_condition = 0, video_target = 0, audio_target = 0;
    unsigned target_video_segments = 0, target_audio_segments = 0;
    for (size_t index = 0; index < layout->segment_count; index++) {
        const h3_segment *segment = &layout->segments[index];
        if (segment->start != cursor || segment->stop < segment->start ||
            segment->stop > layout->seq_len) {
            fail(error, error_size, "DiT layout segments are not contiguous");
            return 0;
        }
        size_t rows = segment->stop - segment->start;
        switch (segment->kind) {
        case H3_SEG_TEXT: text_rows += rows; break;
        case H3_SEG_COND:
        case H3_SEG_REF_IMAGE: video_condition += rows; break;
        case H3_SEG_REF_AUDIO: audio_condition += rows; break;
        case H3_SEG_AUDIO:
            audio_target += rows;
            target_audio_segments++;
            dit->audio_target_start = (uint32_t)segment->start;
            break;
        case H3_SEG_VIDEO:
            video_target += rows;
            target_video_segments++;
            dit->video_target_start = (uint32_t)segment->start;
            break;
        default:
            fail(error, error_size, "DiT layout contains an unknown segment");
            return 0;
        }
        cursor = segment->stop;
    }
    if (cursor != layout->seq_len || text_rows != text->tokens ||
        video_condition != layout->img_cond_rows ||
        audio_condition != layout->audio_cond_rows ||
        video_target != layout->img_target_rows ||
        audio_target != layout->audio_target_rows ||
        target_video_segments != 1 || target_audio_segments != 1 ||
        video_condition > UINT32_MAX - video_target ||
        audio_condition > UINT32_MAX - audio_target) {
        fail(error, error_size, "DiT layout row-source counts are inconsistent");
        return 0;
    }
    if (text->tags) {
        for (size_t index = 0; index < text->tokens; index++) {
            if (text->tags[index] >= H3_DIT_MODALITIES) {
                fail(error, error_size, "DiT text presentation has an invalid tag");
                return 0;
            }
        }
    }
    dit->latent_t = layout->signature[1];
    dit->latent_h = layout->signature[2];
    dit->latent_w = layout->signature[3];
    dit->audio_t = layout->signature[4];
    dit->text_rows = (uint32_t)text->tokens;
    dit->video_condition_rows = (uint32_t)video_condition;
    dit->audio_condition_rows = (uint32_t)audio_condition;
    dit->audio_rows = (uint32_t)layout->audio_target_rows;
    dit->video_rows = (uint32_t)layout->img_target_rows;
    dit->video_total_rows = (uint32_t)(video_condition + video_target);
    dit->audio_total_rows = (uint32_t)(audio_condition + audio_target);
    dit->sequence = (uint32_t)layout->seq_len;
    if (getenv("H3_PROFILE")) {
        fprintf(stderr,
                "h3: DiT rows text=%u video=%u audio=%u sequence=%u "
                "(latent %dx%dx%d, audio_t=%d)\n",
                dit->text_rows, dit->video_total_rows, dit->audio_total_rows,
                dit->sequence, dit->latent_t, dit->latent_h, dit->latent_w,
                dit->audio_t);
    }
    return 1;
}

static int configure_inpaint(h3_dit *dit, const h3_dit_inpaint *inpaint,
                             char *error, size_t error_size) {
    if (!inpaint) return 1;
    size_t video_elements = h3_dit_video_elements(dit);
    size_t audio_elements = h3_dit_audio_elements(dit);
    if (!inpaint->video_source ||
        inpaint->video_source_elements != video_elements ||
        !inpaint->video_generate_rows ||
        inpaint->video_generate_count != (size_t)dit->video_rows ||
        (inpaint->audio_source
             ? inpaint->audio_source_elements != audio_elements ||
               !inpaint->audio_generate_rows ||
               inpaint->audio_generate_count != (size_t)dit->audio_rows
             : inpaint->audio_source_elements != 0 ||
               inpaint->audio_generate_rows ||
               inpaint->audio_generate_count != 0) ||
        dit->token_reduction) {
        fail(error, error_size,
             dit->token_reduction
                 ? "H3 inpainting cannot use token reduction"
                 : "inpainting source rows do not match the target layout");
        return 0;
    }
    for (size_t row = 0; row < inpaint->video_generate_count; row++)
        if (inpaint->video_generate_rows[row] > 1) {
            fail(error, error_size, "video inpainting mask is not hard");
            return 0;
        }
    for (size_t row = 0; row < inpaint->audio_generate_count; row++)
        if (inpaint->audio_generate_rows[row] > 1) {
            fail(error, error_size, "audio inpainting mask is not hard");
            return 0;
        }

    dit->inpaint_video_source = malloc(
        video_elements * sizeof(*dit->inpaint_video_source));
    dit->inpaint_video_generate_rows = malloc(
        (size_t)dit->video_rows * sizeof(*dit->inpaint_video_generate_rows));
    if (inpaint->audio_source) {
        dit->inpaint_audio_source = malloc(
            audio_elements * sizeof(*dit->inpaint_audio_source));
        dit->inpaint_audio_generate_rows = malloc(
            (size_t)dit->audio_rows *
            sizeof(*dit->inpaint_audio_generate_rows));
    }
    if (!dit->inpaint_video_source || !dit->inpaint_video_generate_rows ||
        (inpaint->audio_source &&
         (!dit->inpaint_audio_source || !dit->inpaint_audio_generate_rows))) {
        fail(error, error_size, "out of memory retaining inpainting sources");
        return 0;
    }
    memcpy(dit->inpaint_video_source, inpaint->video_source,
           video_elements * sizeof(*dit->inpaint_video_source));
    memcpy(dit->inpaint_video_generate_rows, inpaint->video_generate_rows,
           (size_t)dit->video_rows *
           sizeof(*dit->inpaint_video_generate_rows));
    dit->inpaint_video_source_elements = video_elements;
    dit->inpaint_video_generate_count = (size_t)dit->video_rows;
    if (inpaint->audio_source) {
        memcpy(dit->inpaint_audio_source, inpaint->audio_source,
               audio_elements * sizeof(*dit->inpaint_audio_source));
        memcpy(dit->inpaint_audio_generate_rows,
               inpaint->audio_generate_rows,
               (size_t)dit->audio_rows *
               sizeof(*dit->inpaint_audio_generate_rows));
        dit->inpaint_audio_source_elements = audio_elements;
        dit->inpaint_audio_generate_count = (size_t)dit->audio_rows;
    }
    return 1;
}

static int configure_token_reduction(h3_dit *dit, int requested,
                                     char *error, size_t error_size) {
    const char *enabled = getenv("H3_TOKEN_REDUCTION");
    if (!requested &&
        (!enabled || !*enabled || !strcmp(enabled, "0"))) return 1;
    unsigned begin = 4, end = 30;
    const char *range = getenv("H3_TOKEN_REDUCTION_BLOCKS");
    if (range && *range) {
        char *middle = NULL;
        unsigned long parsed_begin = strtoul(range, &middle, 10);
        if (middle == range || *middle != ':') {
            fail(error, error_size,
                 "H3_TOKEN_REDUCTION_BLOCKS must be BEGIN:END");
            return 0;
        }
        char *tail = NULL;
        unsigned long parsed_end = strtoul(middle + 1, &tail, 10);
        if (tail == middle + 1 || *tail || parsed_begin >= parsed_end ||
            parsed_end > H3_DIT_BLOCKS) {
            fail(error, error_size,
                 "token-reduction block range must satisfy 0 <= BEGIN < END <= 50");
            return 0;
        }
        begin = (unsigned)parsed_begin;
        end = (unsigned)parsed_end;
    }
    /* Coarse structure is tolerant of a deeper reduced stack while the first
     * noisy samples form. Restore earlier once fine detail starts resolving. */
    unsigned early_steps = end < 40 ? 10 : 0;
    unsigned early_end = end < 40 ? 40 : end;
    const char *early = getenv("H3_TOKEN_REDUCTION_EARLY");
    if (early && *early) {
        if (!strcmp(early, "0")) {
            early_steps = 0;
            early_end = end;
        } else {
            char *middle = NULL;
            unsigned long parsed_steps = strtoul(early, &middle, 10);
            if (middle == early || *middle != ':') {
                fail(error, error_size,
                     "H3_TOKEN_REDUCTION_EARLY must be STEPS:END");
                return 0;
            }
            char *tail = NULL;
            unsigned long parsed_end = strtoul(middle + 1, &tail, 10);
            if (tail == middle + 1 || *tail || !parsed_steps ||
                parsed_steps > 1000 || parsed_end <= end ||
                parsed_end > H3_DIT_BLOCKS) {
                fail(error, error_size,
                     "early token reduction requires STEPS > 0 and "
                     "base END < END <= 50");
                return 0;
            }
            early_steps = (unsigned)parsed_steps;
            early_end = (unsigned)parsed_end;
        }
    }
    float scale = 1.0f;
    const char *scale_text = getenv("H3_TOKEN_REDUCTION_SCALE");
    if (scale_text && *scale_text) {
        char *tail = NULL;
        scale = strtof(scale_text, &tail);
        if (tail == scale_text || *tail || !isfinite(scale) ||
            scale < 0.0f || scale > 2.0f) {
            fail(error, error_size,
                 "H3_TOKEN_REDUCTION_SCALE must be in [0, 2]");
            return 0;
        }
    }
    uint32_t spatial_height = (uint32_t)dit->latent_h / 2;
    uint32_t spatial_width = (uint32_t)dit->latent_w / 2;
    uint32_t reduced_width = (spatial_width + 1) / 2;
    uint64_t reduced_video =
        (uint64_t)(uint32_t)dit->latent_t * spatial_height * reduced_width;
    if (!spatial_height || !spatial_width ||
        (uint64_t)(uint32_t)dit->latent_t * spatial_height * spatial_width !=
            dit->video_rows ||
        dit->video_target_start + dit->video_rows != dit->sequence ||
        reduced_video > UINT32_MAX ||
        reduced_video > UINT32_MAX - dit->video_target_start) {
        fail(error, error_size,
             "token reduction requires the target video to end the packed layout");
        return 0;
    }
    dit->token_reduction = 1;
    dit->token_reduction_begin = begin;
    dit->token_reduction_end = end;
    dit->token_reduction_early_steps = early_steps;
    dit->token_reduction_early_end = early_end;
    dit->token_reduction_scale = scale;
    dit->reduced_video_rows = (uint32_t)reduced_video;
    dit->token_baseline_rows = dit->video_rows - dit->reduced_video_rows;
    dit->reduced_sequence = dit->video_target_start +
                            dit->reduced_video_rows;
    return 1;
}

static int configure_sol_attention(h3_dit *dit,
                                   char *error, size_t error_size) {
    const char *enabled = getenv("H3_SOL_ATTN");
    dit->sol_attention_tau = -1.0f;
    if (!enabled || !*enabled || !strcmp(enabled, "0")) return 1;
    uint32_t minimum_rows = H3_DIT_SOL_ATTENTION_ROWS;
    const char *minimum_text = getenv("H3_SOL_ATTN_MIN_ROWS");
    if (minimum_text && *minimum_text) {
        char *tail = NULL;
        unsigned long parsed = strtoul(minimum_text, &tail, 10);
        if (tail == minimum_text || *tail || parsed > UINT32_MAX) {
            fail(error, error_size,
                 "H3_SOL_ATTN_MIN_ROWS must be an unsigned 32-bit integer");
            return 0;
        }
        minimum_rows = (uint32_t)parsed;
    }
    const char *tau_text = getenv("H3_SOL_ATTN_TAU");
    if (tau_text && *tau_text) {
        char *tail = NULL;
        float tau = strtof(tau_text, &tail);
        if (tail == tau_text || *tail || !isfinite(tau) ||
            tau < -100.0f || tau > 10.0f) {
            fail(error, error_size,
                 "H3_SOL_ATTN_TAU must be finite and in [-100, 10]");
            return 0;
        }
        dit->sol_attention_tau = tau;
    }
    if (dit->sequence < minimum_rows || dit->token_reduction) {
        if (getenv("H3_PROFILE"))
            fprintf(stderr,
                    "h3: Sol-Attn requested but inactive (%u rows, minimum %u%s)\n",
                    dit->sequence, minimum_rows,
                    dit->token_reduction ? ", token reduction enabled" : "");
        return 1;
    }
    if (!h3_gpu_has_sol_attention(dit->gpu)) {
        const char *strict = getenv("H3_SOL_ATTN_STRICT");
        if (strict && *strict && strcmp(strict, "0")) {
            fail(error, error_size,
                 "H3_SOL_ATTN requested but the Sol-Attn Metal library is unavailable");
            return 0;
        }
        if (getenv("H3_PROFILE"))
            fprintf(stderr,
                    "h3: Sol-Attn requested but unavailable; using dense attention\n");
        return 1;
    }
    dit->sol_attention = 1;
    if (getenv("H3_PROFILE"))
        fprintf(stderr, "h3: experimental Sol-Attn enabled (tau %.3f)\n",
                dit->sol_attention_tau);
    return 1;
}

static void token_pool_sources(const h3_dit *dit, uint32_t reduced_row,
                               uint32_t *first, uint32_t *second) {
    if (reduced_row < dit->video_target_start) {
        *first = reduced_row;
        *second = reduced_row;
        return;
    }
    uint32_t spatial_width = (uint32_t)dit->latent_w / 2;
    uint32_t reduced_width = (spatial_width + 1) / 2;
    uint32_t local = reduced_row - dit->video_target_start;
    uint32_t source = dit->video_target_start +
        (local / reduced_width) * spatial_width +
        (local % reduced_width) * 2;
    *first = source;
    *second = source + ((source - dit->video_target_start) % spatial_width + 1 <
                        spatial_width ? 1u : 0u);
}

static uint32_t token_reduced_parent(const h3_dit *dit, uint32_t full_row) {
    if (full_row < dit->video_target_start) return full_row;
    uint32_t spatial_width = (uint32_t)dit->latent_w / 2;
    uint32_t reduced_width = (spatial_width + 1) / 2;
    uint32_t local = full_row - dit->video_target_start;
    return dit->video_target_start + (local / spatial_width) * reduced_width +
           (local % spatial_width) / 2;
}

static int load_block(h3_dit *dit, h3_dit_block *block, const char *prefix,
                      char *error, size_t error_size) {
    char name[160];
#define LOAD1(field, suffix, width) do {                                       \
    snprintf(name, sizeof(name), "%s%s", prefix, suffix);                    \
    block->field = bf1(dit, name, width, error, error_size);                    \
    if (!block->field) return 0;                                                \
} while (0)
#define LOAD2(field, suffix, rows, columns) do {                               \
    snprintf(name, sizeof(name), "%s%s", prefix, suffix);                    \
    block->field = bf2(dit, name, rows, columns, error, error_size);            \
    if (!block->field) return 0;                                                \
} while (0)
    LOAD1(norm1, "norm1.weight", HIDDEN);
    LOAD1(norm2, "norm2.weight", HIDDEN);
    LOAD2(qkv, "attn.qkv_proj.weight", INNER * 3, HIDDEN);
    LOAD1(q_norm, "attn.q_norm.weight", HEAD_DIM);
    LOAD1(k_norm, "attn.k_norm.weight", HEAD_DIM);
    LOAD2(out, "attn.out_proj.weight", HIDDEN, INNER);
    LOAD2(fc1, "mlp.fc1.weight", FFN * 2, HIDDEN);
    LOAD2(fc2, "mlp.fc2.weight", HIDDEN, FFN);
#undef LOAD1
#undef LOAD2
    return 1;
}

static int load_block_norms(h3_dit *dit, h3_dit_block *block,
                            const char *prefix,
                            char *error, size_t error_size) {
    char name[160];
#define LOAD1(field, suffix, width) do {                                       \
    snprintf(name, sizeof(name), "%s%s", prefix, suffix);                    \
    block->field = bf1(dit, name, width, error, error_size);                    \
    if (!block->field) return 0;                                                \
} while (0)
    LOAD1(norm1, "norm1.weight", HIDDEN);
    LOAD1(norm2, "norm2.weight", HIDDEN);
    LOAD1(q_norm, "attn.q_norm.weight", HEAD_DIM);
    LOAD1(k_norm, "attn.k_norm.weight", HEAD_DIM);
#undef LOAD1
    return 1;
}

static int load_prequantized_block(h3_dit *dit, h3_dit_block *block,
                                   const char *prefix,
                                   char *error, size_t error_size) {
    char name[160];
    if (!load_block_norms(dit, block, prefix, error, error_size)) return 0;
#define LOAD_I8(weight_field, scale_field, group_field, suffix, rows, columns) \
do {                                                                            \
    snprintf(name, sizeof(name), "%s%s", prefix, suffix);                    \
    int loaded = dit->input_major_transformer                                \
        ? h3_weight_load_i8_linear_input_major(                               \
              dit->weights, dit->gpu, name, columns, rows,                    \
              &block->weight_field, &block->scale_field,                      \
              error, error_size)                                              \
        : h3_weight_load_i8_linear(                                            \
              dit->weights, dit->gpu, name, rows, columns,                    \
              &block->weight_field, &block->scale_field,                      \
              error, error_size);                                             \
    if (!loaded ||                                                             \
        !h3_weight_i8_linear_convrot_group(                                    \
            dit->weights, name, &block->group_field, error, error_size))       \
        return 0;                                                               \
} while (0)
    LOAD_I8(qkv_int8, qkv_scales, qkv_convrot_group,
            "attn.qkv_proj.weight",
            INNER * 3, HIDDEN);
    LOAD_I8(out_int8, out_scales, out_convrot_group,
            "attn.out_proj.weight", HIDDEN, INNER);
    LOAD_I8(fc1_int8, fc1_scales, fc1_convrot_group,
            "mlp.fc1.weight", FFN * 2, HIDDEN);
    LOAD_I8(fc2_int8, fc2_scales, fc2_convrot_group,
            "mlp.fc2.weight", HIDDEN, FFN);
#undef LOAD_I8
    return 1;
}

static void free_block(h3_dit_block *block) {
    free_tensor(&block->norm1);
    free_tensor(&block->norm2);
    free_tensor(&block->qkv);
    free_tensor(&block->qkv_int8);
    free_tensor(&block->qkv_scales);
    free_tensor(&block->q_norm);
    free_tensor(&block->k_norm);
    free_tensor(&block->out);
    free_tensor(&block->out_int8);
    free_tensor(&block->out_scales);
    free_tensor(&block->fc1);
    free_tensor(&block->fc2);
    free_tensor(&block->fc1_int8);
    free_tensor(&block->fc1_scales);
    free_tensor(&block->fc2_int8);
    free_tensor(&block->fc2_scales);
    free_tensor(&block->qkv_lora_a);
    free_tensor(&block->qkv_lora_b);
    free_tensor(&block->out_lora_a);
    free_tensor(&block->out_lora_b);
    free_tensor(&block->fc1_lora_a);
    free_tensor(&block->fc1_lora_b);
    free_tensor(&block->fc2_lora_a);
    free_tensor(&block->fc2_lora_b);
}

/* --- Optional low-rank adapters -------------------------------------- */

static uint16_t f32_to_bf16_round(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (uint16_t)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

static float bf16_to_f32_value(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

/* ComfyUI's generic LoRA format may fuse independently trained Q, K and V
 * adapters into one projection. Its block-diagonal B and concatenated A then
 * have rank three times the other projections, and an explicit alpha keeps
 * alpha/rank equal to the training scale. Older H3 adapters have no alpha and
 * already bake their scale into B, so absence deliberately means one. */
static int lora_pair_schema(const h3_weight_store *store, const char *base,
                            uint32_t input_dim, uint32_t output_dim,
                            uint32_t *rank, float *scale,
                            char *error, size_t error_size) {
    char name[224];
    snprintf(name, sizeof(name), "%s.lora_A.weight", base);
    const h3_st_tensor *a = h3_weight_find(store, name, NULL);
    if (!a || a->dtype != H3_DTYPE_BF16 || a->ndim != 2 ||
        !a->shape[0] || a->shape[0] > 512 || a->shape[1] != input_dim) {
        fail(error, error_size,
             "adapter tensor %s must be BF16 [rank, %u] with rank in [1, 512]",
             name, input_dim);
        return 0;
    }
    uint32_t pair_rank = (uint32_t)a->shape[0];
    snprintf(name, sizeof(name), "%s.lora_B.weight", base);
    const h3_st_tensor *b = h3_weight_find(store, name, NULL);
    if (!b || b->dtype != H3_DTYPE_BF16 || b->ndim != 2 ||
        b->shape[0] != output_dim || b->shape[1] != pair_rank) {
        fail(error, error_size,
             "adapter tensor %s must be BF16 [%u, %u]",
             name, output_dim, pair_rank);
        return 0;
    }

    float pair_scale = 1.0f;
    snprintf(name, sizeof(name), "%s.alpha", base);
    const h3_st_header *alpha_header = NULL;
    const h3_st_tensor *alpha = h3_weight_find(store, name, &alpha_header);
    if (alpha) {
        if (alpha->dtype != H3_DTYPE_F32 ||
            !((alpha->ndim == 0) ||
              (alpha->ndim == 1 && alpha->shape[0] == 1)) ||
            !h3_st_read_data(alpha_header, alpha, &pair_scale,
                             sizeof(pair_scale), error, error_size) ||
            !isfinite(pair_scale)) {
            if (!error || !*error)
                fail(error, error_size,
                     "adapter alpha %s must be one finite F32 value", name);
            return 0;
        }
        pair_scale /= (float)pair_rank;
    }
    *rank = pair_rank;
    *scale = pair_scale;
    return 1;
}

/* Folding the strength into B keeps the forward pass to existing kernels. */
static int scale_bf16_tensor(h3_gpu_tensor *tensor, size_t elements,
                             float strength) {
    if (strength == 1.0f) return 1;
    uint16_t *host = malloc(elements * sizeof(*host));
    if (!host) return 0;
    if (!h3_gpu_tensor_read_bf16(tensor, host, elements)) {
        free(host);
        return 0;
    }
    for (size_t index = 0; index < elements; index++)
        host[index] = f32_to_bf16_round(
            bf16_to_f32_value(host[index]) * strength);
    int ok = h3_gpu_tensor_write_bf16(tensor, host, elements);
    free(host);
    return ok;
}

static int load_lora_pair(h3_dit *dit, const h3_weight_store *store,
                          const char *base, uint32_t input_dim,
                          uint32_t output_dim, uint32_t convrot_group,
                          float strength,
                          h3_gpu_tensor **down, h3_gpu_tensor **up,
                          uint32_t *rank,
                          char *error, size_t error_size) {
    float pair_scale = 1.0f;
    if (!lora_pair_schema(store, base, input_dim, output_dim,
                          rank, &pair_scale, error, error_size))
        return 0;
    char name[224];
    snprintf(name, sizeof(name), "%s.lora_A.weight", base);
    const uint64_t down_shape[] = {*rank, input_dim};
    h3_gpu_tensor *a = h3_weight_load_bf16(store, dit->gpu, name, 2,
                                           down_shape, error, error_size);
    if (!a) return 0;
    snprintf(name, sizeof(name), "%s.lora_B.weight", base);
    const uint64_t up_shape[] = {output_dim, *rank};
    h3_gpu_tensor *b = h3_weight_load_bf16(store, dit->gpu, name, 2,
                                           up_shape, error, error_size);
    if (!b) {
        h3_gpu_tensor_free(a);
        return 0;
    }
    /* The projection consumes an already-rotated activation, so the adapter
     * input is rotated once here instead of unrotating every forward. */
    int ok = 1;
    if (convrot_group)
        ok = h3_gpu_begin(dit->gpu) &&
             h3_gpu_convrot_bf16(dit->gpu, a, a, *rank, input_dim,
                                 convrot_group) &&
             h3_gpu_submit(dit->gpu);
    if (ok)
        ok = scale_bf16_tensor(b, (size_t)output_dim * *rank,
                               strength * pair_scale);
    if (!ok) {
        fail(error, error_size, "cannot prepare the %s adapter: %s", base,
             h3_gpu_error(dit->gpu));
        h3_gpu_tensor_free(a);
        h3_gpu_tensor_free(b);
        return 0;
    }
    *down = a;
    *up = b;
    return 1;
}

/* The four projections of one block, whether it is a core block reading
 * ConvRot-rotated activations or a plain BF16 refiner block. */
static int load_lora_block(h3_dit *dit, const h3_weight_store *store,
                           const char *base, h3_dit_block *block,
                           uint32_t qkv_group, uint32_t out_group,
                           uint32_t fc1_group, uint32_t fc2_group,
                           float strength, char *error, size_t error_size) {
    char name[192];
#define PAIR(suffix, in_dim, out_dim, group, down, up, rank) do {               \
    snprintf(name, sizeof(name), "%s." suffix, base);                           \
    if (!load_lora_pair(dit, store, name, in_dim, out_dim, group, strength,     \
                        &block->down, &block->up, &block->rank,                 \
                        error, error_size))                                     \
        return 0;                                                               \
} while (0)
    PAIR("attn.qkv_proj", HIDDEN, INNER * 3, qkv_group,
         qkv_lora_a, qkv_lora_b, qkv_lora_rank);
    PAIR("attn.out_proj", INNER, HIDDEN, out_group,
         out_lora_a, out_lora_b, out_lora_rank);
    PAIR("mlp.fc1", HIDDEN, FFN * 2, fc1_group,
         fc1_lora_a, fc1_lora_b, fc1_lora_rank);
    PAIR("mlp.fc2", FFN, HIDDEN, fc2_group,
         fc2_lora_a, fc2_lora_b, fc2_lora_rank);
#undef PAIR
    return 1;
}

/* The refiner runs in plain BF16 with no ConvRot, so its adapters need no
 * rotation. Loaded separately because the refiner weights are transient. */
static int load_refiner_lora(h3_dit *dit, h3_dit_block *blocks,
                             char *error, size_t error_size) {
    if (!dit->lora_path || !*dit->lora_path || dit->lora_strength == 0.0f)
        return 1;
    h3_weight_store *store = h3_weight_store_open(
        dit->lora_path, error, error_size);
    if (!store) return 0;
    int ok = 1;
    for (unsigned index = 0; index < 2 && ok; index++) {
        char base[192];
        snprintf(base, sizeof(base),
                 "diffusion_model.token_refiner.blocks.%u", index);
        char probe[224];
        snprintf(probe, sizeof(probe), "%s.attn.qkv_proj.lora_A.weight", base);
        /* Absent refiner adapters are not an error: some conversions ship
         * only the core stack. */
        if (!h3_weight_find(store, probe, NULL)) continue;
        ok = load_lora_block(dit, store, base, &blocks[index], 0, 0, 0, 0,
                             dit->lora_strength, error, error_size);
    }
    h3_weight_store_free(store);
    return ok;
}

/* Validates the adapter file and sizes shared scratch. Runs before the
 * token refiner so both stacks can use the same buffers. */
static int prepare_lora(h3_dit *dit, const char *path, float strength,
                        char *error, size_t error_size) {
    if (!dit->prequantized_int8) {
        fail(error, error_size,
             "runtime adapters currently require a pre-quantized INT8 "
             "checkpoint; this build cannot apply them to BF16 weights");
        return 0;
    }
    h3_weight_store *store = h3_weight_store_open(path, error, error_size);
    if (!store) return 0;
    const h3_st_tensor *probe = h3_weight_find(
        store, "diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight", NULL);
    if (!probe || probe->ndim != 2 || probe->dtype != H3_DTYPE_BF16 ||
        probe->shape[1] != HIDDEN || !probe->shape[0] ||
        probe->shape[0] > 512) {
        fail(error, error_size,
             "the adapter file must hold BF16 diffusion_model.blocks.N "
             "lora_A/lora_B pairs for this transformer");
        h3_weight_store_free(store);
        return 0;
    }
    /* Individual projections may use different ranks (LightX2V's fused QKV
     * is rank 384 while the other projections are rank 128). A fixed upper
     * bound keeps one reusable scratch allocation without constraining every
     * pair to the first tensor's shape. */
    dit->lora_capacity = 512;
    h3_weight_store_free(store);
    dit->lora_path = path;
    dit->lora_strength = strength;
    dit->lora_hidden = h3_gpu_tensor_new_bf16(
        dit->gpu, (size_t)dit->sequence * dit->lora_capacity);
    dit->lora_delta = h3_gpu_tensor_new_bf16(
        dit->gpu, (size_t)dit->sequence * FFN * 2);
    if (!dit->lora_hidden || !dit->lora_delta) {
        fail(error, error_size, "cannot allocate adapter activations: %s",
             h3_gpu_error(dit->gpu));
        return 0;
    }
    return 1;
}

static int load_lora_adapters(h3_dit *dit, const char *path, float strength,
                              h3_dit_progress progress, void *progress_opaque,
                              char *error, size_t error_size) {
    h3_weight_store *store = h3_weight_store_open(path, error, error_size);
    if (!store) return 0;
    int ok = 1;
    for (unsigned layer = 0; layer < H3_DIT_BLOCKS && ok; layer++) {
        report(progress, progress_opaque, "load adapters", (int)layer,
               H3_DIT_BLOCKS);
        if (!dit->block_active[layer]) continue;
        h3_dit_block *block = &dit->blocks[layer];
        char base[192];
        snprintf(base, sizeof(base), "diffusion_model.blocks.%u", layer);
        ok = load_lora_block(dit, store, base, block,
                             block->qkv_convrot_group, block->out_convrot_group,
                             block->fc1_convrot_group, block->fc2_convrot_group,
                             strength, error, error_size);
    }
    h3_weight_store_free(store);
    if (!ok) return 0;
    report(progress, progress_opaque, "load adapters", H3_DIT_BLOCKS,
           H3_DIT_BLOCKS);
    return 1;
}

static int apply_lora(h3_dit *dit, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *down, const h3_gpu_tensor *up,
                      uint32_t rank, uint32_t rows,
                      uint32_t input_dim, uint32_t output_dim,
                      char *error, size_t error_size) {
    if (!down || !up) return 1;
    if (!rank || rank > dit->lora_capacity) {
        fail(error, error_size, "adapter rank exceeds its scratch capacity");
        return 0;
    }
    return gpu_op(dit, h3_gpu_linear_bf16(
                      dit->gpu, dit->lora_hidden, input, down, NULL, rows,
                      input_dim, rank),
                  error, error_size, "DiT adapter down projection") &&
           gpu_op(dit, h3_gpu_linear_bf16(
                      dit->gpu, dit->lora_delta, dit->lora_hidden, up, NULL,
                      rows, rank, output_dim),
                  error, error_size, "DiT adapter up projection") &&
           gpu_op(dit, h3_gpu_add_bf16(
                      dit->gpu, output, output, dit->lora_delta,
                      rows * output_dim),
                  error, error_size, "DiT adapter residual");
}

static double stream_now(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0.0;
    return (double)value.tv_sec + (double)value.tv_nsec * 1e-9;
}

static int compare_stream_sources(const void *left, const void *right) {
    const h3_dit_stream_source *a = left;
    const h3_dit_stream_source *b = right;
    int path = strcmp(a->path, b->path);
    if (path) return path;
    if (a->file_offset < b->file_offset) return -1;
    return a->file_offset > b->file_offset;
}

enum { H3_INPUT_MAJOR_TRANSFORMER_VERSION = 1 };

static uint32_t little_u32(const unsigned char bytes[4]) {
    uint32_t value = 0;
    for (unsigned index = 0; index < 4; index++)
        value |= (uint32_t)bytes[index] << (index * 8u);
    return value;
}

static int read_singleton_marker(const h3_weight_store *store,
                                 const char *name, h3_dtype dtype,
                                 unsigned char *bytes, size_t byte_count,
                                 char *error, size_t error_size) {
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(store, name, &header);
    if (!tensor || !header || tensor->dtype != dtype || tensor->ndim != 1 ||
        tensor->shape[0] != 1 || h3_dtype_size(dtype) != byte_count ||
        !h3_st_read_data(header, tensor, bytes, byte_count,
                         error, error_size)) {
        if (!error || !error[0])
            fail(error, error_size, "checkpoint marker is invalid: %s", name);
        return 0;
    }
    return 1;
}

static int validate_input_major_transformer_weight(
    const h3_dit *dit, unsigned layer, const char *suffix,
    uint64_t input_columns, uint64_t output_rows,
    char *error, size_t error_size) {
    char name[160];
    int length = snprintf(name, sizeof(name), "blocks.%u.%s", layer, suffix);
    if (length < 0 || (size_t)length >= sizeof(name)) {
        fail(error, error_size, "input-major DiT weight name is too long");
        return 0;
    }
    const h3_st_tensor *tensor = h3_weight_find(dit->weights, name, NULL);
    if (!tensor || tensor->dtype != H3_DTYPE_I8 || tensor->ndim != 2 ||
        tensor->shape[0] != input_columns ||
        tensor->shape[1] != output_rows) {
        fail(error, error_size,
             "input-major DiT tensor has the wrong schema: %s", name);
        return 0;
    }
    return 1;
}

static int configure_input_major_transformer(h3_dit *dit,
                                              char *error,
                                              size_t error_size) {
    static const char marker_name[] = "h3.transformer_input_major.version";
    const h3_st_tensor *marker = h3_weight_find(
        dit->weights, marker_name, NULL);
    if (!marker) {
        const h3_st_tensor *qkv = h3_weight_find(
            dit->weights, "blocks.0.attn.qkv_proj.weight", NULL);
        if (dit->prequantized_int8 && qkv && qkv->ndim == 2 &&
            qkv->shape[0] == HIDDEN && qkv->shape[1] == INNER * 3) {
            fail(error, error_size,
                 "input-major DiT checkpoint is missing its layout marker");
            return 0;
        }
        return 1;
    }
    unsigned char version_bytes[4];
    if (!dit->prequantized_int8 ||
        !read_singleton_marker(
            dit->weights, marker_name, H3_DTYPE_U32, version_bytes,
            sizeof(version_bytes), error, error_size) ||
        little_u32(version_bytes) != H3_INPUT_MAJOR_TRANSFORMER_VERSION) {
        if (!error || !error[0])
            fail(error, error_size,
                 "input-major DiT checkpoint has an unsupported version");
        return 0;
    }
    for (unsigned layer = 0; layer < H3_DIT_BLOCKS; layer++) {
        if (!validate_input_major_transformer_weight(
                dit, layer, "attn.qkv_proj.weight", HIDDEN, INNER * 3,
                error, error_size) ||
            !validate_input_major_transformer_weight(
                dit, layer, "attn.out_proj.weight", INNER, HIDDEN,
                error, error_size) ||
            !validate_input_major_transformer_weight(
                dit, layer, "mlp.fc1.weight", HIDDEN, FFN * 2,
                error, error_size) ||
            !validate_input_major_transformer_weight(
                dit, layer, "mlp.fc2.weight", FFN, HIDDEN,
                error, error_size)) return 0;
    }
    dit->input_major_transformer = 1;
    if (getenv("H3_PROFILE"))
        fprintf(stderr, "h3: full input-major DiT checkpoint enabled\n");
    return 1;
}

enum {
    H3_HYBRID_ADALN_VERSION = 1,
    H3_HYBRID_ADALN_FIRST_BLOCK = 25,
    H3_HYBRID_ADALN_BLOCK_COUNT = 25,
    H3_HYBRID_ADALN_TIME_DIM = 8
};

static int read_u32_marker(const h3_weight_store *store, const char *name,
                           uint32_t *value, char *error,
                           size_t error_size) {
    unsigned char bytes[4];
    if (!read_singleton_marker(store, name, H3_DTYPE_U32, bytes,
                               sizeof(bytes), error, error_size)) return 0;
    *value = little_u32(bytes);
    return 1;
}

static int validate_hybrid_adaln_tensor(const h3_weight_store *store,
                                        unsigned block, const char *suffix,
                                        int ndim, const uint64_t *shape,
                                        char *error, size_t error_size) {
    char name[160];
    int length = snprintf(name, sizeof(name),
                          "blocks.%u.adaln_proj.linear.%s", block, suffix);
    if (length < 0 || (size_t)length >= sizeof(name)) {
        fail(error, error_size, "hybrid AdaLN tensor name is too long");
        return 0;
    }
    const h3_st_tensor *tensor = h3_weight_find(store, name, NULL);
    if (!tensor || tensor->dtype != H3_DTYPE_F16 || tensor->ndim != ndim) {
        fail(error, error_size,
             "hybrid AdaLN tensor has the wrong schema: %s", name);
        return 0;
    }
    for (int dimension = 0; dimension < ndim; dimension++) {
        if (tensor->shape[dimension] != shape[dimension]) {
            fail(error, error_size,
                 "hybrid AdaLN tensor has the wrong schema: %s", name);
            return 0;
        }
    }
    return 1;
}

static int configure_hybrid_adaln(h3_dit *dit, const char *overlay_path,
                                  char *error, size_t error_size) {
    if (!overlay_path || !*overlay_path) return 1;
    const h3_st_tensor *table = h3_weight_find(
        dit->weights, "adaln_t_table", NULL);
    if (!table || table->dtype != H3_DTYPE_F32 || table->ndim != 2 ||
        table->shape[0] < 2 || table->shape[1] != H3_HYBRID_ADALN_TIME_DIM) {
        fail(error, error_size,
             "hybrid AdaLN requires a compact FL2VA base checkpoint");
        return 0;
    }
    dit->late_adaln_overlay = h3_weight_store_open(
        overlay_path, error, error_size);
    if (!dit->late_adaln_overlay) return 0;
    uint32_t version = 0, first_block = 0, block_count = 0;
    if (!read_u32_marker(
            dit->late_adaln_overlay, "h3.hybrid_adaln.version", &version,
            error, error_size) ||
        !read_u32_marker(
            dit->late_adaln_overlay, "h3.hybrid_adaln.first_block",
            &first_block, error, error_size) ||
        !read_u32_marker(
            dit->late_adaln_overlay, "h3.hybrid_adaln.block_count",
            &block_count, error, error_size) ||
        version != H3_HYBRID_ADALN_VERSION ||
        first_block != H3_HYBRID_ADALN_FIRST_BLOCK ||
        block_count != H3_HYBRID_ADALN_BLOCK_COUNT) {
        if (!error || !error[0])
            fail(error, error_size,
                 "hybrid AdaLN checkpoint has an unsupported recipe");
        return 0;
    }
    const uint64_t weight_shape[] = {
        H3_DIT_MODALITIES * H3_DIT_ADALN_SLOTS * H3_DIT_HIDDEN,
        H3_HYBRID_ADALN_TIME_DIM
    };
    const uint64_t bias_shape[] = {
        H3_DIT_MODALITIES * H3_DIT_ADALN_SLOTS * H3_DIT_HIDDEN
    };
    for (unsigned block = first_block;
         block < first_block + block_count; block++) {
        if (!validate_hybrid_adaln_tensor(
                dit->late_adaln_overlay, block, "weight", 2, weight_shape,
                error, error_size) ||
            !validate_hybrid_adaln_tensor(
                dit->late_adaln_overlay, block, "bias", 1, bias_shape,
                error, error_size)) return 0;
    }
    dit->hybrid_adaln = 1;
    if (getenv("H3_PROFILE"))
        fprintf(stderr,
                "h3: FL2VA core + Ref2VA AdaLN blocks 25-49 enabled\n");
    return 1;
}

static int prepare_stream_source_from_store(const h3_weight_store *store,
                                 h3_dit_stream_source *source,
                                 const char *name, h3_dtype dtype,
                                 int ndim, const uint64_t *shape,
                                 unsigned field,
                                 char *error, size_t error_size) {
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(store, name, &header);
    if (!tensor) {
        fail(error, error_size, "required streaming weight is absent: %s",
             name);
        return 0;
    }
    if (!header || tensor->dtype != dtype || tensor->ndim != ndim) {
        fail(error, error_size, "streaming weight has the wrong schema: %s",
             name);
        return 0;
    }
    uint64_t elements = 1;
    for (int dimension = 0; dimension < ndim; dimension++) {
        if (tensor->shape[dimension] != shape[dimension] ||
            (shape[dimension] && elements > UINT64_MAX / shape[dimension])) {
            fail(error, error_size,
                 "streaming weight has the wrong schema: %s", name);
            return 0;
        }
        elements *= shape[dimension];
    }
    if (elements > SIZE_MAX) {
        fail(error, error_size, "streaming weight is too large: %s", name);
        return 0;
    }
    source->path = header->path;
    source->file_offset = tensor->file_offset;
    source->elements = (size_t)elements;
    source->field = field;
    source->dtype = dtype;
    return 1;
}

static int prepare_stream_source(h3_dit *dit,
                                 h3_dit_stream_source *source,
                                 const char *name, h3_dtype dtype,
                                 int ndim, const uint64_t *shape,
                                 unsigned field,
                                 char *error, size_t error_size) {
    return prepare_stream_source_from_store(
        dit->weights, source, name, dtype, ndim, shape, field,
        error, error_size);
}

static int prepare_prequantized_stream_linear(
                                 h3_dit *dit, h3_dit_stream_layer *stream,
                                 const h3_weight_store *weight_store,
                                 const char *weight_name,
                                 uint64_t stored_rows, uint64_t stored_columns,
                                 uint64_t output_rows,
                                 unsigned weight_field, unsigned scale_field,
                                 uint32_t *convrot_group,
                                 char *error, size_t error_size) {
    if (stream->count + 2 > STREAM_TENSORS) {
        fail(error, error_size, "too many tensors in DiT stream layer");
        return 0;
    }
    const uint64_t weight_shape[] = {stored_rows, stored_columns};
    if (!prepare_stream_source_from_store(
            weight_store, &stream->sources[stream->count++], weight_name,
            H3_DTYPE_I8, 2, weight_shape, weight_field,
            error, error_size)) return 0;
    char scale_name[192];
    int length = snprintf(scale_name, sizeof(scale_name), "%s_scale",
                          weight_name);
    if (length < 0 || (size_t)length >= sizeof(scale_name)) {
        fail(error, error_size, "DiT stream scale name is too long");
        return 0;
    }
    const h3_st_tensor *scale = h3_weight_find(
        dit->weights, scale_name, NULL);
    const uint64_t scale_shape_1d[] = {output_rows};
    const uint64_t scale_shape_2d[] = {output_rows, 1};
    if (!scale || scale->dtype != H3_DTYPE_F32 ||
        !((scale->ndim == 1 && scale->shape[0] == output_rows) ||
          (scale->ndim == 2 && scale->shape[0] == output_rows &&
           scale->shape[1] == 1)) ||
        !prepare_stream_source(
            dit, &stream->sources[stream->count++], scale_name,
            H3_DTYPE_F32, scale->ndim,
            scale->ndim == 1 ? scale_shape_1d : scale_shape_2d,
            scale_field, error, error_size)) {
        if (!error || !error[0])
            fail(error, error_size,
                 "streaming scale has the wrong schema: %s", scale_name);
        return 0;
    }
    return h3_weight_i8_linear_convrot_group(
        dit->weights, weight_name, convrot_group, error, error_size);
}

static int prepare_stream_layer(h3_dit *dit, unsigned layer,
                                char *error, size_t error_size) {
    char name[160];
    h3_dit_stream_layer *stream = &dit->stream_layers[layer];
    memset(stream, 0, sizeof(*stream));
    if (dit->prequantized_int8) {
#define I8_SOURCE(suffix, rows, columns, weight_field, scale_field, group) do { \
    snprintf(name, sizeof(name), "blocks.%u.%s", layer, suffix);              \
    if (!prepare_prequantized_stream_linear(                                  \
            dit, stream, dit->weights, name,                                   \
            dit->input_major_transformer ? columns : rows,                     \
            dit->input_major_transformer ? rows : columns, rows,               \
            weight_field, scale_field,                                         \
            &dit->blocks[layer].group, error, error_size)) return 0;           \
} while (0)
        I8_SOURCE("attn.qkv_proj.weight", INNER * 3, HIDDEN,
                  STREAM_QKV, STREAM_QKV_SCALES, qkv_convrot_group);
        I8_SOURCE("attn.out_proj.weight", HIDDEN, INNER,
                  STREAM_OUT, STREAM_OUT_SCALES, out_convrot_group);
        I8_SOURCE("mlp.fc1.weight", FFN * 2, HIDDEN,
                  STREAM_FC1, STREAM_FC1_SCALES, fc1_convrot_group);
        snprintf(name, sizeof(name), "blocks.%u.mlp.fc2.weight", layer);
        if (!prepare_prequantized_stream_linear(
                dit, stream, dit->weights, name,
                dit->input_major_transformer ? FFN : HIDDEN,
                dit->input_major_transformer ? HIDDEN : FFN,
                HIDDEN, STREAM_FC2, STREAM_FC2_SCALES,
                &dit->blocks[layer].fc2_convrot_group,
                error, error_size)) return 0;
#undef I8_SOURCE
    } else {
#define SOURCE(suffix, rows, columns, field) do {                               \
    snprintf(name, sizeof(name), "blocks.%u.%s", layer, suffix);              \
    const uint64_t shape[] = {rows, columns};                                  \
    if (!prepare_stream_source(dit, &stream->sources[stream->count++], name,   \
                               H3_DTYPE_BF16, 2, shape, field,                 \
                               error, error_size))                             \
        return 0;                                                               \
} while (0)
        SOURCE("attn.qkv_proj.weight", INNER * 3, HIDDEN, STREAM_QKV);
        SOURCE("attn.out_proj.weight", HIDDEN, INNER, STREAM_OUT);
        SOURCE("mlp.fc1.weight", FFN * 2, HIDDEN, STREAM_FC1);
        SOURCE("mlp.fc2.weight", HIDDEN, FFN, STREAM_FC2);
#undef SOURCE
    }
    qsort(stream->sources, stream->count, sizeof(stream->sources[0]),
          compare_stream_sources);
    return 1;
}

static int allocate_stream_slot(h3_dit *dit, h3_dit_block *slot,
                                char *error, size_t error_size) {
    if (dit->prequantized_int8) {
        slot->qkv_int8 = h3_gpu_tensor_new_i8(
            dit->gpu, (size_t)INNER * 3 * HIDDEN);
        slot->qkv_scales = h3_gpu_tensor_new_f32(dit->gpu, INNER * 3);
        slot->out_int8 = h3_gpu_tensor_new_i8(
            dit->gpu, (size_t)HIDDEN * INNER);
        slot->out_scales = h3_gpu_tensor_new_f32(dit->gpu, HIDDEN);
        slot->fc1_int8 = h3_gpu_tensor_new_i8(
            dit->gpu, (size_t)FFN * 2 * HIDDEN);
        slot->fc1_scales = h3_gpu_tensor_new_f32(dit->gpu, FFN * 2);
        slot->fc2_int8 = h3_gpu_tensor_new_i8(
            dit->gpu, (size_t)HIDDEN * FFN);
        slot->fc2_scales = h3_gpu_tensor_new_f32(dit->gpu, HIDDEN);
        if (slot->qkv_int8 && slot->qkv_scales &&
            slot->out_int8 && slot->out_scales &&
            slot->fc1_int8 && slot->fc1_scales &&
            slot->fc2_int8 && slot->fc2_scales) return 1;
    } else {
        slot->qkv = h3_gpu_tensor_new_bf16(
            dit->gpu, (size_t)INNER * 3 * HIDDEN);
        slot->out = h3_gpu_tensor_new_bf16(
            dit->gpu, (size_t)HIDDEN * INNER);
        slot->fc1 = h3_gpu_tensor_new_bf16(
            dit->gpu, (size_t)FFN * 2 * HIDDEN);
        slot->fc2 = h3_gpu_tensor_new_bf16(
            dit->gpu, (size_t)HIDDEN * FFN);
        if (slot->qkv && slot->out && slot->fc1 && slot->fc2) return 1;
    }
    fail(error, error_size, "cannot allocate bounded DiT layer slot: %s",
             h3_gpu_error(dit->gpu));
    return 0;
}

static h3_gpu_tensor *stream_slot_target(h3_dit_block *slot,
                                         unsigned field, h3_dtype dtype) {
    if (field == STREAM_QKV)
        return dtype == H3_DTYPE_I8 ? slot->qkv_int8 : slot->qkv;
    if (field == STREAM_QKV_SCALES) return slot->qkv_scales;
    if (field == STREAM_OUT)
        return dtype == H3_DTYPE_I8 ? slot->out_int8 : slot->out;
    if (field == STREAM_OUT_SCALES) return slot->out_scales;
    if (field == STREAM_FC1)
        return dtype == H3_DTYPE_I8 ? slot->fc1_int8 : slot->fc1;
    if (field == STREAM_FC1_SCALES) return slot->fc1_scales;
    if (field == STREAM_FC2)
        return dtype == H3_DTYPE_I8 ? slot->fc2_int8 : slot->fc2;
    if (field == STREAM_FC2_SCALES) return slot->fc2_scales;
    return NULL;
}

typedef struct {
    h3_dit *dit;
    unsigned layer;
    unsigned slot;
    int ok;
    uint64_t bytes;
    double seconds;
    char error[512];
} h3_dit_stream_job;

static int read_stream_layer(h3_dit_stream_job *job) {
    h3_dit_stream_layer *layer = &job->dit->stream_layers[job->layer];
    h3_dit_block *slot = &job->dit->stream_slots[job->slot];
    double started = stream_now();
    job->ok = 1;
    job->bytes = 0;
    job->error[0] = '\0';
    for (unsigned index = 0; index < layer->count; index++) {
        const h3_dit_stream_source *source = &layer->sources[index];
        h3_gpu_tensor *target = stream_slot_target(
            slot, source->field, source->dtype);
        int ok = 0;
        if (target && source->dtype == H3_DTYPE_BF16)
            ok = h3_gpu_tensor_stream_file_bf16(
                target, source->path, source->file_offset, source->elements,
                job->error, sizeof(job->error));
        else if (target && source->dtype == H3_DTYPE_I8)
            ok = h3_gpu_tensor_stream_file_i8(
                target, source->path, source->file_offset, source->elements,
                job->error, sizeof(job->error));
        else if (target && source->dtype == H3_DTYPE_F32)
            ok = h3_gpu_tensor_read_file_f32(
                target, source->path, source->file_offset, source->elements,
                job->error, sizeof(job->error));
        if (!ok) {
            if (!job->error[0])
                snprintf(job->error, sizeof(job->error),
                         "invalid %s streaming destination",
                         h3_dtype_name(source->dtype));
            job->ok = 0;
            break;
        }
        job->bytes += (uint64_t)source->elements *
                      h3_dtype_size(source->dtype);
    }
    job->seconds = stream_now() - started;
    return job->ok;
}

static void *read_stream_layer_thread(void *opaque) {
    read_stream_layer(opaque);
    return NULL;
}

static int quantize_block_mlp(h3_dit *dit, h3_dit_block *block,
                              char *error, size_t error_size) {
    block->fc1_int8 = h3_gpu_tensor_new_i8(
        dit->gpu, (size_t)FFN * 2 * HIDDEN);
    block->fc1_scales = h3_gpu_tensor_new_f32(dit->gpu, FFN * 2);
    block->fc2_int8 = h3_gpu_tensor_new_i8(
        dit->gpu, (size_t)HIDDEN * FFN);
    block->fc2_scales = h3_gpu_tensor_new_f32(dit->gpu, HIDDEN);
    int ok = block->fc1_int8 && block->fc1_scales &&
             block->fc2_int8 && block->fc2_scales &&
             h3_gpu_begin(dit->gpu) &&
             h3_gpu_quantize_weight_int8(
                 dit->gpu, block->fc1_int8, block->fc1_scales, block->fc1,
                 FFN * 2, HIDDEN) &&
             h3_gpu_quantize_weight_int8(
                 dit->gpu, block->fc2_int8, block->fc2_scales, block->fc2,
                 HIDDEN, FFN) &&
             h3_gpu_submit(dit->gpu);
    if (!ok) {
        fail(error, error_size, "cannot quantize DiT MLP weights: %s",
             h3_gpu_error(dit->gpu));
        return 0;
    }
    if (!dit->keep_bf16_mlp) {
        free_tensor(&block->fc1);
        free_tensor(&block->fc2);
    }
    return 1;
}

static int quantize_block_qkv(h3_dit *dit, h3_dit_block *block,
                              char *error, size_t error_size) {
    block->qkv_int8 = h3_gpu_tensor_new_i8(
        dit->gpu, (size_t)INNER * 3 * HIDDEN);
    block->qkv_scales = h3_gpu_tensor_new_f32(dit->gpu, INNER * 3);
    int ok = block->qkv_int8 && block->qkv_scales &&
             h3_gpu_begin(dit->gpu) &&
             h3_gpu_quantize_weight_int8(
                 dit->gpu, block->qkv_int8, block->qkv_scales, block->qkv,
                 INNER * 3, HIDDEN) &&
             h3_gpu_submit(dit->gpu);
    if (!ok) {
        fail(error, error_size, "cannot quantize DiT QKV weight: %s",
             h3_gpu_error(dit->gpu));
        return 0;
    }
    if (!dit->keep_bf16_qkv) free_tensor(&block->qkv);
    return 1;
}

static int quantize_block_attention_out(h3_dit *dit, h3_dit_block *block,
                                        char *error, size_t error_size) {
    block->out_int8 = h3_gpu_tensor_new_i8(
        dit->gpu, (size_t)HIDDEN * INNER);
    block->out_scales = h3_gpu_tensor_new_f32(dit->gpu, HIDDEN);
    int ok = block->out_int8 && block->out_scales &&
             h3_gpu_begin(dit->gpu) &&
             h3_gpu_quantize_weight_int8(
                 dit->gpu, block->out_int8, block->out_scales, block->out,
                 HIDDEN, INNER) &&
             h3_gpu_submit(dit->gpu);
    if (!ok) {
        fail(error, error_size,
             "cannot quantize DiT attention-output weight: %s",
             h3_gpu_error(dit->gpu));
        return 0;
    }
    if (!dit->keep_bf16_attention_out) free_tensor(&block->out);
    return 1;
}

static int run_refiner_block(h3_dit *dit, const h3_dit_block *weight,
                             h3_gpu_tensor *hidden, h3_gpu_tensor *norm,
                             h3_gpu_tensor *qkv, h3_gpu_tensor *query,
                             h3_gpu_tensor *key, h3_gpu_tensor *value,
                             h3_gpu_tensor *heads, h3_gpu_tensor *branch,
                             h3_gpu_tensor *fc1, h3_gpu_tensor *activated,
                             char *error, size_t error_size) {
    uint32_t rows = dit->text_rows;
#define OP(call, label) do {                                                    \
    if (!gpu_op(dit, (call), error, error_size, label)) return 0;               \
} while (0)
    OP(h3_gpu_rms_norm_bf16(dit->gpu, norm, hidden, weight->norm1, rows,
                             HIDDEN, 1e-5f), "refiner attention norm");
    OP(h3_gpu_linear_bf16(dit->gpu, qkv, norm, weight->qkv, NULL, rows,
                           HIDDEN, INNER * 3), "refiner QKV");
    if (!apply_lora(dit, qkv, norm, weight->qkv_lora_a, weight->qkv_lora_b,
                    weight->qkv_lora_rank, rows, HIDDEN, INNER * 3,
                    error, error_size)) return 0;
    OP(h3_gpu_grouped_qkv_rope_bf16(
                             dit->gpu, query, key, value, qkv, weight->q_norm,
                             weight->k_norm, weight->q_norm, weight->q_norm,
                             rows, HEADS, HEAD_DIM, 0, 1e-5f),
       "refiner QK norm");
    OP(h3_gpu_sdpa_bf16(dit->gpu, heads, query, key, value, rows, HEADS,
                         HEAD_DIM, 1.0f / sqrtf((float)HEAD_DIM)),
       "refiner attention");
    OP(h3_gpu_linear_bf16(dit->gpu, branch, heads, weight->out, NULL, rows,
                           INNER, HIDDEN), "refiner attention output");
    if (!apply_lora(dit, branch, heads, weight->out_lora_a, weight->out_lora_b,
                    weight->out_lora_rank, rows, INNER, HIDDEN,
                    error, error_size)) return 0;
    OP(h3_gpu_add_bf16(dit->gpu, hidden, hidden, branch, rows * HIDDEN),
       "refiner attention residual");
    OP(h3_gpu_rms_norm_bf16(dit->gpu, norm, hidden, weight->norm2, rows,
                             HIDDEN, 1e-5f), "refiner MLP norm");
    OP(h3_gpu_linear_bf16(dit->gpu, fc1, norm, weight->fc1, NULL, rows,
                           HIDDEN, FFN * 2), "refiner MLP input");
    if (!apply_lora(dit, fc1, norm, weight->fc1_lora_a, weight->fc1_lora_b,
                    weight->fc1_lora_rank, rows, HIDDEN, FFN * 2,
                    error, error_size)) return 0;
    OP(h3_gpu_swiglu_bf16(dit->gpu, activated, fc1, rows, FFN),
       "refiner SwiGLU");
    OP(h3_gpu_linear_bf16(dit->gpu, branch, activated, weight->fc2, NULL,
                           rows, FFN, HIDDEN), "refiner MLP output");
    if (!apply_lora(dit, branch, activated, weight->fc2_lora_a,
                    weight->fc2_lora_b, weight->fc2_lora_rank,
                    rows, FFN, HIDDEN,
                    error, error_size)) return 0;
    OP(h3_gpu_add_bf16(dit->gpu, hidden, hidden, branch, rows * HIDDEN),
       "refiner MLP residual");
#undef OP
    return 1;
}

static int refine_text(h3_dit *dit, const h3_text_embedding *text,
                       char *error, size_t error_size) {
    h3_gpu_tensor *source = h3_gpu_tensor_from_bf16(
        dit->gpu, text->values, text->tokens * TEXT_DIM);
    h3_gpu_tensor *condition_w = bf2(dit, "condition_proj.weight", HIDDEN,
                                     TEXT_DIM, error, error_size);
    h3_gpu_tensor *condition_b = bf1(dit, "condition_proj.bias", HIDDEN,
                                     error, error_size);
    h3_dit_block refiner[2];
    memset(refiner, 0, sizeof(refiner));
    h3_gpu_tensor *final_norm = NULL;
    h3_gpu_tensor *norm = NULL, *qkv = NULL, *query = NULL, *key = NULL;
    h3_gpu_tensor *value = NULL, *heads = NULL, *branch = NULL, *fc1 = NULL;
    h3_gpu_tensor *activated = NULL;
    int ok = source && condition_w && condition_b &&
        load_block(dit, &refiner[0], "token_refiner.blocks.0.",
                   error, error_size) &&
        load_block(dit, &refiner[1], "token_refiner.blocks.1.",
                   error, error_size);
    if (ok) ok = load_refiner_lora(dit, refiner, error, error_size);
    if (ok) final_norm = bf1(dit, "token_refiner.final_norm.weight", HIDDEN,
                             error, error_size);
    size_t rows = dit->text_rows;
    if (ok && final_norm) {
        dit->refined_text = h3_gpu_tensor_new_bf16(dit->gpu, rows * HIDDEN);
        norm = h3_gpu_tensor_new_bf16(dit->gpu, rows * HIDDEN);
        qkv = h3_gpu_tensor_new_bf16(dit->gpu, rows * INNER * 3);
        query = h3_gpu_tensor_new_bf16(dit->gpu, rows * INNER);
        key = h3_gpu_tensor_new_bf16(dit->gpu, rows * INNER);
        value = h3_gpu_tensor_new_bf16(dit->gpu, rows * INNER);
        heads = h3_gpu_tensor_new_bf16(dit->gpu, rows * INNER);
        branch = h3_gpu_tensor_new_bf16(dit->gpu, rows * HIDDEN);
        fc1 = h3_gpu_tensor_new_bf16(dit->gpu, rows * FFN * 2);
        activated = h3_gpu_tensor_new_bf16(dit->gpu, rows * FFN);
        ok = dit->refined_text && norm && qkv && query && key && value &&
             heads && branch && fc1 && activated;
    }
    if (!ok) {
        if (!error || !*error)
            fail(error, error_size, "cannot allocate token-refiner tensors: %s",
                 h3_gpu_error(dit->gpu));
        goto cleanup;
    }
    ok = gpu_op(dit, h3_gpu_begin(dit->gpu), error, error_size,
                "begin token refinement") &&
         gpu_op(dit, h3_gpu_linear_bf16(
             dit->gpu, dit->refined_text, source, condition_w, condition_b,
             dit->text_rows, TEXT_DIM, HIDDEN), error, error_size,
             "condition projection") &&
         run_refiner_block(dit, &refiner[0], dit->refined_text, norm, qkv,
             query, key, value, heads, branch, fc1, activated,
             error, error_size) &&
         run_refiner_block(dit, &refiner[1], dit->refined_text, norm, qkv,
             query, key, value, heads, branch, fc1, activated,
             error, error_size) &&
         gpu_op(dit, h3_gpu_rms_norm_bf16(
             dit->gpu, dit->refined_text, dit->refined_text, final_norm,
             dit->text_rows, HIDDEN, 1e-5f), error, error_size,
             "refiner final norm") &&
         gpu_op(dit, h3_gpu_submit(dit->gpu), error, error_size,
                "submit token refinement");
cleanup:
    free_tensor(&source);
    free_tensor(&condition_w);
    free_tensor(&condition_b);
    free_block(&refiner[0]);
    free_block(&refiner[1]);
    free_tensor(&final_norm);
    free_tensor(&norm);
    free_tensor(&qkv);
    free_tensor(&query);
    free_tensor(&key);
    free_tensor(&value);
    free_tensor(&heads);
    free_tensor(&branch);
    free_tensor(&fc1);
    free_tensor(&activated);
    return ok;
}

static int prepare_rope(h3_dit *dit, char *error, size_t error_size) {
    h3_gpu_tensor *inverse_tensor = f1(dit, "rope.inv_freq", ROPE_FREQS,
                                       error, error_size);
    float inverse[ROPE_FREQS];
    if (!inverse_tensor ||
        !h3_gpu_tensor_read_f32(inverse_tensor, inverse, ROPE_FREQS)) {
        free_tensor(&inverse_tensor);
        if (!error || !*error) fail(error, error_size, "cannot read RoPE frequencies");
        return 0;
    }
    free_tensor(&inverse_tensor);
    float spatial_scale = dit->spatial_rope_scale;
    size_t count = (size_t)dit->sequence * ROPE_HALF;
    size_t reduced_count = dit->token_reduction ?
        (size_t)dit->reduced_sequence * ROPE_HALF : 0;
    float *cosines = malloc(count * sizeof(*cosines));
    float *sines = malloc(count * sizeof(*sines));
    float *reduced_cosines = reduced_count ?
        malloc(reduced_count * sizeof(*reduced_cosines)) : NULL;
    float *reduced_sines = reduced_count ?
        malloc(reduced_count * sizeof(*reduced_sines)) : NULL;
    if (!cosines || !sines ||
        (reduced_count && (!reduced_cosines || !reduced_sines))) {
        free(cosines);
        free(sines);
        free(reduced_cosines);
        free(reduced_sines);
        fail(error, error_size, "out of memory allocating DiT RoPE tables");
        return 0;
    }
    for (uint32_t row = 0; row < dit->sequence; row++) {
        float axes[] = {(float)dit->layout.positions[row].t,
                        (float)dit->layout.positions[row].h * spatial_scale,
                        (float)dit->layout.positions[row].w * spatial_scale};
        for (uint32_t axis = 0; axis < 3; axis++) {
            for (uint32_t frequency = 0; frequency < ROPE_FREQS; frequency++) {
                size_t index = (size_t)row * ROPE_HALF +
                               axis * ROPE_FREQS + frequency;
                float angle = axes[axis] * inverse[frequency];
                cosines[index] = cosf(angle);
                sines[index] = sinf(angle);
            }
        }
    }
    for (uint32_t row = 0; row < dit->reduced_sequence; row++) {
        uint32_t first, second;
        token_pool_sources(dit, row, &first, &second);
        float axes[] = {
            (float)((dit->layout.positions[first].t +
                     dit->layout.positions[second].t) * 0.5),
            (float)((dit->layout.positions[first].h +
                     dit->layout.positions[second].h) * 0.5) * spatial_scale,
            (float)((dit->layout.positions[first].w +
                     dit->layout.positions[second].w) * 0.5) * spatial_scale
        };
        for (uint32_t axis = 0; axis < 3; axis++) {
            for (uint32_t frequency = 0; frequency < ROPE_FREQS; frequency++) {
                size_t index = (size_t)row * ROPE_HALF +
                               axis * ROPE_FREQS + frequency;
                float angle = axes[axis] * inverse[frequency];
                reduced_cosines[index] = cosf(angle);
                reduced_sines[index] = sinf(angle);
            }
        }
    }
    h3_gpu_tensor *cos_f32 = h3_gpu_tensor_from_f32(dit->gpu, cosines, count);
    h3_gpu_tensor *sin_f32 = h3_gpu_tensor_from_f32(dit->gpu, sines, count);
    h3_gpu_tensor *reduced_cos_f32 = reduced_count ?
        h3_gpu_tensor_from_f32(dit->gpu, reduced_cosines, reduced_count) : NULL;
    h3_gpu_tensor *reduced_sin_f32 = reduced_count ?
        h3_gpu_tensor_from_f32(dit->gpu, reduced_sines, reduced_count) : NULL;
    free(cosines);
    free(sines);
    free(reduced_cosines);
    free(reduced_sines);
    dit->rope_cos = h3_gpu_tensor_new_bf16(dit->gpu, count);
    dit->rope_sin = h3_gpu_tensor_new_bf16(dit->gpu, count);
    if (reduced_count) {
        dit->reduced_rope_cos = h3_gpu_tensor_new_bf16(
            dit->gpu, reduced_count);
        dit->reduced_rope_sin = h3_gpu_tensor_new_bf16(
            dit->gpu, reduced_count);
    }
    int ok = cos_f32 && sin_f32 && dit->rope_cos && dit->rope_sin &&
        (!reduced_count || (reduced_cos_f32 && reduced_sin_f32 &&
                            dit->reduced_rope_cos &&
                            dit->reduced_rope_sin));
    if (ok) {
        ok = gpu_op(dit, h3_gpu_begin(dit->gpu), error, error_size,
                    "begin RoPE setup") &&
             gpu_op(dit, h3_gpu_cast_f32_to_bf16(
                 dit->gpu, dit->rope_cos, cos_f32, (uint32_t)count),
                 error, error_size, "RoPE cosine cast") &&
             gpu_op(dit, h3_gpu_cast_f32_to_bf16(
                 dit->gpu, dit->rope_sin, sin_f32, (uint32_t)count),
                 error, error_size, "RoPE sine cast");
        if (ok && reduced_count) {
            ok = gpu_op(dit, h3_gpu_cast_f32_to_bf16(
                     dit->gpu, dit->reduced_rope_cos, reduced_cos_f32,
                     (uint32_t)reduced_count), error, error_size,
                     "reduced RoPE cosine cast") &&
                 gpu_op(dit, h3_gpu_cast_f32_to_bf16(
                     dit->gpu, dit->reduced_rope_sin, reduced_sin_f32,
                     (uint32_t)reduced_count), error, error_size,
                     "reduced RoPE sine cast");
        }
        if (ok) ok =
             gpu_op(dit, h3_gpu_submit(dit->gpu), error, error_size,
                    "submit RoPE setup");
    } else if (!error || !*error) {
        fail(error, error_size, "cannot allocate DiT RoPE buffers: %s",
             h3_gpu_error(dit->gpu));
    }
    free_tensor(&cos_f32);
    free_tensor(&sin_f32);
    free_tensor(&reduced_cos_f32);
    free_tensor(&reduced_sin_f32);
    return ok;
}

static int prepare_maps(h3_dit *dit, const h3_text_embedding *text,
                        char *error, size_t error_size) {
    int steps = h3_dit_schedule_steps(dit->schedule);
    dit->row_maps = calloc((size_t)steps, sizeof(*dit->row_maps));
    if (dit->token_reduction)
        dit->reduced_row_maps = calloc((size_t)steps,
                                       sizeof(*dit->reduced_row_maps));
    dit->final_audio_maps = calloc((size_t)steps,
                                   sizeof(*dit->final_audio_maps));
    dit->final_video_maps = calloc((size_t)steps,
                                   sizeof(*dit->final_video_maps));
    uint32_t *rows = malloc((size_t)dit->sequence * sizeof(*rows));
    uint32_t *reduced = dit->token_reduction ?
        malloc((size_t)dit->reduced_sequence * sizeof(*reduced)) : NULL;
    uint32_t *audio = malloc((size_t)dit->audio_rows * sizeof(*audio));
    uint32_t *video = malloc((size_t)dit->video_rows * sizeof(*video));
    if (!dit->row_maps || !dit->final_audio_maps || !dit->final_video_maps ||
        (dit->token_reduction && (!dit->reduced_row_maps || !reduced)) ||
        !rows || !audio || !video) {
        fail(error, error_size, "out of memory allocating modulation row maps");
        free(rows); free(reduced); free(audio); free(video);
        return 0;
    }
    for (int step = 0; step < steps; step++) {
        if (!h3_dit_schedule_row_map(
                dit->schedule, step, &dit->layout,
                text->tags, text->tokens,
                dit->inpaint_video_generate_rows,
                dit->inpaint_video_generate_count,
                dit->inpaint_audio_generate_rows,
                dit->inpaint_audio_generate_count,
                rows, dit->sequence)) {
            fail(error, error_size, "cannot construct modulation row map");
            free(rows); free(reduced); free(audio); free(video);
            return 0;
        }
        if (dit->token_reduction) {
            for (uint32_t row = 0; row < dit->reduced_sequence; row++) {
                uint32_t first, second;
                token_pool_sources(dit, row, &first, &second);
                (void)second;
                reduced[row] = rows[first];
            }
            dit->reduced_row_maps[step] = h3_gpu_tensor_from_u32(
                dit->gpu, reduced, dit->reduced_sequence);
        }
        uint32_t audio_row = h3_dit_schedule_audio_row(dit->schedule, step);
        uint32_t video_row = h3_dit_schedule_video_row(dit->schedule, step);
        uint32_t audio_condition_row =
            h3_dit_schedule_audio_condition_row(dit->schedule, step);
        uint32_t video_condition_row =
            h3_dit_schedule_visual_condition_row(dit->schedule, step);
        for (uint32_t index = 0; index < dit->audio_rows; index++)
            audio[index] = dit->inpaint_audio_generate_rows &&
                           !dit->inpaint_audio_generate_rows[index]
                ? audio_condition_row : audio_row;
        for (uint32_t index = 0; index < dit->video_rows; index++)
            video[index] = dit->inpaint_video_generate_rows &&
                           !dit->inpaint_video_generate_rows[index]
                ? video_condition_row : video_row;
        dit->row_maps[step] = h3_gpu_tensor_from_u32(
            dit->gpu, rows, dit->sequence);
        dit->final_audio_maps[step] = h3_gpu_tensor_from_u32(
            dit->gpu, audio, dit->audio_rows);
        dit->final_video_maps[step] = h3_gpu_tensor_from_u32(
            dit->gpu, video, dit->video_rows);
        if (!dit->row_maps[step] ||
            (dit->token_reduction && !dit->reduced_row_maps[step]) ||
            !dit->final_audio_maps[step] ||
            !dit->final_video_maps[step]) {
            fail(error, error_size, "cannot allocate modulation row maps: %s",
                 h3_gpu_error(dit->gpu));
            free(rows); free(reduced); free(audio); free(video);
            return 0;
        }
    }
    free(rows); free(reduced); free(audio); free(video);
    return 1;
}

static int prepare_projection_maps(h3_dit *dit, char *error,
                                   size_t error_size) {
    unsigned video_segments = 0, audio_segments = 0;
    for (size_t index = 0; index < dit->layout.segment_count; index++) {
        h3_segment_kind kind = dit->layout.segments[index].kind;
        if (kind == H3_SEG_COND || kind == H3_SEG_REF_IMAGE ||
            kind == H3_SEG_VIDEO)
            video_segments++;
        else if (kind != H3_SEG_TEXT)
            audio_segments++;
    }
    uint32_t *video = video_segments > 1 ?
        malloc((size_t)dit->video_total_rows * sizeof(*video)) : NULL;
    uint32_t *audio = audio_segments > 1 ?
        malloc((size_t)dit->audio_total_rows * sizeof(*audio)) : NULL;
    if ((video_segments > 1 && !video) || (audio_segments > 1 && !audio)) {
        free(video); free(audio);
        fail(error, error_size, "out of memory allocating projection maps");
        return 0;
    }
    size_t video_offset = 0, audio_offset = 0;
    for (size_t index = 0; index < dit->layout.segment_count; index++) {
        const h3_segment *segment = &dit->layout.segments[index];
        size_t rows = segment->stop - segment->start;
        if (segment->kind == H3_SEG_COND ||
            segment->kind == H3_SEG_REF_IMAGE ||
            segment->kind == H3_SEG_VIDEO) {
            for (size_t row = 0; video && row < rows; row++)
                video[video_offset + row] = (uint32_t)(segment->start + row);
            video_offset += rows;
        } else if (segment->kind != H3_SEG_TEXT) {
            for (size_t row = 0; audio && row < rows; row++)
                audio[audio_offset + row] = (uint32_t)(segment->start + row);
            audio_offset += rows;
        }
    }
    if (video_offset != dit->video_total_rows ||
        audio_offset != dit->audio_total_rows) {
        free(video); free(audio);
        fail(error, error_size, "projection map rows are inconsistent");
        return 0;
    }
    if (video)
        dit->video_projection_map = h3_gpu_tensor_from_u32(
            dit->gpu, video, dit->video_total_rows);
    if (audio)
        dit->audio_projection_map = h3_gpu_tensor_from_u32(
            dit->gpu, audio, dit->audio_total_rows);
    free(video); free(audio);
    if ((video_segments > 1 && !dit->video_projection_map) ||
        (audio_segments > 1 && !dit->audio_projection_map)) {
        fail(error, error_size, "cannot allocate projection map tensors: %s",
             h3_gpu_error(dit->gpu));
        return 0;
    }
    return 1;
}

static int prepare_token_reduction_maps(h3_dit *dit, char *error,
                                        size_t error_size) {
    if (!dit->token_reduction) return 1;
    size_t pair_count = (size_t)dit->reduced_sequence * 2;
    uint32_t *pairs = malloc(pair_count * sizeof(*pairs));
    uint32_t *baseline_indices = malloc(
        (size_t)dit->reduced_sequence * sizeof(*baseline_indices));
    uint32_t *parents = malloc((size_t)dit->sequence * sizeof(*parents));
    if (!pairs || !baseline_indices || !parents) {
        free(pairs);
        free(baseline_indices);
        free(parents);
        fail(error, error_size,
             "out of memory allocating token-reduction maps");
        return 0;
    }
    uint32_t baseline_row = 0;
    for (uint32_t row = 0; row < dit->reduced_sequence; row++) {
        token_pool_sources(dit, row, &pairs[(size_t)row * 2],
                           &pairs[(size_t)row * 2 + 1]);
        baseline_indices[row] =
            row >= dit->video_target_start &&
            pairs[(size_t)row * 2] != pairs[(size_t)row * 2 + 1] ?
                baseline_row++ : UINT32_MAX;
    }
    if (baseline_row != dit->token_baseline_rows) {
        free(pairs);
        free(baseline_indices);
        free(parents);
        fail(error, error_size, "token-reduction baseline map is inconsistent");
        return 0;
    }
    for (uint32_t row = 0; row < dit->sequence; row++)
        parents[row] = token_reduced_parent(dit, row);
    dit->token_pool_pairs = h3_gpu_tensor_from_u32(
        dit->gpu, pairs, pair_count);
    dit->token_baseline_indices = h3_gpu_tensor_from_u32(
        dit->gpu, baseline_indices, dit->reduced_sequence);
    dit->token_expand_parents = h3_gpu_tensor_from_u32(
        dit->gpu, parents, dit->sequence);
    free(pairs);
    free(baseline_indices);
    free(parents);
    if (!dit->token_pool_pairs || !dit->token_baseline_indices ||
        !dit->token_expand_parents) {
        fail(error, error_size,
             "cannot allocate token-reduction map tensors: %s",
             h3_gpu_error(dit->gpu));
        return 0;
    }
    return 1;
}

static void configure_active_blocks(h3_dit *dit, unsigned active) {
    memset(dit->block_active, 1, sizeof(dit->block_active));
    dit->active_block_count = active;
    unsigned skipped = H3_DIT_BLOCKS - active;
    for (unsigned index = 0; index < skipped; index++) {
        unsigned block = ((2 * index + 1) * H3_DIT_BLOCKS) / (2 * skipped);
        if (block == 0) block = 1;
        if (block >= H3_DIT_BLOCKS - 1) block = H3_DIT_BLOCKS - 2;
        dit->block_active[block] = 0;
    }
}

static unsigned first_active_block(const h3_dit *dit) {
    for (unsigned block = 0; block < H3_DIT_BLOCKS; block++)
        if (dit->block_active[block]) return block;
    return H3_DIT_BLOCKS;
}

static unsigned next_active_block(const h3_dit *dit, unsigned current) {
    for (unsigned block = current + 1; block < H3_DIT_BLOCKS; block++)
        if (dit->block_active[block]) return block;
    return H3_DIT_BLOCKS;
}

/* TE-Speed style block cache. On a full step the tail residual — the change
 * the blocks after the warm prefix contribute — is captured; on a gated step
 * only the warm prefix runs and the residual is replayed. The gate is pure
 * schedule arithmetic (sigma deltas), so the full/cached pattern is decided
 * up front and the streaming prefetch chain can follow it exactly. */
#define H3_CACHE_THRESHOLD 0.12f
#define H3_CACHE_WINDOW_START 0.10f
#define H3_CACHE_WINDOW_END 0.90f
#define H3_CACHE_MAX_CONSECUTIVE 2
#define H3_CACHE_WARM_FRACTION 0.25f

int h3_dit_set_block_cache(h3_dit *dit, int enabled,
                           char *error, size_t error_size) {
    if (!dit) return 0;
    dit->block_cache = 0;
    dit->cache_plan_ready = 0;
    dit->cache_residual_ready = 0;
    if (!enabled) return 1;
    if (dit->core_reuse_interval > 1 || dit->token_reduction) {
        fail(error, error_size,
             "the block cache cannot combine with core reuse or token "
             "reduction");
        return 0;
    }
    unsigned active = 0, warm_last = H3_DIT_BLOCKS;
    unsigned warm_target = 0;
    for (unsigned block = 0; block < H3_DIT_BLOCKS; block++)
        if (dit->block_active[block]) active++;
    if (active < 4) {
        fail(error, error_size, "the block cache needs at least four blocks");
        return 0;
    }
    warm_target = (unsigned)((float)active * H3_CACHE_WARM_FRACTION + 0.5f);
    if (warm_target < 1) warm_target = 1;
    for (unsigned block = 0, seen = 0; block < H3_DIT_BLOCKS; block++) {
        if (!dit->block_active[block]) continue;
        if (++seen == warm_target) { warm_last = block; break; }
    }
    if (!dit->cache_snapshot)
        dit->cache_snapshot = h3_gpu_tensor_new_bf16(
            dit->gpu, (size_t)dit->sequence * HIDDEN);
    if (!dit->cache_residual)
        dit->cache_residual = h3_gpu_tensor_new_bf16(
            dit->gpu, (size_t)dit->sequence * HIDDEN);
    if (!dit->cache_snapshot || !dit->cache_residual) {
        fail(error, error_size, "cannot allocate the block cache: %s",
             h3_gpu_error(dit->gpu));
        return 0;
    }
    dit->cache_warm_last = warm_last;
    dit->block_cache = 1;
    return 1;
}

/* Decides full versus cached per step from the sigma schedule. Cached steps
 * need small sigma deltas, sit inside the middle of the schedule, and never
 * run more than H3_CACHE_MAX_CONSECUTIVE in a row; the first and last steps
 * always run full. */
static void h3_dit_block_cache_plan(h3_dit *dit) {
    dit->cache_plan_ready = 0;
    dit->cache_residual_ready = 0;
    if (!dit->block_cache) return;
    int steps = dit->sigmas.steps;
    int consecutive = 0, cached = 0;
    for (int step = 0; step < steps; step++) {
        int full = 1;
        if (step > 0 && step < steps - 1 && steps > 3) {
            float pos = (float)step / (float)(steps - 1);
            float video_delta = fabsf(dit->sigmas.video[step] -
                                      dit->sigmas.video[step - 1]);
            float audio_delta = fabsf(dit->sigmas.audio[step] -
                                      dit->sigmas.audio[step - 1]);
            float delta = video_delta > audio_delta ? video_delta : audio_delta;
            if (pos >= H3_CACHE_WINDOW_START && pos <= H3_CACHE_WINDOW_END &&
                delta < H3_CACHE_THRESHOLD &&
                consecutive < H3_CACHE_MAX_CONSECUTIVE)
                full = 0;
        }
        dit->cache_full_step[step] = (uint8_t)full;
        consecutive = full ? 0 : consecutive + 1;
        cached += !full;
    }
    dit->cache_plan_ready = 1;
    if (getenv("H3_PROFILE"))
        fprintf(stderr,
                "h3: block cache plan: %d of %d steps cached, warm prefix "
                "ends at block %u\n",
                cached, steps, dit->cache_warm_last);
}

static void configure_gate_ranked_blocks(h3_dit *dit) {
    const char *policy = getenv("H3_DIT_LAYER_POLICY");
    if ((policy && !strcmp(policy, "uniform")) ||
        dit->active_block_count == H3_DIT_BLOCKS) return;
    typedef struct { unsigned block; double score; } block_score;
    /* The first two and final blocks establish/close the residual stream.
     * Block 1 has a small gate but proved structurally essential in decoded
     * A/B renders, so magnitude ranking must not treat it as disposable. */
    block_score scores[H3_DIT_BLOCKS - 3];
    for (unsigned block = 2; block + 1 < H3_DIT_BLOCKS; block++) {
        double score = h3_dit_schedule_gate_score(dit->schedule, block);
        if (score < 0.0) return;
        scores[block - 2] = (block_score){block, score};
    }
    unsigned count = H3_DIT_BLOCKS - 3;
    for (unsigned left = 0; left < count; left++) {
        unsigned least = left;
        for (unsigned right = left + 1; right < count; right++)
            if (scores[right].score < scores[least].score) least = right;
        block_score temporary = scores[left];
        scores[left] = scores[least];
        scores[least] = temporary;
    }
    memset(dit->block_active, 1, sizeof(dit->block_active));
    unsigned skipped = H3_DIT_BLOCKS - dit->active_block_count;
    for (unsigned index = 0; index < skipped; index++)
        dit->block_active[scores[index].block] = 0;
    if (getenv("H3_PROFILE")) {
        fprintf(stderr, "h3: gate-ranked DiT skips");
        for (unsigned index = 0; index < skipped; index++)
            fprintf(stderr, " %u(%.4g)", scores[index].block,
                    scores[index].score);
        fputc('\n', stderr);
    }
}

static int load_core(h3_dit *dit, h3_dit_progress progress, void *opaque,
                     char *error, size_t error_size) {
    for (unsigned index = 0; index < H3_DIT_BLOCKS; index++) {
        if (!dit->block_active[index]) {
            report(progress, opaque, "load transformer core", (int)index + 1,
                   H3_DIT_BLOCKS);
            continue;
        }
        char prefix[64];
        snprintf(prefix, sizeof(prefix), "blocks.%u.", index);
        if (dit->prequantized_int8) {
            if (dit->ssd_streaming) {
                if (!load_block_norms(
                        dit, &dit->blocks[index], prefix,
                        error, error_size) ||
                    !prepare_stream_layer(dit, index, error, error_size))
                    return 0;
            } else if (!load_prequantized_block(
                    dit, &dit->blocks[index], prefix,
                    error, error_size)) return 0;
        } else if (dit->ssd_streaming) {
            if (!load_block_norms(dit, &dit->blocks[index], prefix,
                                  error, error_size) ||
                !prepare_stream_layer(dit, index, error, error_size))
                return 0;
        } else {
            if (!load_block(dit, &dit->blocks[index], prefix,
                            error, error_size)) return 0;
            if (dit->int8_mlp &&
                !quantize_block_mlp(dit, &dit->blocks[index],
                                    error, error_size)) return 0;
            if (dit->int8_qkv &&
                !quantize_block_qkv(dit, &dit->blocks[index],
                                    error, error_size)) return 0;
            if (dit->int8_attention_out &&
                !quantize_block_attention_out(
                    dit, &dit->blocks[index], error, error_size)) return 0;
        }
        report(progress, opaque, "load transformer core", (int)index + 1,
               H3_DIT_BLOCKS);
    }
    if (dit->ssd_streaming) {
        if (!allocate_stream_slot(dit, &dit->stream_slots[0],
                                  error, error_size) ||
            !allocate_stream_slot(dit, &dit->stream_slots[1],
                                  error, error_size)) return 0;
        unsigned first = first_active_block(dit);
        if (first == H3_DIT_BLOCKS) {
            fail(error, error_size, "SSD stream has no active DiT block");
            return 0;
        }
        h3_dit_stream_job job = {
            .dit = dit, .layer = first, .slot = 0
        };
        if (!read_stream_layer(&job)) {
            fail(error, error_size, "cannot prime DiT SSD stream: %s",
                 job.error);
            return 0;
        }
        dit->stream_ready_layer = first;
        dit->stream_ready_slot = 0;
        dit->stream_bytes += job.bytes;
        dit->stream_read_seconds += job.seconds;
    }
    dit->video_patch_w = f2(dit, "video_patch_proj.weight", HIDDEN,
                            VIDEO_PATCH, error, error_size);
    dit->video_patch_b = f1(dit, "video_patch_proj.bias", HIDDEN,
                            error, error_size);
    dit->audio_patch_w = f2(dit, "audio_patch_proj.weight", HIDDEN,
                            AUDIO_CHANNELS, error, error_size);
    dit->audio_patch_b = f1(dit, "audio_patch_proj.bias", HIDDEN,
                            error, error_size);
    dit->final_norm = bf1(dit, "final_layer.norm.weight", HIDDEN,
                          error, error_size);
    dit->final_video_w = f2(dit, "final_layer.video_out.weight", VIDEO_PATCH,
                            HIDDEN, error, error_size);
    dit->final_video_b = f1(dit, "final_layer.video_out.bias", VIDEO_PATCH,
                            error, error_size);
    dit->final_audio_w = f2(dit, "final_layer.audio_out.weight", AUDIO_CHANNELS,
                            HIDDEN, error, error_size);
    dit->final_audio_b = f1(dit, "final_layer.audio_out.bias", AUDIO_CHANNELS,
                            error, error_size);
    if (dit->bf16_final && dit->final_video_w && dit->final_video_b &&
        dit->final_audio_w && dit->final_audio_b) {
        h3_gpu_tensor *source[4] = {
            dit->final_video_w, dit->final_video_b,
            dit->final_audio_w, dit->final_audio_b
        };
        size_t elements[4] = {
            (size_t)VIDEO_PATCH * HIDDEN, VIDEO_PATCH,
            (size_t)AUDIO_CHANNELS * HIDDEN, AUDIO_CHANNELS
        };
        h3_gpu_tensor *target[4] = {0};
        int ok = 1;
        for (unsigned index = 0; index < 4; index++) {
            target[index] = h3_gpu_tensor_new_bf16(dit->gpu,
                                                    elements[index]);
            if (!target[index]) ok = 0;
        }
        if (ok) ok = h3_gpu_begin(dit->gpu);
        for (unsigned index = 0; ok && index < 4; index++)
            ok = h3_gpu_cast_f32_to_bf16(dit->gpu, target[index],
                                         source[index],
                                         (uint32_t)elements[index]);
        if (ok) ok = h3_gpu_submit(dit->gpu);
        if (ok) {
            for (unsigned index = 0; index < 4; index++)
                h3_gpu_tensor_free(source[index]);
            dit->final_video_w = target[0];
            dit->final_video_b = target[1];
            dit->final_audio_w = target[2];
            dit->final_audio_b = target[3];
        } else {
            for (unsigned index = 0; index < 4; index++)
                h3_gpu_tensor_free(target[index]);
            fail(error, error_size, "cannot convert DiT final weights: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
    }
    return dit->video_patch_w && dit->video_patch_b && dit->audio_patch_w &&
           dit->audio_patch_b && dit->final_norm && dit->final_video_w &&
           dit->final_video_b && dit->final_audio_w && dit->final_audio_b;
}

static int allocate_activations(h3_dit *dit, char *error, size_t error_size) {
    size_t sequence = dit->sequence;
    size_t audio = dit->audio_rows;
    size_t video = dit->video_rows;
    size_t audio_total = dit->audio_total_rows;
    size_t video_total = dit->video_total_rows;
    dit->activation_aliases = !getenv("H3_DISABLE_DIT_ACTIVATION_ALIAS");
    dit->fused_patch_projection =
        !getenv("H3_DISABLE_FUSED_PATCH_CAST") && !getenv("H3_SCALAR_PATCH");
    dit->fused_patch_pack = dit->fused_patch_projection &&
        !getenv("H3_DISABLE_FUSED_PATCH_PACK");
#define BF(field, elements) (dit->field = h3_gpu_tensor_new_bf16(dit->gpu, (elements)))
#define F32(field, elements) (dit->field = h3_gpu_tensor_new_f32(dit->gpu, (elements)))
    h3_gpu_tensor *all[] = {
        F32(video_input, video_total * VIDEO_PATCH),
        F32(audio_input, audio_total * AUDIO_CHANNELS),
        BF(hidden, sequence * HIDDEN),
        BF(mod_attention, sequence * HIDDEN),
        BF(qkv, sequence * INNER * 3),
        BF(query, sequence * INNER),
        BF(key, sequence * INNER),
        BF(value, sequence * INNER),
        BF(attention_output, sequence * HIDDEN),
        F32(final_audio_inverse, audio),
        F32(final_video_inverse, video),
        BF(audio_output_bf16, audio * AUDIO_CHANNELS),
        BF(video_output_bf16, video * VIDEO_PATCH)
    };
#undef BF
#undef F32
    for (size_t index = 0; index < sizeof(all) / sizeof(*all); index++) {
        if (!all[index]) {
            fail(error, error_size, "cannot allocate DiT activation arena: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
    }
    /* Four f32 buffers for attention, allocated only where it will use them
     * and only where they are affordable.
     *
     * They grow with the sequence: 146 MB each at 5095 rows, 622 MB each at
     * the 21700 a 768-canvas five-second clip runs, which is 2.5 GB on top of
     * 21 GB of weights. That was enough to push a 32 GB machine into swap and
     * get the process killed mid-pass. Speed is worth nothing if the run does
     * not finish, so the trade is refused when it is a large share of the
     * machine; the bf16 path is slower and always correct. */
    const uint64_t f32_attention_bytes =
        (uint64_t)sequence * INNER * sizeof(float) * 4u;
    if (sequence > H3_DIT_F32_ATTENTION_ROWS &&
        f32_attention_bytes <= h3_dit_memory_budget() &&
        !getenv("H3_DISABLE_F32_ATTENTION")) {
        dit->query32 = h3_gpu_tensor_new_f32(dit->gpu, sequence * INNER);
        dit->key32 = h3_gpu_tensor_new_f32(dit->gpu, sequence * INNER);
        dit->value32 = h3_gpu_tensor_new_f32(dit->gpu, sequence * INNER);
        dit->heads32 = h3_gpu_tensor_new_f32(dit->gpu, sequence * INNER);
        if (!dit->query32 || !dit->key32 || !dit->value32 || !dit->heads32) {
            h3_gpu_tensor_free(dit->query32); dit->query32 = NULL;
            h3_gpu_tensor_free(dit->key32); dit->key32 = NULL;
            h3_gpu_tensor_free(dit->value32); dit->value32 = NULL;
            h3_gpu_tensor_free(dit->heads32); dit->heads32 = NULL;
            if (getenv("H3_PROFILE"))
                fprintf(stderr, "h3: no room for f32 attention; using bf16\n");
        }
    }
    if (dit->sol_attention) {
        size_t blocks = (sequence + 63u) / 64u;
        size_t summaries = (size_t)HEADS * blocks * HEAD_DIM;
        size_t statistics = (size_t)HEADS * HEAD_DIM;
        size_t thresholds = (size_t)HEADS * blocks;
        dit->sol_query_centroids = h3_gpu_tensor_new_f32(
            dit->gpu, summaries);
        dit->sol_key_centroids = h3_gpu_tensor_new_bf16(
            dit->gpu, summaries);
        dit->sol_value_sums = h3_gpu_tensor_new_bf16(
            dit->gpu, summaries);
        dit->sol_key_means = h3_gpu_tensor_new_f32(
            dit->gpu, statistics);
        dit->sol_key_variances = h3_gpu_tensor_new_f32(
            dit->gpu, statistics);
        dit->sol_thresholds = h3_gpu_tensor_new_f32(
            dit->gpu, thresholds);
        if (!dit->sol_query_centroids || !dit->sol_key_centroids ||
            !dit->sol_value_sums || !dit->sol_key_means ||
            !dit->sol_key_variances || !dit->sol_thresholds) {
            const char *strict = getenv("H3_SOL_ATTN_STRICT");
            if (strict && *strict && strcmp(strict, "0")) {
                fail(error, error_size,
                     "cannot allocate Sol-Attn routing buffers: %s",
                     h3_gpu_error(dit->gpu));
                return 0;
            }
            free_tensor(&dit->sol_query_centroids);
            free_tensor(&dit->sol_key_centroids);
            free_tensor(&dit->sol_value_sums);
            free_tensor(&dit->sol_key_means);
            free_tensor(&dit->sol_key_variances);
            free_tensor(&dit->sol_thresholds);
            dit->sol_attention = 0;
            if (getenv("H3_PROFILE"))
                fprintf(stderr,
                        "h3: no room for Sol-Attn routing; using dense attention\n");
        }
        const char *profile_routes = getenv("H3_PROFILE_SOL_ROUTES");
        if (dit->sol_attention && profile_routes && *profile_routes &&
            strcmp(profile_routes, "0")) {
            if (blocks > SIZE_MAX / blocks / HEADS) {
                fail(error, error_size,
                     "Sol-Attn route diagnostic size overflows");
                return 0;
            }
            dit->sol_route_count = (size_t)HEADS * blocks * blocks;
            dit->sol_routes = h3_gpu_tensor_new_i8(
                dit->gpu, dit->sol_route_count);
            if (!dit->sol_routes) {
                fail(error, error_size,
                     "cannot allocate Sol-Attn route diagnostics: %s",
                     h3_gpu_error(dit->gpu));
                return 0;
            }
        }
    }
    if (!dit->fused_patch_pack) {
        dit->video_projected = h3_gpu_tensor_new_bf16(
            dit->gpu, video_total * HIDDEN);
        dit->audio_projected = h3_gpu_tensor_new_bf16(
            dit->gpu, audio_total * HIDDEN);
        if (!dit->video_projected || !dit->audio_projected) {
            fail(error, error_size,
                 "cannot allocate packed patch projections: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
    }
    if (!dit->fused_patch_projection) {
        dit->video_projected_f32 = h3_gpu_tensor_new_f32(
            dit->gpu, video_total * HIDDEN);
        dit->audio_projected_f32 = h3_gpu_tensor_new_f32(
            dit->gpu, audio_total * HIDDEN);
        if (!dit->video_projected_f32 || !dit->audio_projected_f32) {
            fail(error, error_size,
                 "cannot allocate separate patch projections: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
    }
    if (dit->activation_aliases) {
        dit->attention_heads = dit->qkv;
        dit->mod_mlp = dit->qkv;
        dit->mlp_output = NULL;
    } else {
        dit->attention_heads = h3_gpu_tensor_new_bf16(
            dit->gpu, sequence * INNER);
        dit->mod_mlp = h3_gpu_tensor_new_bf16(
            dit->gpu, sequence * HIDDEN);
        dit->mlp_output = h3_gpu_tensor_new_bf16(
            dit->gpu, sequence * HIDDEN);
    }
    if (!dit->attention_heads || !dit->mod_mlp ||
        (!dit->activation_aliases && !dit->mlp_output)) {
        fail(error, error_size,
             "cannot allocate DiT activation buffers: %s",
             h3_gpu_error(dit->gpu));
        return 0;
    }
    if (getenv("H3_DISABLE_FUSED_FINAL_SLICE")) {
        dit->final_audio_input = h3_gpu_tensor_new_bf16(
            dit->gpu, audio * HIDDEN);
        dit->final_video_input = h3_gpu_tensor_new_bf16(
            dit->gpu, video * HIDDEN);
        if (!dit->final_audio_input || !dit->final_video_input) {
            fail(error, error_size,
                 "cannot allocate separate final DiT slices: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
    }
    if (!dit->bf16_final || getenv("H3_DISABLE_FUSED_FINAL_HEAD") ||
        getenv("H3_DISABLE_FUSED_FINAL_SLICE")) {
        dit->final_audio_norm = h3_gpu_tensor_new_bf16(
            dit->gpu, audio * HIDDEN);
        dit->final_video_norm = h3_gpu_tensor_new_bf16(
            dit->gpu, video * HIDDEN);
        if (!dit->final_audio_norm || !dit->final_video_norm) {
            fail(error, error_size,
                 "cannot allocate separate final DiT normalization: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
    }
    if (!dit->bf16_final) {
        dit->final_audio_f32 = h3_gpu_tensor_new_f32(
            dit->gpu, audio * HIDDEN);
        dit->final_video_f32 = h3_gpu_tensor_new_f32(
            dit->gpu, video * HIDDEN);
        dit->audio_output = h3_gpu_tensor_new_f32(
            dit->gpu, audio * AUDIO_CHANNELS);
        dit->video_output = h3_gpu_tensor_new_f32(
            dit->gpu, video * VIDEO_PATCH);
        if (!dit->final_audio_f32 || !dit->final_video_f32 ||
            !dit->audio_output || !dit->video_output) {
            fail(error, error_size,
                 "cannot allocate F32 DiT final activations: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
    }
    if (!dit->fused_mlp || dit->prequantized_int8) {
        dit->fc1 = h3_gpu_tensor_new_bf16(dit->gpu, sequence * FFN * 2);
    }
    if (!dit->fused_mlp || dit->nax_mlp || dit->int8_mlp ||
        dit->prequantized_int8) {
        dit->activated = h3_gpu_tensor_new_bf16(dit->gpu, sequence * FFN);
        if (((!dit->fused_mlp || dit->prequantized_int8) && !dit->fc1) ||
            !dit->activated) {
            fail(error, error_size,
                 "cannot allocate diagnostic DiT MLP tensors: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
    }
    if (dit->int8_mlp || dit->int8_qkv || dit->int8_attention_out) {
        size_t padded_sequence = (sequence + 127) & ~(size_t)127;
        dit->int8_activation = h3_gpu_tensor_new_i8(
            dit->gpu, padded_sequence * FFN);
        dit->int8_activation_scales = h3_gpu_tensor_new_f32(
            dit->gpu, padded_sequence * (FFN / 1024));
        if (!dit->int8_activation || !dit->int8_activation_scales) {
            fail(error, error_size,
                 "cannot allocate int8 DiT activation arena: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
    }
    if (dit->token_reduction) {
        size_t full_elements = sequence * HIDDEN;
        size_t qkv_capacity = sequence * INNER * 3;
        size_t qkv_used = (size_t)dit->reduced_sequence * INNER * 3;
        size_t baseline_elements =
            (size_t)dit->token_baseline_rows * HIDDEN;
        size_t attention_capacity = sequence * HIDDEN;
        size_t attention_used =
            (size_t)dit->reduced_sequence * HIDDEN;
        dit->token_original_in_qkv =
            qkv_used <= qkv_capacity &&
            full_elements <= qkv_capacity - qkv_used &&
            qkv_used <= UINT32_MAX &&
            full_elements <= UINT32_MAX - qkv_used;
        if (dit->token_original_in_qkv)
            dit->token_original_offset = qkv_used;
        else
            dit->token_original = h3_gpu_tensor_new_bf16(
                dit->gpu, full_elements);
        dit->token_baseline_offset = attention_used;
        if (attention_used > attention_capacity ||
            baseline_elements > attention_capacity - attention_used ||
            attention_used > UINT32_MAX ||
            baseline_elements > UINT32_MAX - attention_used ||
            (!dit->token_original_in_qkv && !dit->token_original)) {
            fail(error, error_size,
                 "cannot allocate token-reduction residual state: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
    }
    if (dit->core_reuse_interval > 1) {
        dit->core_input = h3_gpu_tensor_new_bf16(
            dit->gpu, sequence * HIDDEN);
        dit->core_residual = h3_gpu_tensor_new_bf16(
            dit->gpu, sequence * HIDDEN);
        if (!dit->core_input || !dit->core_residual) {
            fail(error, error_size,
                 "cannot allocate DiT core residual cache: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
    }
    return 1;
}

typedef struct {
    h3_dit_progress callback;
    void *opaque;
} schedule_progress;

static void schedule_report(int completed, int total, void *opaque) {
    schedule_progress *state = opaque;
    report(state->callback, state->opaque, "precompute AdaLN", completed, total);
}

static h3_dit *load_dit(const char *weight_directory,
                        const char *adaln_overlay_path,
                        const char *shader_source_path,
                        const h3_text_embedding *text,
                        const h3_layout *layout,
                        const h3_sigma_schedule *sigmas,
                        unsigned active_blocks,
                        unsigned core_reuse_interval,
                        int token_reduction,
                        int ssd_streaming,
                        float spatial_rope_scale,
                        int use_slower_bf16_mlp,
                        int use_slower_bf16_qkv,
                        int use_slower_bf16_attention_output,
                        int use_slower_row_major_attention_output,
                        int use_slower_unfused_int8_inputs,
                        int use_slower_unfused_qkv_rope,
                        int use_slower_scalar_qkv_rms,
                        int use_slower_uncached_int8_scales,
                        int use_slower_dynamic_fc1_k,
                        int use_slower_grouped_quantizer,
                        int use_int8_row_fc2,
                        const char *lora_path,
                        float lora_strength,
                        const float *condition_video_rows,
                        size_t condition_video_elements,
                        const float *condition_audio_rows,
                        size_t condition_audio_elements,
                        const h3_dit_inpaint *inpaint,
                        h3_dit_progress progress, void *progress_opaque,
                        char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!weight_directory || !shader_source_path || !layout || !sigmas ||
        (ssd_streaming != 0 && ssd_streaming != 1) ||
        !isfinite(spatial_rope_scale) || spatial_rope_scale <= 0.0f ||
        active_blocks < H3_DIT_BLOCKS / 2 ||
        active_blocks > H3_DIT_BLOCKS || core_reuse_interval < 1 ||
        core_reuse_interval > 6) {
        fail(error, error_size, "invalid DiT load arguments");
        return NULL;
    }
    h3_dit *dit = calloc(1, sizeof(*dit));
    if (!dit) {
        fail(error, error_size, "out of memory creating DiT model");
        return NULL;
    }
    dit->fused_mlp = getenv("H3_DISABLE_FUSED_MLP") == NULL;
    /* The released final heads are F32, but their inputs are already BF16.
     * Converting these small weights once selects the Iris-derived tiled
     * linear and eliminates two full-width casts plus the scalar F32 kernel.
     * Keep the old path available for close-reference diagnosis. */
    dit->bf16_final = getenv("H3_DIT_F32_FINAL") == NULL;
    dit->core_reuse_interval = core_reuse_interval;
    dit->ssd_streaming = ssd_streaming;
    dit->spatial_rope_scale = spatial_rope_scale;
    configure_active_blocks(dit, active_blocks);
    if (!copy_layout(dit, layout, error, error_size) ||
        !validate_layout(dit, text, error, error_size) ||
        !configure_token_reduction(dit, token_reduction,
                                   error, error_size) ||
        !configure_inpaint(dit, inpaint, error, error_size)) goto failed;
    size_t wanted_video_condition =
        (size_t)dit->video_condition_rows * VIDEO_PATCH;
    size_t wanted_audio_condition =
        (size_t)dit->audio_condition_rows * AUDIO_CHANNELS;
    if (condition_video_elements != wanted_video_condition ||
        condition_audio_elements != wanted_audio_condition ||
        (wanted_video_condition && !condition_video_rows) ||
        (wanted_audio_condition && !condition_audio_rows)) {
        fail(error, error_size,
             "condition row elements do not match the packed DiT layout");
        goto failed;
    }
    dit->sigmas = *sigmas;
    dit->weights = h3_weight_store_open(weight_directory, error, error_size);
    if (!dit->weights) goto failed;
    const h3_st_tensor *core_qkv = h3_weight_find(
        dit->weights, "blocks.0.attn.qkv_proj.weight", NULL);
    if (!core_qkv || (core_qkv->dtype != H3_DTYPE_BF16 &&
                      core_qkv->dtype != H3_DTYPE_I8)) {
        fail(error, error_size,
             "DiT core QKV must use BF16 or pre-quantized I8 weights");
        goto failed;
    }
    dit->prequantized_int8 = core_qkv->dtype == H3_DTYPE_I8;
    /* Comfy's compact/pruned curve keeps core QKV rows in the conventional
     * [Q-all | K-all | V-all] order. The original released BF16 tree uses
     * [head, Q/K/V, dimension] instead. Confusing the two still produces
     * finite tensors, but every attention block mixes Q, K and V and the
     * final video/audio decode as structured noise. The compact AdaLN marker
     * is part of that checkpoint schema and avoids relying on a filename. */
    dit->conventional_core_qkv = h3_weight_find(
        dit->weights, "adaln_t_table", NULL) != NULL;
    if (!configure_input_major_transformer(dit, error, error_size))
        goto failed;
    if (!configure_hybrid_adaln(
            dit, adaln_overlay_path, error, error_size)) goto failed;
    dit->gpu = h3_gpu_create(shader_source_path, error, error_size);
    if (!dit->gpu) goto failed;
    if (!configure_sol_attention(dit, error, error_size)) goto failed;
    dit->nax_mlp = dit->fused_mlp && h3_gpu_has_nax_mlp(dit->gpu);
    dit->int8_mlp = !dit->prequantized_int8 && !dit->ssd_streaming &&
                    dit->fused_mlp &&
                    !use_slower_bf16_mlp &&
                    h3_gpu_has_int8_mlp(dit->gpu);
    dit->int8_qkv = !dit->prequantized_int8 &&
                    !dit->conventional_core_qkv && !dit->ssd_streaming &&
                    !use_slower_bf16_qkv &&
                    dit->sequence >= 128 &&
                    h3_gpu_has_int8_mlp(dit->gpu);
    dit->int8_attention_out = !dit->prequantized_int8 &&
                              !dit->ssd_streaming &&
                              !use_slower_bf16_attention_output &&
                              dit->sequence >= 128 &&
                              h3_gpu_has_int8_mlp(dit->gpu);
    dit->use_slower_row_major_attention_output =
        use_slower_row_major_attention_output;
    dit->use_slower_unfused_int8_inputs =
        use_slower_unfused_int8_inputs;
    dit->use_slower_unfused_qkv_rope =
        use_slower_unfused_qkv_rope;
    dit->use_slower_scalar_qkv_rms = use_slower_scalar_qkv_rms;
    dit->use_slower_uncached_int8_scales =
        use_slower_uncached_int8_scales;
    dit->use_slower_dynamic_fc1_k = use_slower_dynamic_fc1_k;
    dit->keep_bf16_attention_out = dit->int8_attention_out &&
        (getenv("H3_INT8_KEEP_BF16_ATTENTION_OUT") ||
         getenv("H3_BENCH_INT8_ATTENTION_OUT_AB"));
    dit->keep_bf16_qkv = dit->int8_qkv &&
        (getenv("H3_INT8_KEEP_BF16_QKV") ||
         getenv("H3_BENCH_INT8_QKV_AB"));
    dit->use_slower_grouped_quantizer = use_slower_grouped_quantizer;
    dit->use_int8_row_fc2 = dit->int8_mlp && use_int8_row_fc2;
    dit->keep_bf16_mlp = dit->int8_mlp &&
        (getenv("H3_INT8_KEEP_BF16_MLP") ||
         getenv("H3_BENCH_INT8_MLP_AB") ||
         getenv("H3_INT8_MLP_STAGE"));
    h3_gpu_profile_set_label(dit->gpu, "H3 DiT");
    if (lora_path && *lora_path && lora_strength != 0.0f &&
        !prepare_lora(dit, lora_path, lora_strength, error, error_size))
        goto failed;
    report(progress, progress_opaque, "refine text", 0, 1);
    if (!refine_text(dit, text, error, error_size)) goto failed;
    report(progress, progress_opaque, "refine text", 1, 1);
    schedule_progress schedule_state = {progress, progress_opaque};
    dit->schedule = h3_dit_schedule_precompute(
        dit->weights, dit->late_adaln_overlay, dit->gpu, sigmas,
        dit->video_condition_rows != 0 || dit->inpaint_video_source != NULL,
        dit->audio_condition_rows != 0 || dit->inpaint_audio_source != NULL,
        schedule_report, &schedule_state,
        error, error_size);
    h3_weight_store_free(dit->late_adaln_overlay);
    dit->late_adaln_overlay = NULL;
    if (dit->schedule) {
        configure_gate_ranked_blocks(dit);
        h3_dit_schedule_prune(dit->schedule, dit->block_active,
                              H3_DIT_BLOCKS);
    }
    if (!dit->schedule || !prepare_rope(dit, error, error_size) ||
        !prepare_maps(dit, text, error, error_size) ||
        !prepare_projection_maps(dit, error, error_size) ||
        !prepare_token_reduction_maps(dit, error, error_size) ||
        !load_core(dit, progress, progress_opaque, error, error_size) ||
        !allocate_activations(dit, error, error_size)) goto failed;
    if (lora_path && *lora_path && lora_strength != 0.0f &&
        !load_lora_adapters(dit, lora_path, lora_strength, progress,
                            progress_opaque, error, error_size)) goto failed;
    if ((wanted_video_condition && !h3_gpu_tensor_write_f32_range(
             dit->video_input, 0, condition_video_rows,
             wanted_video_condition)) ||
        (wanted_audio_condition && !h3_gpu_tensor_write_f32_range(
             dit->audio_input, 0, condition_audio_rows,
             wanted_audio_condition))) {
        fail(error, error_size, "cannot write persistent DiT condition rows");
        goto failed;
    }
    h3_gpu_profile_mark(dit->gpu, "load");
    return dit;
failed:
    h3_dit_free(dit);
    return NULL;
}

h3_dit *h3_dit_load_t2va(const char *weight_directory,
                         const char *shader_source_path,
                         const h3_text_embedding *text,
                         const h3_layout *layout,
                         const h3_sigma_schedule *sigmas,
                         unsigned active_blocks,
                         unsigned core_reuse_interval,
                         int token_reduction,
                         int ssd_streaming,
                         float spatial_rope_scale,
                         int use_slower_bf16_mlp,
                         int use_slower_bf16_qkv,
                         int use_slower_bf16_attention_output,
                         int use_slower_row_major_attention_output,
                         int use_slower_unfused_int8_inputs,
                         int use_slower_unfused_qkv_rope,
                         int use_slower_scalar_qkv_rms,
                         int use_slower_uncached_int8_scales,
                         int use_slower_dynamic_fc1_k,
                         int use_slower_grouped_quantizer,
                         int use_int8_row_fc2,
                         const char *lora_path,
                         float lora_strength,
                         h3_dit_progress progress, void *progress_opaque,
                         char *error, size_t error_size) {
    return load_dit(weight_directory, NULL, shader_source_path, text, layout,
                    sigmas, active_blocks, core_reuse_interval, token_reduction,
                    ssd_streaming,
                    spatial_rope_scale,
                    use_slower_bf16_mlp, use_slower_bf16_qkv,
                    use_slower_bf16_attention_output,
                    use_slower_row_major_attention_output,
                    use_slower_unfused_int8_inputs,
                    use_slower_unfused_qkv_rope,
                    use_slower_scalar_qkv_rms,
                    use_slower_uncached_int8_scales,
                    use_slower_dynamic_fc1_k,
                    use_slower_grouped_quantizer,
                    use_int8_row_fc2,
                    lora_path, lora_strength,
                    NULL, 0, NULL, 0, NULL, progress, progress_opaque,
                    error, error_size);
}

h3_dit *h3_dit_load_conditioned(
                         const char *weight_directory,
                         const char *adaln_overlay_path,
                         const char *shader_source_path,
                         const h3_text_embedding *text,
                         const h3_layout *layout,
                         const h3_sigma_schedule *sigmas,
                         unsigned active_blocks,
                         unsigned core_reuse_interval,
                         int token_reduction,
                         int ssd_streaming,
                         float spatial_rope_scale,
                         int use_slower_bf16_mlp,
                         int use_slower_bf16_qkv,
                         int use_slower_bf16_attention_output,
                         int use_slower_row_major_attention_output,
                         int use_slower_unfused_int8_inputs,
                         int use_slower_unfused_qkv_rope,
                         int use_slower_scalar_qkv_rms,
                         int use_slower_uncached_int8_scales,
                         int use_slower_dynamic_fc1_k,
                         int use_slower_grouped_quantizer,
                         int use_int8_row_fc2,
                         const char *lora_path,
                         float lora_strength,
                         const float *condition_video_rows,
                         size_t condition_video_elements,
                         const float *condition_audio_rows,
                         size_t condition_audio_elements,
                         const h3_dit_inpaint *inpaint,
                         h3_dit_progress progress, void *progress_opaque,
                         char *error, size_t error_size) {
    return load_dit(weight_directory, adaln_overlay_path, shader_source_path,
                    text, layout, sigmas,
                    active_blocks, core_reuse_interval, token_reduction,
                    ssd_streaming,
                    spatial_rope_scale,
                    use_slower_bf16_mlp, use_slower_bf16_qkv,
                    use_slower_bf16_attention_output,
                    use_slower_row_major_attention_output,
                    use_slower_unfused_int8_inputs,
                    use_slower_unfused_qkv_rope,
                    use_slower_scalar_qkv_rms,
                    use_slower_uncached_int8_scales,
                    use_slower_dynamic_fc1_k,
                    use_slower_grouped_quantizer,
                    use_int8_row_fc2,
                    lora_path, lora_strength,
                    condition_video_rows, condition_video_elements,
                    condition_audio_rows, condition_audio_elements,
                    inpaint,
                    progress, progress_opaque, error, error_size);
}

static int enter_token_reduction(h3_dit *dit, char *error,
                                 size_t error_size) {
    h3_gpu_tensor *original = dit->token_original_in_qkv ?
        dit->qkv : dit->token_original;
    if (!gpu_op(dit, h3_gpu_token_pool_bf16(
            dit->gpu, dit->attention_output, dit->hidden, 0,
            original, dit->token_original_offset, dit->attention_output,
            dit->token_baseline_offset, dit->token_baseline_indices,
            dit->token_pool_pairs, dit->sequence, dit->reduced_sequence,
            dit->token_baseline_rows, HIDDEN),
            error, error_size, "snapshot and pool video tokens")) return 0;
    h3_gpu_tensor *swap = dit->hidden;
    dit->hidden = dit->attention_output;
    dit->attention_output = swap;
    dit->token_reduction_active = 1;
    return 1;
}

static int enter_token_reduction_adaln(h3_dit *dit, unsigned block,
                                       int step, char *error,
                                       size_t error_size) {
    h3_gpu_tensor *original = dit->token_original_in_qkv ?
        dit->qkv : dit->token_original;
    h3_dit_block *weight = &dit->blocks[block];
    const h3_gpu_tensor *modulation = h3_dit_schedule_block(
        dit->schedule, block);
    if (!gpu_op(dit, h3_gpu_token_pool_adaln_bf16(
            dit->gpu, dit->attention_output, dit->mod_attention,
            dit->hidden, 0, original, dit->token_original_offset,
            dit->attention_output, dit->token_baseline_offset,
            dit->token_baseline_indices, dit->token_pool_pairs,
            weight->norm1, modulation, dit->reduced_row_maps[step],
            dit->sequence, dit->reduced_sequence, dit->token_baseline_rows,
            HIDDEN, SLOTS, 0, 1, 1e-5f), error, error_size,
            "snapshot, pool, and apply attention AdaLN")) return 0;
    h3_gpu_tensor *swap = dit->hidden;
    dit->hidden = dit->attention_output;
    dit->attention_output = swap;
    dit->token_reduction_active = 1;
    return 1;
}

static int leave_token_reduction(h3_dit *dit, char *error,
                                 size_t error_size) {
    h3_gpu_tensor *original = dit->token_original_in_qkv ?
        dit->qkv : dit->token_original;
    if (!gpu_op(dit, h3_gpu_token_expand_delta_bf16(
            dit->gpu, dit->mod_attention, original,
            dit->token_original_offset, dit->hidden, dit->hidden,
            dit->token_baseline_offset, dit->token_baseline_indices,
            dit->token_expand_parents, dit->sequence,
            dit->reduced_sequence, dit->token_baseline_rows, HIDDEN,
            dit->video_target_start,
            dit->token_reduction_scale), error, error_size,
            "restore full video-token grid")) return 0;
    h3_gpu_tensor *swap = dit->hidden;
    dit->hidden = dit->mod_attention;
    dit->mod_attention = swap;
    dit->token_reduction_active = 0;
    return 1;
}

static int leave_token_reduction_adaln(h3_dit *dit, unsigned block,
                                       int step, char *error,
                                       size_t error_size) {
    h3_gpu_tensor *original = dit->token_original_in_qkv ?
        dit->qkv : dit->token_original;
    h3_dit_block *weight = &dit->blocks[block];
    const h3_gpu_tensor *modulation = h3_dit_schedule_block(
        dit->schedule, block);
    if (!gpu_op(dit, h3_gpu_token_expand_adaln_bf16(
            dit->gpu, dit->attention_output, dit->mod_attention,
            original, dit->token_original_offset, dit->hidden, dit->hidden,
            dit->token_baseline_offset, dit->token_baseline_indices,
            dit->token_expand_parents, weight->norm1, modulation,
            dit->row_maps[step], dit->sequence, dit->reduced_sequence,
            dit->token_baseline_rows, HIDDEN, dit->video_target_start,
            dit->token_reduction_scale, SLOTS, 0, 1, 1e-5f),
            error, error_size, "restore tokens and apply attention AdaLN"))
        return 0;
    h3_gpu_tensor *reduced = dit->hidden;
    dit->hidden = dit->attention_output;
    dit->attention_output = reduced;
    dit->token_reduction_active = 0;
    return 1;
}

/* Development-only activation capture for calibrating an attention backend
 * against real H3 Q/K/V rather than synthetic normals. The explicit prefix
 * env var makes this unreachable in the shipped app. A capture intentionally
 * stops the run after synchronizing the requested block, avoiding a complete
 * generation whose decoded result would be discarded. */
static int capture_sol_attention(h3_dit *dit, unsigned block, int step,
                                 uint32_t rows,
                                 char *error, size_t error_size) {
    const char *prefix = getenv("H3_DUMP_SOL_ATTENTION");
    if (!prefix || !*prefix) return 1;
    unsigned wanted_block = 0;
    int wanted_step = 0;
    const char *block_value = getenv("H3_DUMP_SOL_BLOCK");
    const char *step_value = getenv("H3_DUMP_SOL_STEP");
    if (block_value && *block_value)
        wanted_block = (unsigned)strtoul(block_value, NULL, 10);
    if (step_value && *step_value)
        wanted_step = (int)strtol(step_value, NULL, 10);
    if (block != wanted_block || step != wanted_step) return 1;
    if (!h3_gpu_submit(dit->gpu)) {
        fail(error, error_size, "cannot synchronize Sol-Attn capture: %s",
             h3_gpu_error(dit->gpu));
        return 0;
    }
    size_t elements = (size_t)rows * INNER;
    uint16_t *host = malloc(elements * sizeof(*host));
    if (!host) {
        fail(error, error_size, "out of memory capturing Sol-Attn inputs");
        return 0;
    }
    const h3_gpu_tensor *tensors[3] = {dit->query, dit->key, dit->value};
    const char *suffixes[3] = {"q.bf16", "k.bf16", "v.bf16"};
    char path[4096];
    int ok = 1;
    for (size_t slot = 0; slot < 3 && ok; slot++) {
        int length = snprintf(path, sizeof(path), "%s.%s", prefix,
                              suffixes[slot]);
        if (length < 0 || (size_t)length >= sizeof(path) ||
            !h3_gpu_tensor_read_bf16(tensors[slot], host, elements)) {
            ok = 0;
            break;
        }
        FILE *file = fopen(path, "wb");
        if (!file || fwrite(host, sizeof(*host), elements, file) != elements)
            ok = 0;
        if (file && fclose(file) != 0) ok = 0;
    }
    free(host);
    int length = snprintf(path, sizeof(path), "%s.json", prefix);
    if (ok && length > 0 && (size_t)length < sizeof(path)) {
        FILE *file = fopen(path, "w");
        if (!file || fprintf(file,
                "{\"tokens\":%u,\"heads\":%u,\"head_dim\":%u,"
                "\"sink_tokens\":%u,\"block\":%u,\"step\":%d}\n",
                rows, HEADS, HEAD_DIM, dit->video_target_start, block, step) < 0)
            ok = 0;
        if (file && fclose(file) != 0) ok = 0;
    } else {
        ok = 0;
    }
    fail(error, error_size, ok ? "Sol-Attn capture complete at %s" :
         "cannot write Sol-Attn capture at %s", prefix);
    return 0;
}

static int use_sol_attention(const h3_dit *dit, int step) {
    if (!dit->sol_attention || dit->token_reduction_active) return 0;
    const char *all_steps = getenv("H3_SOL_ATTN_ALL_STEPS");
    if (all_steps && *all_steps && strcmp(all_steps, "0")) return 1;
    int steps = h3_dit_schedule_steps(dit->schedule);
    if (steps < 3 || step < 0 || step >= steps) return 0;
    /* Match the conservative 20%-90% window used by the published H3
     * integration. The noisiest opening and final-detail pass remain dense. */
    int denominator = steps - 1;
    return step * 10 >= denominator * 2 &&
           step * 10 <= denominator * 9;
}

static int report_sol_route_density(h3_dit *dit, unsigned block, int step,
                                    uint32_t rows,
                                    char *error, size_t error_size) {
    if (!dit->sol_routes || dit->sol_routes_reported) return 1;
    if (!h3_gpu_sol_attention_routes_bf16(
            dit->gpu, dit->sol_routes,
            dit->sol_query_centroids, dit->sol_key_centroids,
            dit->sol_thresholds, rows, HEADS, HEAD_DIM,
            dit->video_target_start, 1.0f / sqrtf((float)HEAD_DIM)) ||
        !h3_gpu_submit(dit->gpu)) {
        fail(error, error_size, "cannot profile Sol-Attn routes: %s",
             h3_gpu_error(dit->gpu));
        return 0;
    }
    int8_t *routes = malloc(dit->sol_route_count);
    if (!routes) {
        fail(error, error_size,
             "out of memory reading Sol-Attn route diagnostics");
        return 0;
    }
    int read = h3_gpu_tensor_read_i8(
        dit->sol_routes, routes, dit->sol_route_count);
    size_t selected = 0;
    if (read)
        for (size_t index = 0; index < dit->sol_route_count; index++)
            selected += routes[index] != 0;
    free(routes);
    if (!read || !h3_gpu_begin(dit->gpu)) {
        fail(error, error_size, "cannot read Sol-Attn route diagnostics: %s",
             h3_gpu_error(dit->gpu));
        return 0;
    }
    dit->sol_routes_reported = 1;
    fprintf(stderr,
            "h3: Sol-Attn routes block=%u step=%d selected=%zu/%zu "
            "density=%.4f\n",
            block, step + 1, selected, dit->sol_route_count,
            (double)selected / (double)dit->sol_route_count);
    return 1;
}

static int run_block(h3_dit *dit, unsigned index, int step,
                     h3_dit_block *weight,
                     int attention_adaln_ready,
                     int attention_input_quantized,
                     int fuse_next_attention, unsigned next_index,
                     int *next_attention_adaln_ready,
                     int *next_attention_input_quantized,
                     char *error, size_t error_size) {
    const h3_gpu_tensor *modulation = h3_dit_schedule_block(dit->schedule,
                                                            index);
    h3_gpu_tensor *row_map = dit->token_reduction_active ?
        dit->reduced_row_maps[step] : dit->row_maps[step];
    h3_gpu_tensor *rope_cos = dit->token_reduction_active ?
        dit->reduced_rope_cos : dit->rope_cos;
    h3_gpu_tensor *rope_sin = dit->token_reduction_active ?
        dit->reduced_rope_sin : dit->rope_sin;
    uint32_t rows = dit->token_reduction_active ?
        dit->reduced_sequence : dit->sequence;
#define OP(call, label) do {                                                    \
    if (!gpu_op(dit, (call), error, error_size, label)) return 0;               \
} while (0)
    if (!attention_adaln_ready)
        OP(h3_gpu_adaln_bf16(dit->gpu, dit->mod_attention, dit->hidden,
            weight->norm1, modulation, row_map, rows, HIDDEN, SLOTS,
            0, 1, 1e-5f), "DiT attention AdaLN");
    if (dit->prequantized_int8) {
        if (weight->qkv_convrot_group)
            OP(h3_gpu_convrot_bf16(
                dit->gpu, dit->mod_attention, dit->mod_attention,
                rows, HIDDEN, weight->qkv_convrot_group),
               "DiT QKV ConvRot");
        OP(dit_int8_linear(
            dit->gpu, dit->qkv, dit->mod_attention,
            weight->qkv_int8, weight->qkv_scales, NULL,
            rows, HIDDEN, INNER * 3, dit->input_major_transformer),
           "DiT pre-quantized QKV projection");
        if (!apply_lora(dit, dit->qkv, dit->mod_attention,
                        weight->qkv_lora_a, weight->qkv_lora_b,
                        weight->qkv_lora_rank, rows, HIDDEN, INNER * 3,
                        error, error_size)) return 0;
        /* Comfy's compact checkpoint preserves the projection's ordinary
         * [Q-all-heads | K-all-heads | V-all-heads] row order.  The released
         * MiniMax tree uses the per-head interleaved order handled by the
         * grouped helper below, but applying that interpretation here mixes
         * Q, K, and V rows and leaves denoising as structured noise. */
        OP(h3_gpu_qkv_rope_bf16(
            dit->gpu, dit->query, dit->key, dit->value, dit->qkv,
            weight->q_norm, weight->k_norm, rope_cos, rope_sin,
            rows, HEADS, HEAD_DIM, ROPE_HALF, 1e-5f),
           "DiT pre-quantized QKV norm/RoPE");
    } else if (dit->int8_qkv && !getenv("H3_DISABLE_INT8_QKV")) {
        OP(h3_gpu_grouped_qkv_linear_rope_int8(
            dit->gpu, dit->query, dit->key, dit->value,
            dit->int8_activation, dit->int8_activation_scales,
            dit->mod_attention, weight->qkv_int8, weight->qkv_scales,
            weight->q_norm, weight->k_norm, rope_cos, rope_sin,
            rows, HIDDEN, HEADS, HEAD_DIM, ROPE_HALF, 1e-5f,
            attention_input_quantized,
            dit->use_slower_unfused_qkv_rope,
            dit->use_slower_scalar_qkv_rms,
            dit->use_slower_uncached_int8_scales),
           "DiT int8 QKV projection/norm/RoPE");
    } else if (dit->conventional_core_qkv) {
        OP(h3_gpu_linear_bf16(
            dit->gpu, dit->qkv, dit->mod_attention, weight->qkv, NULL,
            rows, HIDDEN, INNER * 3), "DiT compact BF16 QKV projection");
        OP(h3_gpu_qkv_rope_bf16(
            dit->gpu, dit->query, dit->key, dit->value, dit->qkv,
            weight->q_norm, weight->k_norm, rope_cos, rope_sin,
            rows, HEADS, HEAD_DIM, ROPE_HALF, 1e-5f),
           "DiT compact BF16 QKV norm/RoPE");
    } else {
        OP(h3_gpu_grouped_qkv_linear_rope_bf16(
            dit->gpu, dit->query, dit->key, dit->value, dit->qkv,
            dit->mod_attention, weight->qkv, weight->q_norm, weight->k_norm,
            rope_cos, rope_sin, rows, HIDDEN, HEADS, HEAD_DIM, ROPE_HALF,
            1e-5f), "DiT QKV projection/norm/RoPE");
    }
    if (!capture_sol_attention(dit, index, step, rows, error, error_size))
        return 0;
    int int8_attention_output = dit->int8_attention_out &&
        !getenv("H3_DISABLE_INT8_ATTENTION_OUT");
    int head_major_attention_output = int8_attention_output &&
        !dit->use_slower_row_major_attention_output &&
        !dit->use_slower_uncached_int8_scales &&
        !getenv("H3_DISABLE_HEAD_MAJOR_ATTENTION_OUTPUT");
    int sol_active = use_sol_attention(dit, step);
    if (sol_active)
        OP(h3_gpu_sol_attention_bf16(
            dit->gpu, dit->attention_heads,
            dit->query, dit->key, dit->value,
            dit->sol_query_centroids, dit->sol_key_centroids,
            dit->sol_value_sums, dit->sol_key_means,
            dit->sol_key_variances, dit->sol_thresholds,
            rows, HEADS, HEAD_DIM, dit->video_target_start,
            1.0f / sqrtf((float)HEAD_DIM), dit->sol_attention_tau,
            head_major_attention_output),
           "DiT Sol block-sparse attention");
    else if (head_major_attention_output)
        OP(h3_gpu_sdpa_bf16_head_major_output(
            dit->gpu, dit->attention_heads, dit->query, dit->key, dit->value,
            rows, HEADS, HEAD_DIM, 1.0f / sqrtf((float)HEAD_DIM)),
           "DiT head-major full attention");
    else if (dit->query32 && rows > H3_DIT_F32_ATTENTION_ROWS) {
        /* Attention in f32 though everything around it is bf16. MPSGraph's
         * bf16 SDPA is roughly 1.76x slower than its f32 path on this shape,
         * and casting three operands up and the result back still comes out
         * well ahead: 868 ms a block against 494 at 5095 rows, which is 18.7 s
         * a pass across 50 blocks.
         *
         * Only reached when the head-major variant above is not, which for a
         * pre-quantized package is always — int8_attention_out is false there,
         * so nothing is given up by writing row-major here. */
        const uint32_t span = rows * INNER;
        OP(h3_gpu_cast_bf16_to_f32(dit->gpu, dit->query32, dit->query, span),
           "DiT attention cast q");
        OP(h3_gpu_cast_bf16_to_f32(dit->gpu, dit->key32, dit->key, span),
           "DiT attention cast k");
        OP(h3_gpu_cast_bf16_to_f32(dit->gpu, dit->value32, dit->value, span),
           "DiT attention cast v");
        OP(h3_gpu_sdpa_f32(
            dit->gpu, dit->heads32, dit->query32, dit->key32, dit->value32,
            rows, HEADS, HEAD_DIM, 1.0f / sqrtf((float)HEAD_DIM)),
           "DiT full attention f32");
        OP(h3_gpu_cast_f32_to_bf16(dit->gpu, dit->attention_heads,
                                   dit->heads32, span),
           "DiT attention cast out");
    } else
        OP(h3_gpu_sdpa_bf16(
            dit->gpu, dit->attention_heads, dit->query, dit->key, dit->value,
            rows, HEADS, HEAD_DIM, 1.0f / sqrtf((float)HEAD_DIM)),
           "DiT full attention");
    if (sol_active &&
        !report_sol_route_density(dit, index, step, rows,
                                  error, error_size)) return 0;
    if (dit->prequantized_int8) {
        if (weight->out_convrot_group)
            OP(h3_gpu_convrot_bf16(
                dit->gpu, dit->attention_heads, dit->attention_heads,
                rows, INNER, weight->out_convrot_group),
               "DiT attention-output ConvRot");
        OP(dit_int8_linear(
            dit->gpu, dit->attention_output, dit->attention_heads,
            weight->out_int8, weight->out_scales, NULL,
            rows, INNER, HIDDEN, dit->input_major_transformer),
           "DiT pre-quantized attention output");
        if (!apply_lora(dit, dit->attention_output, dit->attention_heads,
                        weight->out_lora_a, weight->out_lora_b,
                        weight->out_lora_rank, rows, INNER, HIDDEN,
                        error, error_size)) return 0;
    } else if (int8_attention_output) {
        if (head_major_attention_output)
            OP(h3_gpu_linear_int8_head_major_bf16(
                dit->gpu, dit->attention_output, dit->int8_activation,
                dit->int8_activation_scales, dit->attention_heads,
                weight->out_int8, weight->out_scales, rows, HEADS, HEAD_DIM,
                HIDDEN), "DiT head-major int8 attention output");
        else
            OP(h3_gpu_linear_int8_bf16(
                dit->gpu, dit->attention_output, dit->int8_activation,
                dit->int8_activation_scales, dit->attention_heads,
                weight->out_int8, weight->out_scales, rows, INNER, HIDDEN,
                dit->use_slower_uncached_int8_scales),
               "DiT int8 attention output");
    } else {
        OP(h3_gpu_linear_bf16(dit->gpu, dit->attention_output,
            dit->attention_heads, weight->out, NULL, rows, INNER, HIDDEN),
           "DiT attention output");
    }
    int fused_int8_mlp_input = dit->int8_mlp &&
        !dit->use_slower_unfused_int8_inputs &&
        !getenv("H3_DISABLE_FUSED_INT8_MLP_INPUT") &&
        !getenv("H3_INT8_MLP_STAGE");
    if (fused_int8_mlp_input) {
        uint32_t padded_rows = (rows + 127u) & ~127u;
        OP(h3_gpu_gate_adaln_quantize_int8(
            dit->gpu, dit->hidden, dit->int8_activation,
            dit->int8_activation_scales, dit->hidden,
            dit->attention_output, weight->norm2, modulation, modulation,
            row_map, rows, padded_rows, HIDDEN, SLOTS, 2, 3, 4, 1e-5f),
           "DiT fused attention gate, MLP AdaLN and int8 quantization");
    } else if (!getenv("H3_DISABLE_FUSED_GATE_ADALN")) {
        OP(h3_gpu_gate_adaln_bf16(
            dit->gpu, dit->hidden, dit->mod_mlp, dit->hidden,
            dit->attention_output, weight->norm2, modulation, modulation,
            row_map,
            rows, HIDDEN, SLOTS, 2, 3, 4, 1e-5f),
           "DiT fused attention gate and MLP AdaLN");
    } else {
        OP(h3_gpu_gate_bf16(dit->gpu, dit->hidden, dit->hidden,
            dit->attention_output, modulation, row_map, rows, HIDDEN,
            SLOTS, 2), "DiT attention gate");
        OP(h3_gpu_adaln_bf16(
            dit->gpu, dit->mod_mlp, dit->hidden, weight->norm2,
            modulation, row_map, rows, HIDDEN, SLOTS, 3, 4, 1e-5f),
           "DiT MLP AdaLN");
    }
    h3_gpu_tensor *mlp_output = dit->activation_aliases ?
        dit->attention_output : dit->mlp_output;
    if (dit->prequantized_int8) {
        if (weight->fc1_convrot_group)
            OP(h3_gpu_convrot_bf16(
                dit->gpu, dit->mod_mlp, dit->mod_mlp,
                rows, HIDDEN, weight->fc1_convrot_group),
               "DiT FC1 ConvRot");
        OP(dit_int8_linear(
            dit->gpu, dit->fc1, dit->mod_mlp,
            weight->fc1_int8, weight->fc1_scales, NULL,
            rows, HIDDEN, FFN * 2, dit->input_major_transformer),
           "DiT pre-quantized MLP input");
        if (!apply_lora(dit, dit->fc1, dit->mod_mlp,
                        weight->fc1_lora_a, weight->fc1_lora_b,
                        weight->fc1_lora_rank, rows, HIDDEN, FFN * 2,
                        error, error_size)) return 0;
        OP(h3_gpu_swiglu_bf16(
            dit->gpu, dit->activated, dit->fc1, rows, FFN),
           "DiT pre-quantized SwiGLU");
        if (weight->fc2_convrot_group)
            OP(h3_gpu_convrot_bf16(
                dit->gpu, dit->activated, dit->activated,
                rows, FFN, weight->fc2_convrot_group),
               "DiT FC2 ConvRot");
        OP(dit_int8_linear(
            dit->gpu, mlp_output, dit->activated,
            weight->fc2_int8, weight->fc2_scales, NULL,
            rows, FFN, HIDDEN, dit->input_major_transformer),
           "DiT pre-quantized MLP output");
        if (!apply_lora(dit, mlp_output, dit->activated,
                        weight->fc2_lora_a, weight->fc2_lora_b,
                        weight->fc2_lora_rank, rows, FFN, HIDDEN,
                        error, error_size)) return 0;
    } else if (dit->int8_mlp &&
        (!getenv("H3_DISABLE_INT8_MLP") ||
         !weight->fc1 || !weight->fc2)) {
        OP(h3_gpu_mlp_int8_bf16(
            dit->gpu, mlp_output, dit->activated, dit->int8_activation,
            dit->int8_activation_scales, dit->mod_mlp,
            weight->fc1_int8, weight->fc1_scales,
            weight->fc2_int8, weight->fc2_scales,
            weight->fc1, weight->fc2,
            rows, HIDDEN, FFN, HIDDEN,
            dit->use_slower_grouped_quantizer,
            dit->use_slower_dynamic_fc1_k, dit->use_int8_row_fc2,
            fused_int8_mlp_input),
           "DiT int8 fused MLP");
    } else if (dit->nax_mlp && !getenv("H3_DISABLE_NAX_MLP")) {
        OP(h3_gpu_mlp_nax_bf16(dit->gpu, mlp_output, dit->activated,
            dit->mod_mlp, weight->fc1, weight->fc2, rows, HIDDEN, FFN,
            HIDDEN), "DiT NAX fused MLP");
    } else if (dit->fused_mlp) {
        OP(h3_gpu_mlp_bf16(dit->gpu, mlp_output, dit->mod_mlp,
            weight->fc1, weight->fc2, rows, HIDDEN, FFN, HIDDEN),
           "DiT fused MLP");
    } else {
        OP(h3_gpu_linear_bf16(dit->gpu, dit->fc1, dit->mod_mlp, weight->fc1,
            NULL, rows, HIDDEN, FFN * 2), "DiT MLP input");
        OP(h3_gpu_swiglu_bf16(dit->gpu, dit->activated, dit->fc1, rows, FFN),
           "DiT SwiGLU");
        OP(h3_gpu_linear_bf16(dit->gpu, mlp_output, dit->activated,
            weight->fc2, NULL, rows, FFN, HIDDEN), "DiT MLP output");
    }
    if (fuse_next_attention) {
        h3_dit_block *next_weight = &dit->blocks[next_index];
        const h3_gpu_tensor *next_modulation = h3_dit_schedule_block(
            dit->schedule, next_index);
        int fuse_int8_qkv_input = dit->int8_qkv &&
            !dit->use_slower_unfused_int8_inputs &&
            !getenv("H3_DISABLE_INT8_QKV") &&
            !getenv("H3_DISABLE_FUSED_INT8_QKV_INPUT");
        if (fuse_int8_qkv_input) {
            uint32_t padded_rows = (rows + 127u) & ~127u;
            OP(h3_gpu_gate_adaln_quantize_int8(
                dit->gpu, dit->hidden, dit->int8_activation,
                dit->int8_activation_scales, dit->hidden, mlp_output,
                next_weight->norm1, modulation, next_modulation, row_map,
                rows, padded_rows, HIDDEN, SLOTS, 5, 0, 1, 1e-5f),
               "DiT fused MLP gate, next attention AdaLN and int8 quantization");
            *next_attention_input_quantized = 1;
        } else {
            OP(h3_gpu_gate_adaln_bf16(
                dit->gpu, dit->hidden, dit->mod_attention, dit->hidden,
                mlp_output, next_weight->norm1, modulation,
                next_modulation, row_map, rows, HIDDEN, SLOTS, 5, 0, 1,
                1e-5f), "DiT fused MLP gate and next attention AdaLN");
            *next_attention_input_quantized = 0;
        }
        *next_attention_adaln_ready = 1;
    } else {
        OP(h3_gpu_gate_bf16(
            dit->gpu, dit->hidden, dit->hidden, mlp_output,
            modulation, row_map, rows, HIDDEN, SLOTS, 5), "DiT MLP gate");
    }
#undef OP
    return 1;
}

static int encode_forward(h3_dit *dit, int step, int begin, int submit,
                          int disable_command_split,
                          h3_dit_progress progress, void *progress_opaque,
                          char *error, size_t error_size) {
#define OP(call, label) do {                                                    \
    if (!gpu_op(dit, (call), error, error_size, label)) return 0;               \
} while (0)
    if (begin) OP(h3_gpu_begin(dit->gpu), "begin DiT forward");
    size_t video_offset = 0;
    size_t audio_offset = 0;
    if (dit->fused_patch_pack) {
        if (dit->video_projection_map)
            OP(h3_gpu_patch_linear_bf16_map(
                dit->gpu, dit->hidden, dit->video_input, dit->video_patch_w,
                dit->video_patch_b, dit->video_projection_map, dit->sequence,
                dit->video_total_rows, VIDEO_PATCH, HIDDEN),
               "project mapped video sources");
        if (dit->audio_projection_map)
            OP(h3_gpu_patch_linear_bf16_map(
                dit->gpu, dit->hidden, dit->audio_input, dit->audio_patch_w,
                dit->audio_patch_b, dit->audio_projection_map, dit->sequence,
                dit->audio_total_rows, AUDIO_CHANNELS, HIDDEN),
               "project mapped audio sources");
        for (size_t index = 0; index < dit->layout.segment_count; index++) {
            const h3_segment *segment = &dit->layout.segments[index];
            size_t segment_rows = segment->stop - segment->start;
            size_t destination = segment->start * HIDDEN;
            if (segment->kind == H3_SEG_TEXT) {
                OP(h3_gpu_copy_bf16(dit->gpu, dit->hidden, destination,
                    dit->refined_text, 0, segment_rows * HIDDEN),
                   "pack refined text");
            } else if (segment->kind == H3_SEG_COND ||
                       segment->kind == H3_SEG_REF_IMAGE ||
                       segment->kind == H3_SEG_VIDEO) {
                if (!dit->video_projection_map)
                    OP(h3_gpu_patch_linear_bf16_offset(
                        dit->gpu, dit->hidden, destination, dit->video_input,
                        video_offset * VIDEO_PATCH, dit->video_patch_w,
                        dit->video_patch_b, (uint32_t)segment_rows,
                        VIDEO_PATCH, HIDDEN), "project packed video source");
                video_offset += segment_rows;
            } else {
                if (!dit->audio_projection_map)
                    OP(h3_gpu_patch_linear_bf16_offset(
                        dit->gpu, dit->hidden, destination, dit->audio_input,
                        audio_offset * AUDIO_CHANNELS, dit->audio_patch_w,
                        dit->audio_patch_b, (uint32_t)segment_rows,
                        AUDIO_CHANNELS, HIDDEN),
                       "project packed audio source");
                audio_offset += segment_rows;
            }
        }
    } else {
        if (dit->fused_patch_projection) {
            OP(h3_gpu_patch_linear_bf16(
                dit->gpu, dit->video_projected, dit->video_input,
                dit->video_patch_w, dit->video_patch_b,
                dit->video_total_rows, VIDEO_PATCH, HIDDEN),
               "fused video patch projection");
            OP(h3_gpu_patch_linear_bf16(
                dit->gpu, dit->audio_projected, dit->audio_input,
                dit->audio_patch_w, dit->audio_patch_b,
                dit->audio_total_rows, AUDIO_CHANNELS, HIDDEN),
               "fused audio patch projection");
        } else {
            OP(h3_gpu_linear_f32(
                dit->gpu, dit->video_projected_f32, dit->video_input,
                dit->video_patch_w, dit->video_patch_b,
                dit->video_total_rows, VIDEO_PATCH, HIDDEN),
               "video patch projection");
            OP(h3_gpu_linear_f32(
                dit->gpu, dit->audio_projected_f32, dit->audio_input,
                dit->audio_patch_w, dit->audio_patch_b,
                dit->audio_total_rows, AUDIO_CHANNELS, HIDDEN),
               "audio patch projection");
            OP(h3_gpu_cast_f32_to_bf16(
                dit->gpu, dit->video_projected, dit->video_projected_f32,
                dit->video_total_rows * HIDDEN), "video BF16 cast");
            OP(h3_gpu_cast_f32_to_bf16(
                dit->gpu, dit->audio_projected, dit->audio_projected_f32,
                dit->audio_total_rows * HIDDEN), "audio BF16 cast");
        }
        for (size_t index = 0; index < dit->layout.segment_count; index++) {
            const h3_segment *segment = &dit->layout.segments[index];
            size_t segment_rows = segment->stop - segment->start;
            size_t destination = segment->start * HIDDEN;
            if (segment->kind == H3_SEG_TEXT) {
                OP(h3_gpu_copy_bf16(dit->gpu, dit->hidden, destination,
                    dit->refined_text, 0, segment_rows * HIDDEN),
                   "pack refined text");
            } else if (segment->kind == H3_SEG_COND ||
                       segment->kind == H3_SEG_REF_IMAGE ||
                       segment->kind == H3_SEG_VIDEO) {
                OP(h3_gpu_copy_bf16(
                    dit->gpu, dit->hidden, destination, dit->video_projected,
                    video_offset * HIDDEN, segment_rows * HIDDEN),
                   "pack video source");
                video_offset += segment_rows;
            } else {
                OP(h3_gpu_copy_bf16(
                    dit->gpu, dit->hidden, destination, dit->audio_projected,
                    audio_offset * HIDDEN, segment_rows * HIDDEN),
                   "pack audio source");
                audio_offset += segment_rows;
            }
        }
    }
    if (video_offset != dit->video_total_rows ||
        audio_offset != dit->audio_total_rows) {
        fail(error, error_size, "DiT segment packing did not consume row sources");
        return 0;
    }
    int evaluate_core = dit->core_reuse_interval == 1 ||
        !dit->core_residual_ready ||
        dit->core_forward_count % dit->core_reuse_interval == 0 ||
        step == h3_dit_schedule_steps(dit->schedule) - 1;
    dit->cache_forward_is_cached = dit->block_cache && dit->cache_plan_ready &&
        step < H3_MAX_STEPS && !dit->cache_full_step[step] &&
        dit->cache_residual_ready;
    dit->cache_forward_captures = dit->block_cache && dit->cache_plan_ready &&
        !dit->cache_forward_is_cached;
    int use_token_reduction = evaluate_core && dit->token_reduction &&
        !getenv("H3_DISABLE_TOKEN_REDUCTION");
    unsigned token_reduction_end =
        dit->token_reduction_early_steps &&
        (unsigned)step < dit->token_reduction_early_steps ?
            dit->token_reduction_early_end : dit->token_reduction_end;
    uint32_t hidden_elements = dit->sequence * HIDDEN;
    if (evaluate_core && dit->core_reuse_interval > 1)
        OP(h3_gpu_copy_bf16(dit->gpu, dit->core_input, 0, dit->hidden, 0,
                            hidden_elements), "save DiT core input");
    if (evaluate_core) {
        unsigned command_blocks = disable_command_split
            ? 0 : command_block_interval(dit);
        if (dit->ssd_streaming) command_blocks = 0;
        unsigned completed_blocks = 0;
        char block_phase[64];
        snprintf(block_phase, sizeof(block_phase),
                 "denoise step %d/%d transformer", step + 1,
                 h3_dit_schedule_steps(dit->schedule));
        report(progress, progress_opaque, block_phase, 0,
               (int)dit->active_block_count);
        int carried_attention_adaln = 0;
        int carried_attention_input_quantized = 0;
        for (unsigned block = 0; block < H3_DIT_BLOCKS; block++) {
            int fused_token_adaln = carried_attention_adaln;
            int fused_attention_input_quantized =
                carried_attention_input_quantized;
            carried_attention_adaln = 0;
            carried_attention_input_quantized = 0;
            if (use_token_reduction &&
                block == dit->token_reduction_begin) {
                fused_token_adaln = dit->block_active[block] &&
                    !getenv("H3_DISABLE_FUSED_TOKEN_POOL_ADALN");
                if (fused_token_adaln) {
                    if (!enter_token_reduction_adaln(
                            dit, block, step, error, error_size)) return 0;
                    fused_attention_input_quantized = 0;
                } else if (!enter_token_reduction(
                               dit, error, error_size)) return 0;
            }
            if (use_token_reduction && block == token_reduction_end) {
                fused_token_adaln = dit->block_active[block] &&
                    !getenv("H3_DISABLE_FUSED_TOKEN_ADALN");
                if (fused_token_adaln) {
                    if (!leave_token_reduction_adaln(
                            dit, block, step, error, error_size)) return 0;
                    fused_attention_input_quantized = 0;
                } else if (!leave_token_reduction(
                               dit, error, error_size)) return 0;
            }
            if (!dit->block_active[block]) continue;
            unsigned next_block = block + 1;
            int next_is_token_boundary = use_token_reduction &&
                (next_block == dit->token_reduction_begin ||
                 next_block == token_reduction_end);
            int fuse_next_attention =
                !getenv("H3_DISABLE_FUSED_CROSS_BLOCK_ADALN") &&
                next_block < H3_DIT_BLOCKS &&
                dit->block_active[next_block] && !next_is_token_boundary;
            h3_dit_block streamed_weight;
            h3_dit_block *weight = &dit->blocks[block];
            h3_dit_stream_job stream_job;
            pthread_t stream_thread;
            int stream_started = 0;
            if (dit->ssd_streaming) {
                if (dit->stream_ready_layer != block ||
                    dit->stream_ready_slot > 1) {
                    fail(error, error_size,
                         "DiT SSD stream expected block %u, has block %u",
                         block, dit->stream_ready_layer);
                    return 0;
                }
                h3_dit_block *slot =
                    &dit->stream_slots[dit->stream_ready_slot];
                streamed_weight = dit->blocks[block];
                if (dit->prequantized_int8) {
                    streamed_weight.qkv_int8 = slot->qkv_int8;
                    streamed_weight.qkv_scales = slot->qkv_scales;
                    streamed_weight.out_int8 = slot->out_int8;
                    streamed_weight.out_scales = slot->out_scales;
                    streamed_weight.fc1_int8 = slot->fc1_int8;
                    streamed_weight.fc1_scales = slot->fc1_scales;
                    streamed_weight.fc2_int8 = slot->fc2_int8;
                    streamed_weight.fc2_scales = slot->fc2_scales;
                } else {
                    streamed_weight.qkv = slot->qkv;
                    streamed_weight.out = slot->out;
                    streamed_weight.fc1 = slot->fc1;
                    streamed_weight.fc2 = slot->fc2;
                }
                weight = &streamed_weight;

                unsigned future = next_active_block(dit, block);
                if (future == H3_DIT_BLOCKS ||
                    (dit->cache_forward_is_cached &&
                     block == dit->cache_warm_last))
                    future = first_active_block(dit);
                stream_job = (h3_dit_stream_job){
                    .dit = dit,
                    .layer = future,
                    .slot = dit->stream_ready_slot ^ 1u
                };
                int thread_error = pthread_create(
                    &stream_thread, NULL, read_stream_layer_thread,
                    &stream_job);
                if (thread_error) {
                    fail(error, error_size,
                         "cannot start DiT SSD prefetch for block %u: %s",
                         future, strerror(thread_error));
                    return 0;
                }
                stream_started = 1;
            }
            int block_ok = run_block(
                dit, block, step, weight, fused_token_adaln,
                fused_attention_input_quantized,
                fuse_next_attention, next_block,
                &carried_attention_adaln,
                &carried_attention_input_quantized,
                error, error_size);
            if (!block_ok) {
                if (stream_started) (void)pthread_join(stream_thread, NULL);
                return 0;
            }
            completed_blocks++;
            if (command_blocks &&
                completed_blocks < dit->active_block_count &&
                completed_blocks % command_blocks == 0)
                OP(h3_gpu_continue(dit->gpu), "continue DiT command chain");
            if (stream_started) {
                int gpu_ok = gpu_op(dit, h3_gpu_submit(dit->gpu),
                                    error, error_size,
                                    "submit streamed DiT block");
                double wait_started = stream_now();
                int join_error = pthread_join(stream_thread, NULL);
                dit->stream_wait_seconds += stream_now() - wait_started;
                if (!gpu_ok) return 0;
                if (join_error) {
                    fail(error, error_size,
                         "cannot join DiT SSD prefetch: %s",
                         strerror(join_error));
                    return 0;
                }
                dit->stream_bytes += stream_job.bytes;
                dit->stream_read_seconds += stream_job.seconds;
                if (!stream_job.ok) {
                    fail(error, error_size,
                         "cannot stream DiT block %u: %s",
                         stream_job.layer, stream_job.error);
                    return 0;
                }
                dit->stream_ready_layer = stream_job.layer;
                dit->stream_ready_slot = stream_job.slot;
                OP(h3_gpu_begin(dit->gpu),
                   "continue after streamed DiT block");
            }
            report(progress, progress_opaque, block_phase,
                   (int)completed_blocks, (int)dit->active_block_count);
            if (block == dit->cache_warm_last && dit->block_cache &&
                dit->cache_plan_ready) {
                if (dit->cache_forward_is_cached) {
                    OP(h3_gpu_add_bf16(dit->gpu, dit->hidden, dit->hidden,
                                       dit->cache_residual, hidden_elements),
                       "replay DiT block cache residual");
                    break;
                }
                OP(h3_gpu_copy_bf16(dit->gpu, dit->cache_snapshot, 0,
                                    dit->hidden, 0, hidden_elements),
                   "snapshot DiT warm prefix");
            }
        }
        if (dit->cache_forward_captures) {
            OP(h3_gpu_sub_bf16(dit->gpu, dit->cache_residual, dit->hidden,
                               dit->cache_snapshot, hidden_elements),
               "capture DiT block cache residual");
            dit->cache_residual_ready = 1;
        }
        if (use_token_reduction &&
            token_reduction_end == H3_DIT_BLOCKS &&
            !leave_token_reduction(dit, error, error_size)) return 0;
        if (dit->core_reuse_interval > 1) {
            OP(h3_gpu_sub_bf16(dit->gpu, dit->core_residual, dit->hidden,
                               dit->core_input, hidden_elements),
               "cache DiT core residual");
            dit->core_residual_ready = 1;
        }
    } else {
        OP(h3_gpu_add_bf16(dit->gpu, dit->hidden, dit->hidden,
                           dit->core_residual, hidden_elements),
           "reuse DiT core residual");
    }
    dit->core_forward_count++;
    const h3_gpu_tensor *final = h3_dit_schedule_final(dit->schedule);
    int fused_final_head = dit->bf16_final &&
        !getenv("H3_DISABLE_FUSED_FINAL_HEAD") &&
        !getenv("H3_DISABLE_FUSED_FINAL_SLICE");
    if (fused_final_head) {
        OP(h3_gpu_adaln_linear_bf16(
            dit->gpu, dit->audio_output_bf16, dit->final_audio_inverse,
            dit->hidden, (size_t)dit->audio_target_start * HIDDEN,
            dit->final_norm, final, dit->final_audio_maps[step],
            dit->final_audio_w, dit->final_audio_b, dit->audio_rows, HIDDEN,
            AUDIO_CHANNELS, FINAL_SLOTS, 0, 1, 1e-5f),
           "fused final audio AdaLN/head");
        OP(h3_gpu_adaln_linear_bf16(
            dit->gpu, dit->video_output_bf16, dit->final_video_inverse,
            dit->hidden, (size_t)dit->video_target_start * HIDDEN,
            dit->final_norm, final, dit->final_video_maps[step],
            dit->final_video_w, dit->final_video_b, dit->video_rows, HIDDEN,
            VIDEO_PATCH, FINAL_SLOTS, 0, 1, 1e-5f),
           "fused final video AdaLN/head");
    } else if (getenv("H3_DISABLE_FUSED_FINAL_SLICE")) {
        OP(h3_gpu_copy_bf16(dit->gpu, dit->final_audio_input, 0, dit->hidden,
            (size_t)dit->audio_target_start * HIDDEN,
            (size_t)dit->audio_rows * HIDDEN), "slice final audio");
        OP(h3_gpu_copy_bf16(dit->gpu, dit->final_video_input, 0, dit->hidden,
            (size_t)dit->video_target_start * HIDDEN,
            (size_t)dit->video_rows * HIDDEN), "slice final video");
        OP(h3_gpu_adaln_bf16(dit->gpu, dit->final_audio_norm,
            dit->final_audio_input, dit->final_norm, final,
            dit->final_audio_maps[step], dit->audio_rows, HIDDEN, FINAL_SLOTS,
            0, 1, 1e-5f), "final audio AdaLN");
        OP(h3_gpu_adaln_bf16(dit->gpu, dit->final_video_norm,
            dit->final_video_input, dit->final_norm, final,
            dit->final_video_maps[step], dit->video_rows, HIDDEN, FINAL_SLOTS,
            0, 1, 1e-5f), "final video AdaLN");
    } else {
        OP(h3_gpu_adaln_bf16_offset(
            dit->gpu, dit->final_audio_norm, dit->hidden,
            (size_t)dit->audio_target_start * HIDDEN, dit->final_norm, final,
            dit->final_audio_maps[step], dit->audio_rows, HIDDEN, FINAL_SLOTS,
            0, 1, 1e-5f), "fused final audio slice/AdaLN");
        OP(h3_gpu_adaln_bf16_offset(
            dit->gpu, dit->final_video_norm, dit->hidden,
            (size_t)dit->video_target_start * HIDDEN, dit->final_norm, final,
            dit->final_video_maps[step], dit->video_rows, HIDDEN, FINAL_SLOTS,
            0, 1, 1e-5f), "fused final video slice/AdaLN");
    }
    if (dit->bf16_final && !fused_final_head) {
        OP(h3_gpu_linear_bf16(dit->gpu, dit->audio_output_bf16,
            dit->final_audio_norm, dit->final_audio_w, dit->final_audio_b,
            dit->audio_rows, HIDDEN, AUDIO_CHANNELS),
           "BF16 final audio head");
        OP(h3_gpu_linear_bf16(dit->gpu, dit->video_output_bf16,
            dit->final_video_norm, dit->final_video_w, dit->final_video_b,
            dit->video_rows, HIDDEN, VIDEO_PATCH),
           "BF16 final video head");
    } else if (!dit->bf16_final) {
        OP(h3_gpu_cast_bf16_to_f32(dit->gpu, dit->final_audio_f32,
            dit->final_audio_norm, dit->audio_rows * HIDDEN),
           "final audio F32 cast");
        OP(h3_gpu_cast_bf16_to_f32(dit->gpu, dit->final_video_f32,
            dit->final_video_norm, dit->video_rows * HIDDEN),
           "final video F32 cast");
        OP(h3_gpu_linear_f32(dit->gpu, dit->audio_output,
            dit->final_audio_f32, dit->final_audio_w, dit->final_audio_b,
            dit->audio_rows, HIDDEN, AUDIO_CHANNELS), "final audio head");
        OP(h3_gpu_linear_f32(dit->gpu, dit->video_output,
            dit->final_video_f32, dit->final_video_w, dit->final_video_b,
            dit->video_rows, HIDDEN, VIDEO_PATCH), "final video head");
        OP(h3_gpu_cast_f32_to_bf16(dit->gpu, dit->audio_output_bf16,
            dit->audio_output, dit->audio_rows * AUDIO_CHANNELS),
           "final audio output cast");
        OP(h3_gpu_cast_f32_to_bf16(dit->gpu, dit->video_output_bf16,
            dit->video_output, dit->video_rows * VIDEO_PATCH),
           "final video output cast");
    }
    if (submit) OP(h3_gpu_submit(dit->gpu), "submit DiT forward");
#undef OP
    return 1;
}

size_t h3_dit_video_elements(const h3_dit *dit) {
    return dit ? (size_t)VIDEO_CHANNELS * (size_t)dit->latent_t *
        (size_t)dit->latent_h * (size_t)dit->latent_w : 0;
}

size_t h3_dit_audio_elements(const h3_dit *dit) {
    return dit ? (size_t)AUDIO_CHANNELS * AUDIO_STREAMS *
        (size_t)dit->audio_t : 0;
}

int h3_dit_reset_run(h3_dit *dit,
                     const float *condition_video_rows,
                     size_t condition_video_elements,
                     const float *condition_audio_rows,
                     size_t condition_audio_elements,
                     char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!dit) {
        fail(error, error_size, "prepared DiT is absent");
        return 0;
    }
    size_t wanted_video =
        (size_t)dit->video_condition_rows * VIDEO_PATCH;
    size_t wanted_audio =
        (size_t)dit->audio_condition_rows * AUDIO_CHANNELS;
    if (condition_video_elements != wanted_video ||
        condition_audio_elements != wanted_audio ||
        (wanted_video && !condition_video_rows) ||
        (wanted_audio && !condition_audio_rows)) {
        fail(error, error_size, "prepared DiT condition rows do not match");
        return 0;
    }
    if ((wanted_video && !h3_gpu_tensor_write_f32_range(
             dit->video_input, 0, condition_video_rows, wanted_video)) ||
        (wanted_audio && !h3_gpu_tensor_write_f32_range(
             dit->audio_input, 0, condition_audio_rows, wanted_audio))) {
        fail(error, error_size, "cannot refresh prepared DiT conditions");
        return 0;
    }
    dit->core_forward_count = 0;
    dit->core_residual_ready = 0;
    return 1;
}

static int h3_dit_forward_with_progress(
                   h3_dit *dit, int step,
                   const float *video_latent, const float *audio_latent,
                   float *video_velocity, float *audio_velocity,
                   h3_dit_progress progress, void *progress_opaque,
                   char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!dit || step < 0 || step >= h3_dit_schedule_steps(dit->schedule) ||
        !video_latent || !audio_latent || !video_velocity || !audio_velocity) {
        fail(error, error_size, "invalid DiT forward arguments");
        return 0;
    }
    size_t video_row_elements = (size_t)dit->video_rows * VIDEO_PATCH;
    size_t audio_row_elements = (size_t)dit->audio_rows * AUDIO_CHANNELS;
    float *video_rows = malloc(video_row_elements * sizeof(*video_rows));
    float *audio_rows = malloc(audio_row_elements * sizeof(*audio_rows));
    uint16_t *video_out = malloc(video_row_elements * sizeof(*video_out));
    uint16_t *audio_out = malloc(audio_row_elements * sizeof(*audio_out));
    float *video_f32 = malloc(video_row_elements * sizeof(*video_f32));
    float *audio_f32 = malloc(audio_row_elements * sizeof(*audio_f32));
    if (!video_rows || !audio_rows || !video_out || !audio_out ||
        !video_f32 || !audio_f32) {
        fail(error, error_size, "out of memory packing DiT latents");
        free(video_rows); free(audio_rows); free(video_out); free(audio_out);
        free(video_f32); free(audio_f32);
        return 0;
    }
    int ok = h3_dit_patchify_video(video_latent, VIDEO_CHANNELS,
        dit->latent_t, dit->latent_h, dit->latent_w, video_rows,
        video_row_elements) &&
        h3_dit_pack_audio(audio_latent, AUDIO_CHANNELS, dit->audio_t,
                          audio_rows, audio_row_elements) &&
        h3_gpu_tensor_write_f32_range(
            dit->video_input,
            (size_t)dit->video_condition_rows * VIDEO_PATCH,
            video_rows, video_row_elements) &&
        h3_gpu_tensor_write_f32_range(
            dit->audio_input,
            (size_t)dit->audio_condition_rows * AUDIO_CHANNELS,
            audio_rows, audio_row_elements);
    if (!ok) fail(error, error_size, "cannot pack/write DiT input latents");
    if (ok) ok = encode_forward(dit, step, 1, 1, 0,
                                progress, progress_opaque,
                                error, error_size);
    if (ok) ok = h3_gpu_tensor_read_bf16(dit->video_output_bf16, video_out,
                                         video_row_elements) &&
                 h3_gpu_tensor_read_bf16(dit->audio_output_bf16, audio_out,
                                         audio_row_elements);
    if (!ok && (!error || !*error)) fail(error, error_size, "cannot read DiT output");
    if (ok) {
        for (size_t index = 0; index < video_row_elements; index++) {
            uint32_t bits = (uint32_t)video_out[index] << 16;
            memcpy(&video_f32[index], &bits, sizeof(bits));
        }
        for (size_t index = 0; index < audio_row_elements; index++) {
            uint32_t bits = (uint32_t)audio_out[index] << 16;
            memcpy(&audio_f32[index], &bits, sizeof(bits));
        }
    }
    if (ok) ok = h3_dit_unpatchify_video(video_f32, VIDEO_CHANNELS,
        dit->latent_t, dit->latent_h, dit->latent_w, video_velocity,
        h3_dit_video_elements(dit)) &&
        h3_dit_unpack_audio(audio_f32, AUDIO_CHANNELS, dit->audio_t,
                            audio_velocity, h3_dit_audio_elements(dit));
    if (!ok && (!error || !*error)) fail(error, error_size, "cannot unpack DiT output");
    free(video_rows); free(audio_rows); free(video_out); free(audio_out);
    free(video_f32); free(audio_f32);
    return ok;
}

int h3_dit_forward(h3_dit *dit, int step,
                   const float *video_latent, const float *audio_latent,
                   float *video_velocity, float *audio_velocity,
                   char *error, size_t error_size) {
    return h3_dit_forward_with_progress(
        dit, step, video_latent, audio_latent,
        video_velocity, audio_velocity, NULL, NULL, error, error_size);
}

int h3_dit_get_gpu_stats(const h3_dit *dit, h3_gpu_stats *stats) {
    return dit && h3_gpu_get_stats(dit->gpu, stats);
}

static float extrapolation_ratio(float current_sigma, float last_sigma,
                                 float previous_sigma, int have_previous) {
    if (!have_previous) return 0.0f;
    float denominator = last_sigma - previous_sigma;
    float ratio = denominator != 0.0f
        ? (current_sigma - last_sigma) / denominator : 0.0f;
    /* Reuse intervals are deliberately small. This guard prevents malformed
     * custom schedules from turning one cached evaluation into an explosion. */
    if (ratio < -2.0f) ratio = -2.0f;
    if (ratio > 2.0f) ratio = 2.0f;
    return ratio;
}

static void extrapolate_velocity(float *output, const float *last,
                                 const float *previous, size_t count,
                                 float current_sigma, float last_sigma,
                                 float previous_sigma, int have_previous) {
    if (!have_previous) {
        memcpy(output, last, count * sizeof(*output));
        return;
    }
    float ratio = extrapolation_ratio(current_sigma, last_sigma,
                                      previous_sigma, have_previous);
    for (size_t index = 0; index < count; index++)
        output[index] = last[index] +
                        ratio * (last[index] - previous[index]);
}

int h3_dit_reuse_schedule(int steps, int reuse_interval, uint8_t *selected,
                          size_t selected_count) {
    if (steps < 1 || reuse_interval < 1 || reuse_interval > 32 || !selected ||
        selected_count < (size_t)steps) return -1;
    memset(selected, 0, (size_t)steps);

    int count = 0;
    for (int step = 0; step < steps; step++) {
        if (reuse_interval == 1 || step == 0 || step == steps - 1 ||
            step % reuse_interval == 0) {
            selected[step] = 1;
            count++;
        }
    }
    return count;
}

static int parse_reuse_steps(int steps, uint8_t *selected) {
    const char *text = getenv("H3_REUSE_STEPS");
    if (!text || !*text) return 0;
    memset(selected, 0, (size_t)steps);
    int count = 0;
    int previous = -1;
    while (*text) {
        char *end = NULL;
        long value = strtol(text, &end, 10);
        if (end == text || value < 0 || value >= steps ||
            value <= previous) return -1;
        selected[value] = 1;
        previous = (int)value;
        count++;
        if (!*end) break;
        if (*end != ',') return -1;
        text = end + 1;
        if (!*text) return -1;
    }
    return selected[0] && selected[steps - 1] ? count : -1;
}

static int gpu_sampler_requested(const h3_dit *dit) {
    /* Source rows have to be written back after every Euler transition. The
     * host sampler owns those boundaries; the fused GPU sampler deliberately
     * stays on the ordinary generation path until it has the same primitive. */
    if (dit->inpaint_video_source) return 0;
    const char *cpu = getenv("H3_CPU_SAMPLER");
    if (cpu && *cpu && strcmp(cpu, "0")) return 0;
    const char *value = getenv("H3_GPU_SAMPLER");
    if (value) return *value && strcmp(value, "0");
    return h3_gpu_is_m5(dit->gpu);
}

static unsigned gpu_sampler_window(void) {
    const char *value = getenv("H3_GPU_SAMPLER_WINDOW");
    if (!value || !*value) return 1;
    char *end = NULL;
    long parsed = strtol(value, &end, 10);
    return end != value && !*end && parsed >= 0 && parsed <= H3_MAX_STEPS
        ? (unsigned)parsed : 1;
}

static int ensure_previous_velocities(h3_dit *dit, char *error,
                                      size_t error_size) {
    if (dit->previous_video_velocity && dit->previous_audio_velocity) return 1;
    free_tensor(&dit->previous_video_velocity);
    free_tensor(&dit->previous_audio_velocity);
    dit->previous_video_velocity = h3_gpu_tensor_new_bf16(
        dit->gpu, (size_t)dit->video_rows * VIDEO_PATCH);
    dit->previous_audio_velocity = h3_gpu_tensor_new_bf16(
        dit->gpu, (size_t)dit->audio_rows * AUDIO_CHANNELS);
    if (dit->previous_video_velocity && dit->previous_audio_velocity) return 1;
    free_tensor(&dit->previous_video_velocity);
    free_tensor(&dit->previous_audio_velocity);
    fail(error, error_size, "cannot allocate GPU Euler velocity cache: %s",
         h3_gpu_error(dit->gpu));
    return 0;
}

static int denoise_euler_gpu(h3_dit *dit, float *video_latent,
                             float *audio_latent, int reuse_interval,
                             const h3_dit_euler_resume *resume,
                             h3_dit_checkpoint checkpoint,
                             void *checkpoint_opaque,
                             h3_dit_progress progress, void *progress_opaque,
                             h3_dit_preview preview, void *preview_opaque,
                             char *error, size_t error_size) {
    uint8_t selected[H3_MAX_STEPS] = {0};
    int selected_count = h3_dit_reuse_schedule(
        dit->sigmas.steps, reuse_interval, selected, sizeof(selected));
    int custom_count = reuse_interval > 1 ?
        parse_reuse_steps(dit->sigmas.steps, selected) : 0;
    if (selected_count < 0 || custom_count < 0) {
        fail(error, error_size,
             "H3_REUSE_STEPS must be increasing and include 0 and %d",
             dit->sigmas.steps - 1);
        return 0;
    }
    if (custom_count > 0) selected_count = custom_count;
    if (reuse_interval > 1 && getenv("H3_PROFILE"))
        fprintf(stderr, "h3: %s GPU reuse schedule has %d evaluations\n",
                custom_count > 0 ? "custom" : "selected", selected_count);
    unsigned window = gpu_sampler_window();
    int disable_command_split = window == 1 &&
                                getenv("H3_DIT_COMMAND_BLOCKS") == NULL;
    if (getenv("H3_PROFILE"))
        fprintf(stderr, "h3: GPU sampler encode window is %s; internal split "
                "%s\n", window ? "bounded" : "unbounded",
                disable_command_split ? "disabled" : "enabled");

    size_t video_count = (size_t)dit->video_rows * VIDEO_PATCH;
    size_t audio_count = (size_t)dit->audio_rows * AUDIO_CHANNELS;
    size_t video_offset = (size_t)dit->video_condition_rows * VIDEO_PATCH;
    size_t audio_offset = (size_t)dit->audio_condition_rows * AUDIO_CHANNELS;
    if (video_count > UINT32_MAX || audio_count > UINT32_MAX ||
        video_offset > UINT32_MAX - video_count ||
        audio_offset > UINT32_MAX - audio_count ||
        (reuse_interval > 1 &&
         !ensure_previous_velocities(dit, error, error_size))) return 0;

    float *video_rows = malloc(video_count * sizeof(*video_rows));
    float *audio_rows = malloc(audio_count * sizeof(*audio_rows));
    float *last_video = checkpoint && reuse_interval > 1
        ? malloc(video_count * sizeof(*last_video)) : NULL;
    float *last_audio = checkpoint && reuse_interval > 1
        ? malloc(audio_count * sizeof(*last_audio)) : NULL;
    float *previous_video = checkpoint && reuse_interval > 1
        ? malloc(video_count * sizeof(*previous_video)) : NULL;
    float *previous_audio = checkpoint && reuse_interval > 1
        ? malloc(audio_count * sizeof(*previous_audio)) : NULL;
    if (!video_rows || !audio_rows ||
        (checkpoint && reuse_interval > 1 &&
         (!last_video || !last_audio || !previous_video || !previous_audio))) {
        fail(error, error_size, "out of memory packing GPU Euler latents");
        free(video_rows);
        free(audio_rows);
        free(last_video);
        free(last_audio);
        free(previous_video);
        free(previous_audio);
        return 0;
    }
    int ok = h3_dit_patchify_video(video_latent, VIDEO_CHANNELS,
        dit->latent_t, dit->latent_h, dit->latent_w, video_rows, video_count) &&
        h3_dit_pack_audio(audio_latent, AUDIO_CHANNELS, dit->audio_t,
                          audio_rows, audio_count) &&
        h3_gpu_tensor_write_f32_range(dit->video_input, video_offset,
                                      video_rows, video_count) &&
        h3_gpu_tensor_write_f32_range(dit->audio_input, audio_offset,
                                      audio_rows, audio_count);
    if (!ok) fail(error, error_size, "cannot pack/write GPU Euler latents");

    int start_step = resume ? resume->next_step : 0;
    int last_evaluated = resume ? resume->last_evaluated : -1;
    int previous_evaluated = resume ? resume->previous_evaluated : -1;
    if (ok && resume && reuse_interval > 1 && start_step < dit->sigmas.steps) {
        ok = h3_dit_patchify_video(
                 resume->last_video_velocity, VIDEO_CHANNELS,
                 dit->latent_t, dit->latent_h, dit->latent_w,
                 video_rows, video_count) &&
             h3_dit_pack_audio(
                 resume->last_audio_velocity, AUDIO_CHANNELS, dit->audio_t,
                 audio_rows, audio_count) &&
             h3_gpu_tensor_write_f32_range(
                 dit->video_output_bf16, 0, video_rows, video_count) &&
             h3_gpu_tensor_write_f32_range(
                 dit->audio_output_bf16, 0, audio_rows, audio_count);
        if (ok && previous_evaluated >= 0) {
            ok = h3_dit_patchify_video(
                     resume->previous_video_velocity, VIDEO_CHANNELS,
                     dit->latent_t, dit->latent_h, dit->latent_w,
                     video_rows, video_count) &&
                 h3_dit_pack_audio(
                     resume->previous_audio_velocity, AUDIO_CHANNELS,
                     dit->audio_t, audio_rows, audio_count) &&
                 h3_gpu_tensor_write_f32_range(
                     dit->previous_video_velocity, 0,
                     video_rows, video_count) &&
                 h3_gpu_tensor_write_f32_range(
                     dit->previous_audio_velocity, 0,
                     audio_rows, audio_count);
        }
        if (!ok) fail(error, error_size,
                      "cannot restore GPU Euler velocity state");
    }
    unsigned pending_evaluations = 0;
    int command_active = 0;
    for (int step = start_step; step < dit->sigmas.steps && ok; step++) {
        report(progress, progress_opaque, "denoise enqueue", step,
               dit->sigmas.steps);
        if (!command_active) {
            ok = gpu_op(dit, h3_gpu_begin(dit->gpu), error, error_size,
                        "begin GPU Euler command chain");
            command_active = ok;
        }
        if (!ok) break;
        int evaluate = selected[step];
        if (evaluate) {
            if (last_evaluated >= 0 && reuse_interval > 1) {
                ok = gpu_op(dit, h3_gpu_copy_bf16(
                    dit->gpu, dit->previous_video_velocity, 0,
                    dit->video_output_bf16, 0, video_count),
                    error, error_size, "cache previous video velocity") &&
                    gpu_op(dit, h3_gpu_copy_bf16(
                    dit->gpu, dit->previous_audio_velocity, 0,
                    dit->audio_output_bf16, 0, audio_count),
                    error, error_size, "cache previous audio velocity");
                if (ok) previous_evaluated = last_evaluated;
            }
            if (ok) ok = encode_forward(dit, step, 0, 0,
                                        disable_command_split,
                                        progress, progress_opaque,
                                        error, error_size);
            if (ok) {
                last_evaluated = step;
                pending_evaluations++;
            }
        }
        if (!ok) break;

        float video_ratio = evaluate ? 0.0f : extrapolation_ratio(
            dit->sigmas.video[step], dit->sigmas.video[last_evaluated],
            previous_evaluated >= 0
                ? dit->sigmas.video[previous_evaluated] : 0.0f,
            previous_evaluated >= 0);
        float audio_ratio = evaluate ? 0.0f : extrapolation_ratio(
            dit->sigmas.audio[step], dit->sigmas.audio[last_evaluated],
            previous_evaluated >= 0
                ? dit->sigmas.audio[previous_evaluated] : 0.0f,
            previous_evaluated >= 0);
        const h3_gpu_tensor *previous_video_tensor = previous_evaluated >= 0
            ? dit->previous_video_velocity : dit->video_output_bf16;
        const h3_gpu_tensor *previous_audio_tensor = previous_evaluated >= 0
            ? dit->previous_audio_velocity : dit->audio_output_bf16;
        ok = gpu_op(dit, h3_gpu_euler_bf16(
                dit->gpu, dit->video_input, video_offset,
                dit->video_output_bf16, previous_video_tensor,
                (uint32_t)video_count,
                dit->sigmas.video[step] - dit->sigmas.video[step + 1],
                video_ratio), error, error_size, "GPU video Euler step") &&
             gpu_op(dit, h3_gpu_euler_bf16(
                dit->gpu, dit->audio_input, audio_offset,
                dit->audio_output_bf16, previous_audio_tensor,
                (uint32_t)audio_count,
                dit->sigmas.audio[step] - dit->sigmas.audio[step + 1],
                audio_ratio), error, error_size, "GPU audio Euler step");
        if (ok && (evaluate || preview)) {
            int finish = checkpoint || preview ||
                         step + 1 == dit->sigmas.steps ||
                         (window && pending_evaluations >= window);
            ok = gpu_op(dit, finish ? h3_gpu_submit(dit->gpu)
                                    : h3_gpu_continue(dit->gpu),
                        error, error_size,
                        finish ? "submit GPU Euler window"
                               : "continue GPU Euler command chain");
            if (ok && finish) {
                command_active = 0;
                pending_evaluations = 0;
            }
        }
        if (ok && preview) {
            /* This path previews the sample rather than the denoised
             * estimate the CPU sampler shows: the velocity lives in a BF16
             * GPU tensor here and would need its own readback to combine.
             * The GPU sampler runs on M5-class hardware only, which this
             * change could not be exercised on, so the cheaper preview
             * stands until it can be tested there. */
            ok = h3_gpu_tensor_read_f32_range(
                     dit->video_input, video_offset, video_rows, video_count) &&
                 h3_dit_unpatchify_video(
                     video_rows, VIDEO_CHANNELS, dit->latent_t, dit->latent_h,
                     dit->latent_w, video_latent,
                     h3_dit_video_elements(dit));
            if (!ok) {
                fail(error, error_size,
                     "cannot read GPU Euler preview latent at step %d", step);
            } else if (preview(step + 1, dit->sigmas.steps, video_latent,
                               h3_dit_video_elements(dit), preview_opaque)) {
                fail(error, error_size,
                     "denoising preview stopped at step %d", step + 1);
                ok = 0;
            }
        }
        if (ok && checkpoint && evaluate) {
            ok = h3_gpu_tensor_read_f32_range(
                     dit->video_input, video_offset, video_rows, video_count) &&
                 h3_dit_unpatchify_video(
                     video_rows, VIDEO_CHANNELS, dit->latent_t, dit->latent_h,
                     dit->latent_w, video_latent,
                     h3_dit_video_elements(dit)) &&
                 h3_gpu_tensor_read_f32_range(
                     dit->audio_input, audio_offset, audio_rows, audio_count) &&
                 h3_dit_unpack_audio(
                     audio_rows, AUDIO_CHANNELS, dit->audio_t, audio_latent,
                     h3_dit_audio_elements(dit));
            int needs_velocity = reuse_interval > 1 &&
                                 step + 1 < dit->sigmas.steps;
            if (ok && needs_velocity) {
                ok = h3_gpu_tensor_read_f32_range(
                         dit->video_output_bf16, 0,
                         video_rows, video_count) &&
                     h3_dit_unpatchify_video(
                         video_rows, VIDEO_CHANNELS, dit->latent_t,
                         dit->latent_h, dit->latent_w, last_video,
                         video_count) &&
                     h3_gpu_tensor_read_f32_range(
                         dit->audio_output_bf16, 0,
                         audio_rows, audio_count) &&
                     h3_dit_unpack_audio(
                         audio_rows, AUDIO_CHANNELS, dit->audio_t,
                         last_audio, audio_count);
                if (ok && previous_evaluated >= 0) {
                    ok = h3_gpu_tensor_read_f32_range(
                             dit->previous_video_velocity, 0,
                             video_rows, video_count) &&
                         h3_dit_unpatchify_video(
                             video_rows, VIDEO_CHANNELS, dit->latent_t,
                             dit->latent_h, dit->latent_w, previous_video,
                             video_count) &&
                         h3_gpu_tensor_read_f32_range(
                             dit->previous_audio_velocity, 0,
                             audio_rows, audio_count) &&
                         h3_dit_unpack_audio(
                             audio_rows, AUDIO_CHANNELS, dit->audio_t,
                             previous_audio, audio_count);
                }
            }
            if (!ok) {
                fail(error, error_size,
                     "cannot read GPU Euler checkpoint at step %d", step + 1);
            } else {
                h3_dit_euler_checkpoint_state state = {
                    .total_steps = dit->sigmas.steps,
                    .next_step = step + 1,
                    .last_evaluated = last_evaluated,
                    .previous_evaluated = previous_evaluated,
                    .video_count = video_count,
                    .audio_count = audio_count,
                    .video = video_latent,
                    .audio = audio_latent,
                    .last_video_velocity = needs_velocity ? last_video : NULL,
                    .last_audio_velocity = needs_velocity ? last_audio : NULL,
                    .previous_video_velocity =
                        needs_velocity && previous_evaluated >= 0
                        ? previous_video : NULL,
                    .previous_audio_velocity =
                        needs_velocity && previous_evaluated >= 0
                        ? previous_audio : NULL
                };
                if (checkpoint(&state, checkpoint_opaque)) {
                    fail(error, error_size,
                         "checkpoint callback stopped at step %d", step + 1);
                    ok = 0;
                }
            }
        }
        if (ok) report(progress, progress_opaque, "denoise enqueue", step + 1,
                       dit->sigmas.steps);
    }
    if (ok && command_active)
        ok = gpu_op(dit, h3_gpu_submit(dit->gpu), error, error_size,
                    "submit GPU Euler denoise");
    if (ok) ok = h3_gpu_tensor_read_f32_range(
                     dit->video_input, video_offset, video_rows, video_count) &&
                 h3_gpu_tensor_read_f32_range(
                     dit->audio_input, audio_offset, audio_rows, audio_count);
    if (!ok && (!error || !*error))
        fail(error, error_size, "cannot read GPU Euler latents");
    if (ok) ok = h3_dit_unpatchify_video(
                     video_rows, VIDEO_CHANNELS, dit->latent_t, dit->latent_h,
                     dit->latent_w, video_latent, h3_dit_video_elements(dit)) &&
                 h3_dit_unpack_audio(audio_rows, AUDIO_CHANNELS, dit->audio_t,
                                     audio_latent,
                                     h3_dit_audio_elements(dit));
    if (!ok && (!error || !*error))
        fail(error, error_size, "cannot unpack GPU Euler latents");
    free(video_rows);
    free(audio_rows);
    free(last_video);
    free(last_audio);
    free(previous_video);
    free(previous_audio);
    if (ok) report(progress, progress_opaque, "denoise", dit->sigmas.steps,
                   dit->sigmas.steps);
    h3_gpu_profile_mark(dit->gpu, "GPU Euler denoise");
    return ok;
}

int h3_dit_denoise(h3_dit *dit, float *video_latent, float *audio_latent,
                   h3_dit_progress progress, void *progress_opaque,
                   char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!dit || !video_latent || !audio_latent ||
        dit->sigmas.steps != h3_dit_schedule_steps(dit->schedule)) {
        fail(error, error_size, "invalid DiT denoising arguments");
        return 0;
    }
    size_t video_count = h3_dit_video_elements(dit);
    size_t audio_count = h3_dit_audio_elements(dit);
    float *video_velocity = malloc(video_count * sizeof(*video_velocity));
    float *audio_velocity = malloc(audio_count * sizeof(*audio_velocity));
    float *video_denoised = malloc(video_count * sizeof(*video_denoised));
    float *audio_denoised = malloc(audio_count * sizeof(*audio_denoised));
    float *old_video = malloc(video_count * sizeof(*old_video));
    float *old_audio = malloc(audio_count * sizeof(*old_audio));
    float *video_next = malloc(video_count * sizeof(*video_next));
    float *audio_next = malloc(audio_count * sizeof(*audio_next));
    if (!video_velocity || !audio_velocity || !video_denoised ||
        !audio_denoised || !old_video || !old_audio || !video_next ||
        !audio_next) {
        fail(error, error_size, "out of memory allocating RES solver state");
        free(video_velocity); free(audio_velocity); free(video_denoised);
        free(audio_denoised); free(old_video); free(old_audio);
        free(video_next); free(audio_next);
        return 0;
    }
    int ok = 1;
    for (int step = 0; step < dit->sigmas.steps && ok; step++) {
        report(progress, progress_opaque, "denoise", step, dit->sigmas.steps);
        ok = h3_dit_forward_with_progress(
                            dit, step, video_latent, audio_latent,
                            video_velocity, audio_velocity,
                            progress, progress_opaque, error, error_size);
        float sigma = dit->sigmas.video[step];
        float timestep = 1.0f - sigma;
        float sigma_from_timestep = 1.0f - timestep;
        float audio_slope = (float)h3_time_shift_slope(
            sigma, H3_VIDEO_SIGMA_SHIFT, H3_AUDIO_SIGMA_SHIFT);
        if (ok) {
            for (size_t index = 0; index < video_count; index++)
                video_denoised[index] = video_latent[index] +
                    sigma_from_timestep * video_velocity[index];
            for (size_t index = 0; index < audio_count; index++)
                audio_denoised[index] = audio_latent[index] +
                    sigma_from_timestep * audio_velocity[index] * audio_slope;
            ok = h3_res_step(video_next, video_latent, video_denoised,
                             step ? old_video : NULL, video_count,
                             dit->sigmas.video, step, dit->sigmas.steps) &&
                 h3_res_step(audio_next, audio_latent, audio_denoised,
                             step ? old_audio : NULL, audio_count,
                             dit->sigmas.video, step, dit->sigmas.steps);
            if (!ok) fail(error, error_size, "RES solver rejected step %d", step);
        }
        if (ok) {
            memcpy(video_latent, video_next,
                   video_count * sizeof(*video_latent));
            memcpy(audio_latent, audio_next,
                   audio_count * sizeof(*audio_latent));
            memcpy(old_video, video_denoised,
                   video_count * sizeof(*old_video));
            memcpy(old_audio, audio_denoised,
                   audio_count * sizeof(*old_audio));
            report(progress, progress_opaque, "denoise", step + 1,
                   dit->sigmas.steps);
        }
    }
    free(video_velocity); free(audio_velocity); free(video_denoised);
    free(audio_denoised); free(old_video); free(old_audio);
    free(video_next); free(audio_next);
    h3_gpu_profile_mark(dit->gpu, "RES denoise");
    return ok;
}

/* Previews want the model's current estimate of the finished picture, not
 * the sample it is stepping through. This engine's velocity points from the
 * sample toward the clean image as sigma falls, so the estimate is
 * x0 = x_t + sigma * v — the same expression the res sampler already uses to
 * build its denoised buffer. It reads as a blurry version of the result from
 * the first step and sharpens; decoding x_t instead shows latent noise until
 * sigma collapses near the end, which is useless for deciding whether to
 * cancel. Returns the buffer to hand the preview callback. */
static const float *preview_estimate(float *destination, const float *sample,
                                     const float *velocity, size_t count,
                                     float sigma) {
    if (!destination || !velocity || !isfinite(sigma) || sigma <= 0.0f)
        return sample;
    for (size_t index = 0; index < count; index++)
        destination[index] = sample[index] + sigma * velocity[index];
    return destination;
}

static void impose_inpaint_video(h3_dit *dit, float *latent,
                                 const float *noise, float source_level) {
    if (!dit->inpaint_video_source || !dit->inpaint_video_generate_rows)
        return;
    int patch_width = dit->latent_w / 2;
    int patch_height = dit->latent_h / 2;
    for (int channel = 0; channel < VIDEO_CHANNELS; channel++)
        for (int time = 0; time < dit->latent_t; time++)
            for (int y = 0; y < dit->latent_h; y++)
                for (int x = 0; x < dit->latent_w; x++) {
                    size_t row = ((size_t)time * (size_t)patch_height +
                                  (size_t)(y / 2)) * (size_t)patch_width +
                                 (size_t)(x / 2);
                    if (dit->inpaint_video_generate_rows[row]) continue;
                    size_t index = (((size_t)channel * (size_t)dit->latent_t +
                                     (size_t)time) * (size_t)dit->latent_h +
                                    (size_t)y) * (size_t)dit->latent_w +
                                   (size_t)x;
                    latent[index] = source_level *
                                        dit->inpaint_video_source[index] +
                                    (1.0f - source_level) * noise[index];
                }
}

static void impose_inpaint_audio(h3_dit *dit, float *latent) {
    if (!dit->inpaint_audio_source || !dit->inpaint_audio_generate_rows)
        return;
    for (int channel = 0; channel < AUDIO_CHANNELS; channel++)
        for (int stereo = 0; stereo < AUDIO_STREAMS; stereo++)
            for (int time = 0; time < dit->audio_t; time++) {
                size_t row = (size_t)stereo * (size_t)dit->audio_t +
                             (size_t)time;
                if (dit->inpaint_audio_generate_rows[row]) continue;
                size_t index = ((size_t)channel * AUDIO_STREAMS +
                                (size_t)stereo) * (size_t)dit->audio_t +
                               (size_t)time;
                latent[index] = dit->inpaint_audio_source[index];
            }
}

int h3_dit_denoise_euler_resume(
                         h3_dit *dit, float *video_latent,
                         float *audio_latent, int reuse_interval,
                         const h3_dit_euler_resume *resume,
                         h3_dit_checkpoint checkpoint,
                         void *checkpoint_opaque,
                         h3_dit_progress progress, void *progress_opaque,
                         h3_dit_preview preview, void *preview_opaque,
                         char *error, size_t error_size) {
    int start_step = resume ? resume->next_step : 0;
    if (error && error_size) error[0] = '\0';
    if (!dit || !video_latent || !audio_latent || reuse_interval < 1 ||
        reuse_interval > 32 ||
        start_step < 0 || start_step > dit->sigmas.steps ||
        (start_step > 0 && dit->inpaint_video_source) ||
        (resume && reuse_interval > 1 && start_step < dit->sigmas.steps &&
         (resume->last_evaluated < 0 ||
          resume->last_evaluated >= start_step ||
          !resume->last_video_velocity || !resume->last_audio_velocity ||
          resume->previous_evaluated >= resume->last_evaluated ||
          (resume->previous_evaluated >= 0 &&
           (!resume->previous_video_velocity ||
            !resume->previous_audio_velocity)))) ||
        dit->sigmas.steps != h3_dit_schedule_steps(dit->schedule)) {
        fail(error, error_size, "invalid Euler denoising arguments");
        return 0;
    }
    if (gpu_sampler_requested(dit))
        return denoise_euler_gpu(dit, video_latent, audio_latent,
                                 reuse_interval, resume,
                                 checkpoint, checkpoint_opaque,
                                 progress, progress_opaque,
                                 preview, preview_opaque,
                                 error, error_size);
    h3_dit_block_cache_plan(dit);
    uint8_t selected[H3_MAX_STEPS] = {0};
    int selected_count = h3_dit_reuse_schedule(
        dit->sigmas.steps, reuse_interval, selected, sizeof(selected));
    int custom_count = reuse_interval > 1 ?
        parse_reuse_steps(dit->sigmas.steps, selected) : 0;
    if (selected_count < 0 || custom_count < 0) {
        fail(error, error_size,
             "H3_REUSE_STEPS must be increasing and include 0 and %d",
             dit->sigmas.steps - 1);
        return 0;
    }
    if (custom_count > 0) selected_count = custom_count;
    if (reuse_interval > 1 && getenv("H3_PROFILE"))
        fprintf(stderr, "h3: %s reuse schedule has %d evaluations\n",
                custom_count > 0 ? "custom" : "selected", selected_count);
    size_t video_count = h3_dit_video_elements(dit);
    size_t audio_count = h3_dit_audio_elements(dit);
    float *video_velocity = malloc(video_count * sizeof(*video_velocity));
    float *audio_velocity = malloc(audio_count * sizeof(*audio_velocity));
    float *preview_video = preview
        ? malloc(video_count * sizeof(*preview_video)) : NULL;
    float *last_video = reuse_interval > 1
        ? malloc(video_count * sizeof(*last_video)) : NULL;
    float *previous_video = reuse_interval > 1
        ? malloc(video_count * sizeof(*previous_video)) : NULL;
    float *last_audio = reuse_interval > 1
        ? malloc(audio_count * sizeof(*last_audio)) : NULL;
    float *previous_audio = reuse_interval > 1
        ? malloc(audio_count * sizeof(*previous_audio)) : NULL;
    float *inpaint_video_noise = dit->inpaint_video_source
        ? malloc(video_count * sizeof(*inpaint_video_noise)) : NULL;
    if (!video_velocity || !audio_velocity ||
        (preview && !preview_video) ||
        (dit->inpaint_video_source && !inpaint_video_noise) ||
        (reuse_interval > 1 &&
         (!last_video || !previous_video || !last_audio || !previous_audio))) {
        fail(error, error_size, "out of memory allocating Euler velocities");
        free(video_velocity);
        free(audio_velocity);
        free(preview_video);
        free(last_video);
        free(previous_video);
        free(last_audio);
        free(previous_audio);
        free(inpaint_video_noise);
        return 0;
    }
    if (inpaint_video_noise) {
        memcpy(inpaint_video_noise, video_latent,
               video_count * sizeof(*inpaint_video_noise));
        impose_inpaint_video(dit, video_latent, inpaint_video_noise, 0.999f);
        impose_inpaint_audio(dit, audio_latent);
    }
    int ok = 1;
    int last_evaluated = resume ? resume->last_evaluated : -1;
    int previous_evaluated = resume ? resume->previous_evaluated : -1;
    if (resume && reuse_interval > 1 && start_step < dit->sigmas.steps) {
        memcpy(last_video, resume->last_video_velocity,
               video_count * sizeof(*last_video));
        memcpy(last_audio, resume->last_audio_velocity,
               audio_count * sizeof(*last_audio));
        if (previous_evaluated >= 0) {
            memcpy(previous_video, resume->previous_video_velocity,
                   video_count * sizeof(*previous_video));
            memcpy(previous_audio, resume->previous_audio_velocity,
                   audio_count * sizeof(*previous_audio));
        }
    }
    for (int step = start_step; step < dit->sigmas.steps && ok; step++) {
        report(progress, progress_opaque, "denoise", step, dit->sigmas.steps);
        int evaluate = selected[step];
        if (evaluate) {
            ok = h3_dit_forward_with_progress(
                                dit, step, video_latent, audio_latent,
                                video_velocity, audio_velocity,
                                progress, progress_opaque,
                                error, error_size);
            if (ok && reuse_interval > 1) {
                if (last_evaluated >= 0) {
                    memcpy(previous_video, last_video,
                           video_count * sizeof(*previous_video));
                    memcpy(previous_audio, last_audio,
                           audio_count * sizeof(*previous_audio));
                    previous_evaluated = last_evaluated;
                }
                memcpy(last_video, video_velocity,
                       video_count * sizeof(*last_video));
                memcpy(last_audio, audio_velocity,
                       audio_count * sizeof(*last_audio));
                last_evaluated = step;
            }
        } else {
            extrapolate_velocity(
                video_velocity, last_video, previous_video, video_count,
                dit->sigmas.video[step], dit->sigmas.video[last_evaluated],
                previous_evaluated >= 0
                    ? dit->sigmas.video[previous_evaluated] : 0.0f,
                previous_evaluated >= 0);
            extrapolate_velocity(
                audio_velocity, last_audio, previous_audio, audio_count,
                dit->sigmas.audio[step], dit->sigmas.audio[last_evaluated],
                previous_evaluated >= 0
                    ? dit->sigmas.audio[previous_evaluated] : 0.0f,
                previous_evaluated >= 0);
        }
        if (ok) {
            ok = h3_euler_velocity_step(
                     video_latent, video_velocity, video_count,
                     dit->sigmas.video[step], dit->sigmas.video[step + 1]) &&
                 h3_euler_velocity_step(
                     audio_latent, audio_velocity, audio_count,
                     dit->sigmas.audio[step], dit->sigmas.audio[step + 1]);
            if (!ok) fail(error, error_size,
                          "Euler solver rejected step %d", step);
        }
        if (ok && inpaint_video_noise) {
            float level = step + 1 == dit->sigmas.steps ? 1.0f : 0.999f;
            impose_inpaint_video(
                dit, video_latent, inpaint_video_noise, level);
            impose_inpaint_audio(dit, audio_latent);
        }
        if (ok && checkpoint) {
            int needs_velocity = reuse_interval > 1 &&
                                 step + 1 < dit->sigmas.steps;
            h3_dit_euler_checkpoint_state state = {
                .total_steps = dit->sigmas.steps,
                .next_step = step + 1,
                .last_evaluated = last_evaluated,
                .previous_evaluated = previous_evaluated,
                .video_count = video_count,
                .audio_count = audio_count,
                .video = video_latent,
                .audio = audio_latent,
                .last_video_velocity = needs_velocity ? last_video : NULL,
                .last_audio_velocity = needs_velocity ? last_audio : NULL,
                .previous_video_velocity =
                    needs_velocity && previous_evaluated >= 0
                    ? previous_video : NULL,
                .previous_audio_velocity =
                    needs_velocity && previous_evaluated >= 0
                    ? previous_audio : NULL
            };
            if (checkpoint(&state, checkpoint_opaque)) {
                fail(error, error_size,
                     "checkpoint callback stopped at step %d", step + 1);
                ok = 0;
            }
        }
        if (ok && preview) {
            const float *estimate = preview_estimate(
                preview_video, video_latent, video_velocity, video_count,
                dit->sigmas.video[step + 1]);
            if (inpaint_video_noise) {
                impose_inpaint_video(dit, preview_video,
                                     inpaint_video_noise, 1.0f);
                estimate = preview_video;
            }
            if (preview(step + 1, dit->sigmas.steps, estimate,
                        video_count, preview_opaque)) {
                fail(error, error_size,
                     "denoising preview stopped at step %d", step + 1);
                ok = 0;
            }
        }
        if (ok) report(progress, progress_opaque, "denoise", step + 1,
                       dit->sigmas.steps);
    }
    free(video_velocity);
    free(audio_velocity);
    free(preview_video);
    free(last_video);
    free(previous_video);
    free(last_audio);
    free(previous_audio);
    free(inpaint_video_noise);
    h3_gpu_profile_mark(dit->gpu, "Euler denoise");
    return ok;
}

int h3_dit_denoise_euler_preview(
                         h3_dit *dit, float *video_latent,
                         float *audio_latent, int reuse_interval,
                         h3_dit_progress progress, void *progress_opaque,
                         h3_dit_preview preview, void *preview_opaque,
                         char *error, size_t error_size) {
    return h3_dit_denoise_euler_resume(
        dit, video_latent, audio_latent, reuse_interval,
        NULL, NULL, NULL, progress, progress_opaque,
        preview, preview_opaque, error, error_size);
}

int h3_dit_denoise_euler(h3_dit *dit, float *video_latent,
                         float *audio_latent, int reuse_interval,
                         h3_dit_progress progress, void *progress_opaque,
                         char *error, size_t error_size) {
    return h3_dit_denoise_euler_preview(
        dit, video_latent, audio_latent, reuse_interval,
        progress, progress_opaque, NULL, NULL, error, error_size);
}

void h3_dit_free(h3_dit *dit) {
    if (!dit) return;
    int steps = h3_dit_schedule_steps(dit->schedule);
    if (dit->row_maps) for (int step = 0; step < steps; step++)
        h3_gpu_tensor_free(dit->row_maps[step]);
    if (dit->reduced_row_maps) for (int step = 0; step < steps; step++)
        h3_gpu_tensor_free(dit->reduced_row_maps[step]);
    if (dit->final_audio_maps) for (int step = 0; step < steps; step++)
        h3_gpu_tensor_free(dit->final_audio_maps[step]);
    if (dit->final_video_maps) for (int step = 0; step < steps; step++)
        h3_gpu_tensor_free(dit->final_video_maps[step]);
    free(dit->row_maps);
    free(dit->reduced_row_maps);
    free(dit->final_audio_maps);
    free(dit->final_video_maps);
    free(dit->inpaint_video_source);
    free(dit->inpaint_video_generate_rows);
    free(dit->inpaint_audio_source);
    free(dit->inpaint_audio_generate_rows);
    free_tensor(&dit->refined_text);
    free_tensor(&dit->rope_cos);
    free_tensor(&dit->rope_sin);
    free_tensor(&dit->reduced_rope_cos);
    free_tensor(&dit->reduced_rope_sin);
    free_tensor(&dit->video_patch_w); free_tensor(&dit->video_patch_b);
    free_tensor(&dit->audio_patch_w); free_tensor(&dit->audio_patch_b);
    for (unsigned block = 0; block < H3_DIT_BLOCKS; block++)
        free_block(&dit->blocks[block]);
    free_block(&dit->stream_slots[0]);
    free_block(&dit->stream_slots[1]);
    free_tensor(&dit->final_norm);
    free_tensor(&dit->final_video_w); free_tensor(&dit->final_video_b);
    free_tensor(&dit->final_audio_w); free_tensor(&dit->final_audio_b);
#define FREE(field) free_tensor(&dit->field)
    if (dit->activation_aliases) {
        dit->attention_heads = NULL;
        dit->mod_mlp = NULL;
    }
    FREE(video_input); FREE(audio_input);
    FREE(video_projected_f32); FREE(audio_projected_f32);
    FREE(video_projected); FREE(audio_projected);
    FREE(video_projection_map); FREE(audio_projection_map); FREE(hidden);
    FREE(core_input); FREE(core_residual);
    FREE(mod_attention); FREE(qkv); FREE(query); FREE(key); FREE(value);
    FREE(query32); FREE(key32); FREE(value32); FREE(heads32);
    FREE(sol_query_centroids); FREE(sol_key_centroids); FREE(sol_value_sums);
    FREE(sol_key_means); FREE(sol_key_variances); FREE(sol_thresholds);
    FREE(sol_routes);
    FREE(attention_heads); FREE(attention_output);
    FREE(token_pool_pairs); FREE(token_baseline_indices);
    FREE(token_expand_parents); FREE(token_original); FREE(mod_mlp); FREE(fc1);
    FREE(activated); FREE(mlp_output); FREE(int8_activation);
    FREE(int8_activation_scales); FREE(lora_hidden); FREE(lora_delta); FREE(final_audio_input);
    FREE(final_video_input); FREE(final_audio_inverse);
    FREE(final_video_inverse); FREE(final_audio_norm); FREE(final_video_norm);
    FREE(final_audio_f32); FREE(final_video_f32); FREE(audio_output);
    FREE(video_output);
    FREE(audio_output_bf16); FREE(video_output_bf16);
    FREE(previous_audio_velocity); FREE(previous_video_velocity);
#undef FREE
    h3_dit_schedule_free(dit->schedule);
    if (dit->ssd_streaming && getenv("H3_PROFILE")) {
        double gib = (double)dit->stream_bytes / (1024.0 * 1024.0 * 1024.0);
        fprintf(stderr,
                "h3: %s%s%s SSD stream %.3f GiB read in %.3fs (%.3f GiB/s), "
                "unhidden wait %.3fs\n",
                dit->prequantized_int8 ? "I8/F32" : "BF16",
                dit->input_major_transformer ? "+IM" : "",
                dit->hybrid_adaln ? "+HYBRID" : "",
                gib, dit->stream_read_seconds,
                dit->stream_read_seconds > 0.0
                    ? gib / dit->stream_read_seconds : 0.0,
                dit->stream_wait_seconds);
    }
    h3_gpu_free(dit->gpu);
    h3_weight_store_free(dit->late_adaln_overlay);
    h3_weight_store_free(dit->weights);
    h3_layout_free(&dit->layout);
    free(dit);
}

static int video_shape(int channels, int time, int height, int width,
                       size_t *latent_count, size_t *row_count) {
    if (channels < 1 || time < 1 || height < 2 || width < 2 ||
        height % 2 || width % 2) return 0;
    size_t c = (size_t)channels, t = (size_t)time;
    size_t h = (size_t)height, w = (size_t)width;
    if (c > SIZE_MAX / t || c * t > SIZE_MAX / h ||
        c * t * h > SIZE_MAX / w) return 0;
    *latent_count = c * t * h * w;
    *row_count = t * (h / 2) * (w / 2) * c * 4;
    return 1;
}

int h3_dit_patchify_video(const float *latent, int channels, int time,
                          int height, int width, float *rows,
                          size_t row_elements) {
    size_t latent_count, expected;
    if (!latent || !rows ||
        !video_shape(channels, time, height, width, &latent_count, &expected) ||
        row_elements != expected || latent_count != expected) return 0;
    size_t output = 0;
    for (int t = 0; t < time; t++)
        for (int h = 0; h < height; h += 2)
            for (int w = 0; w < width; w += 2)
                for (int c = 0; c < channels; c++)
                    for (int dh = 0; dh < 2; dh++)
                        for (int dw = 0; dw < 2; dw++) {
                            size_t input = (((size_t)c * (size_t)time +
                                (size_t)t) * (size_t)height + (size_t)(h + dh)) *
                                (size_t)width + (size_t)(w + dw);
                            rows[output++] = latent[input];
                        }
    return output == row_elements;
}

int h3_dit_unpatchify_video(const float *rows, int channels, int time,
                            int height, int width, float *latent,
                            size_t latent_elements) {
    size_t expected, row_count;
    if (!rows || !latent ||
        !video_shape(channels, time, height, width, &expected, &row_count) ||
        latent_elements != expected || row_count != expected) return 0;
    size_t input = 0;
    for (int t = 0; t < time; t++)
        for (int h = 0; h < height; h += 2)
            for (int w = 0; w < width; w += 2)
                for (int c = 0; c < channels; c++)
                    for (int dh = 0; dh < 2; dh++)
                        for (int dw = 0; dw < 2; dw++) {
                            size_t output = (((size_t)c * (size_t)time +
                                (size_t)t) * (size_t)height + (size_t)(h + dh)) *
                                (size_t)width + (size_t)(w + dw);
                            latent[output] = rows[input++];
                        }
    return input == row_count;
}

int h3_dit_pack_audio(const float *latent, int channels, int time,
                      float *rows, size_t row_elements) {
    if (!latent || !rows || channels < 1 || time < 1 ||
        (size_t)channels > SIZE_MAX / (2 * (size_t)time) ||
        row_elements != (size_t)channels * 2 * (size_t)time) return 0;
    size_t output = 0;
    for (int stream = 0; stream < 2; stream++)
        for (int t = 0; t < time; t++)
            for (int channel = 0; channel < channels; channel++) {
                size_t input = ((size_t)channel * 2 + (size_t)stream) *
                               (size_t)time + (size_t)t;
                rows[output++] = latent[input];
            }
    return output == row_elements;
}

int h3_dit_unpack_audio(const float *rows, int channels, int time,
                        float *latent, size_t latent_elements) {
    if (!rows || !latent || channels < 1 || time < 1 ||
        (size_t)channels > SIZE_MAX / (2 * (size_t)time) ||
        latent_elements != (size_t)channels * 2 * (size_t)time) return 0;
    size_t input = 0;
    for (int stream = 0; stream < 2; stream++)
        for (int t = 0; t < time; t++)
            for (int channel = 0; channel < channels; channel++) {
                size_t output = ((size_t)channel * 2 + (size_t)stream) *
                                (size_t)time + (size_t)t;
                latent[output] = rows[input++];
            }
    return input == latent_elements;
}
