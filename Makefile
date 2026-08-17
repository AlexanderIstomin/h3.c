CC := clang
AR := ar
CFLAGS := -std=c11 -O3 -MMD -MP -Wall -Wextra -Wpedantic -Wshadow \
	-Wconversion -Wno-sign-conversion -D_DARWIN_C_SOURCE
OBJCFLAGS := $(CFLAGS) -fobjc-arc
FRAMEWORKS := -framework Foundation -framework Metal \
	-framework AVFoundation -framework CoreMedia -framework CoreVideo \
	-framework VideoToolbox -framework AudioToolbox -framework ImageIO \
	-framework CoreGraphics \
	-framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph \
	-framework Accelerate
LDLIBS := $(FRAMEWORKS) -licucore -lm

LIB_C := h3.c h3_host.c h3_safetensors.c h3_weights.c h3_text_encoder.c \
	h3_dit_schedule.c h3_dit.c

LIB_C += h3_video_vae.c h3_video_encoder.c h3_audio_vae.c h3_ffmpeg.c \
	h3_terminal.c h3_vision_encoder.c h3_multimodal.c h3_tae.c
LIB_M := h3_metal.m h3_gpu.m h3_tokenizer.m h3_avwriter.m h3_avreader.m
LIB_OBJ := $(LIB_C:.c=.o) $(LIB_M:.m=.o)
CLI_OBJ := main.o h3_cli.o linenoise.o

.PHONY: all test parity real-parity real-int8 ltx-int8 ltx-tokenizer clean

all: h3 libh3.a

h3: $(CLI_OBJ) $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

libh3.a: $(LIB_OBJ)
	$(AR) rcs $@ $^

h3_tests: tests/test_h3.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_metal_tests: tests/test_metal.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_bf16_tests: tests/test_bf16.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_tokenizer_tests: tests/test_tokenizer.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_text_tests: tests/test_text_metal.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_audio_gpu_tests: tests/test_audio_gpu.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_quantized_weight_tests: tests/test_quantized_weights.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_optimized_int8_test: tests/test_real_optimized_int8.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_ltx_int8_format_test: tests/test_ltx_int8_format.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_gemma_tokenizer_test: tests/test_gemma_tokenizer.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_gemma_block_test: tests/test_gemma_block.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_gemma_attention_gpu_test: tests/test_gemma_attention_gpu.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_ltx_conditioning_test: tests/test_ltx_conditioning.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_ltx_dit_block_test: tests/test_ltx_dit_block.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_ltx_adaln_single_test: tests/test_ltx_adaln_single.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_ltx_rope_test: tests/test_ltx_rope.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_ltx_head_test: tests/test_ltx_head.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_ltx_sampler_test: tests/test_ltx_sampler.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_ltx_text_test: tests/test_real_ltx_text.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_ltx_connector_test: tests/test_real_ltx_connector.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_ltx_dit_test: tests/test_real_ltx_dit.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_ltx_dit_block_test: tests/test_real_ltx_dit_block.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_ltx_dit_top_test: tests/test_real_ltx_dit_top.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_ltx_dit_run_test: tests/test_real_ltx_dit_run.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_ltx_dit_sample_test: tests/test_real_ltx_dit_sample.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_ltx_video_vae_test: tests/test_real_ltx_video_vae.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_optimized_qwen_test: tests/test_real_optimized_qwen.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_optimized_audio_vae_test: tests/test_real_optimized_audio_vae.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_optimized_video_vae_test: tests/test_real_optimized_video_vae.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_optimized_dit_stream_test: tests/test_real_optimized_dit_stream.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_audio_vae_test: tests/test_real_audio_vae.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_audio_encoder_test: tests/test_real_audio_encoder.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_av_mux_test: tests/test_av_mux.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_video_encoder_test: tests/test_real_video_encoder.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_optimized_vision_test: tests/test_optimized_vision.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_prompt_discrimination_test: tests/test_prompt_discrimination.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_tae_test: tests/test_tae.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_gqa_test: tests/test_gqa.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_avwriter_test: tests/test_avwriter.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_avreader_test: tests/test_avreader.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_qwen_vision_test: tests/test_real_qwen_vision.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_multimodal_text_test: tests/test_real_multimodal_text.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_ref_video_text_test: tests/test_real_ref_video_text.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_prompt_test: tests/test_real_prompt.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_dit_block_test: tests/test_real_dit_block.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_dit_schedule_test: tests/test_real_dit_schedule.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_dit_test: tests/test_real_dit.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_semantic_dit_test: tests/test_semantic_dit.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_dit_bench: tests/bench_dit.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_dit_bench_864: tests/bench_dit_864.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

tests/bench_dit_864.o: tests/bench_dit.c
	$(CC) $(CFLAGS) -I. -DH3_BENCH_LATENT_H=30 \
		-DH3_BENCH_LATENT_W=54 -c $< -o $@

h3_int8_video_vae_test: tests/test_int8_video_vae.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_video_vae_test: tests/test_real_video_vae.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_semantic_vae_test: tests/test_semantic_vae.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

test: h3_tests h3_metal_tests h3_bf16_tests h3_tokenizer_tests h3_text_tests \
	h3_audio_gpu_tests h3_quantized_weight_tests \
	h3_real_audio_vae_test h3_real_audio_encoder_test \
	h3_av_mux_test \
	h3_real_video_encoder_test h3_real_qwen_vision_test \
	h3_real_multimodal_text_test h3_real_ref_video_text_test

	./h3_tests
	@if test -f misc/fixtures/h3_dit.safetensors && \
	         test -f misc/fixtures/h3_dit_bf16.safetensors; then \
		./h3_metal_tests misc/fixtures/h3_dit.safetensors; \
		./h3_bf16_tests misc/fixtures/h3_dit_bf16.safetensors; \
	else \
		echo "skip: MLX toy-block fixtures are not installed"; \
	fi
	@if test -f MiniMax-H3/tokenizer/tokenizer.json; then \
		./h3_tokenizer_tests MiniMax-H3/tokenizer/tokenizer.json; \
	else \
		echo "skip: released tokenizer is not installed"; \
	fi
	@if test -f misc/fixtures/h3_text_bf16.safetensors; then \
		./h3_text_tests misc/fixtures/h3_text_bf16.safetensors; \
	else \
		echo "skip: MLX Qwen fixture is not installed"; \
	fi
	./h3_audio_gpu_tests
	./h3_quantized_weight_tests
	@if test -f MiniMax-H3/FL2VA/audio_vae/model.safetensors && \
	         test -f misc/fixtures/h3_real_audio_vae_37.safetensors; then \
		./h3_real_audio_vae_test; \
	else \
		echo "skip: released AudioVAE weights/fixture are not installed"; \
	fi
	@if test -f MiniMax-H3/FL2VA/audio_vae/model.safetensors && \
	         test -f misc/fixtures/h3_real_audio_encoder_64000.safetensors; then \
		./h3_real_audio_encoder_test; \
	else \
		echo "skip: released audio encoder weights/fixture are not installed"; \
	fi
	@if command -v ffmpeg >/dev/null 2>&1; then \
		./h3_av_mux_test; \
	else \
		echo "skip: FFmpeg is not installed"; \
	fi
	@if test -f MiniMax-H3/FL2VA/video_vae/source/model.safetensors && \
	         test -f misc/fixtures/h3_real_video_encoder_256.safetensors; then \
		./h3_real_video_encoder_test; \
	else \
		echo "skip: released visual encoder weights/fixture are not installed"; \
	fi
	@if test -f MiniMax-H3/Ref2VA/video_vae/source/model.safetensors && \
	         test -f misc/fixtures/h3_real_video_encoder_video_22x64.safetensors; then \
		./h3_real_video_encoder_test MiniMax-H3 \
			misc/fixtures/h3_real_video_encoder_video_22x64.safetensors; \
	else \
		echo "skip: released reference-video encoder fixture is not installed"; \
	fi
	@if test -f MiniMax-H3/FL2VA/text_encoder/model-00014-of-00014.safetensors && \
	         test -f misc/fixtures/h3_real_qwen_vision_64.safetensors; then \
		./h3_real_qwen_vision_test; \
	else \
		echo "skip: released Qwen vision weights/fixture are not installed"; \
	fi
	@if test -f MiniMax-H3/Ref2VA/text_encoder/model-00014-of-00014.safetensors && \
	         test -f misc/fixtures/h3_real_qwen_vision_video2x64.safetensors; then \
		./h3_real_qwen_vision_test MiniMax-H3 \
			misc/fixtures/h3_real_qwen_vision_video2x64.safetensors; \
	else \
		echo "skip: released Qwen video-pair fixture is not installed"; \
	fi
	@if test -f MiniMax-H3/FL2VA/text_encoder/model-00001-of-00014.safetensors && \
	         test -f misc/fixtures/h3_real_multimodal_text_64.safetensors; then \
		./h3_real_multimodal_text_test; \
	else \
		echo "skip: released multimodal Qwen weights/fixture are not installed"; \
	fi
	@if test -f MiniMax-H3/Ref2VA/text_encoder/model-00001-of-00014.safetensors && \
	         test -f misc/fixtures/h3_real_ref_video_text_64.safetensors; then \
		./h3_real_ref_video_text_test; \
	else \
		echo "skip: Ref2VA video presentation fixture is not installed"; \
	fi

parity: h3_metal_tests h3_bf16_tests h3_text_tests
	./h3_metal_tests misc/fixtures/h3_dit.safetensors
	./h3_bf16_tests misc/fixtures/h3_dit_bf16.safetensors
	./h3_text_tests misc/fixtures/h3_text_bf16.safetensors

real-parity: h3_real_prompt_test h3_real_dit_block_test
	./h3_real_prompt_test MiniMax-H3 misc/fixtures/h3_real_prompt_bf16.safetensors
	./h3_real_dit_block_test MiniMax-H3 misc/fixtures/h3_real_dit_block0_bf16.safetensors

real-int8: h3_real_optimized_int8_test h3_real_optimized_qwen_test \
	h3_real_optimized_audio_vae_test h3_real_optimized_video_vae_test \
	h3_real_optimized_dit_stream_test
	@test -n "$(H3_OPTIMIZED_MODEL)" || \
		(echo "set H3_OPTIMIZED_MODEL to the installed package root" >&2; exit 2)
	./h3_real_optimized_int8_test "$(H3_OPTIMIZED_MODEL)"
	./h3_real_optimized_qwen_test "$(H3_OPTIMIZED_MODEL)"
	./h3_real_optimized_audio_vae_test "$(H3_OPTIMIZED_MODEL)"
	./h3_real_optimized_video_vae_test "$(H3_OPTIMIZED_MODEL)"
	./h3_real_optimized_dit_stream_test "$(H3_OPTIMIZED_MODEL)"

# LTX-2.5 publishes both its DiT and its Gemma 4 text tower in the same Comfy
# INT8 ConvRot encoding as the optimized H3 packages. Point H3_LTX_FIXTURE at
# a DiT block-0 extract and H3_LTX_TEXT_FIXTURE at a Gemma layer extract to
# check that claim against real weights on this GPU. Either may be omitted.
ltx-int8: h3_ltx_int8_format_test
	@test -n "$(H3_LTX_FIXTURE)$(H3_LTX_TEXT_FIXTURE)" || \
		(echo "set H3_LTX_FIXTURE and/or H3_LTX_TEXT_FIXTURE to LTX-2.5 safetensors extracts" >&2; exit 2)
	@test -z "$(H3_LTX_FIXTURE)" || ./h3_ltx_int8_format_test "$(H3_LTX_FIXTURE)"
	@test -z "$(H3_LTX_TEXT_FIXTURE)" || ./h3_ltx_int8_format_test "$(H3_LTX_TEXT_FIXTURE)"

# Gemma 4's tokenizer is SentencePiece-style rather than the byte-level BPE
# Qwen3-VL uses, so both flavours are checked against ids generated by the
# reference `tokenizers` library. Give each fixture as TOKENIZER:REFERENCE.
ltx-tokenizer: h3_gemma_tokenizer_test
	@test -n "$(H3_TOKENIZER_CASES)" || \
		(echo "set H3_TOKENIZER_CASES to space-separated TOKENIZER.json:REFERENCE.json pairs" >&2; exit 2)
	@for pair in $(H3_TOKENIZER_CASES); do \
		./h3_gemma_tokenizer_test "$${pair%%:*}" "$${pair##*:}" || exit 1; \
	done

%.o: %.c
	$(CC) $(CFLAGS) -I. -c $< -o $@

%.o: %.m
	$(CC) $(OBJCFLAGS) -I. -c $< -o $@

tests/%.o: tests/%.c
	$(CC) $(CFLAGS) -I. -c $< -o $@

# Vendored from Iris. Keep the main project strict without rewriting this small
# terminal editor for conversion diagnostics unrelated to H3.
linenoise.o: CFLAGS += -Wno-conversion -Wno-variadic-macro-arguments-omitted

-include $(wildcard *.d tests/*.d)

clean:
	rm -f h3 h3_tests h3_metal_tests h3_bf16_tests h3_tokenizer_tests \
		h3_text_tests h3_real_prompt_test h3_real_dit_block_test \
		h3_real_optimized_qwen_test \
		h3_real_optimized_audio_vae_test h3_real_optimized_video_vae_test \
		h3_real_optimized_dit_stream_test \
		h3_audio_gpu_tests h3_quantized_weight_tests \
		h3_real_optimized_int8_test \
		h3_real_audio_vae_test h3_real_audio_encoder_test \
		h3_av_mux_test \
		h3_real_video_encoder_test h3_real_qwen_vision_test \
		h3_real_multimodal_text_test h3_real_ref_video_text_test \
		h3_real_dit_schedule_test h3_real_dit_test h3_semantic_dit_test \
		h3_real_video_vae_test h3_semantic_vae_test \
	h3_dit_bench h3_dit_bench_864 h3_int8_video_vae_test \
	libh3.a *.o *.d tests/*.o tests/*.d
