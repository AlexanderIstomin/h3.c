#!/bin/sh
# A prompt to a playable clip, in one command.
#
# Everything LTX-2.5 needs already exists as separate binaries, each written to
# check one stage against the released weights. This chains them, so an
# arbitrary prompt can be generated and *watched* rather than only the two
# whose intermediate files happen to be lying around. That limitation is what
# made the rope's time-axis bug survive a day: the seam is only checkable when
# running the whole thing is one command.
#
# It is deliberately a script and not a C module. The order and the file
# formats are what an in-process pipeline has to reproduce, and writing them
# down first is cheaper than discovering them inside a refactor. The two-phase
# split is real and not scaffolding: the Gemma tower and the DiT do not fit in
# memory together, so the tower runs, writes, and exits before the DiT opens.
#
# usage: LTX=/path/to/snapshot tests/ltx_clip.sh "a prompt" OUT [STEPS] [SEED]

set -e

PROMPT="$1"
OUT="${2:-clip}"
STEPS="${3:-8}"
SEED="${4:-16}"
: "${LTX:?set LTX to the model snapshot directory}"
: "${SCRATCH:=$(dirname "$OUT")}"
GEN="${GEN:-./h3_ltx_generate_long}"

DIT="$LTX/diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
ENCODER="$LTX/text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"
VIDEO_VAE="$LTX/vae/ltx-2.5-video-vae-conv-bf16.safetensors"
AUDIO_VAE="$LTX/vae/ltx-2.5-audio-vae-bf16.safetensors"

echo "== 1/6 tokenize =="
# From $SCRATCH: the tokenizer is loaded by the relative name `gemma_tok`, and
# from anywhere else transformers silently falls through to the network and
# fails there instead of saying it could not find the local copy.
( cd "$SCRATCH" && uv run --python 3.12 --with transformers --with torch \
    --with safetensors --no-project python tokenize_prompt.py "$PROMPT" \
    "$(cd "$(dirname "$OUT")" && pwd)/$(basename "$OUT").ids.safetensors" )

echo "== 2/6 Gemma tower =="
./h3_real_ltx_text_test "$ENCODER" "$OUT.ids.safetensors" "$OUT.states.bin"

echo "== 3/6 connector =="
./h3_real_ltx_connector_test "$DIT" "$OUT.states.bin" "$OUT.context.bin"

echo "== 4/6 denoise =="
"$GEN" "$DIT" "$OUT.context.bin" "$OUT.latent.bin" "$STEPS" "$SEED"

echo "== 5/6 decode both streams =="
./h3_ltx_decode "$VIDEO_VAE" "$OUT.latent.bin" "$OUT.frames.bin"
./h3_ltx_audio_decode "$AUDIO_VAE" "$OUT.latent.bin" "$OUT.wav"

echo "== 6/6 mux =="
uv run --python 3.12 --with numpy --no-project \
    python "$SCRATCH/mux_clip.py" "$OUT.frames.bin" "$OUT.wav" "$OUT.mp4" "${FPS:-24}"

echo "wrote $OUT.mp4"
