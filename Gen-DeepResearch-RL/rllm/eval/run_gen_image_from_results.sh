#!/usr/bin/env bash
# Generate images from a result JSON (with gen_prompt/reference_images) or a prompt-only JSON.
# Supports API backends (nano/seed/gpt with retries) and local diffusers (qwen gen+edit, zimage, zimage_turbo, etc.).
# Output layout: OUTPUT_DIR / {suffix} / images/ and results.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ----------------- I/O (edit here) -----------------
# Paths are relative to this script directory (rllm/eval)
INPUT_JSON="eval_output/results.json"
OUTPUT_DIR="./output"
SUFFIX="qwen_image_test_gen"

# ----------------- Backend (edit here) -----------------
BACKEND="diffuser_qwen"                                   # api | diffuser_qwen | diffuser_longcat | diffuser_zimage | diffuser_zimage_turbo | diffuser_flux | diffuser_lumina2 | diffuser_sd3 | hunyuan_image3

# If BACKEND is api, then the following variables are used:
API_TYPE="nano"                                   # nano | seed | gpt
API_KEY=""
# e.g., "gemini-3-pro-image-preview" "doubao-seedream-4-5-251128"
MODEL_NAME="gemini-3-pro-image-preview"
TIMEOUT=1600
MAX_TRY=50
PARALLEL=10                                     # API-mode parallelism

# If BACKEND is NOT api, then the following variables are used:

# Shared local-diffuser config (used by all diffuser_* backends; selected by backend)
# gen: text-to-image model; edit: image-editing model (only for backends that need both, e.g. qwen)
# If path is empty, Python falls back to the default HuggingFace id
DIFFUSER_GEN_MODEL_PATH="Qwen/Qwen-Image"
DIFFUSER_EDIT_MODEL_PATH="Qwen/Qwen-Image-Edit-2509"
DIFFUSER_GEN_DEVICE="cuda:0"
DIFFUSER_EDIT_DEVICE="cuda:1,cuda:2,cuda:3" # use more GPUs for Qwen-Image-Edit for acceleration

# ----------------- Run -----------------
mkdir -p "${OUTPUT_DIR}"

python3 "${SCRIPT_DIR}/gen_image_from_results.py" \
  --input "${INPUT_JSON}" \
  --output-dir "${OUTPUT_DIR}" \
  --suffix "${SUFFIX}" \
  --backend "${BACKEND}" \
  --api-type "${API_TYPE}" \
  --api-key "${API_KEY}" \
  --model-name "${MODEL_NAME}" \
  --timeout "${TIMEOUT}" \
  --max-try "${MAX_TRY}" \
  --parallel "${PARALLEL}" \
  --diffuser-gen-model-path "${DIFFUSER_GEN_MODEL_PATH}" \
  --diffuser-edit-model-path "${DIFFUSER_EDIT_MODEL_PATH}" \
  --diffuser-gen-device "${DIFFUSER_GEN_DEVICE}" \
  --diffuser-edit-device "${DIFFUSER_EDIT_DEVICE}" \
  --resume \
  --print-log
