#!/usr/bin/env bash
# Gen Image eval: only produce trajectory logs, gen_prompt, and reference_images; no image generation or scoring
#
# Input JSON format (array or JSONL, one object per line):
#   {"id": 7012, "prompt": "...", "meta": {"category": "...", ...}, "gen_image": "/path/to/gt.png"}
# gen_image is the ground-truth path in the legacy format; current logic does not use it.
#
# Environment variables (optional):
#   GEN_EVAL_INPUT_JSON   input JSON path
#   GEN_EVAL_OUTPUT_DIR   output directory
#   OPENAI_BASE_URL       inference service base_url
#   GEN_EVAL_MODEL        model name
#   OPENAI_API_KEY        API Key
set -euo pipefail

# Ensure we run under the rllm directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RLLM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${RLLM_DIR}"

# Load Gen Image environment variables (required by search/browse/image tools)
if [ -f .env.gen_image ]; then
  echo "[GenEval] Loading .env.gen_image"
  set -a
  source .env.gen_image
  set +a
else
  echo "[GenEval] WARNING: .env.gen_image not found; some tools may be unavailable"
fi

INPUT_JSON="KnowGen-Bench.json"
OUTPUT_DIR="./eval_output"
SUFFIX="test"
OUTPUT_DIR="${OUTPUT_DIR}/${SUFFIX}"

# vLLM OpenAI-compatible API base URL for Gen-Searcher-8B (replace host/IP after deployment)
BASE_URL="http://xxx:8001/v1"
MODEL="Gen-Searcher-8B"

mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/gen_eval_$(date +%Y%m%d_%H%M%S).log"

python3 -m eval.gen_image_eval_runner \
  --input "${INPUT_JSON}" \
  --output-dir "${OUTPUT_DIR}" \
  --base-url "${BASE_URL}" \
  --model "${MODEL}" \
  --max-prompt-length 64000 \
  --max-response-length 64000 \
  --parallel-tasks 5 \
  --temperature 0.6 \
  --top-p 0.9 \
  --resume \
  "$@" 2>&1 | tee "${LOG_FILE}"

