#!/usr/bin/env bash
# Evaluate KnowGen results JSON with GPT-4.1 (image vs GT scoring).
# Usage: edit RESULTS_JSON, API_KEY, paths below, then run.
#
# -----------------------------------------------------------------------------
# Input JSON (--results): a JSON array. Each element is one sample.
#
#   id            Required. Sample id (number or string); used for matching and resume.
#   prompt        Required. KnowGen task prompt (the image-generation instruction text).
#                 Must be non-empty; the evaluator sends this to the judge model.
#   meta:          Recommended. Passed through to output; summary uses meta.category.
#     -category    Used for grouped averages (must match CATEGORY_MAP in gpt_eval_knowgen.py,
#                 e.g. Biology -> science_and_knowledge; see script for full list).
#     -difficulty  Optional tag (e.g. easy, multiple subject); informational.
#   gt_image      Required. Ground-truth reference image path (KnowGen GT).
#                 Relative paths are resolved from the directory containing the results JSON file.
#   output_path   Required. Your generated image path (same relative/absolute rules as gt_image).
#   success       Required to enter the eval queue. Must be true, otherwise the row is skipped
#                 (same convention as run_gen_image_from_results.sh output).
#
# If you only have KnowGen-style fields (id, prompt, meta, gt_image), add:
#   "output_path": "<path-to-your-generated-image>",
#   "success": true
#
# Example (paths relative to the folder where results.json lives):
#   [
#     {
#       "id": 3260,
#       "success": true,
#       "prompt": "Full KnowGen prompt text describing what the image must show.",
#       "meta": { "category": "Biology", "difficulty": "easy" },
#       "output_path": "./gen_image/output_3260.png",
#       "gt_image": "./gen_image/answer_3260.png"
#     }
#   ]
# -----------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Input JSON 
RESULTS_JSON="eval_output/qwen_image_test_gen/results.json"
# Output Evaluation Resutls
OUTPUT_JSON="eval_output/qwen_image_test_gen/results_eval.json"

API_KEY=""
API_BASE="https://api.openai.com/v1"
MODEL="gpt-4.1"
MAX_WORKERS=10

if [[ -z "$API_KEY" ]]; then
  echo "Please set API_KEY inside this script"
  exit 1
fi

EXTRA=()
[[ -n "$OUTPUT_JSON" ]] && EXTRA+=(--output-json "$OUTPUT_JSON")

python3 "${SCRIPT_DIR}/gpt_eval_knowgen.py" \
  --results "$RESULTS_JSON" \
  --api-key "$API_KEY" \
  --api-base "$API_BASE" \
  --model "$MODEL" \
  --max-workers "$MAX_WORKERS" \
  --resume \
  "${EXTRA[@]}"
