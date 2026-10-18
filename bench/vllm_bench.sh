#!/usr/bin/env bash
# vLLM serving benchmark on MI300X (PRD §11). Run ON the droplet (inside the vLLM container or a venv with vllm).
#   VLLM_API_KEY=... ./bench/vllm_bench.sh
# Writes bench/vllm_c1.json and bench/vllm_c32.json; then: python bench/collect_results.py
set -euo pipefail
MODEL="${VLLM_MODEL:-google/gemma-4-26B-A4B-it}"
BASE="${VLLM_BENCH_BASE:-http://localhost:8000}"
export OPENAI_API_KEY="${VLLM_API_KEY:?set VLLM_API_KEY}"   # vllm bench reads the key from OPENAI_API_KEY
OUT="$(cd "$(dirname "$0")" && pwd)"
for C in 1 32; do
  vllm bench serve --backend openai-chat --endpoint /v1/chat/completions \
    --model "$MODEL" --base-url "$BASE" \
    --dataset-name random --random-input-len 2000 --random-output-len 400 \
    --num-prompts 32 --max-concurrency "$C" \
    --save-result --result-dir "$OUT" --result-filename "vllm_c${C}.json"
done
echo "done -> $OUT/vllm_c1.json, $OUT/vllm_c32.json"
