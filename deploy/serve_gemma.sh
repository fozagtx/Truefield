#!/usr/bin/env bash
# Serve Gemma 4 26B-A4B-it with vLLM on an AMD Instinct MI300X (ROCm) droplet (PRD §11).
# Usage: HF_TOKEN=hf_xxx ./deploy/serve_gemma.sh      (prints the generated API key; paste it into Evolus "Your models")
set -euo pipefail
: "${HF_TOKEN:?set HF_TOKEN (accept the Gemma licence on Hugging Face first)}"
export VLLM_API_KEY="${VLLM_API_KEY:-$(openssl rand -hex 24)}"
MODEL="${VLLM_MODEL:-google/gemma-4-26B-A4B-it}"   # fallbacks: google/gemma-4-E4B-it (fastest), google/gemma-4-31B-it
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai-rocm:gemma4}" # if the tag is missing: vllm/vllm-openai-rocm:latest
EXTRA="${VLLM_EXTRA_ARGS:-}"                        # e.g. "--attention-backend TRITON_ATTN" if attention errors appear on ROCm
docker rm -f gemma4-rocm >/dev/null 2>&1 || true
docker run -d --name gemma4-rocm \
  --ipc=host --network=host --privileged \
  --cap-add=CAP_SYS_ADMIN --cap-add=SYS_PTRACE \
  --device=/dev/kfd --device=/dev/dri --group-add=video \
  --security-opt=seccomp=unconfined --shm-size 16G \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN="$HF_TOKEN" \
  "$IMAGE" \
    --model "$MODEL" \
    --served-model-name "$MODEL" \
    --host 0.0.0.0 --port 8000 --api-key "$VLLM_API_KEY" \
    --max-model-len 16384 --gpu-memory-utilization 0.75 \
    --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 \
    --chat-template examples/tool_chat_template_gemma4.jinja \
    --limit-mm-per-prompt image=0,audio=0 --async-scheduling $EXTRA
echo "VLLM_API_KEY=$VLLM_API_KEY   (store in .env and in Evolus model settings; never commit)"
echo "logs: docker logs -f gemma4-rocm"
echo "smoke test:"
echo "  curl http://\$DROPLET_IP:8000/v1/chat/completions -H \"Authorization: Bearer \$VLLM_API_KEY\" -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK\"}],\"max_tokens\":5}'"
