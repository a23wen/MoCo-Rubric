#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
: "${SFT_MODEL:?Set SFT_MODEL to the shared SFT checkpoint-300 directory}"
[[ -d "$SFT_MODEL" ]]
export MOMENTUM_PROCESS_ROLE=momentum
unset NPROC_PER_NODE
exec swift rollout \
  --model "$SFT_MODEL" --model_type qwen3_vl --enable_thinking false \
  --external_plugins "${PROJECT_ROOT}/moco/momentum.py" \
  --vllm_use_async_engine true --torch_dtype bfloat16 --attn_impl flash_attn \
  --infer_backend vllm --host "${HOST:-0.0.0.0}" \
  --port "${MOMENTUM_SERVER_PORT}" \
  --vllm_tensor_parallel_size 1 --vllm_data_parallel_size "${#GPU_IDS[@]}" \
  --vllm_gpu_memory_utilization 0.9 --vllm_max_model_len 12288 \
  --vllm_max_num_seqs 32 --max_pixels 262144 \
  --vllm_limit_mm_per_prompt '{"image": 1, "video": 0}' \
  --vllm_mm_processor_cache_gb 0 "$@"
