#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
export NPROC_PER_NODE="${NPROC_PER_NODE:-${#GPU_IDS[@]}}"
: "${SFT_MODEL:?Set SFT_MODEL to the shared SFT checkpoint-300 directory}"
DATASET="${RL_DATA:-${PROJECT_ROOT}/outputs/data/rl.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/grpo}"
export MOMENTUM_PROCESS_ROLE=trainer
export MOMENTUM_AUDIT="${MOMENTUM_AUDIT:-${OUTPUT_DIR}/rollout_audit.jsonl}"
export SWIFT_UPDATE_WEIGHTS_BUCKET_SIZE=512
[[ -d "$SFT_MODEL" && -f "$DATASET" ]]
[[ "$ROLLOUT_GROUP_PORT" != "$MOMENTUM_GROUP_PORT" ]]
if [[ -n "${MOMENTUM_RESUME_CHECKPOINT:-}" ]]; then
  [[ -f "${MOMENTUM_RESUME_CHECKPOINT}/momentum/momentum_state.json" ]]
  set -- --resume_from_checkpoint "$MOMENTUM_RESUME_CHECKPOINT" "$@"
fi
python scripts/check_servers.py \
  --rollout-url "http://${ROLLOUT_SERVER_HOST}:${ROLLOUT_SERVER_PORT}" \
  --momentum-url "http://${MOMENTUM_SERVER_HOST}:${MOMENTUM_SERVER_PORT}" \
  --decay "$MOMENTUM_DECAY" --update-interval "$MOMENTUM_UPDATE_INTERVAL" \
  --expect-step -1 --expect-model "$SFT_MODEL"
exec swift rlhf \
  --model "$SFT_MODEL" --model_type qwen3_vl --rlhf_type grpo \
  --external_plugins "${PROJECT_ROOT}/moco/momentum.py" \
  --reward_funcs moco_reward --multi_turn_scheduler moco_caption \
  --tuner_type full --torch_dtype bfloat16 --dataset "$DATASET" \
  --load_from_cache_file false --max_completion_length 1024 \
  --num_train_epochs 1 --max_length 6144 --max_pixels 262144 \
  --gradient_checkpointing true --per_device_train_batch_size 8 \
  --learning_rate 1e-6 --gradient_accumulation_steps 2 \
  --lr_scheduler_type cosine --warmup_ratio 0.05 \
  --save_strategy steps --eval_strategy no --save_steps 50 --save_total_limit 3 \
  --logging_steps 5 --output_dir "$OUTPUT_DIR" \
  --dataloader_num_workers 16 --dataset_num_proc 8 --no_remove_unused_columns \
  --freeze_vit false --freeze_llm false --freeze_aligner false \
  --use_vllm true --vllm_mode server \
  --vllm_server_host "$ROLLOUT_SERVER_HOST" --vllm_server_port "$ROLLOUT_SERVER_PORT" \
  --vllm_server_group_port "$ROLLOUT_GROUP_PORT" --vllm_server_timeout 600 \
  --vllm_server_pass_dataset true --async_generate false \
  --vllm_limit_mm_per_prompt '{"image": 1, "video": 0}' \
  --num_generations 16 --temperature 0.9 --deepspeed zero3 \
  --log_completions false --beta 0.01 "$@"
