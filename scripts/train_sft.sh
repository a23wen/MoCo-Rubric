#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
export NPROC_PER_NODE="${NPROC_PER_NODE:-${#GPU_IDS[@]}}"
DATASET="${SFT_DATA:-${PROJECT_ROOT}/outputs/data/sft.jsonl}"
[[ -f "$DATASET" ]] || { echo 'Run python data/prepare_sft.py first.' >&2; exit 2; }
exec swift sft \
  --model "${BASE_MODEL:-Qwen/Qwen3-VL-8B-Instruct}" \
  --model_type qwen3_vl --tuner_type full \
  --dataset "$DATASET" --dataset_shuffle false --train_dataloader_shuffle false \
  --data_seed 42 --seed 42 --strict true \
  --torch_dtype bfloat16 --attn_impl flash_attn \
  --freeze_vit false --freeze_llm false --freeze_aligner false \
  --gradient_checkpointing true --num_train_epochs 1 \
  --per_device_train_batch_size 4 --gradient_accumulation_steps 2 \
  --learning_rate 1e-5 --lr_scheduler_type cosine --warmup_ratio 0.05 \
  --max_length 8192 --max_pixels 262144 \
  --save_strategy steps --save_steps 50 --save_total_limit 5 \
  --logging_steps 1 --eval_strategy no \
  --load_from_cache_file false --remove_unused_columns false \
  --dataloader_num_workers 8 --dataset_num_proc 8 --report_to tensorboard \
  --output_dir "${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/sft}" --deepspeed zero3 "$@"
