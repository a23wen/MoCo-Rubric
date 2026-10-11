# MoCo-Rubric

## Repository structure

```text
moco/          momentum plugin, rubric rewards, Judge validation, caption inference
scripts/       training launchers, rollout services, and runtime checks
data/          data preparation scripts and local training inputs
prompts/       caption, rubric, Judge, and evaluation prompts
eval/          pairwise comparison, blind ranking, and caption-only QA
```

| Script | Purpose |
|---|---|
| `scripts/train_sft.sh` | Run shared-model SFT |
| `scripts/start_rollout.sh` | Start the Policy rollout service |
| `scripts/start_momentum.sh` | Start the shared Generator/Judge service |
| `scripts/train_grpo.sh` | Run GRPO training |
| `scripts/caption.sh` | Generate image captions |
| `scripts/check_env.py` | Check dependency versions |
| `scripts/check_servers.py` | Check service health, models, and EMA settings |
| `scripts/common.sh` | Shared paths, GPU configuration, ports, and EMA defaults |

## Installation

Use Python 3.11 and CUDA 12.8. Dependencies are pinned in `requirements.txt`,
including ms-swift `4.2.0.dev0`, vLLM `0.11.0`, and PyTorch `2.8.0`.

```bash
git clone https://github.com/a23wen/MoCo-Rubric.git
cd MoCo-Rubric

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install flash-attn==2.8.3 --no-build-isolation
python scripts/check_env.py
```

## Data preparation

Place training manifests and images under `data/`:

```text
data/
├── sft.jsonl
├── rl.jsonl
└── images/
    ├── pixmo-cap/
    └── densefusion/
```

Run from the repository root:

```bash
python data/prepare_sft.py
python data/prepare_rl.py
```

The scripts validate the records and resolve image paths, producing
`outputs/data/sft.jsonl` and `outputs/data/rl.jsonl`. These are the training
launchers' default inputs. Rerun preparation when the checkout location changes.
See [data/README.md](data/README.md) for archive extraction, input fields, and
`build_sft.py` usage.

## SFT

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/train_sft.sh
```

The default model is `Qwen/Qwen3-VL-8B-Instruct`. The launcher uses full-parameter
training, one epoch, learning rate `1e-5`, per-device batch size 4, gradient
accumulation 2, cosine decay, and 5% warmup. Data shuffling is disabled.
Checkpoints are written under `outputs/sft/`.

## GRPO

Run the rollout service, momentum service, and trainer on separate GPU pools.
Set the checkpoint path and service hosts on each participating node:

```bash
export SFT_MODEL=/absolute/path/to/sft/checkpoint
export ROLLOUT_SERVER_HOST=policy-node
export MOMENTUM_SERVER_HOST=momentum-node
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

Use the same checkpoint path across the nodes. Start the Policy rollout service:

```bash
bash scripts/start_rollout.sh
```

Start the Generator/Judge service on its assigned node:

```bash
bash scripts/start_momentum.sh
```

Once both services are ready, start the trainer:

```bash
bash scripts/train_grpo.sh
```

The trainer checks both services before starting. Service ports default to
8000/8001, with weight-transfer ports 51216/51217. Configure network access
between the participating nodes for these ports.

Training defaults are one epoch, learning rate `1e-6`, per-device batch size 8,
gradient accumulation 2, and 16 generations per image. Checkpoints are saved
under `outputs/grpo/`, including the `momentum/` state.

To resume, restart the services from `SFT_MODEL`, then launch the trainer with
`MOMENTUM_RESUME_CHECKPOINT` pointing to a saved GRPO checkpoint.

## Configuration

| Variable | Purpose or default |
|---|---|
| `BASE_MODEL` | SFT model; defaults to `Qwen/Qwen3-VL-8B-Instruct` |
| `SFT_MODEL` | Checkpoint used to initialize GRPO and both services |
| `SFT_DATA`, `RL_DATA` | Override the prepared training inputs |
| `OUTPUT_DIR` | Override the training output directory |
| `CUDA_VISIBLE_DEVICES` | GPU pool for the current process |
| `MOMENTUM_DECAY` | EMA coefficient; default `0.99` |
| `MOMENTUM_UPDATE_INTERVAL` | EMA update interval; default `1` |
| `MOMENTUM_RESUME_CHECKPOINT` | Saved GRPO checkpoint for resuming training |

Training and service launchers forward additional command-line arguments to
ms-swift. Port defaults are defined in `scripts/common.sh`.

## Caption inference

```bash
bash scripts/caption.sh \
  --model /absolute/path/to/grpo/checkpoint \
  --input data/benchmarks/pixmo-cap.jsonl \
  --output outputs/captions/pixmo-cap.jsonl \
  --gpus 0,1,2,3,4,5,6,7
```

Input rows contain `image` and optionally `image_id` and reference metadata.
Image paths are relative to `data/`. Output rows retain the input fields and
add `response`. Generation uses temperature 0, top-p 1, and a 1,024-token output
limit. Add `--prepare-only` to prepare the inference inputs.

## Evaluation

| Script | Purpose |
|---|---|
| `eval/pairwise.py` | Build caption pairs, run judging, and summarize wins/losses/ties |
| `eval/ranking.py` | Build anonymous caption groups, run ranking, and aggregate scores |
| `eval/qa/run.py` | Prepare caption-based QA inputs, generate answers, and score them |
| `eval/qa/download_metrics.py` | Download the scoring code listed in `eval/qa/sources.json` |

See [eval/README.md](eval/README.md) for commands and API configuration.
