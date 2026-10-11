# MoCo-Rubric

## Code layout

```text
moco/          momentum plugin, rubric rewards, and caption inference
scripts/       training, inference, and service launchers
data/          data preparation scripts
prompts/       training and evaluation prompts
eval/          pairwise comparison, ranking, and caption-based QA
```

## Installation

Use Python 3.11 and CUDA 12.8.

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

Download the [training data](https://huggingface.co/datasets/a23wen/MoCo-Rubric)
and run the preparation scripts from the repository root:

```bash
hf download a23wen/MoCo-Rubric \
  sft.jsonl rl.jsonl images/pixmo-cap.tar images/densefusion.tar \
  --repo-type dataset --local-dir data

cd data
tar -xf images/pixmo-cap.tar
tar -xf images/densefusion.tar
cd ..

python data/prepare_sft.py
python data/prepare_rl.py
```

After extraction:

```text
data/
├── sft.jsonl
├── rl.jsonl
└── images/
    ├── pixmo-cap/
    └── densefusion/
```

The scripts write training inputs to `outputs/data/` with local image paths.
Run them again if you move the repository. See the [data guide](data/README.md)
for input formats and custom paths.

## SFT

Train `Qwen/Qwen3-VL-8B-Instruct` with the shared SFT data:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/train_sft.sh
```

Checkpoints are saved under `outputs/sft/`.

## GRPO

Run the Policy rollout service, Generator/Judge service, and trainer on separate
GPU pools. Set these variables on each node, using the same SFT checkpoint path:

```bash
export SFT_MODEL=/absolute/path/to/sft/checkpoint
export ROLLOUT_SERVER_HOST=policy-node
export MOMENTUM_SERVER_HOST=momentum-node
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

On the Policy rollout node:

```bash
bash scripts/start_rollout.sh
```

On the Generator/Judge node:

```bash
bash scripts/start_momentum.sh
```

On the training node, after both services are ready:

```bash
bash scripts/train_grpo.sh
```

Allow connections between the nodes on service ports 8000/8001 and
weight-transfer ports 51216/51217. Checkpoints are saved under `outputs/grpo/`.

To resume, restart both services from `SFT_MODEL` and set
`MOMENTUM_RESUME_CHECKPOINT` to the saved GRPO checkpoint before launching the trainer.

### Configuration

| Variable | Purpose |
|---|---|
| `BASE_MODEL` | SFT model; default `Qwen/Qwen3-VL-8B-Instruct` |
| `SFT_MODEL` | SFT checkpoint for GRPO and both services |
| `SFT_DATA`, `RL_DATA` | Training input paths |
| `OUTPUT_DIR` | Training output directory |
| `MOMENTUM_DECAY` | EMA coefficient; default `0.99` |
| `MOMENTUM_UPDATE_INTERVAL` | EMA update interval; default `1` |
| `MOMENTUM_RESUME_CHECKPOINT` | GRPO checkpoint to resume |

Shared settings are in `scripts/common.sh`. Training and service launchers
accept additional ms-swift arguments.

## Caption inference

```bash
bash scripts/caption.sh \
  --model /absolute/path/to/grpo/checkpoint \
  --input data/benchmarks/pixmo-cap.jsonl \
  --output outputs/captions/pixmo-cap.jsonl \
  --gpus 0,1,2,3,4,5,6,7
```

Each input row contains `image` and an optional `image_id`. Image paths are
relative to `data/`; the output adds a `response` field with the caption.

## Evaluation

| Script | Purpose |
|---|---|
| `eval/pairwise.py` | Compare two sets of captions |
| `eval/ranking.py` | Rank multiple sets of captions |
| `eval/qa/run.py` | Answer and score questions using captions |

See the [evaluation guide](eval/README.md) for commands and API settings.
