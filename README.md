# MoCo-Rubric

Implementation of **Momentum-Coupled Rubric Adaptation for Detailed Image Captioning**.

MoCo-Rubric learns detailed image captions using image-grounded rubrics that adapt
with the caption policy. A shared model is first trained for three roles through
multi-role supervised fine-tuning (SFT). During reinforcement learning, the
Caption Policy is optimized with GRPO, while a single momentum model serves both
the Rubric Generator and Rubric Judge and tracks the Policy through exponential
moving average (EMA) updates. Inference uses the Caption Policy alone.

| Role | Inputs | Output | Updates during GRPO |
|---|---|---|---|
| Caption Policy | Image and caption prompt | Detailed caption | GRPO gradients |
| Rubric Generator | Image, two reference captions, four sampled Policy captions | Image-grounded rubric | EMA from the Policy |
| Rubric Judge | Candidate caption and rubric; no image | Binary decisions and reasons | Same shared EMA model |

The main configuration generates 16 Policy captions per image, uses four of them
for rubric construction, and updates the momentum model with `m=0.99` every
`H=1` optimizer step.

## Release status

This repository contains the method implementation, prompts, training launchers,
data preparation tools, and evaluation utilities. Training examples, images,
checkpoints, and runtime outputs are excluded from Git. Baseline training,
ablations, and historical experiment results are not included.

**Training data is prepared for a separate Hugging Face release, but is not yet
published.** The download link will be added here when available. The training
commands below require that data and the appropriate model weights.

## Repository structure

```text
moco/          EMA integration, rubric rewards, Judge validation, caption inference
scripts/       SFT/GRPO launchers, rollout services, environment and service checks
data/          data preparation tools and the local training-data directory
prompts/       fixed caption, rubric, Judge, and evaluation prompts
eval/          pairwise comparison, blind ranking, and caption-only QA
```

The two detailed guides describe the [data formats](data/README.md) and
[evaluation protocols](eval/README.md).

## Installation

The recorded training environment uses Python 3.11, CUDA 12.8, PyTorch 2.8.0,
ms-swift **4.2.0.dev0**, vLLM **0.11.0**, Transformers **4.57.1**, and
FlashAttention **2.8.3**. The momentum plugin integrates with specific
ms-swift/vLLM interfaces; use the versions in `requirements.txt`.

```bash
git clone https://github.com/a23wen/MoCo-Rubric.git
cd MoCo-Rubric

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install flash-attn==2.8.3 --no-build-isolation
python scripts/check_env.py
```

Install into a separate environment. `check_env.py` checks package versions; it
does not launch a model or establish GPU-training compatibility by itself.

## Prepare training data

The planned data release contains:

| Component | Records or image files |
|---|---:|
| SFT Caption Policy | 10,000 |
| SFT Rubric Generator | 5,000 |
| SFT Rubric Judge | 5,000 |
| **SFT total** | **20,000** |
| RL PixMo-Cap | 2,500 |
| RL DenseFusion | 2,500 |
| **RL total** | **5,000** |
| PixMo-Cap images | 7,500 |
| DenseFusion images | 7,500 |

Place the downloaded files under the checkout's `data/` directory. The archives
contain paths beginning with `images/pixmo-cap/` and `images/densefusion/`.

```text
data/
├── sft.jsonl
├── rl.jsonl
├── SHA256SUMS
└── images/
    ├── pixmo-cap.tar
    ├── densefusion.tar
    ├── pixmo-cap/       # after extraction
    └── densefusion/     # after extraction
```

Verify and extract the archives from inside `data/`, then prepare local training
inputs from the repository root:

```bash
cd data
sha256sum -c SHA256SUMS
tar -xf images/pixmo-cap.tar
tar -xf images/densefusion.tar
cd ..

python data/prepare_sft.py
python data/prepare_rl.py
```

The preparation scripts validate the records and resolve relative image paths.
They write `outputs/data/sft.jsonl` and `outputs/data/rl.jsonl`, which are the
training launchers' default inputs. Source manifests and images remain unchanged.
Rerun preparation after moving the checkout. Keep approximately 12 GB available
for the manifests, archives, and extracted images.

**Reproduction note.** The original RL run used 4,999 records. The prepared
release restores `pixmo-cap:001577` from the original image selection, reusing
its GPT reference and visual information and adding a
`gemini-3.1-pro-preview` reference. Existing paper checkpoints were not retrained
for this release. The added row records its provenance. DenseFusion image
`011586` uses only the repaired RGB PNG. See the
[data release notes](data/README.md#training) for details.

`data/build_sft.py` is optional: it reconstructs the deterministic 2:1:1 mixture
from four aligned role files. Users of the mixed data release can use
`prepare_sft.py` directly.

## Shared SFT

The default backbone is `Qwen/Qwen3-VL-8B-Instruct`. Set `BASE_MODEL` to use an
existing local copy. Run SFT on the training node:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/train_sft.sh
```

The launcher trains all parameters for one epoch, with learning rate `1e-5`,
per-device batch size 4, gradient accumulation 2, cosine decay, and 5% warmup.
With eight training GPUs, the global batch size is 64. Both dataset and
training-dataloader shuffling are disabled to preserve the mixed-role grouping.

The original run selected **checkpoint-300**. Keep the epoch-based schedule;
setting `max_steps=300` would change the cosine horizon. ms-swift may create a
version subdirectory under `outputs/sft/`; use the actual checkpoint directory
for `SFT_MODEL` in the next step.

## GRPO with momentum adaptation

GRPO uses three processes: a Policy rollout server, a shared momentum server
for Generator/Judge, and a trainer. Assign separate GPU pools to the three
processes. The original configuration uses eight GPUs per process; changing
GPU count changes the effective training setup. The examples below assume
separate nodes with the checkout, data, and checkpoint available at consistent
paths.

Set these variables on each participating node, replacing the example model
path and hostnames:

```bash
export SFT_MODEL=/absolute/path/to/shared-sft/checkpoint-300
export ROLLOUT_SERVER_HOST=policy-node
export MOMENTUM_SERVER_HOST=momentum-node
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

On the Policy rollout node:

```bash
bash scripts/start_rollout.sh
```

On the momentum node:

```bash
bash scripts/start_momentum.sh
```

After both services are ready, on the trainer node:

```bash
bash scripts/train_grpo.sh
```

The trainer first checks service health, roles, initial model, and EMA settings.
The default service ports are 8000/8001; weight-transfer group ports are
51216/51217. These ports must be reachable between the participating nodes.

The main training settings are learning rate `1e-6`, per-device batch size 8,
gradient accumulation 2, one epoch, and 16 generations per image. Only the
Policy receives GRPO gradients. The EMA master is stored in FP32 on CPU and
distributed in BF16 for inference; Generator and Judge share the same momentum
model version within each rollout group.

The original run evaluated **checkpoint-600**. The launcher preserves the
epoch-based cosine schedule and saves under `outputs/grpo/`. Select
checkpoint-600 even if the epoch continues. Checkpoints include a `momentum/`
snapshot. To resume, restart both services from `SFT_MODEL` and set
`MOMENTUM_RESUME_CHECKPOINT` to the complete saved GRPO checkpoint when launching
the trainer.

### Launcher configuration

| Variable | Purpose or default |
|---|---|
| `BASE_MODEL` | SFT backbone; defaults to `Qwen/Qwen3-VL-8B-Instruct` |
| `SFT_MODEL` | Shared SFT checkpoint; required for GRPO and both services |
| `SFT_DATA`, `RL_DATA` | Override the prepared training-manifest paths |
| `OUTPUT_DIR` | Override the selected training launcher's output directory |
| `CUDA_VISIBLE_DEVICES` | GPU pool for the current process |
| `MOMENTUM_DECAY` | EMA coefficient; default `0.99` |
| `MOMENTUM_UPDATE_INTERVAL` | EMA update interval; default `1` |
| `MOMENTUM_RESUME_CHECKPOINT` | GRPO checkpoint with its momentum snapshot |

Launchers also forward additional command-line arguments to ms-swift.

## Generate captions

Run inference with the trained Caption Policy:

```bash
bash scripts/caption.sh \
  --model /absolute/path/to/grpo/checkpoint-600 \
  --input data/benchmarks/pixmo-cap.jsonl \
  --output outputs/captions/pixmo-cap.jsonl \
  --gpus 0,1,2,3,4,5,6,7
```

The input manifest provides `image` and optionally `image_id` and reference
metadata. Image paths are relative to `data/`; the output preserves the input
fields and adds `response`. Generation uses temperature 0, top-p 1, and a
1,024-token output limit. `--prepare-only` validates and prepares inputs without
launching inference. Generator and Judge are not needed for this step.

## Evaluation

The [evaluation guide](eval/README.md) provides commands and input schemas for:

- **Pairwise comparison:** image-conditioned judging and `W/(W+L+T)` summaries,
  excluding invalid judgments.
- **Blind ranking:** anonymous caption ordering, response validation, and
  aggregation under the selected permutation protocol.
- **Caption-only QA:** CaptionQA, BLINK, TextVQA, DocVQA, and ChartQA, using a
  fixed text-only Qwen3-8B-Instruct answerer. It sees captions and questions,
  without images.

Evaluation datasets and comparison captions are supplied separately. API-based
judging takes credentials and endpoints at execution time. Training and local
caption inference do not require an external judge API.
