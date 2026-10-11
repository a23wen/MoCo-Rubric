# Data preparation

Follow the [download instructions](../README.md#data-preparation) to place the
training data under `data/`. Image paths in the JSONL files are relative to this
directory, for example `images/pixmo-cap/000001.jpg`.

## Scripts

| Script | Purpose |
|---|---|
| `prepare_sft.py` | Prepare SFT inputs at `outputs/data/sft.jsonl` |
| `prepare_rl.py` | Prepare GRPO inputs at `outputs/data/rl.jsonl` |
| `build_sft.py` | Mix aligned Policy, Generator, and Judge examples for SFT |

Run scripts from the repository root. To use custom paths:

```bash
python data/prepare_sft.py \
  --input data/sft.jsonl \
  --data-root data \
  --output outputs/data/sft.jsonl
```

`prepare_rl.py` accepts the same options. For SFT mixing options, run
`python data/build_sft.py --help`.

## Training formats

### SFT

- `messages`: conversation ending with an assistant training target.
- `images`: image paths for Policy and Generator examples; Judge examples are text-only.

### RL

| Field | Contents |
|---|---|
| `sample_uid` | Unique identifier, such as `pixmo-cap:000001` |
| `sample_id`, `source_dataset` | Image identifier and source dataset |
| `messages` | Conversation ending with the user caption prompt |
| `images` | One image path |
| `visual_info` | Visual annotations |
| `gpt_caption`, `gemini_caption` | Reference captions |
| `reference_caption` | Combined reference text |

## Evaluation inputs

Place caption benchmark files under `data/benchmarks/`. Each JSONL row contains
`image` and an optional `image_id`. Generated caption files also contain `response`.

Place QA files under `data/benchmarks/qa/`, one JSONL file per benchmark. Each
record has `sample_id` and `bench`, plus the following fields:

| Benchmark | Fields |
|---|---|
| CaptionQA | `image_id`, `question`, `choices`, `answer`, `permutation`, `domain` |
| BLINK | Ordered `image_ids`, `question`, `options`, `official_prompt`, `references`, `category` |
| TextVQA / DocVQA / ChartQA | `image_ids`, `question`, `references`, `category` |

CaptionQA uses the correct option text in `answer`; BLINK uses the correct
letter in `references`. Caption files must provide matching `image_id` values
and a `response` for each image.
