# Data preparation

Run the commands below from the repository root.

## Layout

```text
data/
├── sft.jsonl
├── rl.jsonl
├── SHA256SUMS
├── images/
│   ├── pixmo-cap.tar
│   ├── densefusion.tar
│   ├── pixmo-cap/
│   └── densefusion/
└── benchmarks/
    └── qa/
```

Image paths in the input manifests are relative to `data/`, for example
`images/pixmo-cap/000001.jpg`. Sample identifiers include the source dataset,
such as `pixmo-cap:000001`.

If using the image archives, verify and extract them from inside `data/`:

```bash
cd data
sha256sum -c SHA256SUMS
tar -xf images/pixmo-cap.tar
tar -xf images/densefusion.tar
cd ..
```

## Prepare training inputs

```bash
python data/prepare_sft.py
python data/prepare_rl.py
```

Both scripts use `_prepare.py` to validate records and convert relative image
paths to local absolute paths. They preserve record order and metadata and
write `outputs/data/sft.jsonl` and `outputs/data/rl.jsonl`.

Use `--input`, `--data-root`, and `--output` to specify another location:

```bash
python data/prepare_sft.py \
  --input data/sft.jsonl \
  --data-root data \
  --output outputs/data/sft.jsonl
```

`prepare_rl.py` accepts the same options. Rerun preparation after moving the
checkout so the generated paths match its new location.

### SFT fields

- `messages`: ms-swift conversation ending with an assistant training target.
- `images`: image paths for Policy and Generator records. Judge records are text-only.
- Additional role and source metadata is preserved by the preparation script.

### RL fields

| Field | Contents |
|---|---|
| `sample_uid` | Unique dataset-qualified identifier |
| `sample_id` | Identifier within the source dataset |
| `source_dataset` | Dataset name |
| `messages` | Conversation ending with the user caption prompt |
| `images` | One image path |
| `visual_info` | Structured visual annotations |
| `gpt_caption`, `gemini_caption` | Reference captions |
| `reference_caption` | Combined reference text |

The training plugin generates rubrics during GRPO.

## Build the SFT mixture

`build_sft.py` combines aligned Policy, Generator, and Judge files into grouped
SFT records. It writes the mixed dataset, a grouping manifest, and a summary.

```bash
python data/build_sft.py --help
```

Its input arguments are `--policy-10000`, `--aligned-policy-5000`,
`--generator-5000`, and `--judge-5000`. Set `--output`, `--manifest`, and
`--summary` for the generated files. The default random seed is 42.

## Evaluation inputs

Caption benchmark manifests go under `data/benchmarks/`. Each row contains
`image` and optionally `image_id` and reference metadata. `scripts/caption.sh`
adds a `response` field when generating captions. Pairwise input files are
aligned by their unique `image` identifiers.

QA manifests go under `data/benchmarks/qa/`, with one JSONL file per benchmark.
Each record contains `sample_id` and `bench`, plus:

| Benchmark | Fields |
|---|---|
| CaptionQA | `image_id`, `question`, `choices`, `answer`, `permutation`, `domain` |
| BLINK | Ordered `image_ids`, `question`, `options`, `official_prompt`, `references`, `category` |
| TextVQA / DocVQA / ChartQA | `image_ids`, `question`, `references`, `category` |

CaptionQA uses the exact option text in `answer`; BLINK uses the correct letter
in `references`. QA caption files provide `image_id` and `response` matching
the question manifests.

`eval/qa/download_metrics.py` installs the scoring and prompt code listed in
`eval/qa/sources.json` into `data/metrics/`.

## Dataset sources

- [PixMo-Cap](https://huggingface.co/datasets/allenai/pixmo-cap)
- [DenseFusion-1M](https://huggingface.co/datasets/BAAI/DenseFusion-1M)
