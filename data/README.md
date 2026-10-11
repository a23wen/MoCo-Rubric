---
language:
- en
task_categories:
- image-to-text
pretty_name: MoCo-Rubric Training Data
configs:
- config_name: sft
  data_files:
  - split: train
    path: sft.jsonl
- config_name: rl
  data_files:
  - split: train
    path: rl.jsonl
---

# MoCo-Rubric Training Data

Training data will be released separately on Hugging Face. The repository and
revision will be added after that release; there is no download link yet.
Training manifests, archives, and extracted images belong in this `data/`
directory. They are ignored by Git and will be uploaded separately from the
code. Model weights and evaluation datasets are not included in the training
data release.

Training layout after downloading and extracting the image archives:

```text
data/
├── README.md
├── SHA256SUMS
├── sft.jsonl
├── rl.jsonl
├── images/
│   ├── pixmo-cap.tar
│   ├── densefusion.tar
│   ├── pixmo-cap/
│   └── densefusion/
└── benchmarks/                 # separately supplied evaluation inputs
```

Image paths in all distributed manifests must be relative to `data/`, for
example `images/pixmo-cap/000001.jpg`. Preserve dataset-qualified sample IDs;
images from different datasets must not be aligned by filename alone.
An external download directory is also supported through `--data-root`.
The archives preserve image bytes and formats without resizing or re-encoding.
Keeping both archives and extracted images requires approximately 12 GB.

## Training

`sft.jsonl` contains ms-swift `messages` with an assistant target, an optional
`images` list, and preserved role/provenance fields. Policy and Generator
records include images; Judge records are text-only. The intended mixed set
contains 20,000 records in a 2:1:1 Policy:Generator:Judge ratio. Preserve its
published ordering and metadata.

`rl.jsonl` contains `sample_uid`, `sample_id`, `source_dataset`, `messages`
ending in a user prompt, one image in `images`, structured `visual_info`,
and two reference strings named `gpt_caption` and `gemini_caption`.
`reference_caption` combines the two references. These field names retain
the original implementation's schema.
Do not include precomputed `rubrics`: the momentum Generator constructs them
from current Policy candidates. Other provenance fields are preserved.

Convert downloaded relative image paths into local training inputs:

```bash
python data/prepare_sft.py
python data/prepare_rl.py
```

These commands validate the manifests and write `outputs/data/sft.jsonl` and
`outputs/data/rl.jsonl`. They never edit the downloaded manifests or move images.
For another location, pass `--input`, `--data-root`, and `--output` explicitly.
The generated files contain machine-local paths and are excluded from Git;
regenerate them whenever the checkout location changes.

`build_sft.py` retains the original deterministic mixing procedure for users
who have the four aligned role files. Its `--help` lists the inputs. The normal
training path uses the prepared mixed manifest from the data release, without
regenerating teacher annotations.

The selected SFT source is `shared_sft_2_1_1_20k_seed42.jsonl` (20,000
records). The released RL set contains 5,000 examples: 2,500 PixMo-Cap and
2,500 DenseFusion. It restores `pixmo-cap:001577` from the original image
selection, using its existing GPT reference and visual information plus a
new `gemini-3.1-pro-preview` reference. The added record includes
`release_added` and `reference_sources` metadata. The original training
manifest had 4,999 examples; existing paper checkpoints were not retrained
as part of this release.

The image archives contain 7,500 files per source dataset. Both SFT references
to `DenseFusion:011586` use the repaired RGB image
`images/densefusion/011586.png`; the old JPG is excluded. Original SFT text,
targets, teacher metadata, and record order are preserved. Original RL
records retain all non-path fields and their relative order.

Download `sft.jsonl`, `rl.jsonl`, `SHA256SUMS`, `images/pixmo-cap.tar`, and
`images/densefusion.tar` into `data/`. Keep this README. From inside `data/`,
verify and unpack the images before running the preparation commands above:

```bash
sha256sum -c SHA256SUMS
tar -xf images/pixmo-cap.tar
tar -xf images/densefusion.tar
```

## Caption and QA evaluation inputs

Caption benchmark files go under `data/benchmarks/` (`pixmo-cap.jsonl`,
`densefusion.jsonl`, `caparena.jsonl`, `comprecap.jsonl`, and `docci.jsonl`).
QA inputs go under `data/benchmarks/qa/`, one JSONL file per benchmark listed
below. These files are supplied separately and are not in the training package.

A caption benchmark JSONL row contains `image` and optionally `image_id` and
reference metadata. `scripts/caption.sh` emits the same fields plus `response`.
For pairwise evaluation, both caption files must have identical `image` IDs.

QA files contain one benchmark per file, with a unique `sample_id` and `bench`:

- CaptionQA: `image_id`, `question`, `choices`, `answer`, `permutation`, `domain`.
  `answer` is the exact correct option text, not its letter. Preserve the
  published option permutation; do not shuffle again.
- BLINK: ordered `image_ids`, `question`, `options`, `official_prompt`,
  `references` (the correct letter), and `category`.
- TextVQA, DocVQA, ChartQA: `image_ids`, `question`, `references`, `category`.

The generated caption file for QA must provide `image_id` and `response` for
every requested image. Images/questions/option permutations remain external
data, and their release is separate from these scripts.

`python eval/qa/download_metrics.py` downloads only pinned upstream scoring
and prompt code to ignored `data/metrics/`; it does not download a dataset.

## Source datasets

- [PixMo-Cap](https://huggingface.co/datasets/allenai/pixmo-cap)
- [DenseFusion-1M](https://huggingface.co/datasets/BAAI/DenseFusion-1M)

Refer to the original dataset cards for their notices, licenses, and citations.
This release does not relicense the source images.
