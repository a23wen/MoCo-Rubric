# Evaluation

Run commands from the repository root.

## API settings

Pairwise comparison and ranking use `GEMINI_API_KEY` and `GEMINI_API_URL`.
Set the URL to the full `generateContent` endpoint for the model passed to `--model`.
The default authentication header is `x-goog-api-key`.

Use `--provider openai` for an OpenAI-compatible chat endpoint.
`--auth-header` and `--auth-prefix` adjust the authentication format.

## Pairwise comparison

Each caption file contains `image` and `response`, with the same unique image
identifiers in both files. The candidate is Caption A; the reference is Caption B.

```bash
python eval/pairwise.py build \
  --candidate outputs/captions/pixmo-cap.jsonl \
  --reference data/benchmarks/reference/pixmo-cap.jsonl \
  --out outputs/pairwise/pixmo-cap

python eval/pairwise.py run \
  --bundle outputs/pairwise/pixmo-cap \
  --model gemini-3.1-pro-preview \
  --api-url "$GEMINI_API_URL" --api-key-env GEMINI_API_KEY

python eval/pairwise.py summarize --bundle outputs/pairwise/pixmo-cap
```

The summary reports wins, losses, ties, and `W/(W+L+T)` over valid judgments.
Use `--ids file.json` to select a JSON list of image identifiers.

## Blind ranking

Create `outputs/ranking_config.json`. File paths are relative to this configuration:

```json
{
  "image_root": "../data",
  "permutations": "fixed",
  "alignment": "strict",
  "seed": 42,
  "models": [
    {"id": "ours", "captions": "captions/pixmo-cap.jsonl", "caption_field": "response"},
    {"id": "reference", "captions": "../data/benchmarks/reference/pixmo-cap.jsonl", "caption_field": "response"}
  ]
}
```

Add caption files to `models`. Choose `fixed`, `random`, or `cyclic` for
`permutations`, and `strict` or `intersection` for `alignment`.

```bash
python eval/ranking.py build --config outputs/ranking_config.json --out outputs/rank_bundle
python eval/ranking.py run --bundle outputs/rank_bundle --out outputs/rank_judge \
  --provider gemini --model gemini-3.1-pro-preview \
  --api-url "$GEMINI_API_URL" --api-key-env GEMINI_API_KEY
python eval/ranking.py summarize --bundle outputs/rank_bundle --run outputs/rank_judge
```

Scores use `(accuracy + completeness + clarity)/3 - 1.5*hallucination_penalty`.
The summary aggregates rankings for images with valid responses for every permutation.

## Caption-based QA

`qa/run.py` answers questions from captions with `Qwen/Qwen3-8B-Instruct` by default.
Download the scoring code, then prepare, run, and score a benchmark:

```bash
python eval/qa/download_metrics.py
python eval/qa/run.py prepare \
  --benchmark TextVQA --questions data/benchmarks/qa/TextVQA.jsonl \
  --captions outputs/captions/textvqa.jsonl --out outputs/qa/TextVQA
python eval/qa/run.py infer --bundle outputs/qa/TextVQA --gpus 0,1,2,3,4,5,6,7
python eval/qa/run.py score --bundle outputs/qa/TextVQA
```

Caption files must match the question files by `image_id`.
See the [input formats](../data/README.md#evaluation-inputs).

| Benchmark | Metric |
|---|---|
| CaptionQA | Option accuracy |
| BLINK | 14-category macro accuracy |
| TextVQA | Soft consensus accuracy |
| DocVQA | ANLS with threshold 0.5 |
| ChartQA | Text-only LLM-judge accuracy |

ChartQA scoring and unresolved BLINK answers require an API judge. Set
`QA_JUDGE_URL`, `QA_JUDGE_MODEL`, and `QA_JUDGE_API_KEY`, then run:

```bash
python eval/qa/run.py score --bundle outputs/qa/ChartQA \
  --judge-url "$QA_JUDGE_URL" --judge-model "$QA_JUDGE_MODEL" \
  --judge-key-env QA_JUDGE_API_KEY
```
