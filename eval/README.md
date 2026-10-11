# Evaluation

Run commands from the repository root. Each run writes only to the requested
output directory. Dataset manifests, images, question/option order, and
comparison captions are external inputs; no benchmark results are bundled.

## Pairwise comparison

The candidate is always Caption A; the supplied Qwen3-VL-8B reference is
Caption B. Both files use `image` and `response`, and must contain the same
unique image IDs. Repeat for PixMo-Cap, DenseFusion, CapArena, CompreCap, and
DOCCI using their fixed manifests.

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

Set `GEMINI_API_KEY` and the full native `generateContent` endpoint in
`GEMINI_API_URL`. For native Gemini the model is selected in the URL; it must
match `--model`. The authentication header defaults to `x-goog-api-key`;
`--auth-header` / `--auth-prefix` support compatible endpoints. `--dry-run`
prints the request configuration without making API calls. An OpenAI-compatible
chat endpoint can be selected with `--provider openai` and an appropriate
credential variable; changing the judge creates a different evaluation run.
The GPT-5.6 Responses-API supplementary run is not claimed to be reproduced by
this generic chat adapter.

The summary reports **W/(W+L+T)**, plus all three counts and excluded/missing
judgments. Invalid judgments never enter the denominator. Use `--ids file.json`
with an explicit common-valid ID list when reproducing a comparison on a shared
sample set. Keep each table's ID set separate. Across the five benchmarks,
report the equal-weight mean of benchmark win rates, not pooled counts.

## Blind ranking

`ranking.py` retains the original anonymous ranking and aggregation logic.
Provide our caption file plus externally prepared comparison caption files.
There is no inference or training code for those other methods.

Create a local config, for example `outputs/ranking_config.json` (paths resolve
relative to that config):

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

This two-file example demonstrates the interface. The paper's protocol uses
**five** supplied caption files, one fixed A–E order, and the prompt in
`prompts/ranking.txt`. Preserve that order when reproducing its protocol.

```bash
python eval/ranking.py build --config outputs/ranking_config.json --out outputs/rank_bundle
python eval/ranking.py run --bundle outputs/rank_bundle --out outputs/rank_judge \
  --provider gemini --model gemini-3.1-pro-preview \
  --api-url "$GEMINI_API_URL" --api-key-env GEMINI_API_KEY
python eval/ranking.py summarize --bundle outputs/rank_bundle --run outputs/rank_judge
```

Ranking uses the judge-returned order after validating locally recomputed
scores. The score is `(accuracy + completeness + clarity)/3 - 1.5*hallucination_penalty`.
Only images with all planned rankings valid enter aggregation. Fixed order,
random order, and cyclic permutations are distinct protocols.

## Caption-only QA

The fixed answerer is **Qwen3-8B-Instruct**, with thinking disabled, temperature
0, top-p 1 and seed 42. It sees captions and questions, without images. Benchmark
scoring/prompt sources are downloaded separately, pinned by commit and SHA-256
in `qa/sources.json`:

```bash
python eval/qa/download_metrics.py
python eval/qa/run.py prepare \
  --benchmark TextVQA --questions data/benchmarks/qa/TextVQA.jsonl \
  --captions outputs/captions/textvqa.jsonl --out outputs/qa/TextVQA
python eval/qa/run.py infer --bundle outputs/qa/TextVQA --gpus 0,1,2,3,4,5,6,7
python eval/qa/run.py score --bundle outputs/qa/TextVQA
```

The caption inputs for QA are ordinary image manifests handled by
`scripts/caption.sh`, with `image_id` values matching the question manifest.
See [data schemas](../data/README.md). `infer --dry-run` prepares the command
without loading a model.

| Benchmark | Answer limit | Scoring used by this pipeline |
|---|---:|---|
| CaptionQA | 4 tokens | Official option score; published option permutations |
| BLINK | 1,024 tokens | Official answer extraction; unresolved choices use API mapping; 14-category macro mean |
| TextVQA | 256 tokens | MMF soft consensus accuracy |
| DocVQA | 256 tokens | Challenge ANLS with threshold 0.5 |
| ChartQA | 256 tokens | Text-only LLM judge using `prompts/chartqa.txt` |

ChartQA and unresolved BLINK mappings use the original run's Qwen3.6-27B
judge adaptation. Supply a matching endpoint/model alias explicitly:

```bash
python eval/qa/run.py score --bundle outputs/qa/ChartQA \
  --judge-url "$QA_JUDGE_URL" --judge-model "$QA_JUDGE_MODEL" \
  --judge-key-env QA_JUDGE_API_KEY
```

The API judge uses temperature 0, top-p 1, a 64-token output limit and disabled
thinking. This ChartQA score is **LLM-judge accuracy**, not the official
string-based relaxed-accuracy metric. Benchmark source attribution is preserved
in `qa/sources.json`; this caption-mediated setup is not directly comparable
to an image-input leaderboard.

Empty or unparsable QA answers remain in the denominator. Missing answers or
failed API grading stop scoring instead of silently shrinking the sample set.
The five-benchmark average is an equal-weight mean of their scores.
