# Evaluation

Run commands from the repository root.

## Pairwise comparison

`pairwise.py` aligns two caption files by their unique `image` identifiers.
Each row provides `image` and `response`. The candidate becomes Caption A,
and the reference becomes Caption B.

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

`summarize` reports wins, losses, ties, `W/(W+L+T)`, and the number of invalid
or missing judgments. The win-rate denominator includes valid judgments.
Use `--ids file.json` to select a JSON list of image identifiers for the summary.

## API configuration

Set `GEMINI_API_KEY` and the full `generateContent` endpoint in `GEMINI_API_URL`.
For native Gemini, the model in the endpoint URL should match `--model`.
The default authentication header is `x-goog-api-key`.

- `--auth-header` and `--auth-prefix` configure authentication for compatible endpoints.
- `--provider openai` selects an OpenAI-compatible chat endpoint.
- `--api-key-env` names the environment variable containing the API key.
- `--dry-run` displays the request configuration.

These options are supported by both pairwise judging and blind ranking.

## Blind ranking

`ranking.py` builds anonymous caption groups from a configuration file, calls
the judge, and aggregates the returned rankings and scores.

Create `outputs/ranking_config.json`. Paths are relative to this configuration:

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

`permutations` accepts `fixed`, `random`, or `cyclic`. `alignment` accepts
`strict` or `intersection`. Add caption files through the `models` list.

```bash
python eval/ranking.py build --config outputs/ranking_config.json --out outputs/rank_bundle
python eval/ranking.py run --bundle outputs/rank_bundle --out outputs/rank_judge \
  --provider gemini --model gemini-3.1-pro-preview \
  --api-url "$GEMINI_API_URL" --api-key-env GEMINI_API_KEY
python eval/ranking.py summarize --bundle outputs/rank_bundle --run outputs/rank_judge
```

The script validates scores using
`(accuracy + completeness + clarity)/3 - 1.5*hallucination_penalty` and aggregates
the judge-returned rankings. An image enters aggregation when all configured
permutations have valid responses.

## Caption-only QA

`qa/run.py` provides `prepare`, `infer`, and `score` commands. The default
answerer is `Qwen/Qwen3-8B-Instruct`, with thinking disabled, temperature 0,
top-p 1, and seed 42. Inputs consist of captions and questions.

Download the scoring code, then prepare and run a benchmark:

```bash
python eval/qa/download_metrics.py
python eval/qa/run.py prepare \
  --benchmark TextVQA --questions data/benchmarks/qa/TextVQA.jsonl \
  --captions outputs/captions/textvqa.jsonl --out outputs/qa/TextVQA
python eval/qa/run.py infer --bundle outputs/qa/TextVQA --gpus 0,1,2,3,4,5,6,7
python eval/qa/run.py score --bundle outputs/qa/TextVQA
```

Caption files use `image_id` values matching the question manifests. See the
[data schemas](../data/README.md#evaluation-inputs). `infer --dry-run` displays
the inference command.

| Benchmark | Answer limit | Scoring |
|---|---:|---|
| CaptionQA | 4 tokens | Option accuracy |
| BLINK | 1,024 tokens | Answer extraction and 14-category macro accuracy |
| TextVQA | 256 tokens | Soft consensus accuracy |
| DocVQA | 256 tokens | ANLS with threshold 0.5 |
| ChartQA | 256 tokens | Text-only LLM-judge accuracy |

ChartQA scoring and unresolved BLINK answer mapping use an API judge. Configure
its endpoint, model, and credential variable:

```bash
python eval/qa/run.py score --bundle outputs/qa/ChartQA \
  --judge-url "$QA_JUDGE_URL" --judge-model "$QA_JUDGE_MODEL" \
  --judge-key-env QA_JUDGE_API_KEY
```

The API judge uses temperature 0, top-p 1, a 64-token output limit, and disabled
thinking. Empty or unparsable answers count as incorrect; missing answers or
failed API grading produce an error.

Scoring-code URLs, revisions, and checksums are listed in `qa/sources.json`.
