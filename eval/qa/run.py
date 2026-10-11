#!/usr/bin/env python3
"""Prepare, answer and score questions using captions as the sole visual input."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from moco.io import ROOT, index_rows, read_jsonl, save_json, sha256, write_jsonl
from eval.api import request_text
from eval.qa.adapters import Adapters, clean

BENCHMARKS = ("CaptionQA", "BLINK", "TextVQA", "DocVQA", "ChartQA")
TOKENS = {"CaptionQA": 4, "BLINK": 1024, "TextVQA": 256, "DocVQA": 256, "ChartQA": 256}
REMINDER = '\n\nReturn the correctness label only: 1 (correct) or 0 (incorrect). Do not answer the chart question or output a value, ratio, confidence, or explanation.'


def prepare(args):
    qs = read_jsonl(args.questions)
    index_rows(qs, "sample_id")
    if not qs or any(q.get("bench") != args.benchmark for q in qs):
        raise ValueError("Questions must all belong to the requested benchmark")
    source = index_rows(read_jsonl(args.captions), "image_id")
    captions = {k: v["response"] for k, v in source.items()}
    if any(not isinstance(v, str) or not v.strip() for v in captions.values()):
        raise ValueError("Missing caption text")
    adapter = Adapters(args.metrics)
    inputs = [adapter.qa_input(q, captions) for q in qs]
    args.out.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.out / "questions.jsonl", qs)
    write_jsonl(args.out / "inputs.jsonl", inputs)
    save_json(args.out / "manifest.json", {"benchmark": args.benchmark, "questions": len(qs),
              "questions_sha256": sha256(args.out / "questions.jsonl"),
              "inputs_sha256": sha256(args.out / "inputs.jsonl"),
              "captions_sha256": sha256(args.captions), "sources_sha256": sha256(Path(__file__).with_name('sources.json'))})
    print(f"Prepared {len(qs)} text-only QA inputs; no inference started.")


def verify(bundle):
    m = json.loads((bundle / "manifest.json").read_text())
    for file in ("questions", "inputs"):
        if sha256(bundle / f"{file}.jsonl") != m[f"{file}_sha256"]:
            raise ValueError("Input changed after preparation")
    if sha256(Path(__file__).with_name('sources.json')) != m['sources_sha256']:
        raise ValueError("Scoring sources changed")
    return m


def infer(args):
    meta = verify(args.bundle)
    raw = args.bundle / "answers.jsonl"
    if raw.exists():
        raise ValueError("Answers already exist; use score, or prepare a new output directory")
    cmd = ["swift", "infer", "--model", args.model, "--model_type", "qwen3", "--template", "qwen3",
           "--enable_thinking", "false", "--infer_backend", "vllm", "--vllm_tensor_parallel_size", "1",
           "--vllm_data_parallel_size", str(len(args.gpus.split(','))), "--vllm_gpu_memory_utilization", "0.75",
           "--vllm_max_model_len", "32768", "--vllm_max_num_seqs", "32", "--vllm_enforce_eager", "true",
           "--val_dataset", str(args.bundle / "inputs.jsonl"), "--result_path", str(raw),
           "--load_args", "false", "--load_data_args", "false", "--max_new_tokens", str(TOKENS[meta['benchmark']]),
           "--temperature", "0", "--top_p", "1", "--seed", "42", "--remove_unused_columns", "false",
           "--load_from_cache_file", "false"]
    save_json(args.bundle / "answer_config.json", {"model": args.model, "command": cmd, "text_only": True})
    if args.dry_run:
        print(json.dumps({"command": cmd, "inference_started": False}, indent=2))
        return
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": args.gpus}
    subprocess.run(cmd, check=True, env=env)


def score(args):
    meta = verify(args.bundle)
    ds = meta['benchmark']
    qs = read_jsonl(args.bundle / "questions.jsonl")
    inputs = index_rows(read_jsonl(args.bundle / "inputs.jsonl"), "sample_id")
    answers = index_rows(read_jsonl(args.bundle / "answers.jsonl"), "sample_id")
    if set(answers) != set(inputs):
        raise ValueError("QA answer IDs must match all question IDs, including empty answers")
    a = Adapters(args.metrics)
    rows = []
    pending = []
    for q in qs:
        x = answers[q['sample_id']]
        if x.get('images'):
            raise ValueError("QA answerer must not receive images")
        if 'messages' in x:
            messages = [{k: m[k] for k in ['role', 'content']} for m in x['messages'] if m['role'] != 'assistant']
            if messages != inputs[q['sample_id']]['messages']:
                raise ValueError("QA prompt mismatch")
        prediction = clean(str(x.get('response', '')))
        row = {**q, 'prediction': prediction}
        if ds == 'CaptionQA':
            row.update(a.captionqa_score(q, prediction))
        elif ds == 'TextVQA':
            row['score'] = a.text._compute_answer_scores(q['references']).get(a.text.answer_processor(prediction), 0.)
        elif ds == 'BLINK':
            choice, prompt = a.blink_choice(q, prediction)
            if prompt is None:
                row['score'] = float(choice == '('+q['references'][0]+')')
            else:
                pending.append((row, [{'role': 'user', 'content': prompt}]))
        elif ds == 'ChartQA':
            prompt = (ROOT / 'prompts/chartqa.txt').read_text()
            pending.append((row, [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': json.dumps(
                {'question': q['question'], 'reference_answers': q['references'], 'candidate_answer': prediction}, ensure_ascii=False)}]))
        rows.append(row)
    if ds == 'DocVQA':
        gt = {'dataset_name': 'DocVQA', 'data': [{'questionId': i, 'question': r['question'], 'answers': r['references']} for i, r in enumerate(rows)]}
        sub = [{'questionId': i, 'answer': r['prediction']} for i, r in enumerate(rows)]
        gp, sp = args.bundle / 'docvqa_gt.json', args.bundle / 'docvqa_submission.json'
        save_json(gp, gt); save_json(sp, sub)
        a.doc.validate_data(str(gp), str(sp))
        evaluated = a.doc.evaluate_method(str(gp), str(sp), argparse.Namespace(answer_types=False, anls_threshold=.5))
        for i, row in enumerate(rows): row['score'] = evaluated['per_sample_result'][str(i)]['score']
    if pending:
        if not args.judge_url or not args.judge_model:
            raise ValueError('ChartQA grading / unresolved BLINK answers require --judge-url and --judge-model')
        if not os.environ.get(args.judge_key_env):
            raise ValueError(f'Set {args.judge_key_env}')
        cfg = {'model': args.judge_model, 'api_url': args.judge_url, 'temperature': 0, 'top_p': 1,
               'max_tokens': 64, 'enable_thinking': False, 'answers_sha256': sha256(args.bundle/'answers.jsonl')}
        config_file = args.bundle / 'judge_config.json'
        if config_file.exists() and json.loads(config_file.read_text()) != cfg:
            raise ValueError('Judge configuration changed; use a new output directory')
        save_json(config_file, cfg)
        cache_path = args.bundle / 'judge_outputs.jsonl'
        cached = index_rows(read_jsonl(cache_path), 'sample_id') if cache_path.exists() else {}
        def grade(item):
            row, messages = item
            uid = row['sample_id']
            sig = hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            if uid in cached:
                if cached[uid]['prompt_sha256'] != sig: raise ValueError('Changed grading prompt')
                return row, cached[uid]
            for attempt in range(3):
                sent = messages if attempt == 0 or ds != 'ChartQA' else [messages[0], {'role':'user', 'content':messages[1]['content']+REMINDER}]
                for retry in range(5):
                    try:
                        text, body = request_text(sent, model=args.judge_model, api_url=args.judge_url,
                                  api_key_env=args.judge_key_env, max_tokens=64, disable_thinking=True)
                        break
                    except Exception:
                        if retry == 4: raise
                        time.sleep(min(2*(retry+1), 10))
                if ds == 'BLINK' or text.strip() in {'0','1'}:
                    value = float(text == '('+row['references'][0]+')') if ds == 'BLINK' else int(text.strip())
                    return row, {'sample_id': uid, 'prompt_sha256': sig, 'response': text, 'score': value,
                                 'raw_response': body, 'actual_messages': sent, 'format_retry': attempt}
            raise ValueError('Invalid ChartQA judge response')
        with ThreadPoolExecutor(max_workers=16) as pool:
            for future in as_completed([pool.submit(grade, item) for item in pending]):
                row, record = future.result()
                row['score'] = record['score']
                cached[row['sample_id']] = record
                write_jsonl(cache_path, list(cached.values()))
    write_jsonl(args.bundle / 'predictions.jsonl', rows)
    categories = defaultdict(list)
    for row in rows:
        cat = row.get('domain') if ds == 'CaptionQA' else row.get('category', 'all')
        for value in cat if isinstance(cat, list) else [cat]: categories[value].append(row['score'])
    cat_scores = {k: {'count': len(v), 'score': 100*sum(v)/len(v)} for k,v in categories.items()}
    if ds == 'BLINK' and len(categories) != 14:
        raise ValueError('The paper BLINK protocol requires all 14 categories')
    overall = sum(x['score'] for x in cat_scores.values())/len(cat_scores) if ds == 'BLINK' else 100*sum(r['score'] for r in rows)/len(rows)
    result = {'benchmark': ds, 'questions': len(rows), 'score': overall, 'categories': cat_scores,
              'metric': {'CaptionQA': 'official option score', 'TextVQA': 'soft consensus accuracy', 'DocVQA': 'ANLS',
                         'BLINK': '14-category macro accuracy', 'ChartQA': 'LLM-judge accuracy'}[ds]}
    if ds == 'CaptionQA':
        result.update(accuracy=100*sum(r['is_correct'] for r in rows)/len(rows),
                      cannot_answer_rate=100*sum(r['is_cannot_answer'] for r in rows)/len(rows),
                      parse_failure_rate=100*sum(r['parsed_letter'] is None for r in rows)/len(rows))
    save_json(args.bundle / 'summary.json', result)
    print(json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    b = sub.add_parser('prepare')
    b.add_argument('--benchmark', choices=BENCHMARKS, required=True)
    b.add_argument('--questions', type=Path, required=True)
    b.add_argument('--captions', type=Path, required=True)
    b.add_argument('--out', type=Path, required=True)
    b.add_argument('--metrics', type=Path, default=ROOT/'data/metrics')
    b.set_defaults(func=prepare)
    r = sub.add_parser('infer')
    r.add_argument('--bundle', type=Path, required=True)
    r.add_argument('--model', default='Qwen/Qwen3-8B-Instruct')
    r.add_argument('--gpus', default=os.getenv('CUDA_VISIBLE_DEVICES', '0'))
    r.add_argument('--dry-run', action='store_true')
    r.set_defaults(func=infer)
    s = sub.add_parser('score')
    s.add_argument('--bundle', type=Path, required=True)
    s.add_argument('--metrics', type=Path, default=ROOT/'data/metrics')
    s.add_argument('--judge-url')
    s.add_argument('--judge-model')
    s.add_argument('--judge-key-env', default='QA_JUDGE_API_KEY')
    s.set_defaults(func=score)
    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
