#!/usr/bin/env python3
"""Compare MoCo captions (A) with supplied reference captions (B)."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moco.io import ROOT, image_path, index_rows, read_jsonl, save_json, sha256, write_jsonl
from eval.api import request_text

SCHEMA = {"type": "object", "properties": {"reason": {"type": "string"},
          "judgment": {"type": "string", "enum": ["A", "B", "Tie"]}},
          "required": ["reason", "judgment"], "additionalProperties": False}


def parse_judgment(text):
    text = text.strip()
    if text.startswith("```"):
        text = "\n".join(x for x in text.splitlines() if not x.strip().startswith("```"))
    value = json.loads(text)
    judgment = str(value.get("judgment", "")).strip()
    judgment = "Tie" if judgment.lower() == "tie" else judgment.upper()
    reason = value.get("reason")
    if judgment not in {"A", "B", "Tie"} or not isinstance(reason, str) or not reason.strip():
        raise ValueError("Invalid judgment or empty reason")
    return {"judgment": judgment, "reason": reason}


def build(args):
    a = index_rows(read_jsonl(args.candidate), "image")
    b = index_rows(read_jsonl(args.reference), "image")
    if not a or set(a) != set(b):
        raise ValueError("Candidate and reference image IDs must match exactly")
    template = (ROOT / "prompts/pairwise.txt").read_text()
    rows = []
    for key in sorted(a):
        captions = [x[key].get("response") for x in (a, b)]
        if any(not isinstance(c, str) or not c.strip() for c in captions):
            raise ValueError(f"Missing response: {key}")
        rows.append({"image": key, "image_path": image_path(key, args.data_root),
                     "prompt": template.format(caption_a=captions[0].strip(), caption_b=captions[1].strip())})
    args.out.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.out / "requests.jsonl", rows)
    save_json(args.out / "manifest.json", {"order": "candidate_A_reference_B", "expected": len(rows),
              "candidate_sha256": sha256(args.candidate), "reference_sha256": sha256(args.reference),
              "requests_sha256": sha256(args.out / "requests.jsonl"),
              "prompt_sha256": sha256(ROOT / "prompts/pairwise.txt")})
    print(f"Prepared {len(rows)} pairs; no API calls.")


def run(args):
    manifest = json.loads((args.bundle / "manifest.json").read_text())
    requests_file = args.bundle / "requests.jsonl"
    if manifest["requests_sha256"] != sha256(requests_file):
        raise ValueError("Requests changed after preparation")
    rows = read_jsonl(requests_file)
    config = {k: getattr(args, k) for k in ("provider", "model", "api_url", "api_key_env",
              "auth_header", "auth_prefix", "max_tokens", "max_attempts", "concurrency")}
    config.update(requests_sha256=sha256(requests_file), temperature=0.0, top_p=0.95)
    config_file = args.bundle / "judge_config.json"
    if config_file.exists() and json.loads(config_file.read_text()) != config:
        raise ValueError("Judge settings changed; prepare a new bundle")
    if args.dry_run:
        print(json.dumps({"requests": len(rows), "config": config, "api_calls": 0}, indent=2))
        return
    if min(args.concurrency, args.max_attempts, args.max_tokens) < 1:
        raise ValueError("Execution options must be positive")
    if not os.environ.get(args.api_key_env):
        raise ValueError(f"Set {args.api_key_env}")
    save_json(config_file, config)
    path = args.bundle / "judgments.jsonl"
    previous = index_rows(read_jsonl(path), "image") if path.exists() else {}
    if not set(previous).issubset({x["image"] for x in rows}):
        raise ValueError("Unknown image in cached judgments")
    def one(row):
        if previous.get(row["image"], {}).get("valid"):
            return previous[row["image"]]
        for attempt in range(args.max_attempts):
            try:
                text, body = request_text([{"role": "user", "content": row["prompt"]}],
                    model=args.model, provider=args.provider, api_url=args.api_url,
                    api_key_env=args.api_key_env, image=row["image_path"], schema=SCHEMA,
                    max_tokens=args.max_tokens, top_p=0.95,
                    auth_header=args.auth_header, auth_prefix=args.auth_prefix)
                return {"image": row["image"], "valid": True, **parse_judgment(text), "raw_response": body}
            except Exception as exc:
                error = type(exc).__name__
                if attempt + 1 < args.max_attempts:
                    time.sleep(min(3 * 2**attempt, 30))
        return {"image": row["image"], "valid": False, "error_type": error}
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for future in as_completed([pool.submit(one, row) for row in rows]):
            item = future.result()
            previous[item["image"]] = item
            write_jsonl(path, [previous[k] for k in sorted(previous)])
    summarize(argparse.Namespace(bundle=args.bundle, ids=None))


def summarize(args):
    requests = index_rows(read_jsonl(args.bundle / "requests.jsonl"), "image")
    rows = index_rows(read_jsonl(args.bundle / "judgments.jsonl"), "image")
    if not set(rows).issubset(requests):
        raise ValueError("Unknown judgment IDs")
    ids = set(requests)
    if args.ids:
        selected = json.loads(args.ids.read_text())
        if not isinstance(selected, list) or len(set(selected)) != len(selected) or not set(selected).issubset(ids):
            raise ValueError("--ids must be a unique JSON list of requested image IDs")
        ids = set(selected)
    counts = Counter()
    for uid in ids:
        row = rows.get(uid, {})
        if row.get("valid"):
            parsed = parse_judgment(json.dumps(row))
            counts[parsed["judgment"]] += 1
    valid = sum(counts.values())
    result = {"expected": len(ids), "valid": valid, "excluded_or_missing": len(ids)-valid,
              "wins": counts["A"], "losses": counts["B"], "ties": counts["Tie"],
              "win_rate": counts["A"] / valid if valid else None,
              "win_rate_percent": 100 * counts["A"] / valid if valid else None,
              "metric": "W/(W+L+T)", "scope": "provided_ids" if args.ids else "all_requested"}
    name = "summary_selected.json" if args.ids else "summary.json"
    save_json(args.bundle / name, result)
    print(json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--candidate", required=True, type=Path)
    b.add_argument("--reference", required=True, type=Path)
    b.add_argument("--data-root", type=Path, default=ROOT / "data")
    b.add_argument("--out", type=Path, required=True)
    b.set_defaults(func=build)
    r = sub.add_parser("run")
    r.add_argument("--bundle", type=Path, required=True)
    r.add_argument("--provider", choices=["gemini", "openai"], default="gemini")
    r.add_argument("--model", required=True)
    r.add_argument("--api-url", required=True)
    r.add_argument("--api-key-env", default="GEMINI_API_KEY")
    r.add_argument("--auth-header")
    r.add_argument("--auth-prefix")
    r.add_argument("--concurrency", type=int, default=16)
    r.add_argument("--max-attempts", type=int, default=5)
    r.add_argument("--max-tokens", type=int, default=4096)
    r.add_argument("--dry-run", action="store_true")
    r.set_defaults(func=run)
    s = sub.add_parser("summarize")
    s.add_argument("--bundle", type=Path, required=True)
    s.add_argument("--ids", type=Path, help="Explicit common-valid image ID list")
    s.set_defaults(func=summarize)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
