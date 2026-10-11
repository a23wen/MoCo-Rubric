#!/usr/bin/env python3
"""Blind ranking: build, run, summarize. Standard library only."""
import argparse
import base64
import concurrent.futures
import fcntl
import hashlib
import json
import math
import mimetypes
import os
from pathlib import Path
import random
import time
import urllib.error
import urllib.request

VERSION = "moco-blind-rank-v1"
FIELDS = ("accuracy", "completeness", "clarity", "hallucination_penalty")


def canonical(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(obj):
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def records(path):
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        if line.strip():
            try:
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError("expected object")
                yield obj
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{n}: {exc}") from exc


def write_jsonl(path, rows):
    Path(path).write_text("".join(canonical(row) + "\n" for row in rows))


def label(index):
    result = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result
    return "Caption " + result


def resolve(base, value):
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def prompt_for(template, candidates):
    example = {"assessments": {c["label"]: {
        "justification": "<2-3 sentences citing specific visual evidence from the image>",
        **{f: "<int 0-10>" for f in FIELDS}, "total_score": "<float>"
    } for c in candidates}, "ranking": [c["label"] for c in candidates]}
    return template.format(count=len(candidates), last=len(candidates)-1,
                           captions_text="\n\n".join(c["label"] + ":\n" + c["caption"] for c in candidates),
                           output_example=json.dumps(example, indent=2))


def schema_for(labels):
    assessment = {"type": "object", "properties": {
        "justification": {"type": "string"},
        **{f: {"type": "integer", "minimum": 0, "maximum": 10} for f in FIELDS},
        "total_score": {"type": "number"}},
        "required": ["justification", *FIELDS, "total_score"], "additionalProperties": False}
    return {"type": "object", "properties": {
        "assessments": {"type": "object", "properties": {x: assessment for x in labels},
                        "required": labels, "additionalProperties": False},
        "ranking": {"type": "array", "items": {"type": "string", "enum": labels},
                    "minItems": len(labels), "maxItems": len(labels)}},
        "required": ["assessments", "ranking"], "additionalProperties": False}


def build(args):
    cfg_path = Path(args.config).resolve()
    cfg = read_json(cfg_path)
    base = cfg_path.parent
    models = cfg["models"]
    ids = [m["id"] for m in models]
    if len(ids) < 2 or any(not isinstance(x, str) or not x.strip() for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("models must contain at least two unique nonempty string IDs")
    alignment = cfg.get("alignment", "strict")
    mode = cfg.get("permutations", "fixed")
    seed = cfg.get("seed", 42)
    if alignment not in {"strict", "intersection"} or mode not in {"fixed", "random", "cyclic"}:
        raise ValueError("alignment: strict/intersection; permutations: fixed/random/cyclic")
    if type(seed) is not int:
        raise ValueError("seed must be integer")
    image_root = resolve(base, cfg["image_root"])
    template_path = resolve(base, cfg["prompt_file"]) if cfg.get("prompt_file") else Path(__file__).resolve().parents[1] / "prompts/ranking.txt"
    template = template_path.read_text()
    source_maps, sources = {}, []
    for model in models:
        path = resolve(base, model["captions"])
        by_image = {}
        for row in records(path):
            image = row[model.get("image_field", "image")]
            if isinstance(image, list) and len(image) == 1:
                image = image[0]
            caption = row[model.get("caption_field", "response")]
            if not isinstance(image, str) or not image.strip() or not isinstance(caption, str) or not caption.strip():
                raise ValueError(f"{model['id']}: image/caption must be nonempty strings")
            image_path = str(resolve(image_root, image))
            if image_path in by_image:
                raise ValueError(f"{model['id']}: duplicate image {image_path}")
            by_image[image_path] = caption
        if not by_image:
            raise ValueError(f"empty caption source: {path}")
        source_maps[model["id"]] = by_image
        sources.append({**model, "captions": str(path), "sha256": file_hash(path), "rows": len(by_image)})
    image_sets = [set(source_maps[x]) for x in ids]
    common = set.intersection(*image_sets)
    if alignment == "strict" and any(s != image_sets[0] for s in image_sets):
        raise ValueError("image sets differ; fix inputs or explicitly choose intersection alignment")
    if not common:
        raise ValueError("no common images")
    manifest = {"version": VERSION, "config": cfg, "sources": sources, "image_root": str(image_root),
                "prompt_sha256": hashlib.sha256(template.encode()).hexdigest(),
                "seed": seed, "permutations": mode, "model_ids": ids,
                "excluded_images": {x: sorted(set(source_maps[x])-common) for x in ids},
                "images": len(common), "requests_per_image": len(ids) if mode == "cyclic" else 1}
    protocol_id = digest(manifest)
    rows = []
    for image in sorted(common):
        image_sha = file_hash(image)
        order = ids.copy()
        if mode != "fixed":
            random.Random(digest([seed, image])).shuffle(order)
        for p in range(manifest["requests_per_image"]):
            shifted = order[p:] + order[:p]
            candidates = [{"label": label(j), "model_id": mid, "caption": source_maps[mid][image]} for j, mid in enumerate(shifted)]
            row = {"protocol_id": protocol_id, "image": image, "image_sha256": image_sha,
                   "permutation": p, "candidates": candidates, "prompt": prompt_for(template, candidates)}
            row["request_id"] = digest(row)
            rows.append(row)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    write_jsonl(out / "requests.jsonl", rows)
    (out / "prompt.txt").write_text(template)
    save_json(out / "manifest.json", {**manifest, "protocol_id": protocol_id, "requests": len(rows),
                                       "requests_sha256": file_hash(out / "requests.jsonl")})
    print(f"Built {len(rows)} requests / {len(common)} images; no API calls")


def load_bundle(bundle):
    bundle = Path(bundle)
    manifest = read_json(bundle / "manifest.json")
    if file_hash(bundle / "requests.jsonl") != manifest["requests_sha256"]:
        raise ValueError("request file hash mismatch")
    rows = list(records(bundle / "requests.jsonl"))
    if len(rows) != manifest["requests"] or len({r["request_id"] for r in rows}) != len(rows):
        raise ValueError("request count or duplicate ID error")
    for row in rows:
        if row["request_id"] != digest({k: v for k, v in row.items() if k != "request_id"}):
            raise ValueError("invalid request identity")
        if row["protocol_id"] != manifest["protocol_id"]:
            raise ValueError("protocol identity mismatch")
    return manifest, rows


def validate(text, request):
    data = json.loads(text)
    labels = [c["label"] for c in request["candidates"]]
    assessments, ranking = data["assessments"], data["ranking"]
    if not isinstance(assessments, dict) or set(assessments) != set(labels):
        raise ValueError("assessment labels mismatch")
    if not isinstance(ranking, list) or len(ranking) != len(labels) or set(ranking) != set(labels):
        raise ValueError("ranking must be a permutation of all labels")
    scores, warnings = {}, []
    for name in labels:
        a = assessments[name]
        if not isinstance(a.get("justification"), str) or not a["justification"].strip():
            raise ValueError("missing justification")
        if any(type(a.get(f)) is not int or not 0 <= a[f] <= 10 for f in FIELDS):
            raise ValueError("scores must be integers in [0, 10]")
        total = a.get("total_score")
        if type(total) not in (int, float) or not math.isfinite(total):
            raise ValueError("total_score must be finite numeric")
        q = 2 * (a["accuracy"] + a["completeness"] + a["clarity"]) - 9 * a["hallucination_penalty"]
        if abs(total - q/6) > 0.011:
            warnings.append({"label": name, "type": "total_mismatch", "returned": total, "computed": q/6})
        scores[name] = q
    if any(scores[a] < scores[b] for a, b in zip(ranking, ranking[1:])):
        raise ValueError("returned ranking contradicts recomputed total scores")
    per_model = {}
    for c in request["candidates"]:
        name, q = c["label"], scores[c["label"]]
        better = sum(x > q for x in scores.values())
        equal = sum(x == q for x in scores.values())
        per_model[c["model_id"]] = {**{f: assessments[name][f] for f in FIELDS},
            "total_score": q/6, "score_key": q, "label": name,
            "returned_rank": ranking.index(name)+1, "midrank": better+(equal+1)/2,
            "fractional_top1": 1/equal if better == 0 else 0}
    return {"per_model": per_model, "warnings": warnings,
            "has_score_tie": len(set(scores.values())) < len(scores),
            "top_score_tie": sum(x == max(scores.values()) for x in scores.values()) > 1}


def make_payload(row, config):
    image = Path(row["image"])
    if file_hash(image) != row["image_sha256"]:
        raise ValueError("image hash changed after build")
    mime = mimetypes.guess_type(image.name)[0]
    if mime not in {"image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp"}:
        raise ValueError(f"unsupported image MIME: {mime}")
    encoded = base64.b64encode(image.read_bytes()).decode()
    schema = schema_for([c["label"] for c in row["candidates"]])
    if config["provider"] == "gemini":
        payload = {"contents": [{"role": "user", "parts": [
            {"inlineData": {"mimeType": mime, "data": encoded}}, {"text": row["prompt"]}]}],
            "generationConfig": {"temperature": config["temperature"], "maxOutputTokens": config["max_tokens"]}}
        if config["structured_output"]:
            payload["generationConfig"].update(responseMimeType="application/json", responseJsonSchema=schema)
    else:
        payload = {"model": config["model"], "temperature": config["temperature"], "max_tokens": config["max_tokens"],
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
                {"type": "text", "text": row["prompt"]}]}]}
        if config["structured_output"]:
            payload["response_format"] = {"type": "json_schema", "json_schema": {"name": "blind_rank", "strict": True, "schema": schema}}
    return payload


def extract_text(body, provider):
    if body.get("error") or body.get("Error"):
        raise ValueError("API returned business error; see raw_response")
    if provider == "gemini":
        candidate = body["candidates"][0]
        finish = candidate.get("finishReason")
        if finish != "STOP":
            raise ValueError(f"unsuccessful Gemini finish reason: {finish}")
        return "".join(p.get("text", "") for p in candidate["content"]["parts"] if not p.get("thought"))
    choice = body["choices"][0]
    if choice.get("finish_reason") != "stop":
        raise ValueError(f"unsuccessful chat completion: {choice.get('finish_reason')}")
    return choice["message"]["content"]


def evaluate_one(row, config, key):
    payload = make_payload(row, config)
    header = config["auth_header"]
    headers = {"Content-Type": "application/json", header: config["auth_prefix"] + key}
    request = urllib.request.Request(config["api_url"], data=canonical(payload).encode(), headers=headers, method="POST")
    result = {"request_id": row["request_id"], "image": row["image"], "permutation": row["permutation"], "max_tokens_used": config["max_tokens"]}
    try:
        with urllib.request.urlopen(request, timeout=config["timeout"]) as response:
            result["http_status"] = response.status
            result["raw_response"] = response.read().decode()
        body = json.loads(result["raw_response"])
        result["judge_text"] = extract_text(body, config["provider"])
        result["parsed"] = validate(result["judge_text"], row)
        result["valid"] = True
    except urllib.error.HTTPError as exc:
        result.update(valid=False, http_status=exc.code, raw_response=exc.read().decode(errors="replace"), error=f"HTTP {exc.code}")
    except Exception as exc:
        result.update(valid=False, error=f"{type(exc).__name__}: {exc}")
    return result


def recover_log(path):
    """Discard only an interrupted final JSONL append; archive the fragment."""
    if not path.exists():
        return
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        split = data.rfind(b"\n") + 1
        fragment = data[split:]
        try:
            json.loads(fragment)
        except (ValueError, UnicodeDecodeError):
            fragment_path = path.with_name(f"interrupted_tail_{time.time_ns()}.bin")
            fragment_path.write_bytes(fragment)
            with path.open("r+b") as stream:
                stream.truncate(split)
        else:
            with path.open("ab") as stream:
                stream.write(b"\n")


def run(args):
    _, rows = load_bundle(args.bundle)
    if args.concurrency < 1 or args.max_attempts < 1 or args.max_tokens < 1 or args.timeout <= 0 or args.retry_delay < 0:
        raise ValueError("invalid numeric execution options")
    config = {k: getattr(args, k) for k in ("provider", "model", "api_url", "temperature", "max_tokens", "timeout", "concurrency", "max_attempts", "retry_delay")}
    config.update(retry_max_tokens=getattr(args, "retry_max_tokens", None),
                  structured_output=not args.no_structured_output,
                  auth_header=args.auth_header or ("x-goog-api-key" if args.provider == "gemini" else "Authorization"),
                  auth_prefix=args.auth_prefix if args.auth_prefix is not None else ("" if args.provider == "gemini" else "Bearer "),
                  bundle_sha256=file_hash(Path(args.bundle)/"requests.jsonl"),
                  implementation_sha256=file_hash(__file__))
    # No payload/model/key is selected or invoked by build/dry-run.
    if args.dry_run:
        print(json.dumps({"requests": len(rows), "config": config, "api_calls": 0}, indent=2))
        return
    key = os.environ.get(args.api_key_env)
    if not key:
        raise ValueError(f"set credential environment variable {args.api_key_env}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg_file, log = out/"run_config.json", out/"attempts.jsonl"
        if cfg_file.exists():
            if read_json(cfg_file) != config:
                raise ValueError("run configuration changed; use a new output directory")
        elif log.exists():
            raise ValueError("attempt log exists without run config")
        else:
            save_json(cfg_file, config)
        recover_log(log)
        history = list(records(log)) if log.exists() else []
        known = {r["request_id"] for r in rows}
        if any(r["request_id"] not in known for r in history):
            raise ValueError("unknown request in attempt log")
        done = {r["request_id"] for r in history if r.get("valid")}
        counts = {rid: sum(r["request_id"] == rid for r in history) for rid in known}
        truncated = {r["request_id"] for r in history if "MAX_TOKENS" in r.get("error", "")}
        with log.open("a") as stream, concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            while True:
                pending = [r for r in rows if r["request_id"] not in done and counts[r["request_id"]] < args.max_attempts]
                if not pending:
                    break
                futures = {}
                for row in pending:
                    request_config = config
                    if row["request_id"] in truncated and config.get("retry_max_tokens"):
                        request_config = {**config, "max_tokens": config["retry_max_tokens"]}
                    futures[pool.submit(evaluate_one, row, request_config, key)] = row
                for future in concurrent.futures.as_completed(futures):
                    row = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {"request_id": row["request_id"], "image": row["image"], "permutation": row["permutation"],
                                  "valid": False, "error": f"{type(exc).__name__}: {exc}"}
                    rid = row["request_id"]
                    counts[rid] += 1
                    if "MAX_TOKENS" in result.get("error", ""):
                        truncated.add(rid)
                    result.update(attempt=counts[rid], timestamp=time.time())
                    stream.write(canonical(result)+"\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    if result["valid"]:
                        done.add(rid)
                    print(f"valid={len(done)}/{len(rows)} attempt={counts[rid]} status={result['valid']}", flush=True)
                if any(r["request_id"] not in done and counts[r["request_id"]] < args.max_attempts for r in rows):
                    time.sleep(args.retry_delay)
        save_json(out/"status.json", {"execution_finished": True, "all_valid": len(done) == len(rows),
                                     "requests": len(rows), "valid": len(done), "failed": len(rows)-len(done)})


def summarize(args):
    manifest, rows = load_bundle(args.bundle)
    run_dir = Path(args.run)
    config = read_json(run_dir/"run_config.json")
    if config["bundle_sha256"] != file_hash(Path(args.bundle)/"requests.jsonl"):
        raise ValueError("run and bundle mismatch")
    lookup = {r["request_id"]: r for r in rows}
    valid, counts = {}, {}
    for attempt in records(run_dir/"attempts.jsonl"):
        rid = attempt["request_id"]
        if rid not in lookup:
            raise ValueError("unknown attempt request ID")
        counts[rid] = counts.get(rid, 0) + 1
        if attempt.get("valid"):
            if rid in valid:
                raise ValueError("duplicate successful request")
            valid[rid] = validate(attempt["judge_text"], lookup[rid])
    pending = [rid for rid in lookup if rid not in valid and counts.get(rid, 0) < config["max_attempts"]]
    if pending and not args.allow_partial:
        raise ValueError(f"{len(pending)} requests still pending; finish run or explicitly use --allow-partial")
    by_image = {}
    for row in rows:
        by_image.setdefault(row["image"], []).append(row)
    eligible = {image: group for image, group in by_image.items() if all(r["request_id"] in valid for r in group)}
    if not eligible:
        raise ValueError("no images with all planned permutations valid")
    result = {}
    for mid in manifest["model_ids"]:
        per_image = []
        position = {}
        for image, group in eligible.items():
            values = [valid[r["request_id"]]["per_model"][mid] for r in group]
            for value in values:
                position.setdefault(value["label"], []).append(value)
            per_image.append({**{f: sum(v[f] for v in values)/len(values) for f in (*FIELDS, "total_score", "midrank", "fractional_top1")},
                              "mean_returned_rank": sum(v["returned_rank"] for v in values)/len(values),
                              "rank_distribution": {str(k): sum(v["returned_rank"] == k for v in values)/len(values)
                                                    for k in range(1, len(manifest["model_ids"])+1)}})
        n = len(per_image)
        result[mid] = {f: sum(v[f] for v in per_image)/n for f in (*FIELDS, "total_score", "midrank", "fractional_top1", "mean_returned_rank")}
        result[mid]["rank_distribution"] = {str(k): sum(v["rank_distribution"][str(k)] for v in per_image)/n
                                             for k in range(1, len(manifest["model_ids"])+1)}
        result[mid]["rank1_rate"] = result[mid]["rank_distribution"]["1"]
        result[mid]["by_position"] = {pos: {"requests": len(vs), "total_score": sum(v["total_score"] for v in vs)/len(vs),
                                           "rank1_rate": sum(v["returned_rank"] == 1 for v in vs)/len(vs)} for pos, vs in position.items()}
    included = [valid[r["request_id"]] for group in eligible.values() for r in group]
    summary = {"version": VERSION, "protocol_id": manifest["protocol_id"], "judge": config["model"],
               "provider": config["provider"], "partial": bool(pending), "expected_images": len(by_image),
               "valid_images": len(eligible), "valid_requests": len(valid), "included_requests": len(included),
               "pending_requests": len(pending), "excluded_images": sorted(set(by_image)-set(eligible)),
               "score_tie_rate": sum(r["has_score_tie"] for r in included)/len(included),
               "top_score_tie_rate": sum(r["top_score_tie"] for r in included)/len(included),
               "total_mismatch_count": sum(len(r["warnings"]) for r in included),
               "aggregation": "equal weight per image; include only images with all planned permutations valid",
               "rank1_definition": "first label in valid judge-returned ranking; tie order is judge-returned",
               "fractional_top1_definition": "1/t for t captions tied on maximal locally computed score",
               "attempts_sha256": file_hash(run_dir/"attempts.jsonl"), "models": result}
    save_json(run_dir/"summary.json", summary)
    columns = ["Model", "Rank-1", "Mean rank", "Fractional top-1", "Midrank", *FIELDS, "Total"]
    lines = [f"# Blind Rank ({'PARTIAL' if pending else 'finished'})", "", f"Images: {len(eligible)}/{len(by_image)}. Judge: {config['model']}.", "",
             "Rank-1 follows judge-returned tie order; fractional top-1 and midrank handle score ties.", "",
             "| " + " | ".join(columns) + " |", "|" + "---|"*len(columns)]
    for mid, s in result.items():
        vals = [f"{s['rank1_rate']:.2%}", f"{s['mean_returned_rank']:.3f}", f"{s['fractional_top1']:.2%}", f"{s['midrank']:.3f}",
                *[f"{s[f]:.3f}" for f in FIELDS], f"{s['total_score']:.3f}"]
        lines.append("| " + " | ".join([mid.replace("|", "\\|"), *vals]) + " |")
    (run_dir/"summary.md").write_text("\n".join(lines)+"\n")
    print(f"Summarized {len(eligible)}/{len(by_image)} images -> {run_dir/'summary.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="freeze inputs and anonymous groups; no API calls")
    b.add_argument("--config", required=True)
    b.add_argument("--out", required=True)
    b.set_defaults(func=build)
    r = sub.add_parser("run", help="evaluate a frozen bundle, resuming matching request IDs")
    r.add_argument("--bundle", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--provider", choices=["gemini", "openai"], required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--api-url", required=True, help="Full POST endpoint for Gemini generateContent or OpenAI-compatible chat completions")
    r.add_argument("--api-key-env", default="BLIND_RANK_API_KEY")
    r.add_argument("--auth-header")
    r.add_argument("--auth-prefix")
    r.add_argument("--temperature", type=float, default=0.0)
    r.add_argument("--max-tokens", type=int, default=8192)
    r.add_argument("--retry-max-tokens", type=int, help="larger output limit only for retries after MAX_TOKENS")
    r.add_argument("--timeout", type=float, default=180)
    r.add_argument("--concurrency", type=int, default=4)
    r.add_argument("--max-attempts", type=int, default=2, help="total attempts per request across resumes")
    r.add_argument("--retry-delay", type=float, default=3)
    r.add_argument("--no-structured-output", action="store_true")
    r.add_argument("--dry-run", action="store_true")
    r.set_defaults(func=run)
    s = sub.add_parser("summarize", help="aggregate common complete images")
    s.add_argument("--bundle", required=True)
    s.add_argument("--run", required=True)
    s.add_argument("--allow-partial", action="store_true")
    s.set_defaults(func=summarize)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
