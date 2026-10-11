"""Validate a downloaded training manifest without modifying it."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moco.io import ROOT, image_path, read_jsonl, write_jsonl


def main(stage):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data" / f"{stage}.jsonl")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/data" / f"{stage}.jsonl")
    args = parser.parse_args()
    if args.output.resolve() == args.input.resolve():
        parser.error("Output must differ from the downloaded input")
    rows = read_jsonl(args.input)
    if not rows:
        raise ValueError("Empty training dataset")
    seen = set()
    for n, row in enumerate(rows, 1):
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages or any(
            not isinstance(m, dict) or m.get("role") not in {"system", "user", "assistant"}
            or not isinstance(m.get("content"), str) or not m["content"].strip()
            for m in messages
        ):
            raise ValueError(f"Row {n}: invalid messages")
        if stage == "sft":
            if messages[-1]["role"] != "assistant":
                raise ValueError(f"Row {n}: missing SFT assistant target")
        else:
            uid = row.get("sample_uid")
            if not isinstance(uid, str) or not uid or uid in seen:
                raise ValueError(f"Row {n}: missing/duplicate sample_uid")
            seen.add(uid)
            if any(m["role"] == "assistant" for m in messages) or messages[-1]["role"] != "user":
                raise ValueError(f"Row {n}: RL input must not contain an assistant target")
            if "rubrics" in row:
                raise ValueError(f"Row {n}: online rubrics must not be precomputed")
            for key in ("gpt_caption", "gemini_caption"):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError(f"Row {n}: missing reference field {key}")
            if not isinstance(row.get("images"), list) or len(row["images"]) != 1:
                raise ValueError(f"Row {n}: RL requires exactly one image")
        if "images" in row:
            if not isinstance(row["images"], list):
                raise ValueError(f"Row {n}: images must be a list")
            row["images"] = [image_path(x, args.data_root) for x in row["images"]]
    # Preserve record order, messages, supervision and teacher metadata.
    write_jsonl(args.output, rows)
    print(json.dumps({"stage": stage, "rows": len(rows), "output": str(args.output)}))
