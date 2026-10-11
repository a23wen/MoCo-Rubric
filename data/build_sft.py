#!/usr/bin/env python3
"""Build 5,000 contiguous P_i + G_i + J_i + P_j shared-SFT groups."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-10000", type=Path, required=True)
    parser.add_argument("--aligned-policy-5000", type=Path, required=True)
    parser.add_argument("--generator-5000", type=Path, required=True)
    parser.add_argument("--judge-5000", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_jsonl(path: Path, expected: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            messages = row.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"{path}:{line_number}: missing messages")
            if messages[-1].get("role") != "assistant" or not messages[-1].get("content"):
                raise ValueError(f"{path}:{line_number}: invalid assistant target")
            rows.append(normalize_images(row, path, line_number))
    if len(rows) != expected:
        raise ValueError(f"{path}: expected {expected} rows, got {len(rows)}")
    return rows


def normalize_images(
    row: dict[str, Any], path: Path, line_number: int
) -> dict[str, Any]:
    if "images" not in row:
        return row
    images = row["images"]
    if not isinstance(images, list):
        raise ValueError(f"{path}:{line_number}: images must be a list")
    normalized: list[str] = []
    for image in images:
        if isinstance(image, str):
            normalized.append(image)
        elif isinstance(image, dict) and isinstance(image.get("path"), str):
            normalized.append(image["path"])
        else:
            raise ValueError(f"{path}:{line_number}: unsupported image value {image!r}")
    result = dict(row)
    result["images"] = normalized
    return result


def message_key(row: dict[str, Any]) -> str:
    return json.dumps(
        row["messages"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def index_uid(rows: list[dict[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows, 1):
        uid = row.get("sample_uid")
        if not isinstance(uid, str) or not uid:
            raise ValueError(f"{name}:{index}: missing sample_uid")
        if uid in result:
            raise ValueError(f"{name}: duplicate sample_uid {uid}")
        result[uid] = row
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    policy_10000 = read_jsonl(args.policy_10000, 10000)
    aligned_policy_rows = read_jsonl(args.aligned_policy_5000, 5000)
    generator_rows = read_jsonl(args.generator_5000, 5000)
    judge_rows = read_jsonl(args.judge_5000, 5000)

    aligned_policy = index_uid(aligned_policy_rows, "aligned_policy")
    generator = index_uid(generator_rows, "generator")
    judge = index_uid(judge_rows, "judge")
    if not (set(aligned_policy) == set(generator) == set(judge)):
        raise ValueError("Aligned Policy, Generator, and Judge sample_uid sets differ")

    full_by_message: dict[str, tuple[int, dict[str, Any]]] = {}
    for raw_index, row in enumerate(policy_10000, 1):
        key = message_key(row)
        if key in full_by_message:
            raise ValueError(f"policy_10000: duplicate messages at row {raw_index}")
        full_by_message[key] = (raw_index, row)

    aligned_message_keys: set[str] = set()
    for uid, row in aligned_policy.items():
        key = message_key(row)
        if key not in full_by_message:
            raise ValueError(f"aligned_policy: {uid} not found in policy_10000 by messages")
        if key in aligned_message_keys:
            raise ValueError(f"aligned_policy: duplicate messages for {uid}")
        aligned_message_keys.add(key)

    extra_policy: list[tuple[int, dict[str, Any]]] = [
        value for key, value in full_by_message.items() if key not in aligned_message_keys
    ]
    if len(extra_policy) != 5000:
        raise ValueError(f"Expected 5,000 extra Policy rows, got {len(extra_policy)}")

    core_uids = sorted(aligned_policy)
    rng.shuffle(core_uids)
    rng.shuffle(extra_policy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    task_counts: Counter[str] = Counter()
    order_counts: Counter[str] = Counter()
    with args.output.open("w", encoding="utf-8") as output_handle, args.manifest.open(
        "w", encoding="utf-8"
    ) as manifest_handle:
        for group_index, (core_uid, (extra_raw_index, extra_row)) in enumerate(
            zip(core_uids, extra_policy)
        ):
            members = [
                ("policy_core", aligned_policy[core_uid]),
                ("generator", generator[core_uid]),
                ("judge", judge[core_uid]),
                ("policy_extra", extra_row),
            ]
            rng.shuffle(members)
            order = [task for task, _ in members]
            order_counts["|".join(order)] += 1
            for task, row in members:
                output_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                task_counts[task] += 1
            manifest_handle.write(
                json.dumps(
                    {
                        "group_index": group_index,
                        "line_start": group_index * 4 + 1,
                        "line_end": group_index * 4 + 4,
                        "core_sample_uid": core_uid,
                        "extra_policy_raw_line": extra_raw_index,
                        "order": order,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    summary = {
        "status": "complete",
        "seed": args.seed,
        "scheme": "P_i + G_i + J_i + P_j",
        "groups": 5000,
        "total_records": 20000,
        "unique_policy_records": 10000,
        "task_counts": dict(sorted(task_counts.items())),
        "combined_task_ratio": {"policy": 2, "generator": 1, "judge": 1},
        "aligned_uid_count": len(core_uids),
        "extra_policy_count": len(extra_policy),
        "aligned_policy_matched_to_policy_10000_by_unique_messages": True,
        "distinct_within_group_orders": len(order_counts),
        "dataset_shuffle_required": False,
        "train_dataloader_shuffle_required": False,
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
    }
    args.summary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
