"""Rubric parsing and weighted binary rewards used during GRPO."""
from __future__ import annotations
import json
import re
from typing import Any

NUM_GENERATIONS = 16

def extract_json_object(text: str) -> dict[str, Any]:
    text = str(text).strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        text = fenced.group(1)
    elif not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            text = text[start:end + 1]
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("response is not a JSON object")
    return payload

def normalize_rubrics(payload: Any) -> list[dict[str, Any]]:
    items = payload.get("rubrics") if isinstance(payload, dict) else payload
    if not isinstance(items, list) or not items:
        raise ValueError("rubrics must be a non-empty list")
    result = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"rubric {index} is not an object")
        criterion = str(item.get("criterion", "")).strip()
        rule = str(item.get("evaluation_rule", "")).strip()
        try:
            weight = float(item.get("weight"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"rubric {index} has invalid weight") from exc
        if not criterion or not rule or weight <= 0:
            raise ValueError(f"rubric {index} is incomplete")
        result.append({"criterion": criterion, "evaluation_rule": rule, "weight": weight})
    return result

def weighted_reward(payload: dict[str, Any], rubrics: list[dict[str, Any]]) -> float:
    total = sum(float(item["weight"]) for item in rubrics)
    if total <= 0:
        return 0.0
    passed = sum(
        float(rubric["weight"]) * int(result["pass"])
        for rubric, result in zip(rubrics, payload["items"])
    )
    return max(0.0, min(1.0, passed / total))
