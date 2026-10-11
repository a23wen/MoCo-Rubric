"""Versioned Judge prompt loading, rendering, and output contracts."""
from __future__ import annotations

import hashlib
import json
import os
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT_ROOT = PROJECT_ROOT / "prompts"
DEFAULT_PROMPT_VARIANT = "judge"
OUTPUT_CONTRACTS = {"pass_only", "reason_pass"}
USER_TEMPLATE_FIELDS = {"generated_caption", "rubrics", "example_json", "item_count"}


def _template_fields(template: str) -> set[str]:
    return {
        field_name
        for _, field_name, _, _ in string.Formatter().parse(template)
        if field_name is not None
    }


@dataclass(frozen=True)
class JudgePromptBundle:
    variant: str
    prompt_version: str
    requires_reason: bool
    max_output_tokens: int
    root: Path
    config_path: Path
    system_path: Path
    user_path: Path
    system_prompt: str
    user_template: str
    sha256: str

    def example_items(self, item_count: int) -> list[dict[str, Any]]:
        if self.requires_reason:
            return [
                {
                    "reason": (
                        "The caption states the required subject and attribute."
                        if index % 2 == 0
                        else "The caption omits a required detail."
                    ),
                    "pass": 1 if index % 2 == 0 else 0,
                }
                for index in range(item_count)
            ]
        return [{"pass": 1 if index % 2 == 0 else 0} for index in range(item_count)]

    def zero_items(self, item_count: int) -> list[dict[str, Any]]:
        if self.requires_reason:
            return [
                {"reason": "Dry-run placeholder; no judgment was requested.", "pass": 0}
                for _ in range(item_count)
            ]
        return [{"pass": 0} for _ in range(item_count)]

    def render_user(self, caption: str, rubrics: Sequence[Mapping[str, Any]]) -> str:
        rubric_lines = [
            f'{index}. weight={item["weight"]}\n'
            f'criterion: {item["criterion"]}\n'
            f'evaluation_rule: {item["evaluation_rule"]}'
            for index, item in enumerate(rubrics, 1)
        ]
        example = json.dumps(
            {"items": self.example_items(len(rubrics))},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return self.user_template.format(
            generated_caption=str(caption).strip(),
            rubrics="\n\n".join(rubric_lines),
            example_json=example,
            item_count=len(rubrics),
        )

    def item_regex(self) -> str:
        if self.requires_reason:
            return r'\{"reason":"([^"\\]|\\.)+","pass":[01]\}'
        return r'\{"pass":[01]\}'

    def response_regex(self, expected_items: int | None = None) -> str:
        item = self.item_regex()
        if expected_items is not None:
            if expected_items <= 0:
                raise ValueError("expected_items must be positive")
            body = ",".join([item] * expected_items)
        else:
            body = item + r"(," + item + r")*"
        return r'\{"items":\[' + body + r'\]\}'

    def normalize_items(
        self,
        payload: Any,
        expected_items: int,
        *,
        allow_top_level_list: bool = False,
    ) -> list[dict[str, Any]]:
        if allow_top_level_list and isinstance(payload, list):
            items = payload
        elif isinstance(payload, dict) and set(payload) == {"items"}:
            items = payload["items"]
        else:
            raise ValueError("Judge payload must be an object containing only items")
        if not isinstance(items, list) or len(items) != expected_items:
            size = len(items) if isinstance(items, list) else type(items).__name__
            raise ValueError(f"Judge item count {size} != {expected_items}")
        normalized: list[dict[str, Any]] = []
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                raise ValueError(f"Judge item {index} must be an object")
            expected_keys = ["reason", "pass"] if self.requires_reason else ["pass"]
            if list(item) != expected_keys:
                raise ValueError(
                    f"Judge item {index} fields must be exactly {expected_keys} in that order"
                )
            if self.requires_reason:
                reason = item["reason"]
                if not isinstance(reason, str) or not reason.strip():
                    raise ValueError(
                        f"Judge item {index} reason must be a non-empty string"
                    )
            passed = item["pass"]
            if type(passed) is not int or passed not in {0, 1}:
                raise ValueError(f"Judge item {index} pass must be integer 0 or 1")
            normalized.append(
                {"reason": item["reason"].strip(), "pass": passed}
                if self.requires_reason
                else {"pass": passed}
            )
        return normalized

    def json_schema(self, expected_items: int) -> dict[str, Any]:
        properties: dict[str, Any] = {}
        required: list[str] = []
        if self.requires_reason:
            properties["reason"] = {"type": "string", "minLength": 1}
            required.append("reason")
        properties["pass"] = {"type": "integer", "enum": [0, 1]}
        required.append("pass")
        item_schema: dict[str, Any] = {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }
        if self.requires_reason:
            item_schema["propertyOrdering"] = ["reason", "pass"]
        return {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "minItems": expected_items,
                    "maxItems": expected_items,
                    "items": item_schema,
                }
            },
            "required": ["items"],
            "additionalProperties": False,
        }


def load_judge_prompt(
    variant: str | None = None,
    *,
    prompt_root: str | Path | None = None,
) -> JudgePromptBundle:
    selected = str(variant or DEFAULT_PROMPT_VARIANT).strip()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", selected):
        raise ValueError(f"invalid Judge prompt variant name {selected!r}")
    root = Path(
        prompt_root
        or os.getenv("MOCO_PROMPT_ROOT", "")
        or DEFAULT_PROMPT_ROOT
    ).resolve()
    version_dir = root / selected
    config_path = version_dir / "config.json"
    system_path = version_dir / "system.txt"
    user_path = version_dir / "user.txt"
    if not config_path.is_file() or not system_path.is_file() or not user_path.is_file():
        raise FileNotFoundError(
            f"Judge prompt variant {selected!r} requires config.json, system.txt, and user.txt in {version_dir}"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_config_keys = {"prompt_version", "output_contract", "max_output_tokens"}
    if not isinstance(config, dict) or set(config) != expected_config_keys:
        raise ValueError(
            f"{config_path} keys must be exactly {sorted(expected_config_keys)}"
        )
    output_contract = str(config["output_contract"])
    if output_contract not in OUTPUT_CONTRACTS:
        raise ValueError(
            f"{config_path} output_contract must be one of {sorted(OUTPUT_CONTRACTS)}"
        )
    max_output_tokens = config["max_output_tokens"]
    if type(max_output_tokens) is not int or max_output_tokens <= 0:
        raise ValueError(f"{config_path} max_output_tokens must be a positive integer")
    prompt_version = str(config["prompt_version"]).strip()
    if not prompt_version:
        raise ValueError(f"{config_path} prompt_version is empty")
    system_prompt = system_path.read_text(encoding="utf-8").strip()
    user_template = user_path.read_text(encoding="utf-8").strip()
    fields = _template_fields(user_template)
    if fields != USER_TEMPLATE_FIELDS:
        raise ValueError(
            f"{user_path} fields are {sorted(fields)}, expected {sorted(USER_TEMPLATE_FIELDS)}"
        )
    if not system_prompt:
        raise ValueError(f"{system_path} is empty")
    digest = hashlib.sha256(
        (
            json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\0"
            + system_prompt
            + "\0"
            + user_template
        ).encode("utf-8")
    ).hexdigest()
    return JudgePromptBundle(
        variant=selected,
        prompt_version=prompt_version,
        requires_reason=output_contract == "reason_pass",
        max_output_tokens=max_output_tokens,
        root=root,
        config_path=config_path,
        system_path=system_path,
        user_path=user_path,
        system_prompt=system_prompt,
        user_template=user_template,
        sha256=digest,
    )


def variant_from_env(name: str, default: str = DEFAULT_PROMPT_VARIANT) -> str:
    return os.getenv(name, default).strip() or default
