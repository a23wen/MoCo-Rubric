"""Two-server Momentum GRPO plugin for ms-swift 4.2 / vLLM 0.11.

The rollout server always receives the online policy weights. A second rollout
server serves both rubric-generator and Judge roles and receives an EMA of the
same policy weights. This file is imported as an ms-swift external plugin by
the trainer and by both rollout servers.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import sys
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union
from uuid import uuid4

import aiohttp
import requests
import torch
from pydantic import BaseModel, Field

# ``swift rollout`` imports external plugin files by basename. Its console
# entrypoint does not guarantee that the project root is on sys.path.
_PROJECT_ROOT = Path(
    os.getenv("PROJECT_ROOT", str(Path(__file__).resolve().parents[1]))
).resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from moco.rubric import (
    NUM_GENERATIONS,
    extract_json_object,
    normalize_rubrics,
    weighted_reward,
)
from moco.judge import load_judge_prompt
from swift.infer_engine.protocol import RequestConfig, RolloutInferRequest, RolloutOutput
from swift.rewards.orm import ORM, orms
from swift.rollout.multi_turn import RolloutScheduler, multi_turns


SCHEDULER_NAME = "moco_caption"
REWARD_NAME = "moco_reward"
SUPPORTED_SWIFT_VERSION = "4.2.0.dev0"
SUPPORTED_VLLM_VERSION = "0.11.0"
DEFAULT_DECAY = 0.99
DEFAULT_UPDATE_INTERVAL = 1
RUBRIC_ROLLOUT_COUNT = 4
JUDGE_PROMPT = load_judge_prompt()
MOMENTUM_SCHEDULER_NAMES = {SCHEDULER_NAME}


def _env_float(name: str, default: float) -> float:
    value = float(os.getenv(name, str(default)))
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return value


def _momentum_update_interval() -> int:
    value = int(os.getenv("MOMENTUM_UPDATE_INTERVAL", str(DEFAULT_UPDATE_INTERVAL)))
    if value <= 0:
        raise ValueError(f"MOMENTUM_UPDATE_INTERVAL must be positive, got {value}")
    return value


def _expected_momentum_step(policy_step: int, interval: int | None = None) -> int:
    interval = interval or _momentum_update_interval()
    if policy_step < 0:
        return -1
    return policy_step - policy_step % interval


def _base_url(host: str, port: int) -> str:
    return f"http://{host}:{port}"


def _response_content(item: dict[str, Any]) -> str:
    if "response" in item:
        item = item["response"]
    return str(item["choices"][0]["message"]["content"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _completed_snapshot_state(
    directory: str | Path, expected_version: int | None = None
) -> dict[str, Any] | None:
    """Return state only after the writer's checksum commit marker is visible."""
    target = Path(directory)
    model_path = target / "ema_model.safetensors"
    state_path = target / "momentum_state.json"
    checksum_path = target / "sha256.json"
    if not all(path.is_file() for path in (model_path, state_path, checksum_path)):
        return None
    if model_path.stat().st_size <= 0:
        return None
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        checksums = json.loads(checksum_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    required = {"ema_model.safetensors", "momentum_state.json"}
    if not required.issubset(checksums):
        return None
    version = int(state["model_version"])
    if expected_version is not None and version != expected_version:
        raise RuntimeError(
            f"Momentum snapshot version {version} != expected {expected_version}"
        )
    return state


def validate_runtime_versions() -> None:
    """Fail closed when the monkey-patched runtime is not the audited runtime."""
    import swift
    import vllm

    swift_version = str(getattr(swift, "__version__", "unknown"))
    vllm_version = str(getattr(vllm, "__version__", "unknown"))
    model_type = os.getenv("MOMENTUM_MODEL_TYPE", "qwen3_vl")
    if swift_version != SUPPORTED_SWIFT_VERSION:
        raise RuntimeError(
            f"Momentum plugin requires swift {SUPPORTED_SWIFT_VERSION}, got {swift_version}"
        )
    if vllm_version != SUPPORTED_VLLM_VERSION:
        raise RuntimeError(
            f"Momentum plugin requires vllm {SUPPORTED_VLLM_VERSION}, got {vllm_version}"
        )
    if model_type != "qwen3_vl":
        raise RuntimeError(f"Momentum plugin supports only qwen3_vl, got {model_type}")


class MomentumTransaction:
    """Pure transaction bookkeeping used by the server worker and unit tests."""

    def __init__(self, version: int = -1) -> None:
        self.version = int(version)
        self.active: dict[str, Any] | None = None
        self.last_transaction_id = ""

    def begin(self, transaction_id: str, step: int, decay: float, bucket_count: int) -> None:
        if self.active is not None:
            raise RuntimeError("an EMA transaction is already active")
        if step <= self.version:
            raise RuntimeError(f"EMA step {step} must be greater than committed step {self.version}")
        if not transaction_id or bucket_count <= 0 or not 0.0 <= decay <= 1.0:
            raise ValueError("invalid EMA transaction metadata")
        self.active = {
            "transaction_id": transaction_id,
            "step": int(step),
            "decay": float(decay),
            "bucket_count": int(bucket_count),
            "next_bucket": 0,
            "failed": "",
            "scaled_parameter_ids": set(),
            "touched_parameter_names": set(),
        }

    def note_bucket(self, transaction_id: str, bucket_index: int) -> dict[str, Any]:
        tx = self._require(transaction_id)
        if tx["failed"]:
            raise RuntimeError(f"EMA transaction is failed: {tx['failed']}")
        if bucket_index != tx["next_bucket"]:
            raise RuntimeError(
                f"EMA bucket index {bucket_index} != expected {tx['next_bucket']}"
            )
        if bucket_index >= tx["bucket_count"]:
            raise RuntimeError("EMA bucket index exceeds declared bucket count")
        tx["next_bucket"] += 1
        return tx

    def fail(self, message: str) -> None:
        if self.active is not None:
            self.active["failed"] = str(message)

    def commit(self, transaction_id: str) -> dict[str, Any]:
        tx = self._require(transaction_id)
        if tx["failed"]:
            raise RuntimeError(f"EMA transaction is failed: {tx['failed']}")
        if tx["next_bucket"] != tx["bucket_count"]:
            raise RuntimeError(
                f"EMA received {tx['next_bucket']} buckets, expected {tx['bucket_count']}"
            )
        if not tx["touched_parameter_names"]:
            raise RuntimeError("EMA transaction touched no runtime parameters")
        self.version = tx["step"]
        self.last_transaction_id = tx["transaction_id"]
        result = dict(tx)
        result["scaled_parameter_ids"] = len(tx["scaled_parameter_ids"])
        result["touched_parameter_names"] = sorted(tx["touched_parameter_names"])
        self.active = None
        return result

    def abort(self, transaction_id: str, reason: str) -> None:
        tx = self._require(transaction_id)
        tx["failed"] = str(reason)

    def _require(self, transaction_id: str) -> dict[str, Any]:
        if self.active is None:
            raise RuntimeError("no active EMA transaction")
        if self.active["transaction_id"] != transaction_id:
            raise RuntimeError("EMA transaction id mismatch")
        return self.active


class MomentumRoleClient:
    """Async inference/status client used by the rollout-server scheduler."""

    def __init__(self, base_url: str, timeout: int = 600) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = int(timeout)

    async def infer(
        self, requests_: Sequence[RolloutInferRequest], config: RequestConfig
    ) -> list[str]:
        payload = {
            "infer_requests": [asdict(request) for request in requests_],
            "request_config": asdict(config),
            "metrics": None,
            "use_tqdm": False,
            "adapter_request": None,
        }
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(f"{self.base_url}/infer/", json=payload) as response:
                text = await response.text()
                if response.status != 200:
                    raise RuntimeError(
                        f"Momentum inference failed: HTTP {response.status}: {text[:1000]}"
                    )
                items = json.loads(text)
        if len(items) != len(requests_):
            raise RuntimeError(
                f"Momentum server returned {len(items)} responses for {len(requests_)} requests"
            )
        return [_response_content(item) for item in items]

    async def status(self) -> dict[str, Any]:
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(f"{self.base_url}/role_status/") as response:
                text = await response.text()
                if response.status != 200:
                    raise RuntimeError(
                        f"Role status failed: HTTP {response.status}: {text[:1000]}"
                    )
                return json.loads(text)


def render_four_rollout_prompt(
    template: str, data: dict[str, Any], policy_rollouts: Sequence[str]
) -> str:
    required_data = ("gpt_caption", "gemini_caption")
    missing = [key for key in required_data if not data.get(key)]
    if missing:
        raise ValueError(f"missing four-rollout rubric fields: {missing}")
    if len(policy_rollouts) != RUBRIC_ROLLOUT_COUNT:
        raise ValueError(
            f"expected {RUBRIC_ROLLOUT_COUNT} policy rollouts, got {len(policy_rollouts)}"
        )
    placeholders = ["__GPT_CAPTION__", "__GEMINI_CAPTION__"] + [
        f"__POLICY_ROLLOUT_{index}__"
        for index in range(1, RUBRIC_ROLLOUT_COUNT + 1)
    ]
    invalid = [token for token in placeholders if template.count(token) != 1]
    if invalid:
        raise ValueError(f"four-rollout prompt placeholder mismatch: {invalid}")
    prompt = template.replace("__GPT_CAPTION__", str(data["gpt_caption"]).strip())
    prompt = prompt.replace("__GEMINI_CAPTION__", str(data["gemini_caption"]).strip())
    for index, rollout in enumerate(policy_rollouts, 1):
        prompt = prompt.replace(f"__POLICY_ROLLOUT_{index}__", str(rollout).strip())
    unresolved = [token for token in placeholders if token in prompt]
    if unresolved:
        raise ValueError(f"unresolved four-rollout placeholders: {unresolved}")
    return "<image>\n" + prompt


class _RubricScheduler(RolloutScheduler):
    """Shared request routing, version checks, and Judge scoring."""

    JUDGE_PROMPT = JUDGE_PROMPT
    RUBRIC_PROMPT_ENV = "DYNAMIC_RUBRIC_PROMPT"
    RUBRIC_PROMPT_RELATIVE = Path(
        "prompts/generator.txt"
    )
    JUDGE_MAX_TOKENS_ENV = "DYNAMIC_RUBRIC_JUDGE_MAX_TOKENS"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        project_root = Path(
            os.getenv("PROJECT_ROOT", str(Path(__file__).resolve().parents[1]))
        )
        self.rubric_prompt_path = Path(
            os.getenv(
                self.RUBRIC_PROMPT_ENV,
                str(project_root / self.RUBRIC_PROMPT_RELATIVE),
            )
        ).resolve()
        self.rubric_template = self.rubric_prompt_path.read_text(encoding="utf-8")
        self.rubric_prompt_sha256 = hashlib.sha256(
            self.rubric_template.encode("utf-8")
        ).hexdigest()
        self.max_retries = int(os.getenv("DYNAMIC_RUBRIC_MAX_RETRIES", "3"))
        self.generator_max_tokens = int(
            os.getenv("DYNAMIC_RUBRIC_GENERATOR_MAX_TOKENS", "2048")
        )
        self.judge_max_tokens = int(
            os.getenv(
                self.JUDGE_MAX_TOKENS_ENV,
                str(self.JUDGE_PROMPT.max_output_tokens),
            )
        )
        momentum_url = os.getenv("MOMENTUM_SERVER_URL", "").strip()
        if not momentum_url:
            host = os.environ["MOMENTUM_SERVER_HOST"]
            port = int(os.getenv("MOMENTUM_SERVER_PORT", "8001"))
            momentum_url = _base_url(host, port)
        self.momentum_url = momentum_url.rstrip("/")
        self.momentum_client = MomentumRoleClient(
            self.momentum_url, int(os.getenv("MOMENTUM_SERVER_TIMEOUT", "600"))
        )
        rollout_url = os.getenv("ROLLOUT_SERVER_URL", "").strip()
        if not rollout_url:
            rollout_host = os.getenv("ROLLOUT_SERVER_HOST", "127.0.0.1")
            rollout_port = int(os.getenv("ROLLOUT_SERVER_PORT", "8000"))
            rollout_url = _base_url(rollout_host, rollout_port)
        self.rollout_url = rollout_url.rstrip("/")
        audit = os.getenv("MOMENTUM_AUDIT", "").strip()
        self.audit_path = Path(audit) if audit else None
        if self.audit_path:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self.audit_lock = asyncio.Lock()
        self.batch_id = 0

    @staticmethod
    def _request(value: RolloutInferRequest | dict[str, Any]) -> RolloutInferRequest:
        return value if isinstance(value, RolloutInferRequest) else RolloutInferRequest(**value)

    @staticmethod
    def _logical_groups(
        requests_: list[RolloutInferRequest],
    ) -> list[list[RolloutInferRequest]]:
        grouped: dict[tuple[str, str], list[RolloutInferRequest]] = defaultdict(list)
        for request in requests_:
            data = request.data_dict or {}
            uid = str(data.get("sample_uid", ""))
            prompt_id = str(data.get("prompt_id", ""))
            if not uid or not prompt_id:
                raise ValueError("momentum rollout requires sample_uid and prompt_id")
            grouped[(uid, prompt_id)].append(request)
        result = []
        for key, items in grouped.items():
            if len(items) % NUM_GENERATIONS:
                raise ValueError(
                    f"{key} produced {len(items)} requests, expected a multiple of {NUM_GENERATIONS}"
                )
            result.extend(
                items[index : index + NUM_GENERATIONS]
                for index in range(0, len(items), NUM_GENERATIONS)
            )
        return result

    async def _role_versions(self) -> dict[str, Any]:
        rollout_client = MomentumRoleClient(self.rollout_url, timeout=30)
        rollout_status, momentum_status = await asyncio.gather(
            rollout_client.status(), self.momentum_client.status()
        )
        if rollout_status.get("role") != "rollout":
            raise RuntimeError(f"unexpected rollout role status: {rollout_status}")
        if momentum_status.get("role") != "momentum":
            raise RuntimeError(f"unexpected momentum role status: {momentum_status}")
        rollout_version = int(rollout_status.get("model_version", -1))
        momentum_version = int(momentum_status.get("model_version", -1))
        interval = _momentum_update_interval()
        expected_momentum_version = _expected_momentum_step(rollout_version, interval)
        if rollout_version < 0 or momentum_version != expected_momentum_version:
            raise RuntimeError(
                "server version mismatch: "
                f"rollout={rollout_version}, momentum={momentum_version}, "
                f"expected_momentum={expected_momentum_version}, interval={interval}"
            )
        return {
            "policy_global_step": rollout_version,
            "rollout_model_version": rollout_version,
            "momentum_model_version": momentum_version,
            "momentum_decay": float(momentum_status["momentum_decay"]),
            "ema_transaction_id": str(momentum_status.get("last_transaction_id", "")),
            "generator_endpoint": self.momentum_url,
            "judge_endpoint": self.momentum_url,
        }


    @classmethod
    def _judge_request(
        cls, completion: str, rubrics: list[dict[str, Any]]
    ) -> RolloutInferRequest:
        return RolloutInferRequest(
            messages=[
                {"role": "system", "content": cls.JUDGE_PROMPT.system_prompt},
                {"role": "user", "content": cls.JUDGE_PROMPT.render_user(completion, rubrics)},
            ],
            images=[], audios=[], videos=[], tools=None, objects={}, data_dict={},
            uuid=f"judge-{uuid4().hex}",
        )

    @classmethod
    def _parse_judge_payload(cls, text: str, expected_items: int) -> dict[str, Any]:
        stripped = str(text).strip()
        if not stripped.startswith("{") or not stripped.endswith("}"):
            raise ValueError("Judge response must be a bare JSON object")
        payload = json.loads(stripped)
        return {"items": cls.JUDGE_PROMPT.normalize_items(payload, expected_items)}


    async def _judge_batch(
        self, completions: list[str], rubrics_by_completion: list[list[dict[str, Any]]]
    ) -> list[tuple[dict[str, Any], float, str | None, str]]:
        results: list[tuple[dict[str, Any], float, str | None, str] | None] = [None] * len(completions)
        pending = [i for i, rubrics in enumerate(rubrics_by_completion) if rubrics]
        for i, rubrics in enumerate(rubrics_by_completion):
            if not rubrics:
                results[i] = ({"items": []}, 0.0, "generator_failed", "")
        last_errors = [""] * len(completions)
        last_raw = [""] * len(completions)
        for _ in range(self.max_retries):
            if not pending:
                break
            requests_ = [self._judge_request(completions[i], rubrics_by_completion[i]) for i in pending]
            lengths = {len(rubrics_by_completion[i]) for i in pending}
            only_length = next(iter(lengths)) if len(lengths) == 1 else None
            exact_regex = self.JUDGE_PROMPT.response_regex(only_length)
            config = RequestConfig(
                max_tokens=self.judge_max_tokens,
                temperature=0.0,
                structured_outputs_regex=exact_regex,
            )
            try:
                raws = await self.momentum_client.infer(requests_, config)
            except Exception as exc:
                message = repr(exc)
                for index in pending:
                    last_errors[index] = message
                continue
            retry = []
            for index, raw in zip(pending, raws):
                last_raw[index] = raw
                try:
                    payload = self._parse_judge_payload(
                        raw, len(rubrics_by_completion[index])
                    )
                    results[index] = (
                        payload,
                        weighted_reward(payload, rubrics_by_completion[index]),
                        None,
                        raw,
                    )
                except Exception as exc:
                    last_errors[index] = repr(exc)
                    retry.append(index)
            pending = retry
        for index in pending:
            results[index] = (
                {"items": []}, 0.0,
                last_errors[index] or "judge_failed", last_raw[index],
            )
        return [result for result in results if result is not None]

    async def _write_audit(self, rows: Iterable[dict[str, Any]]) -> None:
        if not self.audit_path:
            return
        async with self.audit_lock:
            with self.audit_path.open("a", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")



class MomentumScheduler(_RubricScheduler):
    """Policy rollouts, candidate-conditioned rubric generation, then Judge scoring."""

    JUDGE_PROMPT = JUDGE_PROMPT
    RUBRIC_PROMPT_ENV = "MOMENTUM_4ROLLOUT_RUBRIC_PROMPT"
    RUBRIC_PROMPT_RELATIVE = Path(
        "prompts/generator.txt"
    )
    JUDGE_MAX_TOKENS_ENV = "MOMENTUM_4ROLLOUT_JUDGE_MAX_TOKENS"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rollout_sample_seed = int(
            os.getenv("MOMENTUM_4ROLLOUT_SAMPLE_SEED", "42")
        )

    @staticmethod
    def _generator_request(
        request: RolloutInferRequest,
        template: str,
        policy_rollouts: Sequence[str],
    ) -> RolloutInferRequest:
        prompt = render_four_rollout_prompt(
            template, request.data_dict or {}, policy_rollouts
        )
        return RolloutInferRequest(
            messages=[{"role": "user", "content": prompt}],
            images=deepcopy(request.images),
            audios=[], videos=[], tools=None, objects={}, data_dict={},
            uuid=f"generator-{uuid4().hex}",
        )

    def _sample_rollout_indices(
        self,
        group: Sequence[RolloutInferRequest],
        versions: dict[str, Any],
        group_index: int,
    ) -> tuple[list[int], int]:
        if len(group) < RUBRIC_ROLLOUT_COUNT:
            raise ValueError(
                f"cannot sample {RUBRIC_ROLLOUT_COUNT} rollouts from {len(group)}"
            )
        data = group[0].data_dict or {}
        seed_payload = {
            "base_seed": self.rollout_sample_seed,
            "rollout_model_version": int(versions["rollout_model_version"]),
            "sample_uid": str(data.get("sample_uid", "")),
            "prompt_id": str(data.get("prompt_id", "")),
            "group_index": group_index,
        }
        seed_bytes = hashlib.sha256(
            json.dumps(
                seed_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).digest()
        derived_seed = int.from_bytes(seed_bytes[:8], "big")
        indices = sorted(
            random.Random(derived_seed).sample(
                range(len(group)), RUBRIC_ROLLOUT_COUNT
            )
        )
        return indices, derived_seed

    async def _generate_rubrics_batch(
        self,
        groups: list[list[RolloutInferRequest]],
        policy_rollouts_by_group: Sequence[Sequence[str]],
    ) -> list[tuple[list[dict[str, Any]], str | None, str]]:
        if len(groups) != len(policy_rollouts_by_group):
            raise ValueError("groups and four-rollout rubric inputs must align")
        requests_ = [
            self._generator_request(group[0], self.rubric_template, policy_rollouts)
            for group, policy_rollouts in zip(groups, policy_rollouts_by_group)
        ]
        config = RequestConfig(max_tokens=self.generator_max_tokens, temperature=0.0)
        results: list[
            tuple[list[dict[str, Any]], str | None, str] | None
        ] = [None] * len(requests_)
        pending = list(range(len(requests_)))
        last_errors = [""] * len(requests_)
        last_raw = [""] * len(requests_)
        for _ in range(self.max_retries):
            if not pending:
                break
            try:
                raws = await self.momentum_client.infer(
                    [requests_[index] for index in pending], config
                )
            except Exception as exc:
                message = repr(exc)
                for index in pending:
                    last_errors[index] = message
                continue
            retry = []
            for index, raw in zip(pending, raws):
                last_raw[index] = raw
                try:
                    results[index] = (
                        normalize_rubrics(extract_json_object(raw)),
                        None,
                        raw,
                    )
                except Exception as exc:
                    last_errors[index] = repr(exc)
                    retry.append(index)
            pending = retry
        for index in pending:
            results[index] = (
                [],
                last_errors[index] or "generator_failed",
                last_raw[index],
            )
        return [result for result in results if result is not None]

    async def async_infer(
        self, infer_requests, request_config, *, use_tqdm=None, **kwargs
    ):
        del use_tqdm
        requests_ = [self._request(value) for value in infer_requests]
        groups = self._logical_groups(requests_)
        versions = await self._role_versions()
        current_batch = self.batch_id
        self.batch_id += 1

        group_by_uuid: dict[str, tuple[str, int]] = {}
        rollout_index_by_uuid: dict[str, int] = {}
        group_records: list[tuple[str, list[RolloutInferRequest]]] = []
        for group_index, group in enumerate(groups):
            uid = str((group[0].data_dict or {})["sample_uid"])
            group_id = f"batch-{current_batch}:group-{group_index}:{uid}"
            group_records.append((group_id, group))
            for rollout_index, request in enumerate(group):
                group_by_uuid[request.uuid] = (group_id, group_index)
                rollout_index_by_uuid[request.uuid] = rollout_index

        policy_responses = await asyncio.gather(
            *[
                self.infer_engine.infer_async(
                    deepcopy(request), request_config, **kwargs
                )
                for request in requests_
            ]
        )
        completion_by_uuid = {
            request.uuid: response.choices[0].message.content
            for request, response in zip(requests_, policy_responses)
        }

        policy_rollouts_by_group: list[list[str]] = []
        selection_by_group_id: dict[str, dict[str, Any]] = {}
        for group_index, (group_id, group) in enumerate(group_records):
            selected_indices, derived_seed = self._sample_rollout_indices(
                group, versions, group_index
            )
            selected_captions = [
                completion_by_uuid[group[index].uuid] for index in selected_indices
            ]
            policy_rollouts_by_group.append(selected_captions)
            selection_by_group_id[group_id] = {
                "rubric_input_rollout_indices": selected_indices,
                "rubric_input_rollouts": selected_captions,
                "rubric_input_unique_text_count": len(set(selected_captions)),
                "rubric_sample_seed": derived_seed,
            }

        generator_results = await self._generate_rubrics_batch(
            groups, policy_rollouts_by_group
        )
        rubric_by_uuid: dict[
            str, tuple[list[dict[str, Any]], str | None, str]
        ] = {}
        for group, generator_result in zip(groups, generator_results):
            for request in group:
                rubric_by_uuid[request.uuid] = generator_result

        completions = [completion_by_uuid[request.uuid] for request in requests_]
        rubrics_by_completion = [
            rubric_by_uuid[request.uuid][0] for request in requests_
        ]
        judge_results = await self._judge_batch(completions, rubrics_by_completion)

        outputs = []
        audits: dict[str, dict[str, Any]] = {}
        for request, response, judge_result in zip(
            requests_, policy_responses, judge_results
        ):
            response.id = request.uuid
            rubrics, generator_error, generator_raw = rubric_by_uuid[request.uuid]
            group_id, _ = group_by_uuid[request.uuid]
            rollout_index = rollout_index_by_uuid[request.uuid]
            selection = selection_by_group_id[group_id]
            payload, reward, judge_error, judge_raw = judge_result
            completion = completion_by_uuid[request.uuid]
            token_ids = list(response.choices[0].token_ids or [])
            messages = deepcopy(request.messages)
            messages.append({"role": "assistant", "content": completion})
            rollout_info = {
                "num_turns": 1,
                "sample_uid": (request.data_dict or {})["sample_uid"],
                "group_id": group_id,
                "rollout_index": rollout_index,
                "num_generations": NUM_GENERATIONS,
                "rubric_source": "policy_rollout_sample_4",
                "rubric_input_rollout_indices": selection[
                    "rubric_input_rollout_indices"
                ],
                "rubric_input_selected": rollout_index
                in selection["rubric_input_rollout_indices"],
                "rubric_prompt_path": str(self.rubric_prompt_path),
                "rubric_prompt_sha256": self.rubric_prompt_sha256,
                "rubrics": rubrics,
                "judge_payload": payload,
                "reward": reward,
                "judge_prompt_variant": self.JUDGE_PROMPT.variant,
                "judge_prompt_version": self.JUDGE_PROMPT.prompt_version,
                "judge_prompt_sha256": self.JUDGE_PROMPT.sha256,
                "generator_error": generator_error,
                "judge_error": judge_error,
                **versions,
            }
            outputs.append(
                RolloutOutput(
                    response=response,
                    messages=messages,
                    response_token_ids=[token_ids] if token_ids else [],
                    response_loss_mask=[[1] * len(token_ids)] if token_ids else [],
                    rollout_infos=rollout_info,
                )
            )
            audit = audits.setdefault(
                group_id,
                {
                    "server_batch_id": current_batch,
                    "group_id": group_id,
                    "sample_uid": (request.data_dict or {})["sample_uid"],
                    "rubric_source": "policy_rollout_sample_4",
                    "rubric_prompt_path": str(self.rubric_prompt_path),
                    "rubric_prompt_sha256": self.rubric_prompt_sha256,
                    **selection,
                    "rubrics": rubrics,
                    "judge_prompt_variant": self.JUDGE_PROMPT.variant,
                    "judge_prompt_version": self.JUDGE_PROMPT.prompt_version,
                    "judge_prompt_sha256": self.JUDGE_PROMPT.sha256,
                    "generator_error": generator_error,
                    "generator_raw": generator_raw,
                    **versions,
                    "rollouts": [],
                },
            )
            audit["rollouts"].append(
                {
                    "rollout_index": rollout_index,
                    "completion": completion,
                    "rubric_input_selected": rollout_index
                    in selection["rubric_input_rollout_indices"],
                    "judge_payload": payload,
                    "judge_error": judge_error,
                    "judge_raw": judge_raw,
                    "reward": reward,
                }
            )
        for audit in audits.values():
            audit["rollouts"].sort(key=lambda row: row["rollout_index"])
        await self._write_audit(audits.values())
        return outputs


class MomentumDynamicRubricReward(ORM):
    def __call__(self, completions, rollout_infos=None, **kwargs) -> list[float]:
        del kwargs
        if not isinstance(rollout_infos, list) or len(rollout_infos) != len(completions):
            raise ValueError("rollout_infos must align with completions")
        rewards = []
        for info in rollout_infos:
            if not isinstance(info, dict) or "reward" not in info:
                raise ValueError("missing precomputed momentum rubric reward")
            rewards.append(float(info["reward"]))
        return rewards


multi_turns[SCHEDULER_NAME] = MomentumScheduler
orms[REWARD_NAME] = MomentumDynamicRubricReward


class BeginEMARequest(BaseModel):
    transaction_id: str
    step: int
    decay: float
    bucket_count: int


class EMABucketRequest(BaseModel):
    transaction_id: str
    bucket_index: int
    metadatas: list[dict[str, Any]] = Field(default_factory=list)


class CommitEMARequest(BaseModel):
    transaction_id: str


class AbortEMARequest(BaseModel):
    transaction_id: str
    reason: str


class SetVersionRequest(BaseModel):
    step: int


class SnapshotRequest(BaseModel):
    directory: str
    policy_model_path: str = ""


def _named_parameters_no_duplicates(model) -> dict[str, torch.nn.Parameter]:
    try:
        iterator = model.named_parameters(remove_duplicate=False)
    except TypeError:
        iterator = model.named_parameters()
    result: dict[str, torch.nn.Parameter] = {}
    seen: set[int] = set()
    for name, parameter in iterator:
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        result[name] = parameter
    return result


class _SingleDPGroup:
    is_first_rank = True

    @staticmethod
    def broadcast(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        del src
        return tensor


def _get_dp_group_or_singleton():
    from vllm.distributed.parallel_state import get_dp_group

    try:
        return get_dp_group()
    except AssertionError:
        # Unit tests and single-process static validation do not initialize
        # vLLM distributed state.
        return _SingleDPGroup()


def _get_fp32_master(worker, model) -> dict[str, torch.Tensor]:
    """Return the single CPU FP32 EMA master owned by DP rank zero."""
    dp_group = _get_dp_group_or_singleton()
    if not dp_group.is_first_rank:
        return {}
    master = getattr(worker, "_momentum_fp32_master", None)
    if master is None:
        master = {
            name: parameter.detach().to(device="cpu", dtype=torch.float32).contiguous()
            for name, parameter in _named_parameters_no_duplicates(model).items()
        }
        worker._momentum_fp32_master = master
    return master


def _ema_load_weights(
    model,
    named_weights: Sequence[tuple[str, torch.Tensor]],
    decay: float,
    scaled_parameter_ids: set[int],
    touched_parameter_names: set[str],
    worker=None,
) -> None:
    """Update one CPU FP32 master, then broadcast one BF16 runtime copy to DPs."""
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader
    parameters = _named_parameters_no_duplicates(model)
    dp_group = _get_dp_group_or_singleton()
    is_writer = dp_group.is_first_rank
    master_owner = worker if worker is not None else model
    fp32_master = _get_fp32_master(master_owner, model)
    originals: list[tuple[torch.nn.Parameter, bool, Any]] = []
    name_by_id = {id(parameter): name for name, parameter in parameters.items()}
    bucket_parameter_names: set[str] = set()

    for parameter in parameters.values():
        had_loader = hasattr(parameter, "weight_loader")
        original_loader = getattr(parameter, "weight_loader", default_weight_loader)

        def ema_loader(param, loaded_weight, *args, _original=original_loader, **kwargs):
            parameter_id = id(param)
            runtime_name = name_by_id.get(parameter_id, f"parameter-{parameter_id}")
            with torch.no_grad():
                live_data = param.data
                # Every DP executes the loader only to obtain the exact vLLM
                # packed-parameter mapping. Only DP0 performs EMA arithmetic.
                contribution = torch.zeros_like(
                    live_data,
                    device="cpu" if is_writer else live_data.device,
                    dtype=torch.float32 if is_writer else live_data.dtype,
                )
                try:
                    param.data = contribution
                    incoming = loaded_weight.to(
                        device=contribution.device, dtype=contribution.dtype
                    )
                    _original(param, incoming, *args, **kwargs)
                finally:
                    param.data = live_data
                if is_writer:
                    master = fp32_master[runtime_name]
                    if parameter_id not in scaled_parameter_ids:
                        master.mul_(decay)
                    master.add_(contribution, alpha=1.0 - decay)
                scaled_parameter_ids.add(parameter_id)
            touched_parameter_names.add(runtime_name)
            bucket_parameter_names.add(runtime_name)

        originals.append((parameter, had_loader, original_loader))
        parameter.weight_loader = ema_loader

    try:
        loaded = model.load_weights(weights=list(named_weights))
        if loaded is not None and not isinstance(loaded, set):
            raise RuntimeError(f"unexpected vLLM load_weights result: {type(loaded)!r}")
    finally:
        for parameter, had_loader, original_loader in originals:
            if had_loader:
                parameter.weight_loader = original_loader
            else:
                delattr(parameter, "weight_loader")

    # Use runtime parameter names here: the vLLM loader has already performed
    # HF-to-packed mapping once. DP0 converts each updated FP32 tensor exactly
    # once; all other DPs receive that BF16 tensor without repeating EMA.
    with torch.no_grad():
        for name in sorted(bucket_parameter_names):
            parameter = parameters[name]
            if is_writer:
                parameter.copy_(fp32_master[name].to(parameter.device, parameter.dtype))
            dp_group.broadcast(parameter.data, src=0)

def _worker_book(worker) -> MomentumTransaction:
    book = getattr(worker, "_momentum_transaction_book", None)
    if book is None:
        book = MomentumTransaction()
        worker._momentum_transaction_book = book
        worker._momentum_decay = _env_float("MOMENTUM_DECAY", DEFAULT_DECAY)
    return book


def _patch_weight_sync_worker() -> None:
    from swift.pipelines.infer.rollout import WeightSyncWorkerExtension
    from swift.rlhf_trainers.utils import FlattenedTensorBucket, FlattenedTensorMetadata
    from swift.utils import get_torch_device, synchronize

    if getattr(WeightSyncWorkerExtension, "_momentum_grpo_patched", False):
        if "MomentumWeightSyncWorkerExtension" not in globals():
            worker_cls = type(
                "MomentumWeightSyncWorkerExtension",
                (WeightSyncWorkerExtension,),
                {"__module__": __name__},
            )
            globals()["MomentumWeightSyncWorkerExtension"] = worker_cls
        return

    def momentum_begin(self, transaction_id: str, step: int, decay: float, bucket_count: int):
        book = _worker_book(self)
        book.begin(transaction_id, step, decay, bucket_count)
        self._momentum_decay = float(decay)
        return {"ok": True, "step": step, "bucket_count": bucket_count}

    def momentum_update_bucket(
        self,
        transaction_id: str,
        bucket_index: int,
        metadatas: list[dict[str, Any]],
    ):
        book = _worker_book(self)
        try:
            if self.communicator is None:
                raise RuntimeError("weight communicator is not initialized")
            metadata_objects = [FlattenedTensorMetadata(**metadata) for metadata in metadatas]
            if not metadata_objects:
                raise RuntimeError("EMA bucket has no metadata")
            total_bytes = metadata_objects[-1].end_idx
            flat = torch.empty(total_bytes, dtype=torch.uint8, device=self.communicator.device)
            self.communicator.broadcast(
                flat,
                src=self.client_rank,
                stream=getattr(get_torch_device(), "current_stream", lambda: None)(),
            )
            synchronize()
            self.communicator.group.barrier()
            # Validate only after receiving the broadcast. Rejecting before the
            # receive would leave the trainer-side NCCL broadcast hanging.
            tx = book.note_bucket(transaction_id, bucket_index)
            named = FlattenedTensorBucket(
                metadata=metadata_objects, flattened_tensor=flat
            ).reconstruct_tensors()
            _ema_load_weights(
                self.model_runner.model,
                list(named.items()),
                tx["decay"],
                tx["scaled_parameter_ids"],
                tx["touched_parameter_names"],
                worker=self,
            )
            del named, flat
            return {"ok": True, "bucket_index": bucket_index}
        except Exception as exc:
            book.fail(repr(exc))
            raise

    def momentum_commit(self, transaction_id: str):
        try:
            result = _worker_book(self).commit(transaction_id)
            return {
                "ok": True,
                "step": result["step"],
                "transaction_id": result["transaction_id"],
                "scaled_parameter_count": result["scaled_parameter_ids"],
                "touched_parameter_count": len(result["touched_parameter_names"]),
            }
        except Exception as exc:
            _worker_book(self).fail(repr(exc))
            raise

    def momentum_abort(self, transaction_id: str, reason: str):
        _worker_book(self).abort(transaction_id, reason)
        return {"ok": True, "aborted": transaction_id, "reason": reason}

    def momentum_save_snapshot(self, directory: str, policy_model_path: str):
        from safetensors.torch import save_file

        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        if not _get_dp_group_or_singleton().is_first_rank:
            return {"ok": True, "writer": False}
        if (target / "sha256.json").is_file():
            return {"ok": True, "writer": False, "already_complete": True}
        lock_path = target / ".snapshot-writer.lock"
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return {"ok": True, "writer": False}
        os.close(descriptor)
        try:
            book = _worker_book(self)
            if book.active is not None:
                raise RuntimeError("cannot snapshot an active EMA transaction")
            parameters = _get_fp32_master(self, self.model_runner.model)
            model_path = target / "ema_model.safetensors"
            save_file(parameters, str(model_path))
            checksum = _sha256(model_path)
            state = {
                "model_version": book.version,
                "policy_step": book.version,
                "momentum_step": book.version,
                "momentum_decay": float(self._momentum_decay),
                "last_transaction_id": book.last_transaction_id,
                "policy_model_path": policy_model_path,
                "swift_version": SUPPORTED_SWIFT_VERSION,
                "vllm_version": SUPPORTED_VLLM_VERSION,
                "model_type": "qwen3_vl",
                "master_dtype": "float32",
                "runtime_dtype": "bfloat16",
                "fp32_master_dp_rank": 0,
                "parameter_count": len(parameters),
                "parameters": {
                    name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
                    for name, tensor in parameters.items()
                },
            }
            (target / "momentum_state.json").write_text(
                json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            state_checksum = _sha256(target / "momentum_state.json")
            (target / "sha256.json").write_text(
                json.dumps(
                    {
                        "ema_model.safetensors": checksum,
                        "momentum_state.json": state_checksum,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            return {
                "ok": True,
                "writer": True,
                "model_version": book.version,
                "sha256": checksum,
                "parameter_count": len(parameters),
            }
        finally:
            lock_path.unlink(missing_ok=True)

    def momentum_load_snapshot(self, directory: str):
        from safetensors.torch import load_file

        target = Path(directory)
        model_path = target / "ema_model.safetensors"
        state_path = target / "momentum_state.json"
        checksum_path = target / "sha256.json"
        if not model_path.is_file() or not state_path.is_file() or not checksum_path.is_file():
            raise FileNotFoundError(f"incomplete momentum snapshot: {target}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        checksums = json.loads(checksum_path.read_text(encoding="utf-8"))
        actual = _sha256(model_path)
        if checksums.get("ema_model.safetensors") != actual:
            raise RuntimeError("momentum snapshot SHA-256 mismatch")
        if checksums.get("momentum_state.json") != _sha256(state_path):
            raise RuntimeError("momentum state SHA-256 mismatch")
        if state.get("swift_version") != SUPPORTED_SWIFT_VERSION:
            raise RuntimeError("momentum snapshot swift version mismatch")
        if state.get("vllm_version") != SUPPORTED_VLLM_VERSION:
            raise RuntimeError("momentum snapshot vLLM version mismatch")
        if state.get("model_type") != "qwen3_vl":
            raise RuntimeError("momentum snapshot model type mismatch")
        version = int(state["model_version"])
        if int(state.get("policy_step", version)) != version:
            raise RuntimeError("momentum snapshot policy step mismatch")
        if int(state.get("momentum_step", version)) != version:
            raise RuntimeError("momentum snapshot momentum step mismatch")
        snapshot_decay = float(state["momentum_decay"])
        expected_decay = _env_float("MOMENTUM_DECAY", DEFAULT_DECAY)
        if abs(snapshot_decay - expected_decay) > 1e-12:
            raise RuntimeError(
                f"momentum snapshot decay {snapshot_decay} != server decay {expected_decay}"
            )
        dp_group = _get_dp_group_or_singleton()
        loaded = load_file(str(model_path), device="cpu") if dp_group.is_first_rank else {}
        runtime = _named_parameters_no_duplicates(self.model_runner.model)
        if dp_group.is_first_rank and set(loaded) != set(runtime):
            missing = sorted(set(runtime) - set(loaded))[:20]
            extra = sorted(set(loaded) - set(runtime))[:20]
            raise RuntimeError(
                f"momentum snapshot parameter mismatch: missing={missing}, extra={extra}"
            )
        with torch.no_grad():
            for name, parameter in runtime.items():
                if dp_group.is_first_rank:
                    tensor = loaded[name]
                    if tuple(tensor.shape) != tuple(parameter.shape):
                        raise RuntimeError(f"shape mismatch for {name}")
                    parameter.copy_(
                        tensor.to(device=parameter.device, dtype=parameter.dtype)
                    )
                dp_group.broadcast(parameter.data, src=0)
        if dp_group.is_first_rank:
            self._momentum_fp32_master = {
                name: tensor.to(dtype=torch.float32).contiguous()
                for name, tensor in loaded.items()
            }
        book = MomentumTransaction(version=version)
        book.last_transaction_id = str(state.get("last_transaction_id", ""))
        self._momentum_transaction_book = book
        self._momentum_decay = snapshot_decay
        return {
            "ok": True,
            "model_version": book.version,
            "sha256": actual,
            "parameter_count": len(runtime),
        }

    WeightSyncWorkerExtension.momentum_begin = momentum_begin
    WeightSyncWorkerExtension.momentum_update_bucket = momentum_update_bucket
    WeightSyncWorkerExtension.momentum_commit = momentum_commit
    WeightSyncWorkerExtension.momentum_abort = momentum_abort
    WeightSyncWorkerExtension.momentum_save_snapshot = momentum_save_snapshot
    WeightSyncWorkerExtension.momentum_load_snapshot = momentum_load_snapshot
    WeightSyncWorkerExtension._momentum_grpo_patched = True

    # vLLM imports worker_extension_cls by its fully-qualified string inside
    # newly spawned worker processes.  A parent-process monkeypatch alone is
    # therefore insufficient: expose an importable plugin-owned class and make
    # the Momentum engine request this class explicitly.
    worker_cls = type(
        "MomentumWeightSyncWorkerExtension",
        (WeightSyncWorkerExtension,),
        {"__module__": __name__},
    )
    globals()["MomentumWeightSyncWorkerExtension"] = worker_cls


def _flatten_outputs_ok(outputs: Any) -> bool:
    if isinstance(outputs, list):
        return bool(outputs) and all(_flatten_outputs_ok(item) for item in outputs)
    if isinstance(outputs, dict):
        return bool(outputs.get("ok", False))
    return outputs is not None


def _flatten_dict_outputs(outputs: Any) -> list[dict[str, Any]]:
    if isinstance(outputs, list):
        result: list[dict[str, Any]] = []
        for item in outputs:
            result.extend(_flatten_dict_outputs(item))
        return result
    if isinstance(outputs, dict):
        return [outputs]
    return []


def _chunk_complete_rollout_groups(
    infer_requests: Sequence[RolloutInferRequest | dict[str, Any]],
    num_connections: int,
) -> list[list[RolloutInferRequest | dict[str, Any]]]:
    """Split server requests without breaking a logical GRPO group.

    Swift's default rollout endpoint chunks individual requests across DP
    workers.  With 16 generations and DP=8 that gives each scheduler two
    requests, making it impossible to construct the required group-wise
    reward.  Keep contiguous groups intact and distribute groups instead.
    """
    if num_connections <= 0:
        raise ValueError("num_connections must be positive")
    requests_ = list(infer_requests)
    if len(requests_) % NUM_GENERATIONS:
        raise ValueError(
            f"received {len(requests_)} rollout requests; expected a multiple "
            f"of {NUM_GENERATIONS}"
        )
    groups = [
        requests_[index : index + NUM_GENERATIONS]
        for index in range(0, len(requests_), NUM_GENERATIONS)
    ]
    for group_index, group in enumerate(groups):
        keys = set()
        for request in group:
            data = (
                request.data_dict
                if isinstance(request, RolloutInferRequest)
                else request.get("data_dict", {})
            ) or {}
            keys.add((str(data.get("sample_uid", "")), str(data.get("prompt_id", ""))))
        if len(keys) != 1 or not next(iter(keys))[0] or not next(iter(keys))[1]:
            raise ValueError(
                f"rollout group {group_index} is not one complete sample_uid/prompt_id group: {keys}"
            )

    base, remainder = divmod(len(groups), num_connections)
    chunks: list[list[RolloutInferRequest | dict[str, Any]]] = []
    cursor = 0
    for connection_index in range(num_connections):
        count = base + (1 if connection_index < remainder else 0)
        assigned = groups[cursor : cursor + count]
        chunks.append([request for group in assigned for request in group])
        cursor += count
    return chunks


def _patch_rollout_server_app() -> None:
    import swift.pipelines.infer.rollout as rollout_module

    SwiftRolloutDeploy = rollout_module.SwiftRolloutDeploy

    if getattr(SwiftRolloutDeploy, "_momentum_grpo_patched", False):
        return
    original_register = SwiftRolloutDeploy._register_rl_rollout_app
    original_get_infer_engine = SwiftRolloutDeploy.get_infer_engine
    original_infer = SwiftRolloutDeploy.infer

    @staticmethod
    def get_infer_engine(args, template=None, **kwargs):
        if os.getenv("MOMENTUM_PROCESS_ROLE", "").strip() != "momentum":
            return original_get_infer_engine(args, template=template, **kwargs)

        kwargs.update(
            {
                "model_id_or_path": args.model,
                "model_type": args.model_type,
                "revision": args.model_revision,
                "torch_dtype": args.torch_dtype,
                "template": template,
                "use_async_engine": args.vllm_use_async_engine,
                "max_lora_rank": args.vllm_max_lora_rank,
            }
        )
        infer_backend = kwargs.pop("infer_backend", None) or args.infer_backend
        if infer_backend != "vllm":
            rollout_module.logger.info(
                "Currently, rollout only supports the vLLM backend. Set vLLM backend"
            )
        kwargs.update(args.get_vllm_engine_kwargs())
        kwargs.update({"enable_lora": args.vllm_enable_lora})
        kwargs["logprobs_mode"] = (
            "processed_logprobs"
            if rollout_module.check_vllm_version_ge("0.10.2")
            else None
        )
        engine_kwargs = kwargs.get("engine_kwargs", {})
        engine_kwargs.update(
            {
                "worker_extension_cls": (
                    f"{__name__}.MomentumWeightSyncWorkerExtension"
                )
            }
        )
        kwargs["load_format"] = engine_kwargs.pop("load_format", "auto")
        if args.vllm_use_async_engine and args.vllm_data_parallel_size > 1:
            engine_kwargs["data_parallel_size"] = args.vllm_data_parallel_size
        kwargs["engine_kwargs"] = engine_kwargs
        return rollout_module.GRPOVllmEngine(**kwargs)

    async def infer(
        self,
        infer_requests: List[Union[Dict, RolloutInferRequest]],
        request_config: Optional[RequestConfig] = None,
        *,
        use_tqdm: Optional[bool] = None,
    ):
        role = os.getenv("MOMENTUM_PROCESS_ROLE", "").strip()
        scheduler = str(getattr(self.args, "multi_turn_scheduler", "") or "")
        if role == "momentum":
            state = getattr(self, "_momentum_parent_state", {})
            if state.get("failed"):
                raise RuntimeError(
                    f"Momentum server is fail-closed: {state['failed']}"
                )
            if state.get("active_transaction_id"):
                raise RuntimeError("Momentum inference is blocked during an EMA transaction")
        if role != "rollout" or scheduler not in MOMENTUM_SCHEDULER_NAMES:
            return await original_infer(
                self,
                infer_requests,
                request_config,
                use_tqdm=use_tqdm,
            )

        chunks = _chunk_complete_rollout_groups(infer_requests, self.num_connections)
        for connection, requests_ in zip(self.connections, chunks):
            # Preserve Swift's DP collective participation contract for idle
            # workers, but discard their placeholder result below.
            worker_requests = requests_ or [
                RolloutInferRequest(messages=[{"role": "user", "content": "<placeholder>"}])
            ]
            connection.send(
                {
                    "type": "call",
                    "method": "async_infer" if self.use_async_engine else "infer",
                    "kwargs": {
                        "infer_requests": worker_requests,
                        "request_config": request_config,
                        "use_tqdm": use_tqdm,
                    },
                }
            )

        # A scheduler worker calls this same server's /role_status/ endpoint.
        # Blocking recv() here would freeze uvicorn's event loop and deadlock
        # that self-request until its 30-second timeout.  Wait for all pipes in
        # executor threads so the HTTP event loop remains responsive.
        loop = asyncio.get_running_loop()
        worker_outputs = await asyncio.gather(
            *[loop.run_in_executor(None, connection.recv) for connection in self.connections]
        )
        outputs = []
        for worker_index, (worker_output, requests_) in enumerate(zip(worker_outputs, chunks)):
            if not requests_:
                continue
            if worker_output is None:
                raise RuntimeError(
                    f"rollout DP worker {worker_index} returned no output; inspect that "
                    "worker's 'Method execution failed: async_infer' traceback"
                )
            if len(worker_output) != len(requests_):
                raise RuntimeError(
                    f"rollout DP worker {worker_index} returned {len(worker_output)} "
                    f"outputs for {len(requests_)} requests"
                )
            outputs.extend(worker_output)
        return outputs

    def register(self):
        original_register(self)
        role = os.getenv("MOMENTUM_PROCESS_ROLE", "").strip()
        self._momentum_parent_state = {
            "role": role,
            "model_version": -1,
            "momentum_decay": _env_float("MOMENTUM_DECAY", DEFAULT_DECAY),
            "last_transaction_id": "",
            "active_transaction_id": "",
            "active_step": -1,
            "failed": "",
            "model_path": str(getattr(self.args, "model", "")),
            "swift_version": SUPPORTED_SWIFT_VERSION,
            "vllm_version": SUPPORTED_VLLM_VERSION,
            "runtime_dtype": "bfloat16",
            "ema_compute_dtype": "float32",
            "ema_master_device": "cpu",
            "ema_master_count": 1,
            "ema_distribution_dtype": "bfloat16",
            "ema_writer_dp_rank": 0,
            "ema_dp_worker_count": int(getattr(self.args, "vllm_data_parallel_size", 1)),
            "momentum_update_interval": int(
                os.getenv("MOMENTUM_UPDATE_INTERVAL", str(DEFAULT_UPDATE_INTERVAL))
            ),
        }

        async def collective_call(method: str, args: tuple[Any, ...]):
            kwargs = {"method": method, "args": args}
            for connection in self.connections:
                connection.send(
                    {"type": "call", "method": "collective_rpc", "kwargs": kwargs}
                )
            outputs = [connection.recv() for connection in self.connections]
            if not _flatten_outputs_ok(outputs):
                raise RuntimeError(f"Momentum worker call failed: {method}: {outputs}")
            return outputs

        async def role_status():
            return dict(self._momentum_parent_state)

        async def set_model_version(request: SetVersionRequest):
            if role != "rollout":
                raise RuntimeError("set_model_version is valid only on the rollout server")
            current = int(self._momentum_parent_state["model_version"])
            if request.step < current:
                raise RuntimeError(f"rollout version rollback: {request.step} < {current}")
            self._momentum_parent_state["model_version"] = int(request.step)
            return dict(self._momentum_parent_state)

        async def begin_ema_update(request: BeginEMARequest):
            if role != "momentum":
                raise RuntimeError("EMA updates are valid only on the momentum server")
            await collective_call(
                "momentum_begin",
                (request.transaction_id, request.step, request.decay, request.bucket_count),
            )
            self._momentum_parent_state["failed"] = ""
            self._momentum_parent_state["active_transaction_id"] = request.transaction_id
            self._momentum_parent_state["active_step"] = int(request.step)
            return {"ok": True, **request.model_dump()}

        async def update_ema_bucket(request: EMABucketRequest):
            if role != "momentum":
                raise RuntimeError("EMA updates are valid only on the momentum server")
            if not request.metadatas:
                raise RuntimeError("EMA bucket metadata must not be empty")
            kwargs = {
                "method": "momentum_update_bucket",
                "args": (
                    request.transaction_id,
                    request.bucket_index,
                    request.metadatas,
                ),
            }
            for connection in self.connections:
                connection.send(
                    {"type": "fire_and_forget", "method": "collective_rpc", "kwargs": kwargs}
                )
            return {"ok": True, "bucket_index": request.bucket_index}

        async def commit_ema_update(request: CommitEMARequest):
            if role != "momentum":
                raise RuntimeError("EMA updates are valid only on the momentum server")
            try:
                if self._momentum_parent_state["active_transaction_id"] != request.transaction_id:
                    raise RuntimeError("Momentum parent transaction id mismatch")
                expected_step = int(self._momentum_parent_state["active_step"])
                outputs = await collective_call("momentum_commit", (request.transaction_id,))
                worker_results = _flatten_dict_outputs(outputs)
                if len(worker_results) != len(self.connections):
                    raise RuntimeError(
                        f"Momentum commit returned {len(worker_results)} worker results, "
                        f"expected {len(self.connections)}"
                    )
                integer_fields = (
                    "step",
                    "scaled_parameter_count",
                    "touched_parameter_count",
                )
                for field in integer_fields:
                    values = {int(item[field]) for item in worker_results}
                    if len(values) != 1:
                        raise RuntimeError(f"Momentum workers disagree on {field}: {values}")
                if int(worker_results[0]["step"]) != expected_step:
                    raise RuntimeError(
                        f"Momentum committed step {worker_results[0]['step']} != {expected_step}"
                    )
                result = worker_results[0]
                self._momentum_parent_state["model_version"] = int(result["step"])
                self._momentum_parent_state["last_transaction_id"] = request.transaction_id
                self._momentum_parent_state["active_transaction_id"] = ""
                self._momentum_parent_state["active_step"] = -1
                await self.reset_prefix_cache()
                return {"ok": True, **self._momentum_parent_state}
            except Exception as exc:
                self._momentum_parent_state["failed"] = repr(exc)
                raise

        async def abort_ema_update(request: AbortEMARequest):
            await collective_call(
                "momentum_abort", (request.transaction_id, request.reason)
            )
            self._momentum_parent_state["failed"] = request.reason
            return {"ok": True, "failed": request.reason}

        async def save_momentum_snapshot(request: SnapshotRequest):
            if role != "momentum":
                raise RuntimeError("snapshots are valid only on the momentum server")
            outputs = await collective_call(
                "momentum_save_snapshot", (request.directory, request.policy_model_path)
            )
            expected_version = int(self._momentum_parent_state["model_version"])
            timeout = float(os.getenv("MOMENTUM_SNAPSHOT_TIMEOUT", "600"))
            deadline = asyncio.get_running_loop().time() + timeout
            while True:
                state = _completed_snapshot_state(
                    request.directory, expected_version=expected_version
                )
                if state is not None:
                    return {
                        "ok": True,
                        "outputs": outputs,
                        "model_version": expected_version,
                        "completion_marker": "sha256.json",
                    }
                if asyncio.get_running_loop().time() >= deadline:
                    worker_results = _flatten_dict_outputs(outputs)
                    writer_count = sum(
                        bool(item.get("writer")) for item in worker_results
                    )
                    raise RuntimeError(
                        "Momentum snapshot completion marker timed out: "
                        f"directory={request.directory}, writers_seen={writer_count}"
                    )
                await asyncio.sleep(1.0)

        async def load_momentum_snapshot(request: SnapshotRequest):
            if role != "momentum":
                raise RuntimeError("snapshots are valid only on the momentum server")
            await collective_call("momentum_load_snapshot", (request.directory,))
            state = json.loads(
                (Path(request.directory) / "momentum_state.json").read_text(encoding="utf-8")
            )
            self._momentum_parent_state["model_version"] = int(state["model_version"])
            self._momentum_parent_state["momentum_decay"] = float(state["momentum_decay"])
            self._momentum_parent_state["last_transaction_id"] = str(
                state.get("last_transaction_id", "")
            )
            self._momentum_parent_state["failed"] = ""
            await self.reset_prefix_cache()
            return {"ok": True, **self._momentum_parent_state}

        self.app.get("/role_status/")(role_status)
        self.app.post("/set_model_version/")(set_model_version)
        self.app.post("/begin_ema_update/")(begin_ema_update)
        self.app.post("/update_ema_bucket/")(update_ema_bucket)
        self.app.post("/commit_ema_update/")(commit_ema_update)
        self.app.post("/abort_ema_update/")(abort_ema_update)
        self.app.post("/save_momentum_snapshot/")(save_momentum_snapshot)
        self.app.post("/load_momentum_snapshot/")(load_momentum_snapshot)

    SwiftRolloutDeploy.get_infer_engine = get_infer_engine
    SwiftRolloutDeploy.infer = infer
    SwiftRolloutDeploy._register_rl_rollout_app = register
    SwiftRolloutDeploy._momentum_grpo_patched = True


def _patch_trainer() -> None:
    from swift.rlhf_trainers.grpo_trainer import GRPOTrainer
    from swift.rlhf_trainers.rollout_mixin import RolloutTrainerMixin
    from swift.rlhf_trainers.utils import (
        FlattenedTensorBucket,
        _create_parameter_buckets,
    )
    from swift.rlhf_trainers.vllm_client import VLLMClient
    from swift.utils import get_current_device, get_torch_device, synchronize
    from transformers import TrainerCallback

    if getattr(RolloutTrainerMixin, "_momentum_grpo_patched", False):
        return

    class MomentumWeightClient(VLLMClient):
        def _json(self, method: str, path: str, payload: dict[str, Any] | None = None):
            url = f"{self.base_urls[0]}{path}"
            response = self.sessions[0].request(method, url, json=payload, timeout=600)
            if response.status_code != 200:
                raise RuntimeError(
                    f"Momentum server {path} failed: HTTP {response.status_code}: "
                    f"{response.text[:2000]}"
                )
            data = response.json()
            if isinstance(data, dict) and data.get("ok") is False:
                raise RuntimeError(f"Momentum server rejected {path}: {data}")
            return data

        def status(self):
            return self._json("GET", "/role_status/")

        def begin(self, transaction_id: str, step: int, decay: float, bucket_count: int):
            return self._json(
                "POST",
                "/begin_ema_update/",
                {
                    "transaction_id": transaction_id,
                    "step": int(step),
                    "decay": float(decay),
                    "bucket_count": int(bucket_count),
                },
            )

        def update_bucket(
            self,
            transaction_id: str,
            bucket_index: int,
            metadatas,
            flat_tensor: torch.Tensor,
        ):
            metadata_dicts = [
                item.model_dump() if hasattr(item, "model_dump") else item.dict()
                for item in metadatas
            ]
            self._json(
                "POST",
                "/update_ema_bucket/",
                {
                    "transaction_id": transaction_id,
                    "bucket_index": int(bucket_index),
                    "metadatas": metadata_dicts,
                },
            )
            synchronize()
            self.pynccl_comms[0].broadcast(
                flat_tensor,
                src=self.pynccl_comms[0].rank,
                stream=getattr(get_torch_device(), "current_stream", lambda: None)(),
            )
            synchronize()
            self.pynccl_comms[0].group.barrier()

        def commit(self, transaction_id: str):
            return self._json(
                "POST", "/commit_ema_update/", {"transaction_id": transaction_id}
            )

        def abort(self, transaction_id: str, reason: str):
            return self._json(
                "POST",
                "/abort_ema_update/",
                {"transaction_id": transaction_id, "reason": reason},
            )

        def save_snapshot(self, directory: str, policy_model_path: str):
            attempts = int(os.getenv("MOMENTUM_SNAPSHOT_RETRIES", "3"))
            if attempts <= 0:
                raise ValueError("MOMENTUM_SNAPSHOT_RETRIES must be positive")
            last_error = None
            for attempt in range(attempts):
                try:
                    return self._json(
                        "POST",
                        "/save_momentum_snapshot/",
                        {"directory": directory, "policy_model_path": policy_model_path},
                    )
                except RuntimeError as exc:
                    last_error = exc
                    if attempt + 1 < attempts:
                        time.sleep(5)
            raise RuntimeError(
                f"Momentum snapshot failed after {attempts} attempts: {last_error}"
            ) from last_error

        def load_snapshot(self, directory: str):
            return self._json(
                "POST",
                "/load_momentum_snapshot/",
                {"directory": directory, "policy_model_path": ""},
            )

    def estimate_bucket_count(trainer) -> int:
        bucket_bytes = int(os.getenv("SWIFT_UPDATE_WEIGHTS_BUCKET_SIZE", "512")) * 1024 * 1024
        named_parameters = dict(trainer.model.named_parameters())
        total_buckets = 0
        for parameter_group in trainer.parameter_groups:
            current = 0
            found = 0
            for name, parameter in named_parameters.items():
                if parameter_group and name not in parameter_group:
                    continue
                found += 1
                numel = int(getattr(parameter, "ds_numel", parameter.numel()))
                size = numel * parameter.element_size()
                if current + size > bucket_bytes and current:
                    total_buckets += 1
                    current = 0
                current += size
            if current:
                total_buckets += 1
            if parameter_group and not found:
                raise RuntimeError("unable to estimate EMA buckets for a parameter group")
        if total_buckets <= 0:
            raise RuntimeError("estimated EMA bucket count is zero")
        return total_buckets

    original_prepare_vllm = RolloutTrainerMixin._prepare_vllm
    original_move_full = RolloutTrainerMixin._move_full_model_to_vllm
    original_load_state = RolloutTrainerMixin._load_state_dict_to_vllm
    original_move_model = RolloutTrainerMixin._move_model_to_vllm

    def prepare_vllm(self):
        original_prepare_vllm(self)
        if self.vllm_mode != "server" or not self.accelerator.is_main_process:
            return
        if self.args.tuner_type != "full":
            raise RuntimeError("Momentum GRPO currently requires --tuner_type full")
        rollout_host = os.environ["ROLLOUT_SERVER_HOST"]
        rollout_port = int(os.getenv("ROLLOUT_SERVER_PORT", "8000"))
        momentum_host = os.environ["MOMENTUM_SERVER_HOST"]
        momentum_port = int(os.getenv("MOMENTUM_SERVER_PORT", "8001"))
        rollout_group_port = int(os.getenv("ROLLOUT_GROUP_PORT", "51216"))
        momentum_group_port = int(os.getenv("MOMENTUM_GROUP_PORT", "51217"))
        if rollout_host == momentum_host and rollout_port == momentum_port:
            raise RuntimeError("rollout and momentum endpoints must be different")
        if rollout_group_port == momentum_group_port:
            raise RuntimeError("ROLLOUT_GROUP_PORT and MOMENTUM_GROUP_PORT must differ")
        self.momentum_weight_client = MomentumWeightClient(
            hosts=[momentum_host],
            server_ports=[momentum_port],
            group_ports=[momentum_group_port],
            connection_timeout=float(os.getenv("MOMENTUM_SERVER_TIMEOUT", "600")),
        )
        status = self.momentum_weight_client.status()
        if status.get("role") != "momentum":
            raise RuntimeError(f"unexpected momentum server status: {status}")
        expected_decay = _env_float("MOMENTUM_DECAY", DEFAULT_DECAY)
        if abs(float(status["momentum_decay"]) - expected_decay) > 1e-12:
            raise RuntimeError(
                f"momentum decay mismatch: server={status['momentum_decay']}, "
                f"trainer={expected_decay}"
            )
        self.momentum_decay = expected_decay
        self.momentum_update_interval = _momentum_update_interval()
        server_interval = int(
            status.get("momentum_update_interval", self.momentum_update_interval)
        )
        if server_interval != self.momentum_update_interval:
            raise RuntimeError(
                f"momentum interval mismatch: server={server_interval}, "
                f"trainer={self.momentum_update_interval}"
            )
        self.momentum_weight_client.init_communicator(device=get_current_device())
        resume = os.getenv("MOMENTUM_RESUME_CHECKPOINT", "").strip()
        if resume:
            checkpoint = Path(resume)
            snapshot = checkpoint
            if not (snapshot / "momentum_state.json").is_file():
                snapshot = snapshot / "momentum"
            state = json.loads(
                (snapshot / "momentum_state.json").read_text(encoding="utf-8")
            )
            try:
                checkpoint_step = int(checkpoint.name.rsplit("-", 1)[1])
            except (IndexError, ValueError) as exc:
                raise RuntimeError(
                    f"resume directory must be named checkpoint-N: {checkpoint}"
                ) from exc
            snapshot_step = int(state["model_version"])
            expected_snapshot_step = _expected_momentum_step(
                checkpoint_step, self.momentum_update_interval
            )
            if snapshot_step != expected_snapshot_step:
                raise RuntimeError(
                    f"momentum snapshot step {snapshot_step} != expected step "
                    f"{expected_snapshot_step} for policy checkpoint {checkpoint_step}"
                )
            self.momentum_weight_client.load_snapshot(str(snapshot))
            self._momentum_resume_expected_step = checkpoint_step

    def load_state_dict_to_vllm(self, state_dict):
        original_load_state(self, state_dict)
        if not self.accelerator.is_main_process:
            return
        transaction = getattr(self, "_momentum_active_transaction", None)
        if not transaction or transaction.get("skip", False):
            return
        bucket_size = int(os.getenv("SWIFT_UPDATE_WEIGHTS_BUCKET_SIZE", "512"))
        buckets = _create_parameter_buckets(
            list(state_dict.items()), bucket_size_mb=bucket_size
        )
        for bucket_params in buckets:
            bucket = FlattenedTensorBucket(named_tensors=bucket_params)
            metadatas = bucket.get_metadata()
            flat_tensor = bucket.get_flattened_tensor()
            index = transaction["next_bucket"]
            self.momentum_weight_client.update_bucket(
                transaction["transaction_id"], index, metadatas, flat_tensor
            )
            transaction["next_bucket"] += 1
            del bucket, metadatas, flat_tensor

    def move_full_model_to_vllm(self):
        transaction = None
        if self.accelerator.is_main_process:
            step = int(self.state.global_step)
            status = self.momentum_weight_client.status()
            server_step = int(status.get("model_version", -1))
            interval = self.momentum_update_interval
            expected_server_step = _expected_momentum_step(step, interval)
            resume_expected = getattr(self, "_momentum_resume_expected_step", None)
            if resume_expected is not None:
                if step != resume_expected:
                    raise RuntimeError(
                        f"restored policy step {step} != resume checkpoint step {resume_expected}"
                    )
                if server_step != expected_server_step:
                    raise RuntimeError(
                        f"restored momentum step {server_step} != expected "
                        f"momentum step {expected_server_step}"
                    )
            if server_step > expected_server_step:
                raise RuntimeError(
                    f"momentum server step {server_step} is ahead of expected "
                    f"step {expected_server_step}"
                )
            should_sync = server_step < 0 or step % interval == 0
            if server_step < expected_server_step and not should_sync:
                raise RuntimeError(
                    f"momentum server step {server_step} is behind expected "
                    f"step {expected_server_step}"
                )
            transaction = {
                "transaction_id": f"policy-step-{step}-{uuid4().hex}",
                "step": step,
                "next_bucket": 0,
                "expected_buckets": 0,
                "skip": not should_sync or server_step == step,
            }
            if not transaction["skip"]:
                transaction["expected_buckets"] = estimate_bucket_count(self)
                self.momentum_weight_client.begin(
                    transaction["transaction_id"],
                    step,
                    self.momentum_decay,
                    transaction["expected_buckets"],
                )
            self._momentum_active_transaction = transaction
        try:
            original_move_full(self)
            if self.accelerator.is_main_process and not transaction["skip"]:
                if transaction["next_bucket"] != transaction["expected_buckets"]:
                    raise RuntimeError(
                        f"sent {transaction['next_bucket']} EMA buckets, expected "
                        f"{transaction['expected_buckets']}"
                    )
                self.momentum_weight_client.commit(transaction["transaction_id"])
            if self.accelerator.is_main_process and resume_expected is not None:
                self._momentum_resume_expected_step = None
        except Exception as exc:
            if self.accelerator.is_main_process and transaction and not transaction["skip"]:
                try:
                    self.momentum_weight_client.abort(
                        transaction["transaction_id"], repr(exc)
                    )
                except Exception:
                    pass
            raise
        finally:
            if self.accelerator.is_main_process:
                self._momentum_active_transaction = None

    def move_model_to_vllm(self, skip_async_check=False):
        result = original_move_model(self, skip_async_check=skip_async_check)
        if self.vllm_mode == "server" and self.accelerator.is_main_process:
            step = int(self.state.global_step)
            rollout_url = _base_url(
                os.environ["ROLLOUT_SERVER_HOST"],
                int(os.getenv("ROLLOUT_SERVER_PORT", "8000")),
            )
            response = requests.post(
                f"{rollout_url}/set_model_version/",
                json={"step": step},
                timeout=30,
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"failed to set rollout model version: {response.status_code}: "
                    f"{response.text[:1000]}"
                )
            momentum_status = self.momentum_weight_client.status()
            momentum_step = int(momentum_status.get("model_version", -1))
            expected_momentum_step = _expected_momentum_step(
                step, self.momentum_update_interval
            )
            if momentum_step != expected_momentum_step:
                raise RuntimeError(
                    f"momentum version {momentum_step} != expected "
                    f"{expected_momentum_step} for policy step {step}"
                )
        return result

    RolloutTrainerMixin._prepare_vllm = prepare_vllm
    RolloutTrainerMixin._load_state_dict_to_vllm = load_state_dict_to_vllm
    RolloutTrainerMixin._move_full_model_to_vllm = move_full_model_to_vllm
    RolloutTrainerMixin._move_model_to_vllm = move_model_to_vllm
    RolloutTrainerMixin._momentum_grpo_patched = True

    class MomentumCheckpointCallback(TrainerCallback):
        def __init__(self, trainer):
            self.trainer = trainer

        def on_save(self, args, state, control, **kwargs):
            del kwargs
            trainer = self.trainer
            step = int(state.global_step)
            if step != trainer._last_loaded_step:
                trainer._move_model_to_vllm()
                trainer._last_loaded_step = step
            trainer.accelerator.wait_for_everyone()
            if trainer.accelerator.is_main_process:
                checkpoint = Path(args.output_dir) / f"checkpoint-{step}" / "momentum"
                policy_model_path = str(
                    getattr(trainer.model, "model_dir", getattr(args, "model", ""))
                )
                trainer.momentum_weight_client.save_snapshot(
                    str(checkpoint), policy_model_path
                )
            trainer.accelerator.wait_for_everyone()
            return control

        def on_train_end(self, args, state, control, **kwargs):
            del args, kwargs
            trainer = self.trainer
            step = int(state.global_step)
            if step != trainer._last_loaded_step:
                trainer._move_model_to_vllm()
                trainer._last_loaded_step = step
            trainer.accelerator.wait_for_everyone()
            return control

    original_grpo_init = GRPOTrainer.__init__

    def grpo_init(self, *args, **kwargs):
        original_grpo_init(self, *args, **kwargs)
        self.add_callback(MomentumCheckpointCallback(self))

    GRPOTrainer.__init__ = grpo_init


_process_role = os.getenv("MOMENTUM_PROCESS_ROLE", "").strip()
if _process_role in {"trainer", "rollout", "momentum"}:
    validate_runtime_versions()
if _process_role in {"rollout", "momentum"}:
    _patch_weight_sync_worker()
    _patch_rollout_server_app()
elif _process_role == "trainer":
    _patch_trainer()
