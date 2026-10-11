#!/usr/bin/env python3
"""Read-only validation for the two Momentum GRPO rollout servers."""
from __future__ import annotations

import argparse
import json
import os
from urllib.request import Request, urlopen


def get_json(url: str) -> dict:
    request = Request(url, method="GET")
    with urlopen(request, timeout=10) as response:
        if response.status != 200:
            raise RuntimeError(f"GET {url} returned HTTP {response.status}")
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-url", required=True)
    parser.add_argument("--momentum-url", required=True)
    parser.add_argument("--decay", type=float, default=0.99)
    parser.add_argument("--update-interval", type=int, default=1)
    parser.add_argument("--expect-step", type=int)
    parser.add_argument("--expect-model")
    parser.add_argument("--expect-swift-version", default="4.2.0.dev0")
    parser.add_argument("--expect-vllm-version", default="0.11.0")
    args = parser.parse_args()

    rollout_url = args.rollout_url.rstrip("/")
    momentum_url = args.momentum_url.rstrip("/")
    if rollout_url == momentum_url:
        raise SystemExit("rollout and momentum URLs must differ")
    for base_url in (rollout_url, momentum_url):
        get_json(f"{base_url}/health/")
    rollout = get_json(f"{rollout_url}/role_status/")
    momentum = get_json(f"{momentum_url}/role_status/")
    if rollout.get("role") != "rollout":
        raise SystemExit(f"unexpected rollout status: {rollout}")
    if momentum.get("role") != "momentum":
        raise SystemExit(f"unexpected momentum status: {momentum}")
    for name, status in (("rollout", rollout), ("momentum", momentum)):
        if status.get("failed"):
            raise SystemExit(f"{name} server has a failed state: {status['failed']}")
        if status.get("swift_version") != args.expect_swift_version:
            raise SystemExit(
                f"{name} swift version mismatch: server={status.get('swift_version')}, "
                f"expected={args.expect_swift_version}"
            )
        if status.get("vllm_version") != args.expect_vllm_version:
            raise SystemExit(
                f"{name} vLLM version mismatch: server={status.get('vllm_version')}, "
                f"expected={args.expect_vllm_version}"
            )
    if abs(float(momentum["momentum_decay"]) - args.decay) > 1e-12:
        raise SystemExit(
            f"momentum decay mismatch: server={momentum['momentum_decay']}, expected={args.decay}"
        )
    if int(momentum.get("momentum_update_interval", -1)) != args.update_interval:
        raise SystemExit(
            "momentum update interval mismatch: "
            f"server={momentum.get('momentum_update_interval')}, "
            f"expected={args.update_interval}"
        )
    if momentum.get("runtime_dtype") != "bfloat16":
        raise SystemExit(f"unexpected momentum runtime dtype: {momentum}")
    expected_ema = {
        "ema_compute_dtype": "float32",
        "ema_master_device": "cpu",
        "ema_master_count": 1,
        "ema_distribution_dtype": "bfloat16",
        "ema_writer_dp_rank": 0,
    }
    for field, expected in expected_ema.items():
        if momentum.get(field) != expected:
            raise SystemExit(
                f"unexpected momentum {field}: {momentum.get(field)!r}, "
                f"expected {expected!r}"
            )
    if rollout.get("runtime_dtype") != "bfloat16":
        raise SystemExit(f"unexpected rollout runtime dtype: {rollout}")
    if rollout.get("model_path") != momentum.get("model_path"):
        raise SystemExit(
            f"initial model mismatch: rollout={rollout.get('model_path')}, "
            f"momentum={momentum.get('model_path')}"
        )
    if args.expect_model is not None:
        expected_model = os.path.realpath(args.expect_model)
        rollout_model = os.path.realpath(str(rollout.get("model_path", "")))
        momentum_model = os.path.realpath(str(momentum.get("model_path", "")))
        if rollout_model != expected_model or momentum_model != expected_model:
            raise SystemExit(
                "policy/server initial model mismatch: "
                f"policy={expected_model}, rollout={rollout_model}, momentum={momentum_model}"
            )
    if args.expect_step is not None:
        versions = (int(rollout["model_version"]), int(momentum["model_version"]))
        if versions != (args.expect_step, args.expect_step):
            raise SystemExit(
                f"model version mismatch: rollout={versions[0]}, momentum={versions[1]}, "
                f"expected={args.expect_step}"
            )
    print(
        json.dumps(
            {"ok": True, "rollout": rollout, "momentum": momentum},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
