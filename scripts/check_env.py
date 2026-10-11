#!/usr/bin/env python3
"""Check package versions without starting a model or modifying the environment."""
from importlib import metadata
import json

EXPECTED = {"ms-swift": "4.2.0.dev0", "vllm": "0.11.0", "torch": "2.8.0",
            "transformers": "4.57.1", "deepspeed": "0.17.6", "accelerate": "1.12.0"}


def main():
    versions = {}
    mismatches = []
    for package, expected in EXPECTED.items():
        try:
            value = metadata.version(package)
        except metadata.PackageNotFoundError:
            value = "missing"
        versions[package] = value
        if value.split('+')[0] != expected:
            mismatches.append(package)
    print(json.dumps({"versions": versions, "mismatches": mismatches}, indent=2))
    if mismatches:
        raise SystemExit("Runtime differs from the recorded server environment; see README.md")
    print("Package versions match. This is not a GPU-training or upstream-build compatibility test.")


if __name__ == "__main__":
    main()
