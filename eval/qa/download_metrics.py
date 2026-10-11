#!/usr/bin/env python3
"""Download exact upstream scoring sources, separately from this repository."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
SOURCES = Path(__file__).with_name("sources.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "data/metrics")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    def download(item):
        name, source = item
        target = args.out / name
        content = target.read_bytes() if target.exists() else urllib.request.urlopen(source["url"], timeout=60).read()
        if hashlib.sha256(content).hexdigest() != source["sha256"]:
            raise ValueError(f"Upstream source checksum mismatch: {name}")
        if not target.exists():
            target.write_bytes(content)
        return name
    with ThreadPoolExecutor(max_workers=6) as pool:
        for name in pool.map(download, json.loads(SOURCES.read_text()).items()):
            print(name)


if __name__ == "__main__":
    main()
