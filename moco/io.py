"""Small, dependency-free I/O helpers shared by the command-line tools."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for n, line in enumerate(stream, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{n}: expected a JSON object")
                rows.append(row)
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def index_rows(rows, field):
    indexed = {}
    for row in rows:
        value = row.get(field)
        if not isinstance(value, str) or not value.strip() or value in indexed:
            raise ValueError(f"Missing or duplicate {field}: {value!r}")
        indexed[value] = row
    return indexed


def image_path(value, root):
    """Published manifests use paths relative to data/, never server paths."""
    if isinstance(value, dict):
        value = value.get("path")
    if not isinstance(value, str) or not value:
        raise ValueError("Image path must be a nonempty string")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Expected a portable path relative to data/: {value}")
    root = Path(root).resolve()
    path = (root / path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise FileNotFoundError(f"Missing image under {root}: {value}")
    return str(path)
