"""Generate captions with a MoCo-Rubric Policy checkpoint via ms-swift."""
import argparse
import os
import subprocess
from pathlib import Path

from moco.io import ROOT, image_path, index_rows, read_jsonl, save_json, write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--gpus", default=os.getenv("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Output must differ from input")
    if args.output.exists():
        parser.error("Output already exists; choose a new output file")
    source = read_jsonl(args.input)
    if not source:
        raise ValueError("Empty caption input")
    system = (ROOT / "prompts/caption_system.txt").read_text().strip()
    user = (ROOT / "prompts/caption_user.txt").read_text().strip()
    inputs = []
    for row in source:
        image = row["image"]
        inputs.append({"sample_id": str(row.get("image_id", image)), "image": image,
                       "images": [image_path(image, args.data_root)],
                       "messages": [{"role": "system", "content": system},
                                    {"role": "user", "content": "<image>" + user}]})
    index_rows(inputs, "sample_id")
    prepared = args.output.with_suffix(".input.jsonl")
    raw = args.output.with_suffix(".raw.jsonl")
    if prepared.exists() or raw.exists():
        parser.error("Intermediate files exist; choose a new output filename")
    write_jsonl(prepared, inputs)
    command = ["swift", "infer", "--model", args.model, "--model_type", "qwen3_vl",
               "--template", "qwen3_vl", "--infer_backend", "vllm",
               "--vllm_tensor_parallel_size", "1", "--vllm_data_parallel_size", str(len(args.gpus.split(','))),
               "--vllm_gpu_memory_utilization", "0.9", "--vllm_limit_mm_per_prompt", '{"image": 1, "video": 0}',
               "--val_dataset", str(prepared), "--result_path", str(raw),
               "--load_args", "false", "--load_data_args", "false", "--max_new_tokens", "1024",
               "--temperature", "0.0", "--top_p", "1.0", "--seed", "42",
               "--remove_unused_columns", "false", "--load_from_cache_file", "false"]
    save_json(args.output.with_suffix(".command.json"), {"command": command})
    if args.prepare_only:
        print(f"Prepared {len(inputs)} images; no inference started.")
        return
    env = os.environ.copy()
    for key in ("MAX_PIXELS", "MIN_PIXELS", "IMAGE_MAX_TOKEN_NUM", "IMAGE_MIN_TOKEN_NUM", "CAPTION_MAX_PIXELS"):
        env.pop(key, None)
    env["CUDA_VISIBLE_DEVICES"] = args.gpus
    subprocess.run(command, check=True, env=env)
    outputs = index_rows(read_jsonl(raw), "sample_id")
    if set(outputs) != {row["sample_id"] for row in inputs}:
        raise ValueError("Caption output IDs differ from input IDs")
    captions = []
    for original, item in zip(source, inputs):
        output = outputs[item["sample_id"]]
        text = output.get("response")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Missing caption: {item['sample_id']}")
        captions.append({**original, "image_id": item["sample_id"], "response": text})
    write_jsonl(args.output, captions)
    print(f"Saved {len(captions)} captions to {args.output}")


if __name__ == "__main__":
    main()
