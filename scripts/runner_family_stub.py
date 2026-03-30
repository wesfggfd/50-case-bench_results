#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


TASKS = ["text2lottie", "text_image2lottie", "video2lottie"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generic family runner stub for benchmark model families")
    parser.add_argument("--sketch_weight", required=True, help="Model path / checkpoint / API identifier")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mmlottie_bench_dir", required=True)
    parser.add_argument("--split", choices=["real", "synthetic"], required=True)
    parser.add_argument("--mmlottie_task", choices=TASKS, required=True)
    parser.add_argument("--num_samples", default="50")
    parser.add_argument("--maxlen", type=int, default=4096)
    parser.add_argument("--text_len", type=int, default=1500)
    parser.add_argument("--tokenizer_name", default=None)
    parser.add_argument("--num_candidates", type=int, default=1)
    parser.add_argument("--use_sampling", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.25)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--repetition_penalty", type=float, default=1.01)
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    task_dir = Path(args.output_dir) / f"mmlottie_bench_{args.split}_{args.mmlottie_task}"
    task_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "runner": "runner_family_stub",
        "model_path": args.sketch_weight,
        "split": args.split,
        "task": args.mmlottie_task,
        "num_samples": args.num_samples,
        "tokenizer_name": args.tokenizer_name,
        "note": "Replace this stub with a concrete family runner that performs real inference and writes per-sample artifacts.",
        "expected_outputs": {
            "per_sample_artifacts": [
                "<sample_id>.json (optional)",
                "<sample_id>.mp4 (optional)",
                "<sample_id>_raw.txt (optional)",
                "<sample_id>_info.txt (optional)"
            ],
            "task_summary": "runner_summary.json"
        }
    }
    write_json(task_dir / "runner_summary.json", manifest)
    print(f"Wrote stub runner summary to {task_dir / 'runner_summary.json'}")


if __name__ == "__main__":
    main()
