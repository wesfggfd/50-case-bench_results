#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_OMNILOTTIE_REPO = Path(
    os.environ.get(
        "OMNILOTTIE_ORIGINAL_ROOT",
        "/opt/liblibai-models/user-workspace2/users/Sean_CHEN/OpenVGLab-OmniLottie",
    )
)
TASKS = ["text2lottie", "text_image2lottie", "video2lottie"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark wrapper runner for the original OmniLottie repository")
    parser.add_argument("--sketch_weight", required=True, help="Original OmniLottie checkpoint path")
    parser.add_argument("--output_dir", required=True, help="Benchmark experiment root")
    parser.add_argument("--mmlottie_bench_dir", required=True, help="Local MMLottieBench dataset directory")
    parser.add_argument("--split", choices=["real", "synthetic"], required=True)
    parser.add_argument("--mmlottie_task", choices=TASKS, required=True)
    parser.add_argument("--num_samples", default="150", help="Unified interface: all or a positive integer")
    parser.add_argument("--maxlen", type=int, default=4096)
    parser.add_argument("--text_len", type=int, default=1500)
    parser.add_argument("--tokenizer_name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--num_candidates", type=int, default=1)
    parser.add_argument("--use_sampling", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.25)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--repetition_penalty", type=float, default=1.01)
    parser.add_argument("--omnilottie_repo", default=str(DEFAULT_OMNILOTTIE_REPO), help="Path to the original OpenVGLab OmniLottie repo")
    return parser.parse_args()


def resolve_max_samples(raw: str) -> int:
    value = str(raw).strip().lower()
    if value == "all":
        return -1
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"--num_samples must be 'all' or a positive integer, got: {raw}")
    return parsed


def ensure_repo(repo_root: Path) -> Path:
    inference_script = repo_root / "inference.py"
    if not repo_root.exists():
        raise FileNotFoundError(f"Original OmniLottie repo not found: {repo_root}")
    if not inference_script.exists():
        raise FileNotFoundError(f"Original OmniLottie inference.py not found: {inference_script}")
    return inference_script


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def copy_task_artifacts(src_dir: Path, dst_dir: Path) -> list[str]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for item in sorted(src_dir.iterdir()):
        target = dst_dir / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)
        copied.append(item.name)
    return copied


def build_original_command(args: argparse.Namespace, inference_script: Path, temp_output_dir: Path, max_samples: int) -> list[str]:
    cmd = [
        sys.executable,
        str(inference_script),
        "--sketch_weight", args.sketch_weight,
        "--output_dir", str(temp_output_dir),
        "--mmlottie_bench_dir", args.mmlottie_bench_dir,
        "--split", args.split,
        "--mmlottie_task", args.mmlottie_task,
        "--max_samples", str(max_samples),
        "--maxlen", str(args.maxlen),
        "--text_len", str(args.text_len),
        "--tokenizer_name", args.tokenizer_name,
        "--num_candidates", str(args.num_candidates),
    ]
    if args.use_sampling:
        cmd.extend([
            "--use_sampling",
            "--temperature", str(args.temperature),
            "--top_p", str(args.top_p),
            "--top_k", str(args.top_k),
            "--repetition_penalty", str(args.repetition_penalty),
        ])
    return cmd


def main() -> None:
    args = parse_args()
    max_samples = resolve_max_samples(args.num_samples)
    repo_root = Path(args.omnilottie_repo).resolve()
    inference_script = ensure_repo(repo_root)

    output_root = Path(args.output_dir).resolve()
    task_output_dir = output_root / f"mmlottie_bench_{args.split}_{args.mmlottie_task}"
    task_output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="omnilottie_original_runner_") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        cmd = build_original_command(args, inference_script, temp_dir, max_samples)
        env = os.environ.copy()
        env.setdefault("TOKENIZERS_PARALLELISM", "false")

        print("$", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd=str(repo_root), env=env)

        original_task_dir = temp_dir / f"mmlottie_bench_{args.split}"
        if not original_task_dir.exists():
            raise FileNotFoundError(
                f"Original OmniLottie did not create the expected output directory: {original_task_dir}"
            )

        copied_files = copy_task_artifacts(original_task_dir, task_output_dir)
        summary = {
            "runner": "run_omnilottie_original",
            "model_path": args.sketch_weight,
            "omnilottie_repo": str(repo_root),
            "original_inference_script": str(inference_script),
            "split": args.split,
            "task": args.mmlottie_task,
            "num_samples": args.num_samples,
            "mapped_max_samples": max_samples,
            "tokenizer_name": args.tokenizer_name,
            "output_dir": str(task_output_dir),
            "copied_files": copied_files,
            "wrapped_command": cmd,
        }
        write_json(task_output_dir / "runner_summary.json", summary)
        print(f"Copied {len(copied_files)} artifacts to {task_output_dir}")
        print(f"Wrote runner summary to {task_output_dir / 'runner_summary.json'}")


if __name__ == "__main__":
    main()
