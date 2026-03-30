#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

from model_registry import get_model_spec


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "pipeline_runs"
TASKS = ["text2lottie", "text_image2lottie", "video2lottie"]
SPLITS = ["real", "synthetic"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified benchmark inference and evaluation pipeline")
    parser.add_argument("--model-type", required=True, help="Registered model type")
    parser.add_argument("--model-path", required=True, help="Checkpoint path or API/model identifier passed to the runner")
    parser.add_argument("--runner-script", default=None, help="Optional custom family runner script. Required for model types without a built-in runner")
    parser.add_argument("--experiment-name", required=True, help="Experiment output directory name under results/pipeline_runs")
    parser.add_argument("--num-samples", default="150", help="all or a positive integer")
    parser.add_argument("--split", choices=["real", "synthetic", "all"], default="all")
    parser.add_argument("--task", choices=["text2lottie", "text_image2lottie", "video2lottie", "all"], default="all")
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT), help="Root directory for experiment outputs")
    parser.add_argument("--bench-dir", default=str(REPO_ROOT / ".." / "OmniLottie_training" / "mmlottie_bench_hf"), help="Local MMLottieBench dataset directory")
    parser.add_argument("--run-core-eval", action="store_true", help="Run official core metrics after inference")
    parser.add_argument("--run-judge-eval", action="store_true", help="Run judge metrics after inference")
    parser.add_argument("--judge-api-url", default=None, help="Optional judge API URL override")
    parser.add_argument("--judge-api-key", default=None, help="Optional judge API key override")
    parser.add_argument("--judge-model", default=None, help="Optional judge model override")
    parser.add_argument("--maxlen", type=int, default=4096)
    parser.add_argument("--text-len", type=int, default=1500)
    parser.add_argument("--use-sampling", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.25)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--repetition-penalty", type=float, default=1.01)
    parser.add_argument("--num-candidates", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_num_samples(raw: str) -> str:
    value = str(raw).strip().lower()
    if value == "all":
        return "all"
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"--num-samples must be 'all' or a positive integer, got: {raw}")
    return str(parsed)


def selected_items(value: str, all_values: Iterable[str]) -> List[str]:
    return list(all_values) if value == "all" else [value]


def run_command(command: list[str], *, env: dict[str, str], dry_run: bool = False) -> None:
    print("$", " ".join(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, check=True, env=env)


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def resolve_inference_script(spec, runner_script: str | None) -> Path:
    if runner_script:
        return Path(runner_script).resolve()
    if spec.inference_script:
        return (SCRIPT_DIR / spec.inference_script).resolve()
    note = " ".join(spec.notes) if spec.notes else "No built-in runner is registered for this model type."
    raise ValueError(f"model_type={spec.model_type} requires --runner-script. {note}")


def build_run_record(*, split: str, task: str, spec, status: str, command: list[str] | None, output_dir: Path, note: str | None = None) -> dict:
    return {
        "split": split,
        "task": task,
        "model_type": spec.model_type,
        "model_family": spec.family,
        "status": status,
        "output_dir": str(output_dir),
        "command": command,
        "note": note,
    }


def main() -> None:
    args = parse_args()
    spec = get_model_spec(args.model_type)
    num_samples = resolve_num_samples(args.num_samples)
    splits = selected_items(args.split, SPLITS)
    requested_tasks = selected_items(args.task, TASKS)

    results_root = Path(args.results_root).resolve()
    experiment_root = results_root / args.experiment_name
    reports_root = experiment_root / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)
    records_path = experiment_root / "predictions.jsonl"
    if records_path.exists() and not args.dry_run:
        records_path.unlink()

    env = os.environ.copy()
    env["MMLOTTIE_RESULTS_ROOT"] = str(experiment_root)
    env["MMLOTTIE_CORE_REPORT_PATH"] = str(reports_root / "core_metrics_report.json")
    env["MMLOTTIE_JUDGE_REPORT_PATH"] = str(reports_root / "judge_metrics_report.json")
    env["MMLOTTIE_JUDGE_LOG_PATH"] = str(reports_root / "judge_metrics.log")
    env["BENCH_MODEL_TYPE"] = spec.model_type
    env["BENCH_MODEL_FAMILY"] = spec.family
    env["BENCH_EXPERIMENT_ROOT"] = str(experiment_root)
    if args.judge_api_url:
        env["BENCH_JUDGE_API_URL"] = args.judge_api_url
    if args.judge_api_key:
        env["BENCH_JUDGE_API_KEY"] = args.judge_api_key
    if args.judge_model:
        env["BENCH_JUDGE_MODEL"] = args.judge_model

    inference_script = resolve_inference_script(spec, args.runner_script)
    runnable_tasks = [task for task in requested_tasks if task in spec.supported_tasks]
    skipped_tasks = [task for task in requested_tasks if task not in spec.supported_tasks]

    manifest = {
        "model_type": spec.model_type,
        "model_family": spec.family,
        "model_description": spec.description,
        "model_path": args.model_path,
        "tokenizer_name": spec.tokenizer_name,
        "runner_script": str(inference_script),
        "experiment_name": args.experiment_name,
        "results_root": str(experiment_root),
        "num_samples": num_samples,
        "splits": splits,
        "requested_tasks": requested_tasks,
        "runnable_tasks": runnable_tasks,
        "skipped_tasks": skipped_tasks,
        "run_core_eval": args.run_core_eval,
        "run_judge_eval": args.run_judge_eval,
        "supports_local_weights": spec.supports_local_weights,
        "notes": list(spec.notes),
    }
    write_json(experiment_root / "manifest.json", manifest)

    for split in splits:
        for task in requested_tasks:
            task_output_dir = experiment_root / f"mmlottie_bench_{split}_{task}"
            if task not in spec.supported_tasks:
                record = build_run_record(
                    split=split,
                    task=task,
                    spec=spec,
                    status="unsupported",
                    command=None,
                    output_dir=task_output_dir,
                    note=f"{spec.model_type} does not declare support for task {task}",
                )
                append_jsonl(records_path, record)
                print(f"Skip unsupported task: {task} for model_type={spec.model_type}")
                continue

            cmd = [
                sys.executable,
                str(inference_script),
                "--sketch_weight", args.model_path,
                "--output_dir", str(experiment_root),
                "--mmlottie_bench_dir", args.bench_dir,
                "--split", split,
                "--mmlottie_task", task,
                "--num_samples", num_samples,
                "--maxlen", str(args.maxlen),
                "--text_len", str(args.text_len),
                "--num_candidates", str(args.num_candidates),
            ]
            if spec.tokenizer_name:
                cmd.extend(["--tokenizer_name", spec.tokenizer_name])
            if args.use_sampling:
                cmd.extend([
                    "--use_sampling",
                    "--temperature", str(args.temperature),
                    "--top_p", str(args.top_p),
                    "--top_k", str(args.top_k),
                    "--repetition_penalty", str(args.repetition_penalty),
                ])

            run_command(cmd, env=env, dry_run=args.dry_run)
            record = build_run_record(
                split=split,
                task=task,
                spec=spec,
                status="launched" if args.dry_run else "completed",
                command=cmd,
                output_dir=task_output_dir,
            )
            append_jsonl(records_path, record)

    if args.run_core_eval:
        core_cmd = [sys.executable, str(SCRIPT_DIR / "eval_official_core_metrics.py")]
        run_command(core_cmd, env=env, dry_run=args.dry_run)

    if args.run_judge_eval:
        judge_cmd = [sys.executable, str(SCRIPT_DIR / "eval_official_judge_metrics.py")]
        run_command(judge_cmd, env=env, dry_run=args.dry_run)

    print(f"Pipeline complete. Outputs: {experiment_root}")


if __name__ == "__main__":
    main()
