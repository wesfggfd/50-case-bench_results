#!/usr/bin/env python3
import csv
import json
from pathlib import Path
from statistics import mean

from transformers import AutoTokenizer

TOTAL = 150
TASK_LABELS = {
    "text2lottie": "Text-to-Lottie",
    "text_image2lottie": "Text-Image-to-Lottie",
    "video2lottie": "Video-to-Lottie",
}
TASK_ORDER = ["text2lottie", "text_image2lottie", "video2lottie"]
RESULTS_ROOT = Path("/root/SVG Generation/results/official_rerun")
TIME_CSV = Path("/root/SVG Generation/results/time_runs/timing.csv")
TOKENIZER_PATH = "/root/SVG Generation/downloads/base/Qwen2.5-VL-3B-Instruct"
CORE_REPORT = Path("/root/SVG Generation/OmniLottie/reproduction_results/core_metrics_report_public_standard.json")
JUDGE_REPORT = Path("/root/SVG Generation/OmniLottie/reproduction_results/judge_metrics_report_public_local_claude35_aihhhl.json")
OUT_JSON = Path("/root/SVG Generation/OmniLottie/reproduction_results/official_150_metrics_public_standard.json")
OUT_MD = Path("/root/SVG Generation/OmniLottie/reproduction_results/official_150_metrics_public_standard.md")


def json_dir(split, task_key):
    return RESULTS_ROOT / f"{split}_{task_key}" / f"mmlottie_bench_{split}"


def avg_tokens(tokenizer, paths):
    vals = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        vals.append(len(tokenizer.encode(text, add_special_tokens=False)))
    return round(mean(vals), 4) if vals else None


def load_times():
    times = {}
    if not TIME_CSV.exists():
        return times
    with TIME_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["split"], row["task"])
            times.setdefault(key, []).append(float(row["duration_seconds"]) / TOTAL)
    return times


def fmt(value, digits):
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def main():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    times = load_times()
    core = json.loads(CORE_REPORT.read_text(encoding="utf-8")) if CORE_REPORT.exists() else {"metrics": {}, "notes": []}
    judge = json.loads(JUDGE_REPORT.read_text(encoding="utf-8")) if JUDGE_REPORT.exists() else {"metrics": {}, "notes": []}
    report = {
        "protocol": {
            "selection": "all 150 public benchmark samples per split-task",
            "fvd_protocol": "public-release reproducible: only Video-to-Lottie has benchmark reference videos",
            "judge_protocol": "local third-party API evaluation on downloaded text-task outputs",
        },
        "notes": list(core.get("notes", [])) + list(judge.get("notes", [])),
        "metrics": {},
    }
    lines = [
        "# Official 150 Metrics (Public Standard)",
        "",
        "Selection: all 150 benchmark samples per split-task, with success measured against the full denominator.",
        "",
    ]
    for split in ["real", "synthetic"]:
        report["metrics"][split] = {}
        lines.append(f"## {split.title()}")
        lines.append("")
        lines.append("| Task | Success | Tokens | Time/sample | FVD | CLIP | Obj | Motion | PSNR | SSIM | DINO |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for task_key in TASK_ORDER:
            label = TASK_LABELS[task_key]
            out_dir = json_dir(split, task_key)
            json_files = sorted(out_dir.glob("*.json")) if out_dir.exists() else []
            core_entry = core.get("metrics", {}).get(split, {}).get(label, {})
            judge_entry = judge.get("metrics", {}).get(split, {}).get(label, {})
            row = {
                "selected_count": TOTAL,
                "success_count": len(json_files),
                "success_rate": round(len(json_files) / TOTAL * 100, 4),
                "avg_qwen_tokens": avg_tokens(tokenizer, json_files),
                "avg_time_seconds": round(mean(times.get((split, task_key), [])), 4) if times.get((split, task_key)) else None,
                "time_run_count": len(times.get((split, task_key), [])),
                "fvd": core_entry.get("fvd"),
                "clip": core_entry.get("clip"),
                "object_alignment": judge_entry.get("object_alignment"),
                "motion_alignment": judge_entry.get("motion_alignment"),
                "psnr": core_entry.get("psnr"),
                "ssim": core_entry.get("ssim"),
                "dino": core_entry.get("dino"),
                "fvd_reference_count": core_entry.get("fvd_reference_count"),
                "fvd_generated_count": core_entry.get("fvd_generated_count"),
                "clip_count": core_entry.get("clip_count"),
                "pair_count": core_entry.get("pair_count"),
                "judge_count": judge_entry.get("count"),
            }
            report["metrics"][split][label] = row
            lines.append(
                "| {task} | {succ}/{total} ({rate:.1f}%) | {tok} | {timev} | {fvd} | {clip} | {obj} | {mot} | {psnr} | {ssim} | {dino} |".format(
                    task=label,
                    succ=row["success_count"],
                    total=row["selected_count"],
                    rate=row["success_rate"],
                    tok="-" if row["avg_qwen_tokens"] is None else f"{row['avg_qwen_tokens']:.1f}",
                    timev="-" if row["avg_time_seconds"] is None else f"{row['avg_time_seconds']:.2f}",
                    fvd=fmt(row["fvd"], 2),
                    clip=fmt(row["clip"], 4),
                    obj=fmt(row["object_alignment"], 2),
                    mot=fmt(row["motion_alignment"], 2),
                    psnr=fmt(row["psnr"], 2),
                    ssim=fmt(row["ssim"], 3),
                    dino=fmt(row["dino"], 3),
                )
            )
        lines.append("")
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("WROTE", OUT_JSON)
    print("WROTE", OUT_MD)
    print(json.dumps(report, ensure_ascii=False, indent=2)[:4000])


if __name__ == "__main__":
    main()
