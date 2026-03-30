#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
import scipy.linalg

CORE_ORIGINAL = Path("/root/SVG Generation/OmniLottie/reproduction_results/core_metrics_report.json")
JUDGE_ORIGINAL = Path("/root/SVG Generation/OmniLottie/reproduction_results/judge_metrics_report_public_local_claude35_aihhhl.json")
CORE_FIXED_VIDEO = Path("/root/SVG Generation/OmniLottie/reproduction_results/core_metrics_report_public_standard.json")
FVD_CACHE = Path("/root/SVG Generation/results/fvd_cache")
OUT_JSON = Path("/root/SVG Generation/OmniLottie/reproduction_results/final_150_metrics_fixed_psnr_ssim_dino.json")
OUT_MD = Path("/root/SVG Generation/OmniLottie/reproduction_results/final_150_metrics_fixed_psnr_ssim_dino.md")

TASKS = ["Text-to-Lottie", "Text-Image-to-Lottie", "Video-to-Lottie"]
GEN_FVD_FILES = {
    ("real", "Text-to-Lottie"): "generated_real_text2lottie_i3d_stats.npz",
    ("real", "Text-Image-to-Lottie"): "generated_real_text_image2lottie_i3d_stats.npz",
    ("real", "Video-to-Lottie"): "generated_real_video2lottie_i3d_stats.npz",
    ("synthetic", "Text-to-Lottie"): "generated_synthetic_text2lottie_i3d_stats.npz",
    ("synthetic", "Text-Image-to-Lottie"): "generated_synthetic_text_image2lottie_i3d_stats.npz",
    ("synthetic", "Video-to-Lottie"): "generated_synthetic_video2lottie_i3d_stats.npz",
}
GT_FVD_STEM = "mmlottie2m_seed42_count{count}_i3d_stats.npz"


def fmt(value, digits):
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def load_stats(path: Path):
    arr = np.load(path)
    return arr["mean"], arr["cov"], int(arr["count"])


def frechet_distance(mu1, sigma1, mu2, sigma2):
    eps = 1e-6
    covmean, _ = scipy.linalg.sqrtm((sigma1 + np.eye(sigma1.shape[0]) * eps) @ (sigma2 + np.eye(sigma2.shape[0]) * eps), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu1 - mu2
    return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean))


def compute_corrected_fvd(split: str, task: str, sample_count: int):
    gt_name = GT_FVD_STEM.format(count=int(sample_count))
    gen_name = GEN_FVD_FILES[(split, task)]
    mu1, cov1, gt_count = load_stats(FVD_CACHE / gt_name)
    mu2, cov2, gen_count = load_stats(FVD_CACHE / gen_name)
    return round(frechet_distance(mu1, cov1, mu2, cov2), 6), gt_count, gen_count


def main():
    core_original = json.loads(CORE_ORIGINAL.read_text(encoding="utf-8"))
    judge_original = json.loads(JUDGE_ORIGINAL.read_text(encoding="utf-8"))
    core_fixed_video = json.loads(CORE_FIXED_VIDEO.read_text(encoding="utf-8"))

    report = {
        "notes": [
            "Time/Tokens/Success/FVD/CLIP keep the original pipeline outputs.",
            "Obj/Motion use the user-specified ai.hhhl.cc + claude-sonnet-3-5-latest judge outputs.",
            "PSNR/SSIM/DINO are replaced with the corrected compute_video_pair_metrics results.",
        ],
        "metrics": {},
    }

    lines = [
        "# Final 150 Metrics",
        "",
        "Time/Tokens/Success/FVD/CLIP/Obj/Motion come from the original pipeline. PSNR/SSIM/DINO are refreshed from the corrected video-pair metric implementation.",
        "",
    ]

    for split in ["real", "synthetic"]:
        report["metrics"][split] = {}
        lines.append(f"## {split.title()}")
        lines.append("")
        lines.append("| Task | Success | Tokens | Time/sample | FVD | CLIP | Obj | Motion | PSNR | SSIM | DINO |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for task in TASKS:
            orig = core_original["metrics"][split][task]
            judge = judge_original.get("metrics", {}).get(split, {}).get(task, {})
            fixed = core_fixed_video["metrics"][split][task]
            corrected_fvd, gt_count, gen_count = compute_corrected_fvd(split, task, fixed.get("fvd_generated_count", fixed.get("generated_count", orig.get("success_count"))))
            row = {
                "success_count": orig.get("success_count"),
                "total_count": orig.get("total_count"),
                "success_rate": orig.get("success_rate"),
                "avg_qwen_tokens": orig.get("avg_qwen_tokens"),
                "avg_time_seconds": orig.get("approx_time_seconds", orig.get("avg_time_seconds")),
                "fvd": corrected_fvd,
                "clip": orig.get("clip"),
                "object_alignment": judge.get("object_alignment"),
                "motion_alignment": judge.get("motion_alignment"),
                "psnr": fixed.get("psnr"),
                "ssim": fixed.get("ssim"),
                "dino": fixed.get("dino"),
                "pair_count": fixed.get("pair_count", orig.get("pair_count")),
                "judge_count": judge.get("count"),
                "fvd_gt_count": gt_count,
                "fvd_generated_count": gen_count,
            }
            report["metrics"][split][task] = row
            lines.append(
                "| {task} | {succ}/{total} ({rate:.1f}%) | {tok} | {timev} | {fvd} | {clip} | {obj} | {mot} | {psnr} | {ssim} | {dino} |".format(
                    task=task,
                    succ=row["success_count"],
                    total=row["total_count"],
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
