# 50-case-bench_results

This repository stores the **benchmark inference / evaluation scripts** and the **preserved experiment results** used during the OmniLottie benchmark reproduction workflow.

The goal of this repo is twofold:
1. keep the scripts that were used to run inference and compute metrics
2. keep the original benchmark outputs and final reproduced reports for later inspection

---

## What is kept in this repository

This repo intentionally **preserves the original benchmark artifacts** that were already generated.
That includes the final reports and result files under `results/`.

In particular, the following experiment outputs are kept and should remain in the repository:
- `results/reproduction/`
- `results/single_sample_preview/`
- final merged metrics and markdown reports

So unlike the training-code repo, this benchmark repo is expected to keep the evaluation results themselves.

---

## Repository layout

```text
50-case-bench_results/
├── results/
│   ├── reproduction/
│   └── single_sample_preview/
├── scripts/
├── LICENSE
└── README.md
```

### `results/reproduction/`
Final benchmark reproduction outputs, including:
- `table1_reproduction_report.md`
- `final_150_metrics_fixed_psnr_ssim_dino.json`
- `final_150_metrics_fixed_psnr_ssim_dino.md`
- `judge_metrics_report_public_local_claude35_aihhhl.json`

### `scripts/`
All benchmark inference and evaluation logic is now kept directly under `scripts/`.
This includes:
- inference
- official metric evaluation
- judge-based evaluation
- metric recomputation / correction
- sync / merge helper scripts

---

## Included benchmark reports

Main final reports:
- `results/reproduction/table1_reproduction_report.md`
  Table 1 style summary and comparison against the paper values.
- `results/reproduction/final_150_metrics_fixed_psnr_ssim_dino.json`
  Final merged metrics after corrected `PSNR / SSIM / DINO` and final `FVD` processing.
- `results/reproduction/final_150_metrics_fixed_psnr_ssim_dino.md`
  Human-readable final summary.
- `results/reproduction/judge_metrics_report_public_local_claude35_aihhhl.json`
  Local judge results for `Obj / Motion` using the user-specified API setup.

---

## Metric notes

The reproduced benchmark covers the standard subset / task combinations over:
- `real`
- `synthetic`

and the task types:
- Text-to-Lottie
- Text-Image-to-Lottie
- Video-to-Lottie

The reports include metrics such as:
- time
- generated token count
- success rate
- FVD
- CLIP
- Obj
- Motion
- PSNR
- SSIM
- DINO

Notes:
- `Obj / Motion` were computed with the user-specified `https://ai.hhhl.cc` setup using `claude-sonnet-3-5-latest`
- some counts can be lower than 150 if generation, rendering, decoding, or judge evaluation failed for particular cases

---

## Preservation policy

This repo should **keep the benchmark results**.
When updating the repository, do not remove the existing reproduced outputs unless they are clearly obsolete and intentionally replaced.

The current README is written to make that explicit: the scripts and the benchmark result artifacts are both first-class content of this repository.

---

## Usage

Typical workflow:
1. run remote inference / metric scripts from `scripts/`
2. sync or post-process results locally with `scripts/`
3. keep the resulting reports in `results/reproduction/`

This makes the repository a combined:
- benchmark script archive
- benchmark result archive

---

## License

See `LICENSE`.
