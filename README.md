# 50-case-bench_results

This repository collects the OmniLottie benchmark artifacts prepared during the current reproduction workflow.

## Contents

- `results/reproduction/`: final reproduction reports exported from the remote benchmark environment
- `scripts/remote/`: inference and evaluation scripts used on the remote machine
- `scripts/local/`: local judge and sync scripts used for `Obj / Motion`

## Included Reports

- `results/reproduction/table1_reproduction_report.md`: Table 1-style summary and paper comparison for the final selected benchmark results
- `results/reproduction/final_150_metrics_fixed_psnr_ssim_dino.json`: final merged metrics with corrected `FVD` and `PSNR/SSIM/DINO`
- `results/reproduction/final_150_metrics_fixed_psnr_ssim_dino.md`: human-readable final 150 summary
- `results/reproduction/judge_metrics_report_public_local_claude35_aihhhl.json`: locally computed `Obj / Motion` report using the user-specified `ai.hhhl.cc` API

## Notes

- The final `results/reproduction/` reports summarize the selected 150 benchmark results for each `subset-task`.
- `Obj / Motion` in the final reproduction report come from `https://ai.hhhl.cc` with `claude-sonnet-3-5-latest`, following the user-specified evaluation setup.
