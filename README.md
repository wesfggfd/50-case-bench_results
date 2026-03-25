# 50-case-bench_results

This repository collects the OmniLottie benchmark artifacts prepared during the current reproduction workflow.

## Contents

- `results/first50/`: interim aggregation on the first 50 benchmark cases for each `split-task`, using existing outputs without re-running generation
- `results/reproduction/`: reproduction reports exported from the remote benchmark environment
- `scripts/remote/`: inference and evaluation scripts used on the remote machine
- `scripts/local/`: local judge rerun script used for `Obj / Motion`

## Included Reports

- `results/first50/first50_interim_report.md`: human-readable summary for the first-50-case aggregation
- `results/first50/first50_interim_metrics.json`: machine-readable first-50 metrics
- `results/first50/first50_vs_table1.csv`: flattened comparison table against paper Table 1
- `results/first50/first50_vs_table1_heatmap.png`: visual comparison against paper Table 1
- `results/reproduction/table1_reproduction_report.md`: full reproduction report exported from the remote environment
- `results/reproduction/core_metrics_report.json`: core metric report exported from the remote environment
- `results/reproduction/judge_metrics_report_official_local_claude46.json`: locally computed judge report for `Obj / Motion`

## Notes

- The first-50 aggregation uses the benchmark's first 50 samples in dataset order for each `subset-task`.
- Success rate keeps failed generations in the denominator.
- Metrics are aggregated from existing outputs only.
- One `Video-to-Lottie` sample in the real split rendered successfully but failed video decoding during pairwise metric aggregation, so its pairwise counts are `49` instead of `50`.
