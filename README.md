# 50-case-bench_results

This repository stores the **benchmark inference / evaluation scripts** together with the **preserved experiment results** produced during the OmniLottie benchmark reproduction workflow.

Its purpose is twofold:
1. keep the scripts used to run benchmark inference and compute metrics
2. keep the original benchmark outputs and final reproduced reports for later inspection

---

## Scope of this repository

This repository intentionally **preserves the benchmark artifacts** that were already generated.
That includes the final reports and result files under `results/`.

In particular, the following outputs are meant to remain in the repository:
- `results/reproduction/`
- `results/single_sample_preview/`
- final merged metrics and markdown reports

So unlike the training-code repository, this benchmark repository is expected to version the evaluation results themselves.

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
Only the current unified benchmark interface is kept under `scripts/`.
Legacy one-off helper scripts have been removed.
The remaining entrypoints are:
- `benchmark_pipeline.py` — unified inference + evaluation launcher
- `benchmark_model_registry.py` — model family registry
- `runner_omnilottie_qwen35.py` — current OmniLottie Qwen3.5 runner
- `runner_omnilottie_original.py` — original OmniLottie wrapper runner
- `runner_family_stub.py` — placeholder runner for new model families
- `benchmark_eval_core.py` — automated core metrics evaluation
- `benchmark_eval_judge.py` — automated judge-based evaluation

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
1. run `scripts/benchmark_pipeline.py`
2. optionally enable automated core / judge evaluation in the same command
3. keep the resulting reports under `results/reproduction/` or `results/pipeline_runs/`

This makes the repository both a:
- benchmark script archive
- benchmark result archive

### Unified pipeline

This repository now also provides a unified pipeline entrypoint.
It is designed around **model families**, not just one fixed OmniLottie implementation.
Current registered model types include:
- `omnilottie_qwen35`
- `omnilottie_original`
- `deepseekv3`
- `qwen35_base`
- `recraft`

Example for the built-in OmniLottie Qwen3.5 path:

```bash
python scripts/benchmark_pipeline.py \
  --model-type omnilottie_qwen35 \
  --model-path /path/to/checkpoint \
  --experiment-name omnilottie_real_synth_50 \
  --num-samples 50 \
  --split all \
  --task all \
  --run-core-eval
```

For model types without a built-in runner yet, provide a custom family runner:

```bash
python scripts/benchmark_pipeline.py \
  --model-type deepseekv3 \
  --model-path /path/to/model_or_api_config \
  --runner-script scripts/runner_family_stub.py \
  --experiment-name deepseek_text_20 \
  --num-samples 20 \
  --split all \
  --task text2lottie
```

Supported sample count values are selectable, for example:
- `--num-samples all`
- `--num-samples 10`
- `--num-samples 20`
- `--num-samples 50`
- `--num-samples N`

Outputs are written under:

```text
results/pipeline_runs/<experiment-name>/
```

including:
- per-task inference outputs
- `manifest.json`
- `predictions.jsonl`
- `reports/core_metrics_report.json`
- `reports/judge_metrics_report.json` when judge evaluation is enabled

The pipeline does **not** require every model family to successfully generate renderable Lottie JSON on every sample.
It only requires each family runner to execute normal inference and write its outputs/results honestly, including failures or unsupported tasks.

### Judge evaluation credentials

Judge evaluation no longer reads API credentials from source code.
Set them through environment variables before running:

```bash
export BENCH_JUDGE_API_URL="https://.../v1/messages"
export BENCH_JUDGE_API_KEY="your_api_key"
export BENCH_JUDGE_MODEL="claude-sonnet-4-6"
```

Then run:

```bash
python scripts/benchmark_pipeline.py \
  --model-type omnilottie_qwen35 \
  --model-path /path/to/checkpoint \
  --experiment-name omnilottie_50_with_judge \
  --num-samples 50 \
  --split all \
  --task all \
  --run-core-eval \
  --run-judge-eval
```

---

## License

See `LICENSE`.
