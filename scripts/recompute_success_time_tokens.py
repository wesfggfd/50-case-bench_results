#!/usr/bin/env python3
import csv
import json
import os
from pathlib import Path
from statistics import mean
from transformers import AutoTokenizer

TOTAL = 150
TASK_LABELS = {
    'text2lottie': 'Text-to-Lottie',
    'text_image2lottie': 'Text-Image-to-Lottie',
    'video2lottie': 'Video-to-Lottie',
}
TASK_ORDER = ['text2lottie', 'text_image2lottie', 'video2lottie']
RESULTS_ROOT = Path(os.environ.get('MMLOTTIE_RESULTS_ROOT', '/root/SVG Generation/results/official_rerun'))
TIME_ROOT = Path(os.environ.get('MMLOTTIE_TIME_ROOT', '/root/SVG Generation/results/time_runs'))
TIME_CSV = Path(os.environ.get('MMLOTTIE_TIME_CSV', str(TIME_ROOT / 'timing.csv')))
TOKENIZER_PATH = os.environ.get('MMLOTTIE_TOKENIZER_PATH', 'Qwen/Qwen3.5-9B')
OUT_JSON = Path('/root/SVG Generation/OmniLottie/reproduction_results/official_success_time_tokens.json')


def json_dir(split, task_key):
    return RESULTS_ROOT / f'mmlottie_bench_{split}_{task_key}'


def avg_tokens(tokenizer, paths):
    vals = []
    for path in paths:
        text = path.read_text(encoding='utf-8')
        vals.append(len(tokenizer.encode(text, add_special_tokens=False)))
    return round(mean(vals), 4) if vals else None


def load_times():
    times = {}
    if not TIME_CSV.exists():
        return times
    with TIME_CSV.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row['split'], row['task'])
            times.setdefault(key, []).append(float(row['duration_seconds']) / TOTAL)
    return times


def main():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    times = load_times()
    report = {'notes': ['Computed from official_rerun outputs and time_runs/timing.csv.'], 'metrics': {}}
    for split in ['real', 'synthetic']:
        report['metrics'][split] = {}
        for task_key in TASK_ORDER:
            label = TASK_LABELS[task_key]
            out_dir = json_dir(split, task_key)
            json_files = sorted(out_dir.glob('*.json')) if out_dir.exists() else []
            report['metrics'][split][label] = {
                'success_count': len(json_files),
                'total_count': TOTAL,
                'success_rate': round(len(json_files) / TOTAL * 100, 4),
                'avg_qwen_tokens': avg_tokens(tokenizer, json_files),
                'avg_time_seconds': round(mean(times.get((split, task_key), [])), 4) if times.get((split, task_key)) else None,
                'time_run_count': len(times.get((split, task_key), [])),
            }
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('WROTE', OUT_JSON)
    print(json.dumps(report, ensure_ascii=False, indent=2)[:4000])

if __name__ == '__main__':
    main()
