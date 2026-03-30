#!/usr/bin/env python3
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, '/root/SVG Generation/OmniLottie')
import eval_official_core_metrics as core

FIRST50 = Path('/root/SVG Generation/results/first50_manifest.json')
JUDGE_CACHE_TAR = Path('/root/SVG Generation/OmniLottie/reproduction_results/judge_cache_official_local_claude46.tar.gz')
JUDGE_CACHE_DIR = Path('/root/SVG Generation/OmniLottie/reproduction_results/judge_cache_official_local_claude46')
JUDGE_REPORT = Path('/root/SVG Generation/OmniLottie/reproduction_results/judge_metrics_report_official_local_claude46.json')
OUT_JSON = Path('/root/SVG Generation/OmniLottie/reproduction_results/first50_interim_metrics.json')
OUT_MD = Path('/root/SVG Generation/OmniLottie/reproduction_results/first50_interim_report.md')
TIME_CSV = Path(os.environ.get('MMLOTTIE_TIME_CSV', '/root/SVG Generation/results/time_runs/timing.csv'))
TOKENIZER_PATH = os.environ.get('MMLOTTIE_TOKENIZER_PATH', 'Qwen/Qwen3.5-9B')
TASK_KEY_BY_LABEL = {
    'Text-to-Lottie': 'text2lottie',
    'Text-Image-to-Lottie': 'text_image2lottie',
    'Video-to-Lottie': 'video2lottie',
}
PAPER = {
    'real': {
        'Text-to-Lottie': {'time': 65.76, 'tokens': 13.4, 'success': 88.3, 'fvd': 202.14, 'clip': 0.3029, 'obj': 4.62, 'motion': 5.89},
        'Text-Image-to-Lottie': {'time': 65.55, 'tokens': 16.5, 'success': 93.3, 'fvd': 180.27, 'clip': 0.2997, 'obj': 4.96, 'motion': 4.24},
        'Video-to-Lottie': {'time': 111.90, 'tokens': 40.2, 'success': 88.1, 'fvd': 227.11, 'psnr': 16.08, 'ssim': 0.82, 'dino': 0.92},
    },
    'synthetic': {
        'Text-to-Lottie': {'time': 37.93, 'tokens': 13.4, 'success': 82.1, 'fvd': 206.35, 'clip': 0.2748, 'obj': 4.31, 'motion': 5.63},
        'Text-Image-to-Lottie': {'time': 84.80, 'tokens': 16.3, 'success': 92.9, 'fvd': 225.45, 'clip': 0.2666, 'obj': 4.44, 'motion': 3.98},
        'Video-to-Lottie': {'time': 109.53, 'tokens': 41.4, 'success': 80.7, 'fvd': 342.65, 'psnr': 15.76, 'ssim': 0.79, 'dino': 0.88},
    },
}


def ensure_judge_cache():
    if JUDGE_CACHE_DIR.exists():
        return
    if JUDGE_CACHE_TAR.exists():
        import tarfile
        with tarfile.open(JUDGE_CACHE_TAR, 'r:gz') as tar:
            tar.extractall(JUDGE_CACHE_TAR.parent)


def load_first50():
    return json.loads(FIRST50.read_text(encoding='utf-8'))


def load_time_map():
    import csv
    out = defaultdict(list)
    if not TIME_CSV.exists():
        return out
    with TIME_CSV.open('r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            out[(row['split'], row['task'])].append(float(row['duration_seconds']) / 150.0)
    return out


def load_judge_map():
    ensure_judge_cache()
    out = {}
    if not JUDGE_CACHE_DIR.exists():
        return out
    for path in JUDGE_CACHE_DIR.rglob('*.json'):
        rec = json.loads(path.read_text(encoding='utf-8'))
        if 'object_consistency_score' in rec and 'motion_consistency_score' in rec:
            out[rec['id']] = rec
    return out


def load_gt_stats():
    arr = np.load(core.GT_STATS_PATH)
    return {'mean': arr['mean'], 'cov': arr['cov'], 'count': int(arr['count'])}


def subset_generated_fvd(split, task_key, rows, device):
    detector = core.get_detector(device)
    stats = core.RunningStats()
    clips = []
    ok_ids = []
    failed_ids = []
    result_dir = core.RESULTS[split][task_key]
    for row in rows:
        sample_id = row['id']
        if not (result_dir / f'{sample_id}.json').exists():
            continue
        try:
            render_path = core.ensure_render(split, task_key, sample_id, result_dir)
            frames = core.load_video_frames_cv2(str(render_path))
            clips.append(core.frames_to_clip_tensor(frames))
            ok_ids.append(sample_id)
        except Exception as exc:
            failed_ids.append({'id': sample_id, 'error': str(exc)})
            continue
        if len(clips) >= core.BATCH_SIZE:
            core.update_stats_from_clip_tensors(stats, detector, device, clips)
            clips.clear()
    if clips:
        core.update_stats_from_clip_tensors(stats, detector, device, clips)
    return {'mean': stats.mean, 'cov': stats.covariance(), 'count': stats.count, 'ids': ok_ids, 'failed_ids': failed_ids}


def main():
    device = 'cuda' if core.torch.cuda.is_available() else 'cpu'
    first50 = load_first50()
    bench = core.load_bench()
    gt = load_gt_stats()
    judge_map = load_judge_map()
    times = load_time_map()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    report = {
        'protocol': {
            'selection': 'first 50 benchmark samples per split-task in dataset order, regardless of success',
            'note': 'metrics are aggregated from existing generated outputs only; denominators/counts are reported explicitly',
        },
        'metrics': {},
    }
    lines = ['# First-50 Interim Report', '', 'Selection: first 50 benchmark samples for each split-task, keeping failed generations in the denominator for success rate.', '']
    for split in ['real', 'synthetic']:
        rows = list(bench[split])
        by_id = {row['id']: row for row in rows}
        report['metrics'][split] = {}
        lines.append(f'## {split.title()}')
        lines.append('')
        lines.append('| Task | Success | Tokens | Time/sample | FVD | CLIP | Obj | Motion | PSNR | SSIM | DINO |')
        lines.append('| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
        for label in ['Text-to-Lottie', 'Text-Image-to-Lottie', 'Video-to-Lottie']:
            task_key = TASK_KEY_BY_LABEL[label]
            ids = first50[split][label]
            subset_rows = [by_id[i] for i in ids]
            result_dir = core.RESULTS[split][task_key]
            success_ids = [i for i in ids if (result_dir / f'{i}.json').exists()]
            token_vals = []
            for sample_id in success_ids:
                text = (result_dir / f'{sample_id}.json').read_text(encoding='utf-8')
                token_vals.append(len(tokenizer.encode(text, add_special_tokens=False)))
            row_out = {
                'selected_count': len(ids),
                'success_count': len(success_ids),
                'success_rate': round(len(success_ids) / len(ids) * 100, 4),
                'avg_qwen_tokens': round(sum(token_vals) / len(token_vals), 4) if token_vals else None,
                'avg_time_seconds': round(sum(times.get((split, task_key), [])) / len(times.get((split, task_key), [])), 4) if times.get((split, task_key)) else None,
                'time_run_count': len(times.get((split, task_key), [])),
                'fvd': None,
                'clip': None,
                'object_alignment': None,
                'motion_alignment': None,
                'psnr': None,
                'ssim': None,
                'dino': None,
            }
            if label in {'Text-to-Lottie', 'Text-Image-to-Lottie'}:
                clip_scores = core.compute_clip_for_rows(split, task_key, subset_rows, device=device)
                row_out['clip'] = clip_scores['clip']
                row_out['clip_count'] = clip_scores['clip_count']
                obj = [judge_map[i]['object_consistency_score'] for i in ids if i in judge_map]
                mot = [judge_map[i]['motion_consistency_score'] for i in ids if i in judge_map]
                row_out['object_alignment'] = round(sum(obj) / len(obj), 6) if obj else None
                row_out['motion_alignment'] = round(sum(mot) / len(mot), 6) if mot else None
                row_out['judge_count'] = len(obj)
            else:
                video_scores = core.compute_video_pair_metrics(split, task_key, subset_rows, device=device)
                row_out['psnr'] = video_scores['psnr']
                row_out['ssim'] = video_scores['ssim']
                row_out['dino'] = video_scores['dino']
                row_out['pair_count'] = video_scores['pair_count']
            gen_stats = subset_generated_fvd(split, task_key, subset_rows, device=device)
            if gen_stats['count'] > 1:
                row_out['fvd'] = round(core.frechet_distance(gt['mean'], gt['cov'], gen_stats['mean'], gen_stats['cov']), 6)
                row_out['fvd_count'] = gen_stats['count']
            report['metrics'][split][label] = row_out
            lines.append('| {task} | {succ}/{sel} ({rate:.1f}%) | {tok} | {timev} | {fvd} | {clip} | {obj} | {mot} | {psnr} | {ssim} | {dino} |'.format(
                task=label,
                succ=row_out['success_count'],
                sel=row_out['selected_count'],
                rate=row_out['success_rate'],
                tok='-' if row_out['avg_qwen_tokens'] is None else f"{row_out['avg_qwen_tokens']:.1f}",
                timev='-' if row_out['avg_time_seconds'] is None else f"{row_out['avg_time_seconds']:.2f}",
                fvd='-' if row_out['fvd'] is None else f"{row_out['fvd']:.2f}",
                clip='-' if row_out['clip'] is None else f"{row_out['clip']:.4f}",
                obj='-' if row_out['object_alignment'] is None else f"{row_out['object_alignment']:.2f}",
                mot='-' if row_out['motion_alignment'] is None else f"{row_out['motion_alignment']:.2f}",
                psnr='-' if row_out['psnr'] is None else f"{row_out['psnr']:.2f}",
                ssim='-' if row_out['ssim'] is None else f"{row_out['ssim']:.3f}",
                dino='-' if row_out['dino'] is None else f"{row_out['dino']:.3f}",
            ))
        lines.append('')
    lines.append('## Table 1 Reference')
    lines.append('')
    for split in ['real', 'synthetic']:
        lines.append(f'### {split.title()}')
        lines.append('')
        lines.append('| Task | Time | Tokens | Success | FVD | CLIP | Obj | Motion | PSNR | SSIM | DINO |')
        lines.append('| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
        for label in ['Text-to-Lottie', 'Text-Image-to-Lottie', 'Video-to-Lottie']:
            ref = PAPER[split][label]
            lines.append('| {task} | {timev} | {tok} | {succ}% | {fvd} | {clip} | {obj} | {mot} | {psnr} | {ssim} | {dino} |'.format(
                task=label,
                timev=ref.get('time', '-'),
                tok=ref.get('tokens', '-'),
                succ=ref.get('success', '-'),
                fvd=ref.get('fvd', '-'),
                clip=ref.get('clip', '-'),
                obj=ref.get('obj', '-'),
                mot=ref.get('motion', '-'),
                psnr=ref.get('psnr', '-'),
                ssim=ref.get('ssim', '-'),
                dino=ref.get('dino', '-'),
            ))
        lines.append('')
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    OUT_MD.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('WROTE', OUT_JSON)
    print('WROTE', OUT_MD)
    print(json.dumps(report, ensure_ascii=False, indent=2)[:4000])

if __name__ == '__main__':
    main()
