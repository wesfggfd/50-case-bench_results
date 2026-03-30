#!/usr/bin/env python3
import base64
import io
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import requests
from datasets import Video, load_dataset
from PIL import Image
from lottie.exporters.video import export_video
from lottie.parsers.tgs import parse_tgs

API_URL = os.environ.get('BENCH_JUDGE_API_URL', 'https://aiapi.cxyquan.com/v1/messages')
API_KEY = os.environ.get('BENCH_JUDGE_API_KEY')
MODEL = os.environ.get('BENCH_JUDGE_MODEL', 'claude-sonnet-4-6')


def build_headers():
    if not API_KEY:
        raise RuntimeError('Missing BENCH_JUDGE_API_KEY environment variable for judge evaluation')
    return {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json',
    }
BENCH_FILES = {
    'real': os.environ.get('MMLOTTIE_BENCH_REAL', '/root/SVG Generation/downloads/bench/MMLottieBench/data/real-00000-of-00001.parquet'),
    'synthetic': os.environ.get('MMLOTTIE_BENCH_SYNTHETIC', '/root/SVG Generation/downloads/bench/MMLottieBench/data/synthetic-00000-of-00001.parquet'),
}
RESULTS_ROOT = Path(os.environ.get('MMLOTTIE_RESULTS_ROOT', '/root/SVG Generation/results/official_rerun'))
RESULTS = {
    split: {
        task_key: RESULTS_ROOT / f'mmlottie_bench_{split}_{task_key}'
        for task_key in ['text2lottie', 'text_image2lottie']
    }
    for split in ['real', 'synthetic']
}
TASK_ORDER = [
    ('text2lottie', 'Text-to-Lottie'),
    ('text_image2lottie', 'Text-Image-to-Lottie'),
]
BENCH_VIDEO_FPS = 8.0
BENCH_VIDEO_WIDTH = 336
BENCH_VIDEO_HEIGHT = 336
BENCH_VIDEO_FRAME_COUNT = 16
JUDGE_FRAME_SIZE = 224
RENDER_CACHE = Path('/root/SVG Generation/results/render_cache_official')
CACHE_ROOT = Path('/root/SVG Generation/results/judge_cache_official_claude46')
REPORT_PATH = Path(os.environ.get('MMLOTTIE_JUDGE_REPORT_PATH', '/root/SVG Generation/OmniLottie/reproduction_results/judge_metrics_report.json'))
LOG_PATH = Path(os.environ.get('MMLOTTIE_JUDGE_LOG_PATH', '/root/SVG Generation/results/judge_metrics_official.log'))

OBJ_PROMPT = """You are a professional animation evaluator tasked with assessing AI-generated Lottie animations. Your response must strictly follow this JSON format:
{"object_consistency_score": <score>, "object_reasoning": "..."}

Evaluate the generated Lottie animation against the given caption on the Object Consistency dimension. Rate from 0 to 10 based on whether the objects described in the caption are present in the animation and how accurately they are represented.
Scoring Criteria:
0: No objects from the caption are present, or the animation is blank.
1-2: Objects are barely recognizable or severely inaccurate.
3-4: Some objects are present but with major inaccuracies in type, appearance, quantity, or visual characteristics.
5-6: Main objects are present and somewhat recognizable, but with notable errors in details.
7-8: Objects are accurately represented with only minor inaccuracies.
9: Objects are very accurately represented with only extremely subtle imperfections.
10: Objects perfectly match the caption description in all aspects.
Assess whether all objects are present, types are correct, quantities are correct, colors/shapes/styles are accurate, and spatial relations are correct.
Return JSON only.
"""

MOTION_PROMPT = """You are a professional animation evaluator tasked with assessing AI-generated Lottie animations. Your response must strictly follow this JSON format:
{"motion_consistency_score": <score>, "motion_reasoning": "..."}

Evaluate whether the motion/animation described in the caption is correctly executed, regardless of object accuracy. Rate from 0 to 10.
Scoring Criteria:
0: No objects visible or no motion when described.
1-2: Motion completely wrong or absent.
3-4: Major errors in motion type, direction, or magnitude.
5-6: Motion type correct but notable execution errors.
7-8: Accurately executed with only minor detail errors.
9: Very accurate with extremely subtle imperfections.
10: Perfect match in type, direction, magnitude, and target.
Assess motion type, direction, magnitude, target object accuracy, and smoothness. Motion can be scored independently of object accuracy.
Return JSON only.
"""


def log(message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    with LOG_PATH.open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def load_bench():
    dataset = load_dataset('parquet', data_files=BENCH_FILES)
    return {split: dataset[split].cast_column('video', Video(decode=False)) for split in BENCH_FILES}


def ensure_render(split: str, task_key: str, sample_id: str, result_dir: Path) -> Path:
    out_path = RENDER_CACHE / split / task_key / f'{sample_id}.mp4'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    anim = parse_tgs(str(result_dir / f'{sample_id}.json'))
    export_video(anim, str(out_path), format='mp4')
    return out_path


def sample_frames(video_path: str, num_frames: int = BENCH_VIDEO_FRAME_COUNT):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f'No frames decoded from {video_path}')
    idx = sorted(set(int(x) for x in np.linspace(0, len(frames) - 1, min(num_frames, len(frames)))))
    encoded = []
    for i in idx:
        img = Image.fromarray(frames[i]).convert('RGB').resize((JUDGE_FRAME_SIZE, JUDGE_FRAME_SIZE))
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        encoded.append(base64.b64encode(buf.getvalue()).decode())
    return encoded


def parse_anthropic_json(response_json):
    content = response_json.get('content', [])
    text = ''.join(block.get('text', '') for block in content if block.get('type') == 'text')
    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1:
        raise ValueError(f'Model response did not contain JSON: {text[:300]}')
    return json.loads(text[start : end + 1])


def call_claude(prompt_prefix: str, caption: str, frame_b64_list, max_retries: int = 8):
    payload = {
        'model': MODEL,
        'max_tokens': 512,
        'messages': [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': prompt_prefix + '\nCaption: ' + caption},
                    *[
                        {
                            'type': 'image',
                            'source': {
                                'type': 'base64',
                                'media_type': 'image/png',
                                'data': b64,
                            },
                        }
                        for b64 in frame_b64_list
                    ],
                ],
            }
        ],
    }
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(API_URL, headers=build_headers(), json=payload, timeout=120)
            resp.raise_for_status()
            return parse_anthropic_json(resp.json())
        except Exception as exc:
            last_error = exc
            sleep_s = min(60, 2 ** attempt)
            log(f'retry {attempt}/{max_retries} after error: {exc}')
            time.sleep(sleep_s)
    raise RuntimeError(f'Claude request failed after {max_retries} attempts: {last_error}')


def cache_path(split: str, task_key: str, sample_id: str) -> Path:
    path = CACHE_ROOT / split / task_key / f'{sample_id}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def valid_record(rec):
    return isinstance(rec, dict) and 'object_consistency_score' in rec and 'motion_consistency_score' in rec


def write_report(records):
    aggregate = {
        'model': MODEL,
        'api_url': API_URL,
        'notes': [
            f'Object/Motion were evaluated with 8 uniformly sampled rendered frames on outputs under {RESULTS_ROOT}.',
            'Judge requests use the Anthropic-style messages API endpoint provided by the user.',
        ],
        'metrics': {},
    }
    grouped = defaultdict(list)
    for rec in records:
        grouped[(rec['split'], rec['task_type'])].append(rec)
    for split in ['real', 'synthetic']:
        aggregate['metrics'][split] = {}
        for _, task in TASK_ORDER:
            bucket = grouped.get((split, task), [])
            obj = [float(r['object_consistency_score']) for r in bucket if 'object_consistency_score' in r]
            mot = [float(r['motion_consistency_score']) for r in bucket if 'motion_consistency_score' in r]
            aggregate['metrics'][split][task] = {
                'count': len(bucket),
                'object_alignment': round(sum(obj) / len(obj), 6) if obj else None,
                'motion_alignment': round(sum(mot) / len(mot), 6) if mot else None,
                'evaluated_count_obj': len(obj),
                'evaluated_count_motion': len(mot),
            }
    REPORT_PATH.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    bench = load_bench()
    final_records = []
    all_items = []
    for split in ['real', 'synthetic']:
        rows = list(bench[split])
        for task_key, task_label in TASK_ORDER:
            result_dir = RESULTS[split][task_key]
            for row in rows:
                if row['task_type'] != task_label:
                    continue
                if not (result_dir / f"{row['id']}.json").exists():
                    continue
                all_items.append((split, task_key, task_label, row))
    total = len(all_items)
    for idx, (split, task_key, task_label, row) in enumerate(all_items, 1):
        sample_id = row['id']
        cpath = cache_path(split, task_key, sample_id)
        if cpath.exists():
            rec = json.loads(cpath.read_text(encoding='utf-8'))
            if valid_record(rec):
                final_records.append(rec)
                continue
        try:
            render_path = ensure_render(split, task_key, sample_id, RESULTS[split][task_key])
            frames = sample_frames(str(render_path), num_frames=8)
            obj = call_claude(OBJ_PROMPT, row['text'], frames)
            mot = call_claude(MOTION_PROMPT, row['text'], frames)
            rec = {
                'id': sample_id,
                'split': split,
                'task_type': task_label,
                'object_consistency_score': float(obj['object_consistency_score']),
                'object_reasoning': obj['object_reasoning'],
                'motion_consistency_score': float(mot['motion_consistency_score']),
                'motion_reasoning': mot['motion_reasoning'],
            }
            cpath.write_text(json.dumps(rec, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            final_records.append(rec)
        except Exception as exc:
            rec = {
                'id': sample_id,
                'split': split,
                'task_type': task_label,
                'error': str(exc),
            }
            cpath.write_text(json.dumps(rec, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        if idx % 10 == 0:
            write_report(final_records)
            log(f'processed {idx}/{total}')
    write_report(final_records)
    log('DONE')


if __name__ == '__main__':
    main()
