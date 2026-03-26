#!/usr/bin/env python3
import argparse
import glob
import hashlib
import io
import json
import os
import random
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import open_clip
import pyarrow.parquet as pq
import requests
import scipy.linalg
import timm
import torch
import torch.nn.functional as F
from datasets import Video, load_dataset
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

from lottie.exporters.cairo import export_png
from lottie.exporters.video import export_video
from lottie.parsers.tgs import parse_tgs

BENCH_FILES = {
    'real': '/root/SVG Generation/downloads/bench/MMLottieBench/data/real-00000-of-00001.parquet',
    'synthetic': '/root/SVG Generation/downloads/bench/MMLottieBench/data/synthetic-00000-of-00001.parquet',
}
RESULTS = {
    'real': {
        'text2lottie': Path('/root/SVG Generation/results/official_rerun/real_text2lottie/mmlottie_bench_real'),
        'text_image2lottie': Path('/root/SVG Generation/results/official_rerun/real_text_image2lottie/mmlottie_bench_real'),
        'video2lottie': Path('/root/SVG Generation/results/official_rerun/real_video2lottie/mmlottie_bench_real'),
    },
    'synthetic': {
        'text2lottie': Path('/root/SVG Generation/results/official_rerun/synthetic_text2lottie/mmlottie_bench_synthetic'),
        'text_image2lottie': Path('/root/SVG Generation/results/official_rerun/synthetic_text_image2lottie/mmlottie_bench_synthetic'),
        'video2lottie': Path('/root/SVG Generation/results/official_rerun/synthetic_video2lottie/mmlottie_bench_synthetic'),
    },
}
TASK_LABELS = {
    'text2lottie': 'Text-to-Lottie',
    'text_image2lottie': 'Text-Image-to-Lottie',
    'video2lottie': 'Video-to-Lottie',
}
TEXT_TASKS = {'Text-to-Lottie', 'Text-Image-to-Lottie'}
VIDEO_TASK = 'Video-to-Lottie'
TASK_ORDER = ['text2lottie', 'text_image2lottie', 'video2lottie']
MML2M_GLOB = '/root/SVG Generation/downloads/datasets/MMLottie-2M/data/**/*.parquet'
RENDER_CACHE = Path('/root/SVG Generation/results/render_cache_official')
FVD_CACHE = Path('/root/SVG Generation/results/fvd_cache')
REPORT_PATH = Path('/root/SVG Generation/OmniLottie/reproduction_results/core_metrics_report.json')
I3D_URL = 'https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1'
MODEL_CACHE_DIR = '/root/SVG Generation/OmniLottie/loaded_models'
SAMPLE_SEED = 42
NUM_FRAMES = 16
FRAME_SIZE = 224
BATCH_SIZE = 4
FEATURE_DETECTOR_CACHE = {}


@dataclass
class RunningStats:
    count: int = 0
    mean: np.ndarray | None = None
    m2: np.ndarray | None = None

    def update(self, features: np.ndarray) -> None:
        features = np.asarray(features, dtype=np.float64)
        if features.ndim == 1:
            features = features[None, :]
        if features.size == 0:
            return
        batch_count = features.shape[0]
        batch_mean = features.mean(axis=0)
        centered = features - batch_mean
        batch_m2 = centered.T @ centered
        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.m2 = self.m2 + batch_m2 + np.outer(delta, delta) * self.count * batch_count / total
        self.mean = self.mean + delta * batch_count / total
        self.count = total

    def covariance(self) -> np.ndarray:
        if self.count <= 1:
            dim = 1 if self.mean is None else len(self.mean)
            return np.zeros((dim, dim), dtype=np.float64)
        return self.m2 / (self.count - 1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--gt-stats-only', action='store_true')
    parser.add_argument('--force-gt', action='store_true')
    parser.add_argument('--force-generated', action='store_true')
    parser.add_argument('--max-gt-videos', type=int, default=None)
    return parser.parse_args()


def load_bench():
    dataset = load_dataset('parquet', data_files=BENCH_FILES)
    return {split: dataset[split].cast_column('video', Video(decode=False)) for split in BENCH_FILES}


def load_video_frames_cv2(path: str, num_frames: int = NUM_FRAMES, size: int = FRAME_SIZE) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    if not frames:
        raise ValueError(f'No frames decoded from {path}')
    idx = np.linspace(0, len(frames) - 1, num_frames).astype(int)
    picked = []
    for i in idx:
        img = Image.fromarray(frames[i]).convert('RGB')
        if size is not None:
            if isinstance(size, tuple):
                img = img.resize(size)
            else:
                img = img.resize((size, size))
        picked.append(np.array(img))
    return np.stack(picked)


def load_video_frames_from_bytes(video_bytes: bytes, num_frames: int = NUM_FRAMES, size: int = FRAME_SIZE) -> np.ndarray:
    tmp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    try:
        tmp.write(video_bytes)
        tmp.close()
        return load_video_frames_cv2(tmp.name, num_frames=num_frames, size=size)
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def load_video_frames_from_field(video_field, num_frames: int = NUM_FRAMES, size: int | tuple[int, int] | None = FRAME_SIZE) -> np.ndarray:
    if isinstance(video_field, dict) and video_field.get('bytes') is not None:
        return load_video_frames_from_bytes(video_field['bytes'], num_frames=num_frames, size=size)
    if isinstance(video_field, dict) and video_field.get('path'):
        return load_video_frames_cv2(video_field['path'], num_frames=num_frames, size=size)
    if isinstance(video_field, bytes):
        return load_video_frames_from_bytes(video_field, num_frames=num_frames, size=size)
    raise ValueError('Video field does not contain bytes/path')


def composite_on_background(img: Image.Image, background=(255, 255, 255)) -> Image.Image:
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        rgba = img.convert('RGBA')
        base = Image.new('RGB', rgba.size, background)
        base.paste(rgba, mask=rgba.getchannel('A'))
        return base
    return img.convert('RGB')


def render_sampled_lottie_frames(
    json_path: Path,
    num_frames: int = NUM_FRAMES,
    target_size: tuple[int, int] | None = None,
    background=(255, 255, 255),
) -> np.ndarray:
    anim = parse_tgs(str(json_path))
    start = int(anim.in_point)
    end = int(anim.out_point)
    frame_ids = np.linspace(start, end, num_frames).astype(int)
    picked = []
    for frame_id in frame_ids:
        buf = io.BytesIO()
        export_png(anim, buf, frame=frame_id)
        buf.seek(0)
        img = composite_on_background(Image.open(buf), background=background)
        if target_size is not None:
            img = img.resize(target_size)
        picked.append(np.array(img))
    if not picked:
        raise ValueError(f'No rendered frames from {json_path}')
    return np.stack(picked)


def ensure_render(split: str, task_key: str, sample_id: str, result_dir: Path) -> Path:
    out_path = RENDER_CACHE / split / task_key / f'{sample_id}.mp4'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    anim = parse_tgs(str(result_dir / f'{sample_id}.json'))
    export_video(anim, str(out_path), format='mp4')
    return out_path


def open_url(url: str, num_attempts: int = 10, cache_dir: str = MODEL_CACHE_DIR):
    url_md5 = hashlib.md5(url.encode('utf-8')).hexdigest()
    cache_files = glob.glob(os.path.join(cache_dir, url_md5 + '_*'))
    if len(cache_files) == 1:
        return open(cache_files[0], 'rb')
    with requests.Session() as session:
        for attempts_left in reversed(range(num_attempts)):
            try:
                with session.get(url) as res:
                    res.raise_for_status()
                    url_data = res.content
                    break
            except Exception:
                if not attempts_left:
                    raise
        safe_name = re.sub(r'[^0-9a-zA-Z-._]', '_', url)
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, url_md5 + '_' + safe_name)
        with open(cache_file, 'wb') as f:
            f.write(url_data)
        return io.BytesIO(url_data)


def get_detector(device: str):
    key = (I3D_URL, device)
    if key not in FEATURE_DETECTOR_CACHE:
        with open_url(I3D_URL) as f:
            FEATURE_DETECTOR_CACHE[key] = torch.jit.load(f).eval().to(device)
    return FEATURE_DETECTOR_CACHE[key]


def frames_to_clip_tensor(frames: np.ndarray) -> torch.Tensor:
    return torch.tensor(frames).permute(3, 0, 1, 2).float()


def update_stats_from_clip_tensors(stats: RunningStats, detector, device: str, clips: list[torch.Tensor]) -> None:
    if not clips:
        return
    batch = torch.stack(clips).to(device)
    with torch.no_grad():
        features = detector(batch, rescale=True, resize=True, return_features=True)
    stats.update(features.detach().cpu().numpy())


def frechet_distance(mu1, sigma1, mu2, sigma2):
    mu1 = np.asarray(mu1, dtype=np.float64)
    mu2 = np.asarray(mu2, dtype=np.float64)
    sigma1 = np.asarray(sigma1, dtype=np.float64)
    sigma2 = np.asarray(sigma2, dtype=np.float64)
    eps = 1e-6
    covmean, _ = scipy.linalg.sqrtm((sigma1 + np.eye(sigma1.shape[0]) * eps) @ (sigma2 + np.eye(sigma2.shape[0]) * eps), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu1 - mu2
    return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean))


def parquet_paths_and_counts():
    paths = sorted(glob.glob(MML2M_GLOB, recursive=True))
    if not paths:
        raise FileNotFoundError(f'No parquet files matched {MML2M_GLOB}')
    counts = [pq.ParquetFile(path).metadata.num_rows for path in paths]
    return paths, counts


def iter_sampled_gt_video_fields(target_count: int):
    paths, counts = parquet_paths_and_counts()
    total = sum(counts)
    if target_count <= 0:
        return
    sample_size = min(target_count, total)
    selected = sorted(random.Random(SAMPLE_SEED).sample(range(total), sample_size))
    pointer = 0
    global_row = 0
    for path, row_count in zip(paths, counts):
        if pointer >= len(selected):
            break
        parquet = pq.ParquetFile(path)
        local_start = global_row
        for batch in parquet.iter_batches(columns=['video'], batch_size=64):
            batch_start = local_start
            batch_end = batch_start + batch.num_rows
            video_col = batch.column(0)
            while pointer < len(selected) and selected[pointer] < batch_end:
                local_index = selected[pointer] - batch_start
                yield video_col[local_index].as_py(), selected[pointer], len(selected), total
                pointer += 1
            local_start = batch_end
        global_row += row_count


def save_stats(path: Path, stats: RunningStats, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, mean=stats.mean, cov=stats.covariance(), count=stats.count)
    meta_path = path.with_suffix('.json')
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def load_stats(path: Path):
    arr = np.load(path)
    return {
        'mean': arr['mean'],
        'cov': arr['cov'],
        'count': int(arr['count']),
    }


def gt_stats_paths(target_count: int):
    stem = f'mmlottie2m_seed{SAMPLE_SEED}_count{target_count}_i3d_stats'
    return FVD_CACHE / f'{stem}.npz', FVD_CACHE / f'{stem}.json'


def compute_gt_stats(device: str, target_count: int, force: bool = False):
    gt_stats_path, gt_meta_path = gt_stats_paths(target_count)
    if gt_stats_path.exists() and not force:
        print('Using cached GT stats', gt_stats_path)
        cached = load_stats(gt_stats_path)
        meta = json.loads(gt_meta_path.read_text(encoding='utf-8')) if gt_meta_path.exists() else {}
        cached['meta'] = meta
        return cached
    detector = get_detector(device)
    stats = RunningStats()
    clips = []
    ok = 0
    failed = 0
    total_available = None
    total_rows = None
    for video_field, global_index, sample_total, dataset_total in iter_sampled_gt_video_fields(target_count=target_count):
        total_available = sample_total
        total_rows = dataset_total
        try:
            frames = load_video_frames_from_field(video_field)
            clips.append(frames_to_clip_tensor(frames))
        except Exception as exc:
            failed += 1
            print('GT_DECODE_FAIL', global_index, exc, flush=True)
            continue
        ok += 1
        if len(clips) >= BATCH_SIZE:
            update_stats_from_clip_tensors(stats, detector, device, clips)
            clips.clear()
        if ok % 200 == 0:
            print('GT_PROGRESS', ok, 'decoded', 'failed', failed, 'sample_total', total_available, flush=True)
    if clips:
        update_stats_from_clip_tensors(stats, detector, device, clips)
    meta = {
        'source': 'MMLottie-2M',
        'sample_seed': SAMPLE_SEED,
        'dataset_total_rows': total_rows,
        'sample_target_count': total_available,
        'feature_count': stats.count,
        'decode_failures': failed,
        'num_frames': NUM_FRAMES,
        'frame_size': FRAME_SIZE,
        'device': device,
    }
    save_stats(gt_stats_path, stats, meta)
    gt_meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('WROTE', gt_stats_path)
    return {'mean': stats.mean, 'cov': stats.covariance(), 'count': stats.count, 'meta': meta}


def compute_generated_fvd_stats(split: str, task_key: str, rows, device: str, force: bool = False):
    cache_path = FVD_CACHE / f'generated_{split}_{task_key}_i3d_stats.npz'
    if cache_path.exists() and not force:
        print('Using cached generated stats', cache_path)
        cached = load_stats(cache_path)
        meta_path = cache_path.with_suffix('.json')
        meta = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {}
        cached['meta'] = meta
        return cached
    detector = get_detector(device)
    stats = RunningStats()
    clips = []
    ok = 0
    failed = 0
    for row in rows:
        sample_id = row['id']
        result_dir = RESULTS[split][task_key]
        if not (result_dir / f'{sample_id}.json').exists():
            continue
        try:
            render_path = ensure_render(split, task_key, sample_id, result_dir)
            frames = load_video_frames_cv2(str(render_path))
            clips.append(frames_to_clip_tensor(frames))
        except Exception as exc:
            failed += 1
            print('GEN_DECODE_FAIL', split, task_key, sample_id, exc, flush=True)
            continue
        ok += 1
        if len(clips) >= BATCH_SIZE:
            update_stats_from_clip_tensors(stats, detector, device, clips)
            clips.clear()
        if ok % 25 == 0:
            print('GEN_PROGRESS', split, task_key, ok, 'failed', failed, flush=True)
    if clips:
        update_stats_from_clip_tensors(stats, detector, device, clips)
    meta = {'split': split, 'task_key': task_key, 'feature_count': stats.count, 'decode_failures': failed, 'device': device}
    save_stats(cache_path, stats, meta)
    return {'mean': stats.mean, 'cov': stats.covariance(), 'count': stats.count, 'meta': meta}


def cosine_mean(a, b):
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return float((a * b).sum(dim=-1).mean().item())


def compute_clip_for_rows(split: str, task_key: str, rows, device: str):
    clip_model, _, clip_pre = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    clip_model = clip_model.to(device).eval()
    clip_tok = open_clip.get_tokenizer('ViT-B-32')
    scores = []
    result_dir = RESULTS[split][task_key]
    for row in rows:
        sample_id = row['id']
        if not (result_dir / f'{sample_id}.json').exists():
            continue
        try:
            render_path = ensure_render(split, task_key, sample_id, result_dir)
            frames = load_video_frames_cv2(str(render_path), num_frames=8, size=FRAME_SIZE)
            with torch.no_grad():
                imgs = torch.stack([clip_pre(Image.fromarray(x)) for x in frames]).to(device)
                txt = clip_tok([row['text']]).to(device)
                imf = clip_model.encode_image(imgs)
                tf = clip_model.encode_text(txt)
                score = float((F.normalize(imf, dim=-1) @ F.normalize(tf, dim=-1).T).mean().item())
            scores.append(score)
        except Exception as exc:
            print('CLIP_FAIL', split, task_key, sample_id, exc, flush=True)
    return {
        'clip': round(float(sum(scores) / len(scores)), 6) if scores else None,
        'clip_count': len(scores),
    }


def compute_video_pair_metrics(split: str, task_key: str, rows, device: str):
    trans = Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dino_model = timm.create_model('vit_base_patch16_224.dino', pretrained=True, num_classes=0).to(device).eval()
    psnrs = []
    ssims = []
    dinos = []
    result_dir = RESULTS[split][task_key]
    for row in rows:
        sample_id = row['id']
        if not (result_dir / f'{sample_id}.json').exists():
            continue
        try:
            ref_frames = load_video_frames_from_field(row['video'], size=None)
            target_size = (ref_frames.shape[2], ref_frames.shape[1])
            gen_frames = render_sampled_lottie_frames(
                result_dir / f'{sample_id}.json',
                num_frames=len(ref_frames),
                target_size=target_size,
                background=(255, 255, 255),
            )
            local_psnr = []
            local_ssim = []
            for gen_frame, ref_frame in zip(gen_frames, ref_frames):
                gen_float = gen_frame.astype(np.float32) / 255.0
                ref_float = ref_frame.astype(np.float32) / 255.0
                local_psnr.append(peak_signal_noise_ratio(ref_float, gen_float, data_range=1.0))
                local_ssim.append(structural_similarity(ref_float, gen_float, data_range=1.0, channel_axis=2))
            # Per-sample PSNR: average only over frames with finite PSNR; omit inf (e.g. MSE=0).
            psnr_for_avg = [p for p in local_psnr if not np.isinf(p)]
            if psnr_for_avg:
                psnrs.append(float(np.mean(psnr_for_avg)))
            else:
                psnrs.append(float('nan'))
            ssims.append(float(sum(local_ssim) / len(local_ssim)))
            with torch.no_grad():
                gen_tensor = torch.stack([trans(Image.fromarray(x)) for x in gen_frames]).to(device)
                ref_tensor = torch.stack([trans(Image.fromarray(x)) for x in ref_frames]).to(device)
                gen_feat = dino_model(gen_tensor)
                ref_feat = dino_model(ref_tensor)
                dinos.append(cosine_mean(gen_feat, ref_feat))
        except Exception as exc:
            print('VIDEO_PAIR_FAIL', split, task_key, sample_id, exc, flush=True)
    return {
        'psnr': round(float(np.nanmean(psnrs)), 6) if psnrs and np.any(np.isfinite(psnrs)) else None,
        'ssim': round(float(sum(ssims) / len(ssims)), 6) if ssims else None,
        'dino': round(float(sum(dinos) / len(dinos)), 6) if dinos else None,
        'pair_count': len(psnrs),
    }


def main():
    args = parse_args()
    device = args.device
    FVD_CACHE.mkdir(parents=True, exist_ok=True)
    RENDER_CACHE.mkdir(parents=True, exist_ok=True)
    if args.gt_stats_only:
        raise SystemExit('--gt-stats-only is no longer supported without a target count')
    bench = load_bench()
    report = {
        'protocol': {
            'results_root': '/root/SVG Generation/results/official_rerun',
            'fvd_gt_source': 'MMLottie-2M',
            'fvd_sample_seed': SAMPLE_SEED,
            'num_frames': NUM_FRAMES,
            'frame_size': FRAME_SIZE,
            'device': device,
        },
        'notes': [
            'FVD uses the user-specified protocol: compare each generated video set against a seed=42 random sample from MMLottie-2M with matched sample count.',
            'CLIP is computed only for Text-to-Lottie and Text-Image-to-Lottie.',
            'PSNR, SSIM, and DINO are computed only for Video-to-Lottie against the benchmark reference video.',
        ],
        'metrics': {},
    }
    for split in ['real', 'synthetic']:
        split_rows = list(bench[split])
        report['metrics'][split] = {}
        for task_key in TASK_ORDER:
            label = TASK_LABELS[task_key]
            task_rows = [row for row in split_rows if row['task_type'] == label]
            result_dir = RESULTS[split][task_key]
            generated = compute_generated_fvd_stats(split, task_key, task_rows, device=device, force=args.force_generated)
            gt_stats = compute_gt_stats(device=device, target_count=generated['count'], force=args.force_gt) if generated['count'] > 0 else None
            entry = {
                'generated_count': generated['count'],
                'fvd': round(frechet_distance(gt_stats['mean'], gt_stats['cov'], generated['mean'], generated['cov']), 6) if gt_stats and generated['count'] > 1 and gt_stats['count'] > 1 else None,
                'fvd_feature_count': generated['count'],
            }
            if label in TEXT_TASKS:
                entry.update(compute_clip_for_rows(split, task_key, task_rows, device=device))
                entry['psnr'] = None
                entry['ssim'] = None
                entry['dino'] = None
                entry['pair_count'] = None
            else:
                entry.update(compute_video_pair_metrics(split, task_key, task_rows, device=device))
                entry['clip'] = None
                entry['clip_count'] = None
            report['metrics'][split][label] = entry
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('WROTE', REPORT_PATH)
    print(json.dumps(report, ensure_ascii=False, indent=2)[:4000])


if __name__ == '__main__':
    main()
