#!/usr/bin/env python3
import argparse
import base64
import io
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
# Prefer the repo-bundled lottie package over any site-packages install.
sys.path.insert(0, str(WORKSPACE_ROOT / "OpenVGLab-OmniLottie"))
sys.path.insert(1, str(WORKSPACE_ROOT / "OmniLottie"))

from lottie.exporters.video import export_video
from lottie.parsers.tgs import parse_tgs

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_BUNDLE_DIR = BASE_DIR / "official_bundle" / "public_standard_bundle"

JOINT_PROMPT = """You are a professional animation evaluator tasked with assessing AI-generated Lottie animations.
Your response must strictly follow this JSON format:
{"object_consistency_score": <score>, "object_reasoning": "...", "motion_consistency_score": <score>, "motion_reasoning": "..."}

Evaluate the generated Lottie animation against the given caption on two independent dimensions:

1. Object Consistency (0-10)
0: No objects from the caption are present, or the animation is blank.
1-2: Objects are barely recognizable or severely inaccurate.
3-4: Some objects are present but with major inaccuracies in type, appearance, quantity, or visual characteristics.
5-6: Main objects are present and somewhat recognizable, but with notable errors in details.
7-8: Objects are accurately represented with only minor inaccuracies.
9: Objects are very accurately represented with only extremely subtle imperfections.
10: Objects perfectly match the caption description in all aspects.
Assess whether all objects are present, types are correct, quantities are correct, colors/shapes/styles are accurate, and spatial relations are correct.

2. Motion Consistency (0-10)
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


class RateLimitError(RuntimeError):
    def __init__(self, message: str, retry_after_seconds: int | None = None):
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--render-root", type=Path, default=None)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("--log-path", type=Path, default=None)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--sleep-after-success", type=float, default=0.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def normalize_paths(args):
    manifest = args.manifest or args.bundle_dir / "judge_manifest_public_standard.json"
    results_root = args.results_root or args.bundle_dir / "official_rerun_text"
    render_root = args.render_root or BASE_DIR / "render_cache_public_standard_local"
    cache_root = args.cache_root or BASE_DIR / "judge_cache_public_standard_claude35_aihhhl"
    report_path = args.report_path or BASE_DIR / "judge_metrics_report_public_local_claude35_aihhhl.json"
    log_path = args.log_path or BASE_DIR / "judge_metrics_public_local_claude35_aihhhl.log"
    return manifest, results_root, render_root, cache_root, report_path, log_path


def get_api_config():
    base_url = os.environ.get("JUDGE_API_URL", "https://ai.hhhl.cc").rstrip("/")
    api_key = os.environ.get("JUDGE_API_KEY")
    model = os.environ.get("JUDGE_MODEL", "claude-sonnet-3-5-latest")
    if not api_key:
        raise RuntimeError("Missing JUDGE_API_KEY in environment")
    return {
        "base_url": base_url,
        "messages_url": base_url if base_url.endswith("/v1/messages") else f"{base_url}/v1/messages",
        "chat_url": base_url if base_url.endswith("/v1/chat/completions") else f"{base_url}/v1/chat/completions",
        "api_key": api_key,
        "model": model,
    }


def log(log_path: Path, message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_manifest(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sample_frames(video_path: Path, num_frames: int = 8):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"No frames decoded from {video_path}")
    idx = sorted(set(int(x) for x in np.linspace(0, len(frames) - 1, min(num_frames, len(frames)))))
    encoded = []
    for i in idx:
        img = Image.fromarray(frames[i]).convert("RGB").resize((224, 224))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        encoded.append(base64.b64encode(buf.getvalue()).decode())
    return encoded


def parse_json_block(text: str):
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"Model response did not contain JSON: {text[:300]}")
    return json.loads(text[start : end + 1])


def parse_anthropic_json(response_json):
    content = response_json.get("content", [])
    text = "".join(block.get("text", "") for block in content if block.get("type") == "text")
    return parse_json_block(text)


def parse_openai_json(response_json):
    choices = response_json.get("choices") or []
    if not choices:
        raise ValueError(f"OpenAI-style response missing choices: {str(response_json)[:300]}")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        text = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    else:
        text = content or ""
    return parse_json_block(text)


def call_messages_api(api_cfg, prompt_prefix: str, caption: str, frame_b64_list):
    payload = {
        "model": api_cfg["model"],
        "max_tokens": 512,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_prefix + "\nCaption: " + caption},
                    *[
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": b64,
                            },
                        }
                        for b64 in frame_b64_list
                    ],
                ],
            }
        ],
    }
    headers = {"Authorization": f"Bearer {api_cfg['api_key']}", "Content-Type": "application/json"}
    resp = requests.post(api_cfg["messages_url"], headers=headers, json=payload, timeout=120)
    if resp.status_code == 429:
        try:
            body = resp.json()
        except Exception:
            body = {}
        retry_after = body.get("retry_after_seconds")
        raise RateLimitError(resp.text[:300], retry_after_seconds=retry_after)
    resp.raise_for_status()
    return parse_anthropic_json(resp.json()), "anthropic_messages"


def call_chat_api(api_cfg, prompt_prefix: str, caption: str, frame_b64_list):
    payload = {
        "model": api_cfg["model"],
        "max_tokens": 512,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_prefix + "\nCaption: " + caption},
                    *[
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        }
                        for b64 in frame_b64_list
                    ],
                ],
            }
        ],
    }
    headers = {"Authorization": f"Bearer {api_cfg['api_key']}", "Content-Type": "application/json"}
    resp = requests.post(api_cfg["chat_url"], headers=headers, json=payload, timeout=120)
    if resp.status_code == 429:
        try:
            body = resp.json()
            retry_after = body.get("retry_after_seconds") or body.get("error", {}).get("retry_after_seconds")
        except Exception:
            retry_after = None
        raise RateLimitError(resp.text[:300], retry_after_seconds=retry_after)
    resp.raise_for_status()
    return parse_openai_json(resp.json()), "openai_chat"


def call_judge(api_cfg, log_path: Path, prompt_prefix: str, caption: str, frame_b64_list, max_retries: int = 8):
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            try:
                return call_messages_api(api_cfg, prompt_prefix, caption, frame_b64_list)
            except RateLimitError:
                raise
            except Exception as exc:
                last_error = exc
                return call_chat_api(api_cfg, prompt_prefix, caption, frame_b64_list)
        except RateLimitError as exc:
            last_error = exc
            sleep_s = max(15, int(exc.retry_after_seconds or 15))
            log(log_path, f"retry {attempt}/{max_retries} after rate limit: sleep {sleep_s}s")
            time.sleep(sleep_s)
        except Exception as exc:
            last_error = exc
            sleep_s = min(60, 2**attempt)
            log(log_path, f"retry {attempt}/{max_retries} after error: {exc}")
            time.sleep(sleep_s)
    raise RuntimeError(f"Judge request failed after {max_retries} attempts: {last_error}")


def local_cache_path(cache_root: Path, item):
    path = cache_root / item["split"] / item["task_key"] / f'{item["id"]}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def valid_record(rec):
    return isinstance(rec, dict) and "object_consistency_score" in rec and "motion_consistency_score" in rec


def ensure_render(render_root: Path, results_root: Path, item):
    render_path = render_root / item["split"] / item["task_key"] / f'{item["id"]}.mp4'
    render_path.parent.mkdir(parents=True, exist_ok=True)
    if render_path.exists() and render_path.stat().st_size > 0:
        return render_path
    result_dir = results_root / f'{item["split"]}_{item["task_key"]}' / f'mmlottie_bench_{item["split"]}'
    json_path = result_dir / f'{item["id"]}.json'
    if not json_path.exists():
        raise FileNotFoundError(f"Missing generated JSON: {json_path}")
    anim = parse_tgs(str(json_path))
    export_video(anim, str(render_path), format="mp4")
    return render_path


def write_report(report_path: Path, api_cfg, records):
    aggregate = {
        "model": api_cfg["model"],
        "api_url": api_cfg["base_url"],
        "notes": [
            "Object/Motion were evaluated locally with 8 uniformly sampled rendered frames on downloaded official_rerun text-task outputs.",
            "Requests target the user-provided third-party API, keep only public-standard text tasks, and score object/motion in a single request per sample.",
        ],
        "metrics": {},
    }
    grouped = defaultdict(list)
    for rec in records:
        grouped[(rec["split"], rec["task_type"])].append(rec)
    for split in ["real", "synthetic"]:
        aggregate["metrics"][split] = {}
        for task in ["Text-to-Lottie", "Text-Image-to-Lottie"]:
            bucket = grouped.get((split, task), [])
            obj = [float(r["object_consistency_score"]) for r in bucket if "object_consistency_score" in r]
            mot = [float(r["motion_consistency_score"]) for r in bucket if "motion_consistency_score" in r]
            aggregate["metrics"][split][task] = {
                "count": len(bucket),
                "object_alignment": round(sum(obj) / len(obj), 6) if obj else None,
                "motion_alignment": round(sum(mot) / len(mot), 6) if mot else None,
                "evaluated_count_obj": len(obj),
                "evaluated_count_motion": len(mot),
            }
    report_path.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    api_cfg = get_api_config()
    manifest_path, results_root, render_root, cache_root, report_path, log_path = normalize_paths(args)
    cache_root.mkdir(parents=True, exist_ok=True)
    render_root.mkdir(parents=True, exist_ok=True)
    items = load_manifest(manifest_path)
    if args.offset:
        items = items[args.offset :]
    if args.max_items is not None:
        items = items[: args.max_items]
    final_records = []
    for idx, item in enumerate(items, 1):
        out_path = local_cache_path(cache_root, item)
        if out_path.exists() and not args.force:
            rec = json.loads(out_path.read_text(encoding="utf-8"))
            if valid_record(rec):
                final_records.append(rec)
                continue
        try:
            video_path = ensure_render(render_root, results_root, item)
            frames = sample_frames(video_path, num_frames=8)
            judge, backend = call_judge(api_cfg, log_path, JOINT_PROMPT, item["text"], frames)
            rec = {
                "id": item["id"],
                "split": item["split"],
                "task_type": item["task_type"],
                "task_key": item["task_key"],
                "judge_backend": backend,
                "object_consistency_score": float(judge["object_consistency_score"]),
                "object_reasoning": judge.get("object_reasoning", ""),
                "motion_consistency_score": float(judge["motion_consistency_score"]),
                "motion_reasoning": judge.get("motion_reasoning", ""),
            }
            out_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            final_records.append(rec)
            if args.sleep_after_success > 0:
                time.sleep(args.sleep_after_success)
        except Exception as exc:
            rec = {
                "id": item["id"],
                "split": item["split"],
                "task_type": item["task_type"],
                "task_key": item["task_key"],
                "error": str(exc),
            }
            out_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if idx % 10 == 0:
            write_report(report_path, api_cfg, final_records)
            log(log_path, f"processed {idx}/{len(items)}")
    write_report(report_path, api_cfg, final_records)
    log(log_path, "DONE")


if __name__ == "__main__":
    main()
