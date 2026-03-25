import argparse
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
from PIL import Image

from lottie.exporters.video import export_video
from lottie.parsers.tgs import parse_tgs


BASE_DIR = Path("/home/xinyu-chen/SVG Generation/judge-local")
BUNDLE_DIR = BASE_DIR / "official_bundle" / "judge_official_bundle"
MANIFEST_PATH = BUNDLE_DIR / "judge_manifest_official.json"
RESULTS_ROOT = BUNDLE_DIR / "official_rerun_text"
REMOTE_CACHE_ROOT = BUNDLE_DIR / "judge_cache_official_claude46"
RENDER_ROOT = BASE_DIR / "render_cache_official_local"
LOCAL_CACHE_ROOT = BASE_DIR / "judge_cache_official_local_claude46"
REPORT_PATH = BASE_DIR / "judge_metrics_report_official_local_claude46.json"
LOG_PATH = BASE_DIR / "judge_metrics_official_local.log"

API_URL = "https://aiapi.cxyquan.com/v1/messages"
API_KEY = "sk-SmgUodnjVrKR6OGrZ51GcZL8G7QoFuoUMiFS5wOGL7iKGfM5"
MODEL = "claude-sonnet-4-6"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

RESULT_DIRS = {
    ("real", "text2lottie"): RESULTS_ROOT / "real_text2lottie" / "mmlottie_bench_real",
    ("real", "text_image2lottie"): RESULTS_ROOT / "real_text_image2lottie" / "mmlottie_bench_real",
    ("synthetic", "text2lottie"): RESULTS_ROOT / "synthetic_text2lottie" / "mmlottie_bench_synthetic",
    ("synthetic", "text_image2lottie"): RESULTS_ROOT / "synthetic_text_image2lottie" / "mmlottie_bench_synthetic",
}

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-items", type=int, default=None)
    return parser.parse_args()


def log(message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_manifest():
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def parse_anthropic_json(response_json):
    content = response_json.get("content", [])
    text = "".join(block.get("text", "") for block in content if block.get("type") == "text")
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"Model response did not contain JSON: {text[:300]}")
    return json.loads(text[start : end + 1])


def call_claude(prompt_prefix: str, caption: str, frame_b64_list, max_retries: int = 8):
    payload = {
        "model": MODEL,
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

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(API_URL, headers=HEADERS, json=payload, timeout=120)
            resp.raise_for_status()
            return parse_anthropic_json(resp.json())
        except Exception as exc:
            last_error = exc
            sleep_s = min(60, 2 ** attempt)
            log(f"retry {attempt}/{max_retries} after error: {exc}")
            time.sleep(sleep_s)
    raise RuntimeError(f"Claude request failed after {max_retries} attempts: {last_error}")


def remote_cache_path(item):
    return REMOTE_CACHE_ROOT / item["split"] / item["task_key"] / f'{item["id"]}.json'


def local_cache_path(item):
    path = LOCAL_CACHE_ROOT / item["split"] / item["task_key"] / f'{item["id"]}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def valid_record(rec):
    return (
        isinstance(rec, dict)
        and "object_consistency_score" in rec
        and "motion_consistency_score" in rec
    )


def ensure_render(item):
    render_path = RENDER_ROOT / item["split"] / item["task_key"] / f'{item["id"]}.mp4'
    render_path.parent.mkdir(parents=True, exist_ok=True)
    if render_path.exists() and render_path.stat().st_size > 0:
        return render_path
    result_dir = RESULT_DIRS[(item["split"], item["task_key"])]
    json_path = result_dir / f'{item["id"]}.json'
    if not json_path.exists():
        raise FileNotFoundError(f"Missing generated JSON: {json_path}")
    anim = parse_tgs(str(json_path))
    export_video(anim, str(render_path), format="mp4")
    return render_path


def write_report(records):
    aggregate = {
        "model": MODEL,
        "api_url": API_URL,
        "notes": [
            "Object/Motion were evaluated locally with 8 uniformly sampled rendered frames on official_rerun outputs.",
            "Official manifest contains only successful text-task outputs from official_rerun.",
            "Existing valid remote official cache entries were reused before local re-evaluation.",
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

    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    LOCAL_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    RENDER_ROOT.mkdir(parents=True, exist_ok=True)
    items = load_manifest()
    if args.max_items is not None:
        items = items[: args.max_items]
    final_records = []

    for idx, item in enumerate(items, 1):
        out_path = local_cache_path(item)

        if out_path.exists():
            with out_path.open("r", encoding="utf-8") as f:
                rec = json.load(f)
            if valid_record(rec):
                final_records.append(rec)
                continue

        rpath = remote_cache_path(item)
        if rpath.exists():
            with rpath.open("r", encoding="utf-8") as f:
                rec = json.load(f)
            if valid_record(rec):
                rec["split"] = item["split"]
                rec["task_type"] = item["task_type"]
                out_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                final_records.append(rec)
                continue

        try:
            video_path = ensure_render(item)
            frames = sample_frames(video_path, num_frames=8)
            obj = call_claude(OBJ_PROMPT, item["text"], frames)
            mot = call_claude(MOTION_PROMPT, item["text"], frames)
            rec = {
                "id": item["id"],
                "split": item["split"],
                "task_type": item["task_type"],
                "task_key": item["task_key"],
                "object_consistency_score": float(obj["object_consistency_score"]),
                "object_reasoning": obj["object_reasoning"],
                "motion_consistency_score": float(mot["motion_consistency_score"]),
                "motion_reasoning": mot["motion_reasoning"],
            }
            out_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            final_records.append(rec)
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
            write_report(final_records)
            log(f"processed {idx}/{len(items)}")

    write_report(final_records)
    log("DONE")


if __name__ == "__main__":
    main()
