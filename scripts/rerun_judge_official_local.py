import argparse
import base64
import io
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from lottie.exporters.cairo import export_png
from lottie.parsers.tgs import parse_tgs


BASE_DIR = Path("/home/xinyu-chen/SVG Generation/judge-local")
BUNDLE_DIR = BASE_DIR / "official_bundle" / "judge_official_bundle"
MANIFEST_PATH = BUNDLE_DIR / "judge_manifest_official.json"
RESULTS_ROOT = BUNDLE_DIR / "official_rerun_text"
REMOTE_CACHE_ROOT = BUNDLE_DIR / "judge_cache_official_claude46"
LOCAL_CACHE_ROOT = BASE_DIR / "judge_cache_official_local_claude46"
REPORT_PATH = BASE_DIR / "judge_metrics_report_official_local_claude46.json"
LOG_PATH = BASE_DIR / "judge_metrics_official_local.log"

API_URL = os.environ.get("JUDGE_API_URL", "")
API_KEY = os.environ.get("JUDGE_API_KEY", "")
MODEL = os.environ.get("JUDGE_MODEL", "claude-sonnet-3-5-latest")

RESULT_DIRS = {
    ("real", "text2lottie"): RESULTS_ROOT / "mmlottie_bench_real_text2lottie",
    ("real", "text_image2lottie"): RESULTS_ROOT / "mmlottie_bench_real_text_image2lottie",
    ("synthetic", "text2lottie"): RESULTS_ROOT / "mmlottie_bench_synthetic_text2lottie",
    ("synthetic", "text_image2lottie"): RESULTS_ROOT / "mmlottie_bench_synthetic_text_image2lottie",
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
    parser.add_argument("--manifest-path", type=str, default=str(MANIFEST_PATH))
    parser.add_argument("--report-path", type=str, default=str(REPORT_PATH))
    parser.add_argument("--log-path", type=str, default=str(LOG_PATH))
    parser.add_argument("--local-cache-root", type=str, default=str(LOCAL_CACHE_ROOT))
    parser.add_argument("--remote-cache-root", type=str, default=str(REMOTE_CACHE_ROOT))
    parser.add_argument("--api-url", type=str, default=API_URL)
    parser.add_argument("--api-key", type=str, default=API_KEY)
    parser.add_argument("--model", type=str, default=MODEL)
    return parser.parse_args()


def log(message: str, log_path: Path) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_manifest(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def composite_on_background(img: Image.Image, background=(255, 255, 255)) -> Image.Image:
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        base = Image.new("RGB", rgba.size, background)
        base.paste(rgba, mask=rgba.getchannel("A"))
        return base
    return img.convert("RGB")


def render_sampled_frames_b64(item, num_frames: int = 8, target_size=(224, 224), background=(255, 255, 255)):
    result_dir = RESULT_DIRS[(item["split"], item["task_key"])]
    json_path = result_dir / f'{item["id"]}.json'
    if not json_path.exists():
        raise FileNotFoundError(f"Missing generated JSON: {json_path}")
    anim = parse_tgs(str(json_path))
    frame_ids = np.linspace(int(anim.in_point), int(anim.out_point), num_frames).astype(int)
    encoded = []
    for frame_id in frame_ids:
        buf = io.BytesIO()
        export_png(anim, buf, frame=int(frame_id))
        buf.seek(0)
        img = composite_on_background(Image.open(buf), background=background).resize(target_size)
        out = io.BytesIO()
        img.save(out, format="PNG")
        out.seek(0)
        encoded.append(base64.b64encode(out.read()).decode())
    return encoded


def parse_anthropic_json(response_json):
    content = response_json.get("content", [])
    text = "".join(block.get("text", "") for block in content if block.get("type") == "text")
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"Model response did not contain JSON: {text[:300]}")
    return json.loads(text[start : end + 1])


def normalize_api_url(url: str) -> str:
    url = url.rstrip("/")
    if url.endswith("/v1/messages"):
        return url
    return url + "/v1/messages"


def make_headers(api_key: str):
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def call_claude(prompt_prefix: str, caption: str, frame_b64_list, api_url: str, api_key: str, model: str, log_path: Path, max_retries: int = 8):
    payload = {
        "model": model,
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
            resp = requests.post(normalize_api_url(api_url), headers=make_headers(api_key), json=payload, timeout=120)
            resp.raise_for_status()
            return parse_anthropic_json(resp.json())
        except Exception as exc:
            last_error = exc
            sleep_s = min(60, 2 ** attempt)
            log(f"retry {attempt}/{max_retries} after error: {exc}", log_path)
            time.sleep(sleep_s)
    raise RuntimeError(f"Claude request failed after {max_retries} attempts: {last_error}")


def remote_cache_path(item, remote_cache_root: Path | None):
    if remote_cache_root is None:
        return None
    return remote_cache_root / item["split"] / item["task_key"] / f'{item["id"]}.json'


def local_cache_path(item, local_cache_root: Path):
    path = local_cache_root / item["split"] / item["task_key"] / f'{item["id"]}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def valid_record(rec):
    return (
        isinstance(rec, dict)
        and "object_consistency_score" in rec
        and "motion_consistency_score" in rec
    )


def write_report(records, report_path: Path, api_url: str, model: str):
    aggregate = {
        "model": model,
        "api_url": normalize_api_url(api_url),
        "notes": [
            "Object/Motion were evaluated locally with 8 uniformly sampled rendered PNG frames on official_rerun outputs.",
            "Manifest is user-selected and may contain fewer than 100 items for a subset-task when the generated JSON is missing locally.",
            "Rendered frames are composited onto a white background before judge scoring.",
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

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    manifest_path = Path(args.manifest_path)
    report_path = Path(args.report_path)
    log_path = Path(args.log_path)
    local_cache_root = Path(args.local_cache_root)
    remote_cache_root = Path(args.remote_cache_root) if args.remote_cache_root else None

    if not args.api_url or not args.api_key:
        raise ValueError("api_url/api_key are required, pass via args or JUDGE_API_URL/JUDGE_API_KEY")

    local_cache_root.mkdir(parents=True, exist_ok=True)
    items = load_manifest(manifest_path)
    if args.max_items is not None:
        items = items[: args.max_items]
    final_records = []

    for idx, item in enumerate(items, 1):
        out_path = local_cache_path(item, local_cache_root)

        if out_path.exists():
            with out_path.open("r", encoding="utf-8") as f:
                rec = json.load(f)
            if valid_record(rec):
                final_records.append(rec)
                continue

        rpath = remote_cache_path(item, remote_cache_root)
        if rpath is not None and rpath.exists():
            with rpath.open("r", encoding="utf-8") as f:
                rec = json.load(f)
            if valid_record(rec):
                rec["split"] = item["split"]
                rec["task_type"] = item["task_type"]
                out_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                final_records.append(rec)
                continue

        try:
            frames = render_sampled_frames_b64(item, num_frames=8)
            obj = call_claude(OBJ_PROMPT, item["text"], frames, args.api_url, args.api_key, args.model, log_path)
            mot = call_claude(MOTION_PROMPT, item["text"], frames, args.api_url, args.api_key, args.model, log_path)
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
            write_report(final_records, report_path, args.api_url, args.model)
            log(f"processed {idx}/{len(items)}", log_path)

    write_report(final_records, report_path, args.api_url, args.model)
    log("DONE", log_path)


if __name__ == "__main__":
    main()
