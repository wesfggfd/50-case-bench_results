import os
import sys
import time
import torch
import argparse
import numpy as np
import re
import json
import datetime
import traceback
import shutil
import random
import pandas as pd
import tempfile
import copy
from PIL import Image
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple

from safetensors.torch import load_file
from huggingface_hub import snapshot_download
from datasets import load_dataset, load_from_disk
from transformers import AutoConfig, AutoTokenizer, AutoProcessor
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info

REMOTE_FILE_DIR = Path(__file__).resolve().parent
OMNILOTTIE_ROOT = Path(os.environ.get("OMNILOTTIE_ROOT", "/opt/liblibai-models/user-workspace2/users/Sean_CHEN/OmniLottie_training/OmniLottie"))
for candidate in (OMNILOTTIE_ROOT, OMNILOTTIE_ROOT.parent, REMOTE_FILE_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from decoder import LottieDecoder
try:
    from decord import VideoReader, cpu
except Exception:
    VideoReader = None
    cpu = None

from lottie.objects.lottie_tokenize import LottieTensor
from lottie.objects.lottie_param import (
    from_sequence, ShapeLayer, NullLayer, PreCompLayer, TextLayer,
    SolidColorLayer, Font, Chars,
    shape_layer_to_json, null_layer_to_json, precomp_layer_to_json,
    text_layer_to_json, solid_layer_to_json, font_to_json, char_to_json
)
from lottie.exporters.video import export_video
from lottie.parsers.tgs import parse_tgs

from PIL import Image as PILImage

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# ========== Constants ==========
TASK_VIDEO = "video"
TASK_IMAGE = "image"
TASK_TEXT = "text"
BENCH_TASK_LABELS = {
    "text2lottie": "Text-to-Lottie",
    "text_image2lottie": "Text-Image-to-Lottie",
    "video2lottie": "Video-to-Lottie",
}

SYSTEM_PROMPT = "You are a Lottie animation expert."
VIDEO_PROMPT = "Turn this video into Lottie code."
BENCH_RENDER_FPS = 8.0
BENCH_RENDER_WIDTH = 336
BENCH_RENDER_HEIGHT = 336
BENCH_RENDER_FRAME_COUNT = 16
BENCH_DATA_ROOT = Path(os.environ.get("MMLOTTIE_BENCH_DATA_ROOT", "/opt/liblibai-models/user-workspace2/dataset/MMLottieBench/data"))

# Lottie token IDs — Qwen3.5-9B
# config.text_config.vocab_size=248320, ORIGINAL_BASE=151643 → shift=96677
# ORIGINAL offsets: CMD=151936, NUM=173186, BOS=192398, EOS=192399, PAD=151643
_QWEN35_BASE_VOCAB_SIZE = 248320   # Qwen3.5-9B text_config.vocab_size
_ORIGINAL_BASE_VOCAB_SIZE = 151643
_LOTTIE_SHIFT = _QWEN35_BASE_VOCAB_SIZE - _ORIGINAL_BASE_VOCAB_SIZE  # 96677
LOTTIE_BOS     = 192398 + _LOTTIE_SHIFT  # 289075
LOTTIE_EOS     = 192399 + _LOTTIE_SHIFT  # 289076
PAD_TOKEN      = 151643 + _LOTTIE_SHIFT  # 248320
COMMAND_OFFSET = 151936 + _LOTTIE_SHIFT  # 248613
NUM_COMMANDS   = 282  # unchanged

def sanitize_filename(text, max_length=180):
    text = re.sub(r'[<>:"/\\|?*\n\r\t]', '_', text)
    text = re.sub(r'[\s_]+', '_', text)
    text = text.strip('_ ')
    if len(text) > max_length:
        text = text[:max_length]
    return text if text else "unnamed"

def simplify_to_animation_description(text):
    if pd.isna(text) or text == '':
        return ""
    text = str(text)
    patterns = [
        r"The video features?", r"The video shows?", r"The image features?",
        r"The image shows?", r"This image", r"In this image,?",
    ]
    for p in patterns:
        text = re.sub(p, "", text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    if text:
        text = text[0].upper() + text[1:]
    return text

def add_random_background(img):
    if img.mode != 'RGBA':
        return img.convert('RGB')
    light_colors = [
        (255, 255, 255), (245, 245, 245), (250, 250, 250),
        (255, 250, 240), (240, 248, 255),
    ]
    bg_color = random.choice(light_colors)
    background = PILImage.new('RGB', img.size, bg_color)
    background.paste(img, (0, 0), img)
    return background

def load_frames_from_video(video_path, num_frames=8, target_size=(336, 336)):

    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    if total_frames < 1:
        raise ValueError(f"Video has no frames: {video_path}")
    indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    frames_np = vr.get_batch(indices).asnumpy()

    frames = []
    for f in frames_np:
        img = PILImage.fromarray(f)
        if target_size:
            img = img.resize(target_size, PILImage.LANCZOS)
        frames.append(img)

    return frames

def build_video_messages(frames: List[PILImage.Image], fps: float = 8.0, text_description: Optional[str] = None):
    prompt = text_description or VIDEO_PROMPT
    return [{
        "role": "system",
        "content": SYSTEM_PROMPT
    }, {
        "role": "user",
        "content": [
            {"type": "video", "video": frames, "fps": fps},
            {"type": "text", "text": prompt}
        ]
    }]

def build_image_messages(image, text_description):
    prompt = text_description or "Animate this image."
    return [{
        "role": "system",
        "content": SYSTEM_PROMPT
    }, {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": f"Animate this image: {prompt}"}
        ]
    }]

def build_text_messages(text_description):

    prompt = text_description or "Generate a vivid Lottie animation."
    messages = [{
        "role": "system",
        "content": SYSTEM_PROMPT
    }, {
        "role": "user",
        "content": [
            {"type": "text", "text": f"Generate Lottie code: {prompt}"}
        ]
    }]

    return messages

def prepare_inference_input(
    processor,
    messages,
    device,
    text_len: int = 1500,
    apply_left_padding: bool = True,
    target_context_len: int = 1500):

    text_input = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    image_inputs, video_inputs = process_vision_info(messages)
    
    if video_inputs:
        inputs = processor(
            text=[text_input],
            images=None,
            videos=video_inputs,
            padding=False,
            truncation=False,
            max_length=text_len,
            return_tensors="pt"
        )
        task_type = TASK_VIDEO
    elif image_inputs:
        inputs = processor(
            text=[text_input],
            images=image_inputs,
            videos=None,
            padding=False,
            truncation=False,
            max_length=text_len,
            return_tensors="pt"
        )
        task_type = TASK_IMAGE
    else:
        inputs = processor(
            text=[text_input],
            images=None,
            videos=None,
            padding=False,
            truncation=False,
            max_length=text_len,
            return_tensors="pt"
        )
        task_type = TASK_TEXT
    
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    if apply_left_padding and target_context_len is not None:
        current_len = input_ids.shape[1]
        if current_len < target_context_len:
            pad_len = target_context_len - current_len
            pad_ids = torch.full((1, pad_len), PAD_TOKEN, dtype=torch.long)
            pad_mask = torch.zeros((1, pad_len), dtype=torch.long)
            
            input_ids = torch.cat([pad_ids, input_ids], dim=1)
            attention_mask = torch.cat([pad_mask, attention_mask], dim=1)
    
    result = {
        'input_ids': input_ids.to(device),
        'attention_mask': attention_mask.to(device),
        'pixel_values': None,
        'image_grid_thw': None,
        'pixel_values_videos': None,
        'video_grid_thw': None,
        'task_type': task_type,
        'context_len': input_ids.shape[1],
    }
    
    if video_inputs and inputs.get('pixel_values_videos') is not None:
        result['pixel_values_videos'] = inputs['pixel_values_videos'].to(device)
        if inputs.get('video_grid_thw') is not None:
            result['video_grid_thw'] = inputs['video_grid_thw'].to(device)
    elif image_inputs and inputs.get('pixel_values') is not None:
        result['pixel_values'] = inputs['pixel_values'].to(device)
        if inputs.get('image_grid_thw') is not None:
            result['image_grid_thw'] = inputs['image_grid_thw'].to(device)
    
    return result

def generate_lottie(
    model,
    inputs: dict,
    max_new_tokens: int,
    device,
    use_sampling: bool = False,
    temperature: float = 0.9,
    top_p: float = 0.25,
    top_k: int = 5,
    repetition_penalty: float = 1.01,
    num_candidates: int = 1,
    verbose: bool = True) -> List[Tuple[List[int], dict]]:
    
    info = {
        'input_len': inputs['input_ids'].shape[1],
        'task_type': inputs.get('task_type', 'unknown'),
        'generated_len': 0,
        'has_bos': False,
        'has_eos': False,
        'valid_lottie_tokens': 0,
    }
    
    model.transformer.rope_deltas = None
    position_ids = None
    rope_index_fn = getattr(model.transformer, 'get_rope_index', None)
    if callable(rope_index_fn):
        position_ids, _ = rope_index_fn(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            image_grid_thw=inputs.get('image_grid_thw'),
            video_grid_thw=inputs.get('video_grid_thw'))
        position_ids = position_ids * inputs['attention_mask'][None, ]

    generate_kwargs = {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask'],
        'max_new_tokens': max_new_tokens,
        'min_new_tokens': 20,  
        'num_return_sequences': num_candidates,  
        'eos_token_id': LOTTIE_EOS,
        'pad_token_id': PAD_TOKEN,
        'use_cache': True,
        'return_dict_in_generate': True,
    }

    if inputs.get('pixel_values') is not None:
        generate_kwargs['pixel_values'] = inputs['pixel_values']
    if inputs.get('image_grid_thw') is not None:
        generate_kwargs['image_grid_thw'] = inputs['image_grid_thw']
    if inputs.get('pixel_values_videos') is not None:
        generate_kwargs['pixel_values_videos'] = inputs['pixel_values_videos']
    if inputs.get('video_grid_thw') is not None:
        generate_kwargs['video_grid_thw'] = inputs['video_grid_thw']
    
    if repetition_penalty > 1.0:
        generate_kwargs['repetition_penalty'] = repetition_penalty

    if use_sampling:
        generate_kwargs.update({
            'do_sample': True,
            'temperature': temperature,
            'top_p': top_p,
            'top_k': top_k,
        })
        if verbose:
            print(f"  Using sampling: temp={temperature}, top_p={top_p}, top_k={top_k}")
    else:
        generate_kwargs.update({
            'do_sample': False,
            'num_beams': 1,
        })
        if verbose:
            print("  Using greedy decoding")
    
    if verbose:
        print(f"  Input length: {info['input_len']}")
        print(f"  Max new tokens: {max_new_tokens}")
        print(f"  Repetition penalty: {repetition_penalty}")
    
    with torch.no_grad():
        outputs = model.transformer.generate(**generate_kwargs)

        candidates_results = []

        if hasattr(outputs, 'sequences'):
            sequences = outputs.sequences 
        else:
            sequences = outputs

        input_len = inputs['input_ids'].shape[1]

        for candidate_idx in range(num_candidates):
            generated_sequence = sequences[candidate_idx]
            generated_ids = generated_sequence[input_len:].tolist()

            cand_info = {
                'candidate_idx': candidate_idx,
                'input_len': input_len,
                'task_type': inputs.get('task_type', 'unknown'),
                'generated_len': len(generated_ids),
                'has_bos': LOTTIE_BOS in generated_ids,
                'has_eos': LOTTIE_EOS in generated_ids,
                'valid_lottie_tokens': sum(1 for t in generated_ids if t >= COMMAND_OFFSET),
                'raw_tokens': generated_ids.copy(),
            }

            clean_ids = clean_generated_tokens(generated_ids)
            cand_info['clean_len'] = len(clean_ids)

            candidates_results.append((clean_ids, cand_info))

            if verbose and num_candidates > 1:
                print(f"  Candidate {candidate_idx}: {cand_info['generated_len']} tokens, "
                      f"BOS={cand_info['has_bos']}, EOS={cand_info['has_eos']}")

        if verbose and num_candidates == 1:
            info = candidates_results[0][1]
            print(f"  Generated {info['generated_len']} tokens")
            print(f"  Has BOS: {info['has_bos']}, Has EOS: {info['has_eos']}")
            print(f"  Valid Lottie tokens: {info['valid_lottie_tokens']}")
            if info['raw_tokens']:
                print(f"  First 30 tokens: {info['raw_tokens'][:30]}")

    
    return candidates_results

def clean_generated_tokens(generated_ids: List[int]) -> List[int]:
    if not generated_ids:
        return []
    
    if generated_ids[0] == LOTTIE_BOS:
        generated_ids = generated_ids[1:]
    
    if LOTTIE_EOS in generated_ids:
        eos_idx = generated_ids.index(LOTTIE_EOS)
        generated_ids = generated_ids[:eos_idx]
    
    generated_ids = [t for t in generated_ids if t != PAD_TOKEN]
    
    return generated_ids


def fix_lottie_json(anim):

    anim_ip = int(round(anim.get("ip", 0)))
    anim_op = int(round(anim.get("op", 16)))
    anim["ip"]  = anim_ip
    anim["op"]  = anim_op
    anim["fr"]  = int(round(anim.get("fr", 8)))
    anim["ddd"] = int(anim.get("ddd", 0))

    def fix_t_recursive(obj):
        if isinstance(obj, dict):
            if obj.get("a") == 1 and isinstance(obj.get("k"), list):
                for kf in obj["k"]:
                    if isinstance(kf, dict) and "t" in kf:
                        kf["t"] = int(round(kf["t"]))
            for v in obj.values():
                fix_t_recursive(v)
        elif isinstance(obj, list):
            for item in obj:
                fix_t_recursive(item)

    fix_t_recursive(anim)

    max_x = float(anim.get("w", 512))
    max_y = float(anim.get("h", 512))

    def collect_pos(layer):
        nonlocal max_x, max_y
        p = layer.get("ks", {}).get("p", {})
        if isinstance(p, dict):
            if p.get("a", 0) == 0:
                pv = p.get("k", [0, 0])
                if isinstance(pv, list) and len(pv) >= 2:
                    max_x = max(max_x, float(pv[0]))
                    max_y = max(max_y, float(pv[1]))
            else:
                for kf in p.get("k", []):
                    if isinstance(kf, dict):
                        for sv in (kf.get("s", []), kf.get("e", [])):
                            if isinstance(sv, list) and len(sv) >= 2:
                                max_x = max(max_x, float(sv[0]))
                                max_y = max(max_y, float(sv[1]))
        for sub in layer.get("layers", []):
            collect_pos(sub)

    for layer in anim.get("layers", []):
        collect_pos(layer)

    anim["w"] = max(512, int((max_x * 1.1 + 15) // 16 * 16))
    anim["h"] = max(512, int((max_y * 1.1 + 15) // 16 * 16))

    valid_inds = set()
    for layer in anim.get("layers", []):
        if "ind" in layer:
            valid_inds.add(int(layer["ind"]))

    def clean_shapes(shapes):

        if not isinstance(shapes, list):
            return shapes
        cleaned = []
        for sh in shapes:
            if not isinstance(sh, dict):
                continue
            if sh.get("ty") == "gr":
                sh["it"] = clean_shapes(sh.get("it", []))
                if not sh["it"]:
                    continue
                has_tr = any(item.get("ty") == "tr" for item in sh["it"] if isinstance(item, dict))
                if not has_tr:
                    sh["it"].append({
                        "ty": "tr", "nm": "",
                        "a": {"a": 0, "k": [0, 0], "ix": 1},
                        "p": {"a": 0, "k": [0, 0], "ix": 2},
                        "s": {"a": 0, "k": [100, 100], "ix": 3},
                        "r": {"a": 0, "k": 0, "ix": 6},
                        "o": {"a": 0, "k": 100, "ix": 7},
                        "sk": {"a": 0, "k": 0, "ix": 4},
                        "sa": {"a": 0, "k": 0, "ix": 5},
                        "hd": False
                    })
            cleaned.append(sh)
        return cleaned

    def fix_layer(layer):
        ip = int(round(layer.get("ip", anim_ip)))
        op = int(round(layer.get("op", anim_op)))
        layer["ip"] = max(anim_ip, ip)
        layer["op"] = min(anim_op, max(layer["ip"] + 1, op))
        layer["st"] = int(round(layer.get("st", anim_ip)))
        if "ind" in layer:
            layer["ind"] = int(layer["ind"])
        if "parent" in layer:
            p = int(layer["parent"])
            if p in valid_inds:
                layer["parent"] = p
            else:
                del layer["parent"]
        layer.pop("ct", None)
        if "shapes" in layer:
            layer["shapes"] = clean_shapes(layer["shapes"])
        for sub in layer.get("layers", []):
            fix_layer(sub)
        return layer

    fixed_layers = []
    for l in anim.get("layers", []):
        fix_layer(l)
        shapes = l.get("shapes", [])
        if l.get("ty") == 4 and not shapes:
            continue  
        fixed_layers.append(l)
    anim["layers"] = fixed_layers

    for asset in anim.get("assets", []):
        if "layers" in asset:
            fixed = []
            for l in asset["layers"]:
                fix_layer(l)
                if l.get("ty") == 4 and not l.get("shapes"):
                    continue
                fixed.append(l)
            asset["layers"] = fixed

    return anim

def check_lottie_validity(json_animation):
    issues = []
    
    def check_layers(layers, prefix=""):
        nonlocal issues
        if not layers:
            return False
        has_visible = False
        
        for layer in layers:
            ty = layer.get("ty")
            if ty == 4 and layer.get("shapes"):
                has_visible = True
            elif ty == 1:
                has_visible = True
            elif ty == 0:
                has_visible = True
        
        return has_visible
    
    has_main = check_layers(json_animation.get("layers", []), "Main: ")
    
    for asset in json_animation.get("assets", []):
        if "layers" in asset:
            check_layers(asset["layers"], f"Asset {asset.get('id', '?')}: ")
    
    return has_main and len(issues) == 0, issues

def tokens_to_lottie_json(generated_ids: List[int], default_json: dict = None, verbose: bool = True):
    if default_json is None:
        default_json = {
            "v": "5.5.2", "fr": 8, "ip": 0, "op": 16,
            "w": 512, "h": 512, "nm": "Animation", "ddd": 0
        }

    if verbose:
        print(f"  Converting {len(generated_ids)} tokens to Lottie JSON...")

    reconstructed_tensor = LottieTensor.from_list(generated_ids)
    reconstructed_sequence = reconstructed_tensor.to_sequence()
    reconstructed = from_sequence(reconstructed_sequence)

    json_animation = {
        "v": reconstructed.get("v", default_json.get("v", "5.5.2")),
        "fr": reconstructed.get("fr", default_json.get("fr", 8)),
        "ip": reconstructed.get("ip", default_json.get("ip", 0)),
        "op": reconstructed.get("op", default_json.get("op", 16)),
        "w": reconstructed.get("w", default_json.get("w", 512)),
        "h": reconstructed.get("h", default_json.get("h", 512)),
        "nm": reconstructed.get("nm", default_json.get("nm", "Animation")),
        "ddd": reconstructed.get("ddd", default_json.get("ddd", 0)),
        "assets": [],
        "layers": [],
    }

    if "markers" in reconstructed:
        json_animation["markers"] = reconstructed.get("markers", [])
    if "props" in reconstructed:
        json_animation["props"] = reconstructed.get("props", {})

    if "fonts" in reconstructed and reconstructed["fonts"]:
        fonts_data = reconstructed["fonts"]
        if isinstance(fonts_data, dict) and "list" in fonts_data:
            fonts_json = {"list": []}
            for font in fonts_data["list"]:
                if isinstance(font, Font):
                    fonts_json["list"].append(font_to_json(font))
                else:
                    fonts_json["list"].append(font)
            json_animation["fonts"] = fonts_json

    if "chars" in reconstructed and reconstructed["chars"]:
        chars_json = []
        for char in reconstructed["chars"]:
            if isinstance(char, Chars):
                chars_json.append(char_to_json(char))
            else:
                chars_json.append(char)
        json_animation["chars"] = chars_json

    for asset in reconstructed.get("assets", []):
        asset_json = dict(asset)
        if "layers" in asset:
            asset_json["layers"] = []
            for layer in asset["layers"]:
                if isinstance(layer, ShapeLayer):
                    asset_json["layers"].append(shape_layer_to_json(layer))
                elif isinstance(layer, NullLayer):
                    asset_json["layers"].append(null_layer_to_json(layer))
                elif isinstance(layer, PreCompLayer):
                    asset_json["layers"].append(precomp_layer_to_json(layer))
                elif isinstance(layer, TextLayer):
                    asset_json["layers"].append(text_layer_to_json(layer))
                elif isinstance(layer, SolidColorLayer):
                    asset_json["layers"].append(solid_layer_to_json(layer))
                else:
                    asset_json["layers"].append(layer)
        json_animation["assets"].append(asset_json)

    # 处理layers
    for layer in reconstructed.get("layers", []):
        if isinstance(layer, ShapeLayer):
            json_animation["layers"].append(shape_layer_to_json(layer))
        elif isinstance(layer, NullLayer):
            json_animation["layers"].append(null_layer_to_json(layer))
        elif isinstance(layer, PreCompLayer):
            json_animation["layers"].append(precomp_layer_to_json(layer))
        elif isinstance(layer, TextLayer):
            json_animation["layers"].append(text_layer_to_json(layer))
        elif isinstance(layer, SolidColorLayer):
            json_animation["layers"].append(solid_layer_to_json(layer))
        else:
            json_animation["layers"].append(layer)

    json_animation = fix_lottie_json(json_animation)

    return json_animation


def create_lottie_html(animation_data, height=600):
    animation_json = json.dumps(animation_data, ensure_ascii=False)
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <script src=\"https://cdnjs.cloudflare.com/ajax/libs/bodymovin/5.12.2/lottie.min.js\"></script>
  <style>
    body {{ margin: 0; padding: 0; background: #111; color: #eee; font-family: sans-serif; }}
    #lottie-animation {{ width: 100%; max-width: 900px; height: {height}px; margin: 0 auto; }}
    .meta {{ padding: 12px 16px; font-size: 14px; }}
  </style>
</head>
<body>
  <div class=\"meta\">Lottie preview</div>
  <div id=\"lottie-animation\"></div>
  <script>
    const animationData = {animation_json};
    lottie.loadAnimation({{
      container: document.getElementById('lottie-animation'),
      renderer: 'svg',
      loop: true,
      autoplay: true,
      animationData: animationData
    }});
  </script>
</body>
</html>"""


def probe_video_metadata(video_path: str) -> dict:
    try:
        import cv2 as _cv2
    except Exception as exc:
        raise ImportError("probe_video_metadata requires cv2/opencv-python") from exc
    cap = _cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise ValueError(f'Unable to open video: {video_path}')
        fps = float(cap.get(_cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(_cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(_cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = (frame_count / fps) if fps > 0 else None
        return {
            'fps': fps if fps > 0 else BENCH_RENDER_FPS,
            'frame_count': frame_count if frame_count > 0 else BENCH_RENDER_FRAME_COUNT,
            'width': width if width > 0 else BENCH_RENDER_WIDTH,
            'height': height if height > 0 else BENCH_RENDER_HEIGHT,
            'duration': duration if duration is not None else BENCH_RENDER_FRAME_COUNT / BENCH_RENDER_FPS,
        }
    finally:
        cap.release()


def _apply_render_metadata(animation_data: dict, *, fps: Optional[float] = None, width: Optional[int] = None, height: Optional[int] = None, frame_count: Optional[int] = None) -> dict:
    anim = copy.deepcopy(animation_data)
    fps = BENCH_RENDER_FPS if fps is None or fps <= 0 else fps
    width = BENCH_RENDER_WIDTH if width is None or width <= 0 else width
    height = BENCH_RENDER_HEIGHT if height is None or height <= 0 else height
    frame_count = BENCH_RENDER_FRAME_COUNT if frame_count is None or frame_count <= 0 else frame_count
    anim['fr'] = float(fps)
    anim['w'] = int(width)
    anim['h'] = int(height)
    anim['ip'] = int(round(float(anim.get('ip', 0))))
    anim['op'] = anim['ip'] + int(frame_count)
    if anim['op'] <= anim['ip']:
        anim['op'] = anim['ip'] + int(frame_count)
    return anim


def render_lottie_mp4(animation_data: dict, output_path: str, *, reference_video_path: Optional[str] = None) -> str:
    render_anim = animation_data
    meta = None
    if reference_video_path:
        meta = probe_video_metadata(reference_video_path)
        render_anim = _apply_render_metadata(
            animation_data,
            fps=meta['fps'],
            width=meta['width'],
            height=meta['height'],
            frame_count=meta['frame_count'],
        )
    else:
        render_anim = _apply_render_metadata(animation_data)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as tmp_json:
        json.dump(render_anim, tmp_json, ensure_ascii=False)
        tmp_json_path = tmp_json.name
    try:
        anim = parse_tgs(tmp_json_path)
        export_video(anim, output_path, format='mp4')
        return output_path
    finally:
        try:
            os.unlink(tmp_json_path)
        except OSError:
            pass


def _apply_lora_to_decoder(model, lora_rank=64, lora_alpha=128, lora_dropout=0.05):
    """Apply LoRA wrapping to the decoder's transformer, matching training config."""
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LottieDecoder.default_lora_target_modules(),
    )
    model.transformer = get_peft_model(model.transformer, lora_config)
    print(f"  Applied LoRA wrapping (rank={lora_rank}, alpha={lora_alpha})")
    return model


def load_model_weights_into_decoder(model, sketch_weight: str, *, strict: bool = False):
    if os.path.isfile(sketch_weight) and sketch_weight.endswith('.bin'):
        model_path = sketch_weight
        safetensors_path = sketch_weight.replace('.bin', '.safetensors')
    else:
        model_path = os.path.join(sketch_weight, 'pytorch_model.bin')
        safetensors_path = os.path.join(sketch_weight, 'model.safetensors')

    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location='cpu')
        source_path = model_path
    elif os.path.exists(safetensors_path):
        state_dict = load_file(safetensors_path)
        source_path = safetensors_path
    else:
        raise FileNotFoundError(f"Model not found in {sketch_weight}")

    # Detect LoRA checkpoint: keys contain 'base_model.model.' prefix
    has_lora_keys = any('lora_' in k for k in state_dict.keys())
    has_peft_prefix = any('base_model.model.' in k for k in state_dict.keys())

    if has_lora_keys and has_peft_prefix:
        # Checkpoint is from LoRA-wrapped training — apply LoRA wrapping first
        print(f"  Detected LoRA checkpoint ({sum(1 for k in state_dict if 'lora_' in k)} LoRA keys)")
        model = _apply_lora_to_decoder(model)

    incompatible = model.load_state_dict(state_dict, strict=strict)
    missing = list(getattr(incompatible, 'missing_keys', []))
    unexpected = list(getattr(incompatible, 'unexpected_keys', []))
    print(f"Loaded from {source_path} (strict={strict})")
    if missing or unexpected:
        print(f"  Missing keys: {len(missing)}")
        print(f"  Unexpected keys: {len(unexpected)}")
        if missing:
            print(f"  Missing preview: {missing[:12]}")
        if unexpected:
            print(f"  Unexpected preview: {unexpected[:12]}")

    # Merge LoRA weights into base model for faster inference
    if has_lora_keys and has_peft_prefix:
        print("  Merging LoRA weights into base model...")
        model.transformer = model.transformer.merge_and_unload()
        print("  LoRA merged successfully")

    return source_path

def run_inference(
    model,
    processor,
    task_type: str,
    device,
    cfg: dict,
    uid: str = None,
    video_path: str = None,
    image_path: str = None,
    text_description: str = None,
    use_sampling: bool = False,
    temperature: float = 0.9,
    top_p: float = 0.25,
    top_k: int = 5,
    repetition_penalty: float = 1.01,
    output_path: str = None,
    save_mp4: bool = True,
    render_reference_video_path: str = None,
    verbose: bool = True) -> Tuple[dict, dict]:


    prompt_info = None
    task_type = _normalize_task_type(task_type)
    if task_type == "video2lottie":
        if not video_path:
            raise ValueError("video_path required for video task")
        if VideoReader is None or cpu is None:
            raise ImportError("video task requires decord to be installed")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video path does not exist: {video_path}")
        video_path = os.path.abspath(video_path)
        frames = load_frames_from_video(video_path, num_frames=8)
        messages = build_video_messages(frames, fps=8.0, text_description=text_description)

    elif task_type == "text_image2lottie":
        if image_path is None:
            raise ValueError("image_path required for text-image task")
        img = PILImage.open(image_path)
        img = add_random_background(img) if img.mode == 'RGBA' else img.convert('RGB')
        img = img.resize((448, 448), PILImage.LANCZOS)
        desc = text_description or "Describe the image and animate it."
        messages = build_image_messages(img, desc)
    elif task_type == "text2lottie":
        desc = text_description or "A simple animation"
        messages = build_text_messages(desc)
        if len(messages) > 2 and "prompt_info" in messages[-1]:
            prompt_info = messages[-1]["prompt_info"]
            messages = messages[:-1]
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    inputs = prepare_inference_input(
        processor=processor,
        messages=messages,
        device=device,
        text_len=cfg.get('text_len', 1500),
        apply_left_padding=True,
        target_context_len=1500)

    if verbose:
        print(f"\nTask: {task_type}")
        print(f"Context length: {inputs['context_len']}")
        print(f"Prompt length: {len(text_description or '')}")

    num_candidates = cfg.get('num_candidates', 1)
    candidates_list = generate_lottie(
        model=model,
        inputs=inputs,
        max_new_tokens=cfg.get('pix_len', 4096),
        device=device,
        use_sampling=use_sampling,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        num_candidates=num_candidates,  
        verbose=verbose)

    processed_candidates = []

    for token_ids, gen_info in candidates_list:

        if len(token_ids) < 10:
            if verbose and num_candidates > 1:
                print(f"  Candidate {gen_info['candidate_idx']}: Too short ({len(token_ids)} tokens), skipping")
            continue

        try:
            lottie_json = tokens_to_lottie_json(
                token_ids,
                verbose=False)

            is_valid, issues = check_lottie_validity(lottie_json)
            gen_info['is_valid'] = is_valid
            gen_info['issues'] = issues

            processed_candidates.append((
                lottie_json,
                token_ids,
                gen_info['has_eos'],
                gen_info['candidate_idx'],
                gen_info
            ))

        except Exception as e:
            if verbose and num_candidates > 1:
                print(f"  Candidate {gen_info['candidate_idx']}: Conversion failed: {e}")
            continue

    if len(processed_candidates) == 0:
        if verbose:
            print(f"  ERROR: All {num_candidates} candidates failed")
        if candidates_list:
            fallback_ids, fallback_info = candidates_list[0]
            fallback_json = None
            try:
                fallback_json = tokens_to_lottie_json(fallback_ids, verbose=False)
            except Exception:
                fallback_json = None
            if fallback_json is not None:
                fallback_info['is_valid'] = False
                fallback_info['fallback_used'] = True
                if output_path:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(fallback_json, f, indent=2, ensure_ascii=False)
                    html_path = output_path.replace('.json', '.html')
                    with open(html_path, 'w', encoding='utf-8') as f:
                        f.write(create_lottie_html(fallback_json))
                    if save_mp4:
                        mp4_path = output_path.replace('.json', '.mp4')
                        render_lottie_mp4(fallback_json, mp4_path, reference_video_path=render_reference_video_path)
                return fallback_json, fallback_info
        return None, candidates_list[0][1] if candidates_list else {}

    if len(processed_candidates) == 1:
        best_idx = 0
        best_score = None
        best_details = None
        if verbose:
            print(f"  Only 1 valid candidate, using it")
    else:
        candidates_for_scoring = [
            (lottie_json, token_ids, has_eos, cand_idx)
            for lottie_json, token_ids, has_eos, cand_idx, _ in processed_candidates
        ]
        best_idx = 0
        best_score = None
        best_details = None

    lottie_json, generated_ids, has_eos, selected_cand_idx, gen_info = processed_candidates[best_idx]

    if verbose and num_candidates > 1:
        print(f"  ✅ Selected candidate {selected_cand_idx} (score: {best_score})")

    if num_candidates > 1:
        gen_info['num_candidates'] = num_candidates
        gen_info['selected_candidate'] = selected_cand_idx
        gen_info['best_score'] = best_score
        gen_info['best_details'] = best_details

    if not gen_info.get('is_valid') and verbose:
        print(f"  WARNING: Lottie may be invalid: {gen_info.get('issues', [])}")

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(lottie_json, f, indent=2, ensure_ascii=False)
        html_path = output_path.replace('.json', '.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(create_lottie_html(lottie_json))
        mp4_path = None
        if save_mp4:
            mp4_path = output_path.replace('.json', '.mp4')
            render_lottie_mp4(
                lottie_json,
                mp4_path,
                reference_video_path=render_reference_video_path,
            )
        if verbose:
            print(f"  Saved to: {output_path}")
            print(f"  Saved preview to: {html_path}")
            if mp4_path:
                print(f"  Saved video to: {mp4_path}")

        # Save all candidates when num_candidates > 1
        if num_candidates > 1 and len(processed_candidates) > 1:
            base_path = output_path.replace('.json', '')
            for idx, (cand_lottie, cand_tokens, cand_has_eos, cand_idx, cand_info) in enumerate(processed_candidates):
                cand_path = f"{base_path}_candidate_{cand_idx}.json"
                with open(cand_path, 'w', encoding='utf-8') as f:
                    json.dump(cand_lottie, f, indent=2, ensure_ascii=False)
                if verbose:
                    print(f"  Saved candidate {cand_idx} to: {cand_path}")

        info_path = output_path.replace('.json', '_info.txt')
        with open(info_path, 'w', encoding='utf-8') as f:
            f.write(f"=== Generation Info ===\n")
            f.write(f"UID: {uid}\n")
            f.write(f"Task: {task_type}\n")
            f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            if task_type == TASK_TEXT and prompt_info:
                f.write(f"Original prompt ({prompt_info['original_word_count']} words):\n")
                f.write(f"  {prompt_info['original']}\n\n")
                f.write(f"Prompt was NOT enhanced (already detailed enough)\n\n")

            f.write(f"=== Sampling Parameters ===\n")
            f.write(f"Use sampling: {use_sampling}\n")
            if use_sampling:
                f.write(f"Temperature: {temperature}\n")
                f.write(f"Top-p: {top_p}\n")
                f.write(f"Top-k: {top_k}\n")
            f.write(f"Repetition penalty: {repetition_penalty}\n")
            f.write(f"Max new tokens: {cfg.get('pix_len', 4096)}\n\n")

            f.write(f"=== Generation Results ===\n")
            f.write(f"Generated tokens: {len(generated_ids)}\n")
            f.write(f"Valid Lottie: {gen_info.get('is_valid', 'unknown')}\n")
            f.write(f"Has BOS: {gen_info.get('has_bos', False)}\n")
            f.write(f"Has EOS: {gen_info.get('has_eos', False)}\n")
            f.write(f"Valid Lottie tokens: {gen_info.get('valid_lottie_tokens', 0)}\n\n")

            if gen_info.get('num_candidates', 1) > 1:
                f.write(f"=== Candidate Selection ===\n")
                f.write(f"Total candidates generated: {gen_info['num_candidates']}\n")
                f.write(f"Valid candidates: {len(processed_candidates)}\n")
                f.write(f"Selected candidate: {gen_info['selected_candidate']}\n")
                if gen_info.get('best_score') is not None:
                    f.write(f"Quality score: {gen_info['best_score']}\n")
                if gen_info.get('best_details'):
                    f.write(f"Quality details:\n")
                    for key, value in gen_info['best_details'].items():
                        f.write(f"  {key}: {value}\n")
                f.write(f"\n")

    return lottie_json, gen_info


def run_batch_text_file_inference(args, cfg):
    """
    Generate Lottie from text file.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "xpu:0" if torch.xpu.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(args.batch_text_file):
        raise FileNotFoundError(f"Batch text file not found: {args.batch_text_file}")

    print("Loading model...")
    processor = AutoProcessor.from_pretrained(cfg['tokenizer_name'], padding_side="left")
    processor.tokenizer.padding_side = "left"

    model = LottieDecoder(pix_len=cfg['pix_len'], text_len=cfg['text_len'])


    if os.path.isfile(args.sketch_weight) and args.sketch_weight.endswith('.bin'):
        model_path = args.sketch_weight
        safetensors_path = args.sketch_weight.replace('.bin', '.safetensors')
    else:
        model_path = os.path.join(args.sketch_weight, 'pytorch_model.bin')
        safetensors_path = os.path.join(args.sketch_weight, 'model.safetensors')

    load_model_weights_into_decoder(model, args.sketch_weight, strict=False)

    model = model.to(device).eval()


    print(f"\nReading prompts from: {args.batch_text_file}")
    with open(args.batch_text_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]

    print(f"Total prompts: {len(prompts)}")

    output_dir = os.path.join(args.output_dir, 'batch_text2lottie')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    stats = {'success': 0, 'fail': 0, 'total': len(prompts)}

    print(f"\n{'='*60}")
    print(f"Starting batch text2lottie generation...")
    print(f"{'='*60}\n")

    for idx, prompt in enumerate(prompts, 1):
        print(f"\n[{idx}/{len(prompts)}] Processing:")
        print(f"  Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")

        try:
            base_filename = sanitize_filename(prompt)
            output_path = os.path.join(output_dir, f"{base_filename}.json")

            lottie_json, gen_info = run_inference(
                model=model,
                processor=processor,
                task_type=TASK_TEXT,
                device=device,
                cfg=cfg,
                uid=f"batch_{idx:04d}",
                text_description=prompt,
                use_sampling=args.use_sampling,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                output_path=output_path,
                verbose=False,  
            )

            if lottie_json:
                print(f"  ✅ Success: {output_path}")
                print(f"     Layers: {len(lottie_json.get('layers', []))}, Tokens: {gen_info.get('generated_len', 0)}")
                stats['success'] += 1
            else:
                print(f"  ❌ Generation failed")
                stats['fail'] += 1

        except Exception as e:
            print(f"  ❌ Error: {e}")
            if args.debug:
                traceback.print_exc()
            stats['fail'] += 1

    print(f"\n{'='*60}")
    print(f"Batch Processing Complete!")
    print(f"{'='*60}")
    print(f"Total prompts: {stats['total']}")
    print(f"  ✅ Success: {stats['success']}")
    print(f"  ❌ Failed: {stats['fail']}")
    print(f"  Success rate: {stats['success']/stats['total']*100:.1f}%")
    print(f"\nOutput directory: {output_dir}")
    print(f"{'='*60}")

# ========== MMLottie Benchmark 推理 ==========
def _normalize_task_type(value: Optional[str]) -> str:
    if value is None:
        return "text2lottie"
    value = str(value).strip().lower().replace("-", "_")
    aliases = {
        "text": "text2lottie",
        "text2lottie": "text2lottie",
        "text_to_lottie": "text2lottie",
        "text_image": "text_image2lottie",
        "text_image2lottie": "text_image2lottie",
        "text_image_to_lottie": "text_image2lottie",
        "image": "text_image2lottie",
        "video": "video2lottie",
        "video2lottie": "video2lottie",
        "video_to_lottie": "video2lottie",
    }
    return aliases.get(value, value)


def _bench_task_label(task_key: str) -> str:
    task_key = _normalize_task_type(task_key)
    if task_key not in BENCH_TASK_LABELS:
        raise ValueError(f"Unsupported benchmark task key: {task_key}")
    return BENCH_TASK_LABELS[task_key]


def _sample_task_key(sample: dict) -> Optional[str]:
    raw_task = sample.get("task_type")
    if raw_task is None:
        return None
    return _normalize_task_type(raw_task)


def _infer_text_prompt(sample: dict) -> str:
    for key in ("text", "prompt", "description", "caption", "instruction"):
        value = sample.get(key)
        if value:
            return str(value)
    return "Generate a vivid Lottie animation."


def _build_text_only_prompt(sample: dict) -> str:
    prompt = _infer_text_prompt(sample)
    return prompt if prompt else "Generate a vivid Lottie animation."


def _sample_task_matches(sample: dict, requested_task: str) -> bool:
    sample_task = _sample_task_key(sample)
    if sample_task is None:
        return False
    return sample_task == _normalize_task_type(requested_task)


def _validate_bench_sample_fields(sample: dict, task_key: str) -> None:
    task_key = _normalize_task_type(task_key)
    text_value = sample.get('text')
    image_value = sample.get('image')
    video_value = sample.get('video')
    if task_key == 'text2lottie':
        if not text_value:
            raise ValueError('text2lottie sample is missing text')
        if image_value is not None or video_value is not None:
            raise ValueError('text2lottie sample must not contain image or video inputs')
    elif task_key == 'text_image2lottie':
        if not text_value:
            raise ValueError('text_image2lottie sample is missing text')
        if image_value is None:
            raise ValueError('text_image2lottie sample is missing image')
        if video_value is not None:
            raise ValueError('text_image2lottie sample must not contain video input')
    elif task_key == 'video2lottie':
        if not text_value:
            raise ValueError('video2lottie sample is missing text')
        if video_value is None:
            raise ValueError('video2lottie sample is missing video')
        if image_value is not None:
            raise ValueError('video2lottie sample must not contain image input')
    else:
        raise ValueError(f'Unsupported benchmark task key: {task_key}')


def _coerce_image_path(image_value, *, output_dir: str, sample_id: str) -> Tuple[str, List[str]]:
    cleanup_paths: List[str] = []
    if isinstance(image_value, str):
        if not os.path.exists(image_value):
            raise FileNotFoundError(f"Image path does not exist: {image_value}")
        return image_value, cleanup_paths
    if isinstance(image_value, dict) and image_value.get("path"):
        image_path = image_value["path"]
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image path does not exist: {image_path}")
        return image_path, cleanup_paths
    if isinstance(image_value, PILImage.Image):
        pil_image = image_value
    else:
        pil_image = PILImage.fromarray(np.array(image_value))
    image_path = os.path.join(output_dir, f"{sample_id}_image.png")
    pil_image.save(image_path)
    cleanup_paths.append(image_path)
    return image_path, cleanup_paths


def _coerce_video_path(video_value, *, output_dir: str, sample_id: str) -> Tuple[str, List[str]]:
    cleanup_paths: List[str] = []
    if isinstance(video_value, str):
        if not os.path.exists(video_value):
            raise FileNotFoundError(f"Video path does not exist: {video_value}")
        return video_value, cleanup_paths
    if isinstance(video_value, dict) and video_value.get("path"):
        video_path = video_value["path"]
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video path does not exist: {video_path}")
        return video_path, cleanup_paths
    if isinstance(video_value, bytes):
        video_path = os.path.join(output_dir, f"{sample_id}_video.mp4")
        with open(video_path, 'wb') as f:
            f.write(video_value)
        cleanup_paths.append(video_path)
        return video_path, cleanup_paths
    raise ValueError(f"Unsupported video format for sample {sample_id}: {type(video_value)}")


def _resolve_render_reference_video_path(rows, *, output_dir: str) -> Tuple[Optional[str], List[str]]:
    cleanup_paths: List[str] = []
    for idx, sample in enumerate(rows):
        video_value = sample.get('video')
        if video_value is None:
            continue
        try:
            video_path, extra_cleanup = _coerce_video_path(video_value, output_dir=output_dir, sample_id=f'render_ref_{idx}')
            cleanup_paths.extend(extra_cleanup)
            return video_path, cleanup_paths
        except Exception:
            continue
    return None, cleanup_paths


def _select_bench_sample(split: str, task_label: str, max_samples: int = 1) -> Tuple[dict, List[str]]:
    data_root = Path(BENCH_DATA_ROOT)
    parquet_path = data_root / f"{split}-00000-of-00001.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Benchmark parquet not found: {parquet_path}")
def _run_single_bench_task_inference(args, cfg, model, processor, subset, split: str, task_key: str, device):
    requested_label = _bench_task_label(task_key)
    print(f"  Requested task: {task_key} ({requested_label})")

    selected = []
    table = subset.data.table
    for idx in range(len(subset)):
        row = table.slice(idx, 1).to_pydict()
        sample = {k: v[0] for k, v in row.items()}
        if not _sample_task_matches(sample, task_key):
            continue
        selected.append((idx, sample))
        if args.max_samples is not None and args.max_samples > 0 and len(selected) >= args.max_samples:
            break

    print(f"  Selected samples: {len(selected)}")
    if not selected:
        raise RuntimeError(f"No samples found for task '{task_key}' in split '{split}'")

    output_base = os.path.join(args.output_dir, f'mmlottie_bench_{split}_{task_key}')
    os.makedirs(output_base, exist_ok=True)

    stats = {task_key: {'success': 0, 'fail': 0, 'total': 0}}

    print()
    print('=' * 60)
    print(f"Starting benchmark inference for {requested_label}...")
    print('=' * 60)
    print()

    for idx, sample in selected:
        sample_task_key = _normalize_task_type(sample.get('task_type'))
        sample_id = sample.get('id', f'sample_{idx}')
        _validate_bench_sample_fields(sample, task_key)
        stats[task_key]['total'] += 1
        print(f"[{idx+1}/{len(subset)}] Processing {sample_id} ({sample_task_key})...")
        cleanup_paths = []
        try:
            output_path = os.path.join(output_base, f'{sample_id}.json')
            reference_video_path = None
            if task_key == 'video2lottie' and sample.get('video') is not None:
                try:
                    reference_video_path, extra_cleanup = _coerce_video_path(sample['video'], output_dir=output_base, sample_id=f'ref_{sample_id}')
                    cleanup_paths.extend(extra_cleanup)
                except Exception:
                    reference_video_path = None

            if task_key == 'text2lottie':
                text_prompt = _build_text_only_prompt(sample)
                print(f"  Text: {text_prompt[:120]}{'...' if len(text_prompt) > 120 else ''}")
                lottie_json, info = run_inference(
                    model=model,
                    processor=processor,
                    task_type=TASK_TEXT,
                    device=device,
                    cfg=cfg,
                    uid=str(sample_id),
                    text_description=text_prompt,
                    use_sampling=args.use_sampling,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    repetition_penalty=args.repetition_penalty,
                    output_path=output_path,
                    save_mp4=True,
                    render_reference_video_path=None,
                    verbose=False,
                )
            elif task_key == 'text_image2lottie':
                image_value = sample.get('image')
                text_prompt = _infer_text_prompt(sample)
                if image_value is None:
                    raise ValueError('Missing image field for text_image2lottie sample')
                image_path, extra_cleanup = _coerce_image_path(image_value, output_dir=output_base, sample_id=str(sample_id))
                cleanup_paths.extend(extra_cleanup)
                print(f"  Text: {text_prompt[:120]}{'...' if len(text_prompt) > 120 else ''}")
                print(f"  Image: {image_path}")
                lottie_json, info = run_inference(
                    model=model,
                    processor=processor,
                    task_type=TASK_IMAGE,
                    device=device,
                    cfg=cfg,
                    uid=str(sample_id),
                    image_path=image_path,
                    text_description=text_prompt,
                    use_sampling=args.use_sampling,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    repetition_penalty=args.repetition_penalty,
                    output_path=output_path,
                    save_mp4=True,
                    render_reference_video_path=None,
                    verbose=False,
                )
            elif task_key == 'video2lottie':
                video_value = sample.get('video')
                if video_value is None:
                    raise ValueError('Missing video field for video2lottie sample')
                if VideoReader is None or cpu is None:
                    raise ImportError('video2lottie requires decord to be installed')
                video_path, extra_cleanup = _coerce_video_path(video_value, output_dir=output_base, sample_id=str(sample_id))
                cleanup_paths.extend(extra_cleanup)
                print(f"  Video: {video_path}")
                lottie_json, info = run_inference(
                    model=model,
                    processor=processor,
                    task_type=TASK_VIDEO,
                    device=device,
                    cfg=cfg,
                    uid=str(sample_id),
                    video_path=video_path,
                    text_description=None,
                    use_sampling=args.use_sampling,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    repetition_penalty=args.repetition_penalty,
                    output_path=output_path,
                    save_mp4=True,
                    render_reference_video_path=reference_video_path,
                    verbose=False,
                )
            else:
                raise ValueError(f"Unsupported task type: {task_key}")

            if lottie_json is not None:
                print(f"  ✅ Saved to {output_path}")
                print(f"  ✅ Preview: {output_path.replace('.json', '.html')}")
                stats[task_key]['success'] += 1
            else:
                print("  ❌ Generation failed")
                stats[task_key]['fail'] += 1
        except Exception as e:
            print(f"  ❌ Error: {e}")
            if args.debug:
                traceback.print_exc()
            stats[task_key]['fail'] += 1
        finally:
            for path in cleanup_paths:
                try:
                    if os.path.exists(path):
                        os.unlink(path)
                except Exception:
                    pass

    print()
    print('=' * 60)
    print(f"Benchmark Inference Complete for {requested_label}!")
    print('=' * 60)
    task_stats = stats[task_key]
    if task_stats['total'] > 0:
        success_rate = task_stats['success'] / task_stats['total'] * 100
        print(f"{task_key}:")
        print(f"  Success: {task_stats['success']}/{task_stats['total']} ({success_rate:.1f}%)")
        print(f"  Failed: {task_stats['fail']}/{task_stats['total']}")
    print()
    print(f"Output directory: {output_base}")
    print('=' * 60)
    return stats


def run_mmlottie_bench_inference(args, cfg):
    """
    Inference on MMLottieBench dataset from HuggingFace.
    Runs one benchmark task per invocation.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "xpu:0" if torch.xpu.is_available() else "cpu")
    print(f"Using device: {device}")

    print()
    print("Loading MMLottieBench dataset...")
    try:
        if os.path.exists(args.mmlottie_bench_dir) and os.path.isdir(args.mmlottie_bench_dir):
            try:
                print(f"  Attempting to load from local: {args.mmlottie_bench_dir}")
                dataset = load_from_disk(args.mmlottie_bench_dir)
                print("  ✅ Loaded from local directory")
            except Exception as e:
                print(f"  ⚠️  Local load failed: {e}")
                print("  Downloading from HuggingFace...")
                dataset = load_dataset("OmniLottie/MMLottieBench")
        else:
            print("  Local directory not found, downloading from HuggingFace...")
            dataset = load_dataset("OmniLottie/MMLottieBench")
        print(f"  Available splits: {list(dataset.keys())}")
    except Exception as e:
        print()
        print(f"❌ Failed to load dataset: {e}")
        print("Please check your network or download manually using:")
        print("  python download_mmlottie_bench.py")
        raise

    if args.split not in dataset:
        raise ValueError(f"Split '{args.split}' not found in dataset. Available: {list(dataset.keys())}")

    subset = dataset[args.split]
    print()
    print(f"Processing split: {args.split}")
    print(f"  Total samples: {len(subset)}")

    print()
    print("Loading model...")
    processor = AutoProcessor.from_pretrained(cfg['tokenizer_name'], padding_side="left")
    processor.tokenizer.padding_side = "left"
    model = LottieDecoder(pix_len=cfg['pix_len'], text_len=cfg['text_len'])
    load_model_weights_into_decoder(model, args.sketch_weight, strict=False)
    model = model.to(device).eval()

    requested_task = _normalize_task_type(args.mmlottie_task)
    if requested_task not in BENCH_TASK_LABELS:
        raise ValueError(f"Unsupported benchmark task: {requested_task}")

    stats = _run_single_bench_task_inference(args, cfg, model, processor, subset, args.split, requested_task, device)
    task_stats = stats[requested_task]
    print()
    print("Benchmark task summary:")
    if task_stats['total'] > 0:
        success_rate = task_stats['success'] / task_stats['total'] * 100
        print(f"{requested_task}: {task_stats['success']}/{task_stats['total']} ({success_rate:.1f}%), failed={task_stats['fail']}")
def run_single_inference(args, cfg):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "xpu:0" if torch.xpu.is_available() else "cpu")

    print("Loading model...")
    processor = AutoProcessor.from_pretrained(cfg['tokenizer_name'], padding_side="left")
    processor.tokenizer.padding_side = "left"

    model = LottieDecoder(pix_len=cfg['pix_len'], text_len=cfg['text_len'])
    load_model_weights_into_decoder(model, args.sketch_weight, strict=False)
    model = model.to(device).eval()

    os.makedirs(args.output_dir, exist_ok=True)

    task_mode = _normalize_task_type(getattr(args, "task_mode", None))
    if task_mode == "text2lottie":
        text_prompt = args.single_text_prompt or args.single_text or "Generate a vivid Lottie animation."
        out_path = os.path.join(args.output_dir, 'single_text_result.json')
        lottie_json, info = run_inference(
            model=model,
            processor=processor,
            task_type=TASK_TEXT,
            device=device,
            cfg=cfg,
            uid=None,
            text_description=text_prompt,
            use_sampling=args.use_sampling,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            output_path=out_path,
            save_mp4=True,
            verbose=True,
        )
    elif task_mode == "text_image2lottie":
        image_path = args.single_image_path or args.single_image
        if not image_path:
            raise ValueError("text_image2lottie requires --single_image_path/--single_image")
        text_prompt = args.single_text_prompt or args.single_text or "Describe this image and animate it."
        out_path = os.path.join(args.output_dir, 'single_text_image_result.json')
        lottie_json, info = run_inference(
            model=model,
            processor=processor,
            task_type=TASK_IMAGE,
            device=device,
            cfg=cfg,
            uid=None,
            image_path=image_path,
            text_description=text_prompt,
            use_sampling=args.use_sampling,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            output_path=out_path,
            save_mp4=True,
            verbose=True,
        )
    elif task_mode == "video2lottie":
        video_path = args.single_video_path or args.single_video
        if not video_path:
            raise ValueError("video2lottie requires --single_video_path/--single_video")
        out_path = os.path.join(args.output_dir, 'single_video_result.json')
        lottie_json, info = run_inference(
            model=model,
            processor=processor,
            task_type=TASK_VIDEO,
            device=device,
            cfg=cfg,
            uid=None,
            video_path=video_path,
            text_description=None,
            use_sampling=args.use_sampling,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            output_path=out_path,
            save_mp4=True,
            verbose=True,
        )
    else:
        raise ValueError(f"Unsupported task_mode: {task_mode}")

    if lottie_json:
        print("\n✓ Generation successful!")
        print(f"  Output: {out_path}")
        print(f"  HTML Preview: {out_path.replace('.json', '.html')}")
        print(f"  Layers: {len(lottie_json.get('layers', []))}")
        print(f"  Tokens generated: {info.get('generated_len', 0)}")
    else:
        print("\n✗ Generation failed")
        print(f"  Info: {info}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lottie Generation Inference")
    
    parser.add_argument("--sketch_weight", type=str, required=True,
                        help="Path to model checkpoint directory")
    parser.add_argument("--tokenizer_name", type=str,
                        default="Qwen/Qwen3.5-9B",
                        help="Processor/tokenizer path. Must match the base model used by LottieDecoder (Qwen3.5-9B).")
    
    parser.add_argument("--output_dir", type=str, default="./output")
    
    parser.add_argument("--maxlen", type=int, default=4096,
                        help="Maximum token length for generation")
    parser.add_argument("--text_len", type=int, default=1500,
                        help="Maximum instruction context length")

    parser.add_argument("--use_sampling", action="store_true",
                        help="Use sampling instead of greedy decoding")
    parser.add_argument("--temperature", type=float, default=0.9,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.25,
                        help="Top-p (nucleus) sampling")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Top-k sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.01,
                        help="Repetition penalty (1.0 = disabled)")


    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Maximum samples to process (-1 = all)")
    parser.add_argument("--task_filter", type=str,
                        choices=['video', 'image', 'text', 'text2lottie', 'text_image2lottie', 'video2lottie'],
                        default=None, help="Only process specific task type")
    parser.add_argument("--shuffle", action="store_true", default=True,
                        help="Shuffle samples before processing")

    parser.add_argument("--single_video", type=str, default=None,
                        help="Path to single video for inference")
    parser.add_argument("--single_image", type=str, default=None,
                        help="Path to single image for inference")
    parser.add_argument("--single_text", type=str, default=None,
                        help="Text prompt for single inference")

    # MMLottie Benchmark模式
    parser.add_argument("--mmlottie_bench_dir", type=str, default="./mmlottie_bench",
                        help="Path to mmlottie_bench directory (default: ./mmlottie_bench)")
    parser.add_argument("--split", type=str, choices=['real', 'synthetic'], default=None,
                        help="Split to use from mmlottie_bench (real or synthetic)")
    parser.add_argument("--mmlottie_task", type=str,
                        choices=['text2lottie', 'text_image2lottie', 'video2lottie'],
                        default='text2lottie',
                        help="Specific task to run in mmlottie_bench")
    parser.add_argument("--task_mode", type=str,
                        choices=['text2lottie', 'text_image2lottie', 'video2lottie'],
                        default='text2lottie',
                        help="Explicit task mode for single-sample inference")

    parser.add_argument("--single_image_path", type=str, default=None,
                        help="Alias of --single_image for task-mode inference")
    parser.add_argument("--single_video_path", type=str, default=None,
                        help="Alias of --single_video for task-mode inference")
    parser.add_argument("--single_text_prompt", type=str, default=None,
                        help="Alias of --single_text for task-mode inference")

    parser.add_argument("--batch_text_file", type=str, default=None,
                        help="Path to text file with prompts (one per line) for batch text2lottie generation")

    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode with full tracebacks")
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="Verbose output")

    parser.add_argument("--num_candidates", type=int, default=1,
                        help="Number of candidates to generate (for Best-of-N selection, default: 1)")
    parser.add_argument("--no_save_mp4", action="store_true",
                        help="Disable MP4 rendering for generated Lottie JSON")

    args = parser.parse_args()
    
    cfg = {
        'tokenizer_name': args.tokenizer_name,
        'text_len': args.text_len,
        'pix_len': args.maxlen,
        'num_candidates': args.num_candidates,  
    }
    
    print("=" * 60)
    print("Lottie Generation Inference (Improved v2 + Multi-Candidate)")
    print("=" * 60)
    print(f"Model: {args.sketch_weight}")
    print(f"Max tokens: {args.maxlen}")
    print(f"Sampling: {args.use_sampling}")
    if args.use_sampling:
        print(f"  Temperature: {args.temperature}")
        print(f"  Top-p: {args.top_p}")
    print(f"Repetition penalty: {args.repetition_penalty}")
    if args.num_candidates > 1:
        print(f"🆕 Num candidates: {args.num_candidates} (Best-of-{args.num_candidates})")
    print("=" * 60)
    
    if args.single_video or args.single_image or args.single_text or args.single_video_path or args.single_image_path or args.single_text_prompt:
        print("\nRunning single sample inference...")
        run_single_inference(args, cfg)
    elif args.batch_text_file:
        if not os.path.exists(args.batch_text_file):
            raise FileNotFoundError(f"Batch text file not found: {args.batch_text_file}")
        print(f"\nRunning batch text file inference")
        print(f"  Input file: {args.batch_text_file}")
        run_batch_text_file_inference(args, cfg)
    elif args.split:
        print(f"\nRunning MMLottie Benchmark inference")
        print(f"  Split: {args.split}")
        if args.mmlottie_bench_dir and os.path.exists(args.mmlottie_bench_dir):
            print(f"  Local dataset: {args.mmlottie_bench_dir}")
        else:
            print(f"  Will download from HuggingFace if needed")
        run_mmlottie_bench_inference(args, cfg)
    else:
        print("\nError: No input specified!")
        print("Please provide one of:")
        print("  - --task_mode text2lottie|text_image2lottie|video2lottie and matching single_* args")
        print("  - --batch_text_file <path> for batch text2lottie generation")
        print("  - --split [real|synthetic] for MMLottie benchmark inference")
        exit(1)
