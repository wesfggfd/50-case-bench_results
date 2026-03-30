from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass(frozen=True)
class ModelSpec:
    model_type: str
    family: str
    description: str
    tokenizer_name: str | None = None
    inference_script: str | None = None
    supported_tasks: Tuple[str, ...] = ("text2lottie",)
    supports_local_weights: bool = True
    notes: Tuple[str, ...] = field(default_factory=tuple)


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "omnilottie_qwen35": ModelSpec(
        model_type="omnilottie_qwen35",
        family="omnilottie_family",
        description="OmniLottie-style Qwen3.5 benchmark inference backend",
        tokenizer_name="Qwen/Qwen3.5-9B",
        inference_script="runner_omnilottie_qwen35.py",
        supported_tasks=("text2lottie", "text_image2lottie", "video2lottie"),
        supports_local_weights=True,
    ),
    "omnilottie_original": ModelSpec(
        model_type="omnilottie_original",
        family="omnilottie_family",
        description="Original OmniLottie family model with benchmark wrapper runner",
        tokenizer_name="Qwen/Qwen2.5-VL-3B-Instruct",
        inference_script="runner_omnilottie_original.py",
        supported_tasks=("text2lottie", "text_image2lottie", "video2lottie"),
        supports_local_weights=True,
        notes=("Requires the original OmniLottie repository to be present locally.",),
    ),
    "deepseekv3": ModelSpec(
        model_type="deepseekv3",
        family="generic_text_family",
        description="Generic text-generation family entry for DeepSeek-V3 style models",
        tokenizer_name=None,
        inference_script=None,
        supported_tasks=("text2lottie",),
        supports_local_weights=True,
        notes=("Provide --runner-script for the concrete DeepSeek-V3 inference implementation.",),
    ),
    "qwen35_base": ModelSpec(
        model_type="qwen35_base",
        family="generic_text_family",
        description="Generic text-generation family entry for base Qwen3.5 models",
        tokenizer_name="Qwen/Qwen3.5-9B",
        inference_script=None,
        supported_tasks=("text2lottie",),
        supports_local_weights=True,
        notes=("Provide --runner-script for the concrete base-model inference implementation.",),
    ),
    "recraft": ModelSpec(
        model_type="recraft",
        family="api_family",
        description="API-driven model family entry for Recraft-like services",
        tokenizer_name=None,
        inference_script=None,
        supported_tasks=("text2lottie", "text_image2lottie"),
        supports_local_weights=False,
        notes=("Provide --runner-script that wraps the target API and writes benchmark artifacts.",),
    ),
}


def get_model_spec(model_type: str) -> ModelSpec:
    key = str(model_type).strip().lower()
    if key not in MODEL_REGISTRY:
        supported = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(f"Unsupported model_type={model_type}. Supported: {supported}")
    return MODEL_REGISTRY[key]
