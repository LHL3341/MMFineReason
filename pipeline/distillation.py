"""
Stage 3: Distillation — CoT distillation and caption generation.
Uses teacher models to produce long chain-of-thought reasoning and dense image descriptions.
Ref: qwen3vl_distill/distill.txt, distill_tool/caption/local_api.py
"""
import os
import re
import io
import math
import base64
from typing import Dict, List, Optional, Any
from pathlib import Path
from datasets import Dataset
from PIL import Image, ImageOps

DEFAULT_NUM_PROC = 8

COL_RESPONSE = "qwen3vl_235b_thinking_response"
COL_CAPTION = "qwen3vl_235b_instruct_caption"

SHORT_MIN = 32
LONG_MAX = 2048


def _load_prompt(name: str) -> str:
    prompt_path = Path(__file__).resolve().parent.parent / "prompts" / name
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    return ""


def compute_scale(w: int, h: int) -> float:
    """Compute image scaling factor (aligned with caption/local_api.py logic)."""
    short_side = min(w, h)
    long_side = max(w, h)

    need_up = short_side < SHORT_MIN
    need_down = long_side > LONG_MAX

    if not need_up and not need_down:
        return 1.0

    up_min = (SHORT_MIN / short_side) if need_up else 1.0
    down_max = (LONG_MAX / long_side) if need_down else 1.0

    if need_up and not need_down:
        return float(up_min)
    if need_down and not need_up:
        return float(down_max)
    if up_min <= down_max:
        return float(up_min)
    else:
        return float(down_max)


def preprocess_image(image: Image.Image) -> Image.Image:
    """Preprocess image: EXIF transpose + scaling + RGB conversion."""
    image = ImageOps.exif_transpose(image)
    w, h = image.size
    scale = compute_scale(w, h)

    if scale != 1.0:
        new_w = max(1, int(math.ceil(w * scale)))
        new_h = max(1, int(math.ceil(h * scale)))
        image = image.resize((new_w, new_h), resample=Image.BICUBIC)

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def encode_image_to_base64(image: Image.Image) -> str:
    """Encode a PIL Image to a base64 data URI."""
    image = preprocess_image(image)
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=90)
    buffered.seek(0)
    encoded = base64.b64encode(buffered.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


class CoTDistiller:
    """
    Chain-of-thought distiller.
    Ref: qwen3vl_distill/distill.yaml + distill.txt
    """

    def __init__(self, model_client, prompt_template: Optional[str] = None,
                 temperature: float = 1.0, top_p: float = 0.95,
                 max_tokens: int = 16384, thinking: bool = True):
        self.model_client = model_client
        self.prompt_template = prompt_template or _load_prompt("distill.txt")
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.thinking = thinking

    def distill(self, image: Image.Image, question: str) -> str:
        prompt = self.prompt_template.replace("{question}", question)
        prompt = prompt.replace("{image}", "")

        kwargs = {
            "prompt": prompt,
            "image": image,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if hasattr(self.model_client, 'generate'):
            return self.model_client.generate(**kwargs).strip()
        return ""

    @staticmethod
    def validate_format(cot_text: str) -> bool:
        """Validate CoT output format."""
        if not isinstance(cot_text, str):
            return False
        if cot_text.count("<think>") != 1 or cot_text.count("</think>") != 1:
            return False
        idx_start = cot_text.find("<think>")
        idx_end = cot_text.find("</think>")
        if idx_start >= idx_end:
            return False
        has_answer = bool(re.search(r'<answer>.*?</answer>', cot_text, re.DOTALL))
        return has_answer and len(cot_text.split()) >= 100

    @staticmethod
    def extract_answer(cot_text: str) -> Optional[str]:
        match = re.search(r'<answer>(.*?)</answer>', cot_text, re.DOTALL)
        return match.group(1).strip() if match else None

    def distill_batch(self, dataset: Dataset, max_retries: int = 1) -> Dataset:
        """Batch CoT distillation."""
        def _distill(example):
            image = example.get("image")
            question = example.get("question", "")
            if not image or not question:
                return example

            existing = example.get(COL_RESPONSE, "")
            if existing and self.validate_format(existing):
                return example

            for attempt in range(max_retries + 1):
                try:
                    response = self.distill(image, question)
                    if self.validate_format(response):
                        example[COL_RESPONSE] = response
                        break
                    if attempt == max_retries:
                        print(f"  Invalid CoT format (id={example.get('id')}), skipping")
                except Exception as e:
                    if attempt == max_retries:
                        print(f"  CoT distillation failed (id={example.get('id')}): {e}")
            return example

        return dataset.map(_distill)


class CaptionGenerator:
    """
    Image caption generator.
    Ref: distill_tool/caption/local_api.py PROMPT_TEMPLATE
    """

    def __init__(self, model_client, prompt_template: Optional[str] = None,
                 temperature: float = 0.6, top_p: float = 0.95,
                 max_tokens: int = 16384):
        self.model_client = model_client
        self.prompt_template = prompt_template or _load_prompt("caption.txt")
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

    def generate(self, image: Image.Image, question: str = "") -> str:
        prompt = self.prompt_template.replace("{question}", question or "")
        prompt = prompt.replace("{image}", "")

        kwargs = {
            "prompt": prompt,
            "image": image,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if hasattr(self.model_client, 'generate'):
            return self.model_client.generate(**kwargs).strip()
        return ""

    def generate_batch(self, dataset: Dataset, min_caption_len: int = 50) -> Dataset:
        def _caption(example):
            image = example.get("image")
            if not image:
                return example

            existing = example.get(COL_CAPTION, "")
            if existing and len(str(existing)) > min_caption_len:
                return example

            try:
                caption = self.generate(image, example.get("question", ""))
                example[COL_CAPTION] = caption
            except Exception as e:
                print(f"  Caption generation failed (id={example.get('id')}): {e}")
            return example

        return dataset.map(_caption)


class DistillationPipeline:
    """Stage 3: CoT distillation + caption generation."""

    def __init__(self, cot_model_client=None, caption_model_client=None,
                 cot_config: Optional[Dict[str, Any]] = None,
                 caption_config: Optional[Dict[str, Any]] = None):
        cot_cfg = cot_config or {}
        caption_cfg = caption_config or {}

        self.cot_distiller = CoTDistiller(
            cot_model_client, **cot_cfg
        ) if cot_model_client else None

        self.caption_generator = CaptionGenerator(
            caption_model_client, **caption_cfg
        ) if caption_model_client else None

    def run(self, dataset: Dataset, output_path: Optional[str] = None,
            distill_cot: bool = True, generate_caption: bool = True) -> Dataset:
        print("=" * 70)
        print("Stage 3: Distillation")
        print("=" * 70)

        if distill_cot and self.cot_distiller:
            print("\n[1/2] CoT distillation (Teacher Model)...")
            dataset = self.cot_distiller.distill_batch(dataset)
            valid = sum(
                1 for ex in dataset
                if CoTDistiller.validate_format(ex.get(COL_RESPONSE, ""))
            )
            print(f"  Valid CoT: {valid}/{len(dataset)}")
        else:
            has_cot = sum(1 for ex in dataset if ex.get(COL_RESPONSE, ""))
            print(f"\n[1/2] Skipping CoT distillation (existing CoT: {has_cot}/{len(dataset)})")

        if generate_caption and self.caption_generator:
            print("\n[2/2] Caption generation (Teacher Model)...")
            dataset = self.caption_generator.generate_batch(dataset)
            has_cap = sum(1 for ex in dataset if ex.get(COL_CAPTION, ""))
            print(f"  Valid captions: {has_cap}/{len(dataset)}")
        else:
            has_cap = sum(1 for ex in dataset if ex.get(COL_CAPTION, ""))
            print(f"\n[2/2] Skipping caption generation (existing captions: {has_cap}/{len(dataset)})")

        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            dataset.to_parquet(output_path)
            print(f"\n  Distillation complete, saved to: {output_path}")

        return dataset
