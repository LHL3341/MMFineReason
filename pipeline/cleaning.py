"""
Stage 2: Cleaning — Data cleaning.
Calls an LLM to translate, denoise, refine instructions, and filter unsuitable tasks.
Saves the original question to the ori_question field. Image cleaning handled separately.
"""
import json
import os
import re
from typing import Optional
from PIL import Image
from datasets import Dataset


PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts", "clean.txt")


def _load_prompt() -> str:
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


def _parse_llm_response(response: str):
    """Parse the LLM cleaning response.

    Returns:
        (keep: bool, cleaned_text: str | None)
        - keep=False means the sample should be discarded (non-answer question)
        - cleaned_text=None means the original text needs no modification
    """
    text = response.strip()
    if text.startswith("No Problem") or text.startswith("no problem"):
        return True, None

    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        return True, None
    try:
        obj = json.loads(match.group())
    except json.JSONDecodeError:
        return True, None

    error_types = obj.get("error_type", [])
    if isinstance(error_types, str):
        error_types = [error_types]

    if "non-answer question" in error_types:
        return False, None

    corrected = obj.get("corrected_text", "").strip()
    if corrected:
        return True, corrected
    return True, None


class ImageCleaner:
    """Image cleaner — discard corrupt images, resize, convert to RGB."""

    def __init__(self, max_size: int = 2048):
        self.max_size = max_size

    def clean_image(self, image) -> Optional[Image.Image]:
        try:
            if isinstance(image, str):
                img = Image.open(image)
            elif isinstance(image, Image.Image):
                img = image
            else:
                return None

            if img.mode != "RGB":
                img = img.convert("RGB")

            w, h = img.size
            if max(w, h) > self.max_size:
                scale = self.max_size / max(w, h)
                img = img.resize(
                    (int(w * scale), int(h * scale)),
                    Image.Resampling.LANCZOS,
                )
            return img
        except Exception as e:
            print(f"  Image processing failed: {e}")
            return None


class CleaningPipeline:
    """Stage 2: LLM-based question cleaning + image cleaning."""

    def __init__(self, llm_client=None, max_image_size: int = 2048):
        self.llm_client = llm_client
        self.image_cleaner = ImageCleaner(max_image_size)
        self.prompt_template = _load_prompt()

    def _clean_question(self, question: str) -> tuple:
        """Call LLM to clean a single question.

        Returns:
            (keep, cleaned_question)
        """
        if not question:
            return False, question

        if not self.llm_client:
            return True, question

        prompt = self.prompt_template.replace("{question}", question)
        response = self.llm_client.generate(
            prompt=prompt, max_tokens=2048, temperature=0.0,
        )
        keep, corrected = _parse_llm_response(response)
        return keep, corrected if corrected else question

    def run(
        self,
        dataset: Dataset,
        output_path: Optional[str] = None,
        clean_images: bool = True,
    ) -> Dataset:
        print("=" * 70)
        print("Stage 2: Cleaning")
        print("=" * 70)

        initial_count = len(dataset)

        # ── Text cleaning via LLM ──────────────────────────
        print("\n[1/2] Text cleaning (LLM)...")

        def _clean_text(example):
            q = example.get("question", "") or ""
            example["ori_question"] = q
            keep, cleaned = self._clean_question(q)
            example["question"] = cleaned
            example["_keep"] = keep
            return example

        dataset = dataset.map(_clean_text)
        dataset = dataset.filter(lambda x: x["_keep"])
        dataset = dataset.remove_columns(["_keep"])
        print(f"  Text cleaning: {initial_count} -> {len(dataset)} samples")

        # ── Image cleaning ─────────────────────────────────
        if clean_images:
            print("\n[2/2] Image cleaning...")
            pre_img = len(dataset)

            def _clean_image(example):
                img = example.get("image")
                if img is None:
                    example["_has_image"] = False
                    return example
                cleaned = self.image_cleaner.clean_image(img)
                if cleaned is None:
                    example["_has_image"] = False
                else:
                    example["image"] = cleaned
                    example["_has_image"] = True
                return example

            dataset = dataset.map(_clean_image)
            dataset = dataset.filter(lambda x: x["_has_image"])
            dataset = dataset.remove_columns(["_has_image"])
            print(f"  Image cleaning: {pre_img} -> {len(dataset)} samples")
        else:
            print("\n[2/2] Skipping image cleaning")

        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            dataset.to_parquet(output_path)
            print(f"\n  Cleaning complete, saved to: {output_path}")

        return dataset
