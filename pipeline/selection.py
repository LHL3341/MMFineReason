"""
Stage 4: Selection — Data selection.
Content filtering (error/length) + structure validation (think/answer tags)
+ correctness verification + difficulty filtering.
Ref: check_merged_hf.py
"""
import os
import re
from typing import Dict, List, Optional, Any, Tuple
from collections import Counter
from pathlib import Path
from datasets import Dataset


RE_ANSWER_PAIR = re.compile(r"<answer>[\s\S]*?</answer>")

COL_RESPONSE = "qwen3vl_235b_thinking_response"
COL_CAPTION = "qwen3vl_235b_instruct_caption"

NO_ANSWER_CHECK_DATASETS = [
    "FineVision-visualwebinstruct(filtered)",
    "LLaVA-CoT",
    "FineVision-ai2d_merged",
]

DEFAULT_NUM_PROC = 8


def _ngrams(tokens: List[str], n: int) -> List[tuple]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def _normalize_answer(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r'[^\w\s]', '', s)
    return re.sub(r'\s+', ' ', s)


def _load_prompt(name: str) -> str:
    prompt_path = Path(__file__).resolve().parent.parent / "prompts" / name
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    return ""


def check_structure_validity(text: str, skip_answer_check: bool = False) -> bool:
    """
    CoT structure validation (aligned with check_merged_hf.py):
    1. Exactly one <think>...</think> pair, with open tag before close tag
    2. If answer check is not skipped: <answer> tags are paired and at least one is outside <think>
    """
    if not isinstance(text, str):
        return False

    if text.count("<think>") != 1 or text.count("</think>") != 1:
        return False

    idx_think_start = text.find("<think>")
    idx_think_end_tag_start = text.find("</think>")
    idx_think_complete_end = idx_think_end_tag_start + len("</think>")

    if idx_think_start >= idx_think_end_tag_start:
        return False

    if skip_answer_check:
        return True

    count_ans_start = text.count("<answer>")
    if count_ans_start == 0:
        return False
    if count_ans_start != text.count("</answer>"):
        return False

    answer_matches = list(RE_ANSWER_PAIR.finditer(text))
    if len(answer_matches) != count_ans_start:
        return False

    has_outer_answer = False
    for match in answer_matches:
        ans_start, ans_end = match.span()
        if ans_end <= idx_think_start or ans_start >= idx_think_complete_end:
            has_outer_answer = True
            break

    return has_outer_answer


class ContentFilter:
    """
    Content filter — aligned with check_merged_hf.py filter_content_fn.
    Checks [ERROR] keyword, response word count, and caption word count.
    """

    def __init__(self,
                 min_response_words: int = 200,
                 min_caption_words: int = 100,
                 error_keyword: str = "[ERROR]"):
        self.min_response_words = min_response_words
        self.min_caption_words = min_caption_words
        self.error_keyword = error_keyword

    def filter_fn(self, example: Dict[str, Any], has_caption_col: bool = True) -> bool:
        r_text = example.get(COL_RESPONSE) or ""
        r_text = str(r_text)

        if self.error_keyword in r_text:
            return False

        if len(r_text.split()) < self.min_response_words:
            return False

        if has_caption_col:
            c_text = example.get(COL_CAPTION) or ""
            c_text = str(c_text)
            if len(c_text.split()) < self.min_caption_words:
                return False

        return True


class StructureFilter:
    """
    Structure filter — <think>/<answer> tag validation.
    Skips answer check for specific datasets.
    """

    def __init__(self, no_answer_check_datasets: Optional[List[str]] = None):
        self.no_answer_check_datasets = no_answer_check_datasets or NO_ANSWER_CHECK_DATASETS

    def should_skip_answer_check(self, source: str) -> bool:
        return any(kw in source for kw in self.no_answer_check_datasets)

    def filter_fn(self, example: Dict[str, Any]) -> bool:
        r_text = example.get(COL_RESPONSE) or ""
        r_text = str(r_text)
        source = str(example.get("source", ""))
        skip = self.should_skip_answer_check(source)
        return check_structure_validity(r_text, skip_answer_check=skip)


class QualityFilter:
    """Reasoning quality filter — N-gram deduplication + correctness verification."""

    def __init__(self,
                 ngram_n: int = 50,
                 ngram_freq: int = 3,
                 llm_client=None):
        self.ngram_n = ngram_n
        self.ngram_freq = ngram_freq
        self.llm_client = llm_client
        self._verify_prompt = _load_prompt("verify.txt")

    def detect_repetition(self, cot_text: str) -> bool:
        """N-gram repetition detection. Returns True if repetition detected (should discard)."""
        think_match = re.search(r'<think>(.*?)</think>', cot_text, re.DOTALL)
        if not think_match:
            return False

        words = think_match.group(1).split()
        if len(words) < self.ngram_n:
            return False

        counter = Counter(_ngrams(words, self.ngram_n))
        return any(count >= self.ngram_freq for count in counter.values())

    def extract_answer(self, cot_text: str) -> Optional[str]:
        """Extract the answer outside <think> tags from the CoT text."""
        think_match = re.search(r'<think>.*?</think>', cot_text, re.DOTALL)
        if think_match:
            remainder = cot_text[:think_match.start()] + cot_text[think_match.end():]
            ans = re.search(r'<answer>(.*?)</answer>', remainder, re.DOTALL)
            if ans:
                return ans.group(1).strip()

        ans = re.search(r'<answer>(.*?)</answer>', cot_text, re.DOTALL)
        return ans.group(1).strip() if ans else None

    def verify_correctness(self, ground_truth: str, extracted: str,
                           question: str = "") -> Tuple[bool, str]:
        if not ground_truth:
            return True, "No ground truth available"
        if not extracted:
            return False, "No answer extracted from CoT"

        norm_gt = _normalize_answer(ground_truth)
        norm_ext = _normalize_answer(extracted)

        if norm_gt == norm_ext:
            return True, "Exact match"
        if norm_gt in norm_ext or norm_ext in norm_gt:
            return True, "Substring match"

        if self.llm_client and question and self._verify_prompt:
            try:
                prompt = self._verify_prompt.format(
                    question=question, reference=ground_truth, generated=extracted
                )
                response = self.llm_client.generate(
                    prompt=prompt, max_tokens=512, temperature=0.0
                )
                is_equiv = "EQUIVALENT" in response.upper()
                return is_equiv, response.strip()
            except Exception:
                pass

        return False, f"No match: GT='{ground_truth[:50]}' vs Ext='{extracted[:50]}'"


class DifficultyFilter:
    """Difficulty filter — based on small model pass rate."""

    def __init__(self, small_model_client=None, num_samples: int = 4):
        self.small_model_client = small_model_client
        self.num_samples = num_samples

    def compute_pass_rate(self, question: str, image, ground_truth: str) -> float:
        if not self.small_model_client or not question or not ground_truth:
            return 1.0

        correct = 0
        for _ in range(self.num_samples):
            try:
                response = self.small_model_client.generate(
                    prompt=f"Question: {question}\nPlease provide your answer.",
                    image=image, max_tokens=512, temperature=0.7
                )
                ans_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
                extracted = ans_match.group(1).strip() if ans_match else response.strip().split('\n')[-1]

                if _normalize_answer(ground_truth) == _normalize_answer(extracted):
                    correct += 1
                elif _normalize_answer(ground_truth) in _normalize_answer(extracted):
                    correct += 1
            except Exception:
                continue

        return correct / self.num_samples


class SelectionPipeline:
    """
    Stage 4 full pipeline:
    1. Content filter ([ERROR] keyword, word count thresholds)
    2. Structure validation (<think>/<answer> tags)
    3. N-gram deduplication
    4. Correctness verification (is_consistent)
    5. Difficulty filtering (pass_rate)
    """

    def __init__(self,
                 content_filter: Optional[ContentFilter] = None,
                 structure_filter: Optional[StructureFilter] = None,
                 quality_filter: Optional[QualityFilter] = None,
                 difficulty_filter: Optional[DifficultyFilter] = None,
                 num_proc: int = DEFAULT_NUM_PROC):
        self.content_filter = content_filter or ContentFilter()
        self.structure_filter = structure_filter or StructureFilter()
        self.quality_filter = quality_filter or QualityFilter()
        self.difficulty_filter = difficulty_filter
        self.num_proc = num_proc

    def run(self, dataset: Dataset, output_path: Optional[str] = None,
            quality: bool = True,
            consistency: bool = True,
            difficulty: Optional[float] = None,
            use_existing_fields: bool = True) -> Dataset:
        """
        Run data selection.

        Args:
            dataset: Input dataset
            output_path: Output path
            quality: Whether to run quality filtering (content + structure + dedup)
            consistency: Whether to verify CoT answer vs ground-truth consistency
            difficulty: Difficulty threshold, None=skip, 0.0=hardest only, 0.5=moderately hard
            use_existing_fields: Whether to use existing is_consistent / pass_rate fields
        """
        print("=" * 70)
        print("Stage 4: Selection")
        print("=" * 70)

        initial_count = len(dataset)
        has_caption_col = COL_CAPTION in dataset.column_names

        if quality:
            # Step 1: Content filter
            print("\n[1/5] Content filter ([ERROR] check + word count)...")
            pre = len(dataset)
            dataset = dataset.filter(
                lambda x: self.content_filter.filter_fn(x, has_caption_col=has_caption_col),
                num_proc=self.num_proc,
                desc="Content Filter",
            )
            print(f"  Content filter: {pre} -> {len(dataset)} (removed {pre - len(dataset)})")

            # Step 2: Structure validation
            print("\n[2/5] Structure validation (<think>/<answer> tags)...")
            pre = len(dataset)
            dataset = dataset.filter(
                self.structure_filter.filter_fn,
                num_proc=self.num_proc,
                desc="Structure Filter",
            )
            print(f"  Structure filter: {pre} -> {len(dataset)} (removed {pre - len(dataset)})")

            # Step 3: N-gram deduplication
            print("\n[3/5] N-gram deduplication...")
            pre = len(dataset)
            dataset = dataset.filter(
                lambda x: not self.quality_filter.detect_repetition(
                    x.get(COL_RESPONSE, "")
                ),
                num_proc=self.num_proc,
                desc="N-gram Dedup",
            )
            print(f"  N-gram dedup: {pre} -> {len(dataset)} (removed {pre - len(dataset)})")

        else:
            print("\n[1-3/5] Skipping quality filtering")

        # Step 4: Correctness verification
        if consistency:
            print("\n[4/5] Correctness verification...")
            pre = len(dataset)
            if use_existing_fields and "is_consistent" in dataset.column_names:
                dataset = dataset.filter(
                    lambda x: x.get("is_consistent", False),
                    num_proc=self.num_proc,
                    desc="Consistency Filter (existing)",
                )
                print(f"  Correctness (existing field): {pre} -> {len(dataset)} (removed {pre - len(dataset)})")
            else:
                def _verify(example):
                    cot = example.get(COL_RESPONSE, "")
                    extracted = self.quality_filter.extract_answer(cot)
                    gt = example.get("answer", "") or example.get("original_answer", "")
                    is_correct, analysis = self.quality_filter.verify_correctness(
                        gt, extracted or "", example.get("question", "")
                    )
                    example["is_consistent"] = is_correct
                    example["consistency_analysis"] = analysis
                    return example

                dataset = dataset.map(_verify, num_proc=self.num_proc, desc="Verify Correctness")
                dataset = dataset.filter(
                    lambda x: x.get("is_consistent", False),
                    num_proc=self.num_proc,
                )
                print(f"  Correctness (computed): {pre} -> {len(dataset)} (removed {pre - len(dataset)})")
        else:
            print("\n[4/5] Skipping correctness verification")

        # Step 5: Difficulty filtering
        if difficulty is not None:
            print(f"\n[5/5] Difficulty filter (threshold={difficulty})...")
            pre = len(dataset)

            if use_existing_fields and "qwen3vl_4b_pass_rate" in dataset.column_names:
                threshold = difficulty
                dataset = dataset.filter(
                    lambda x: (x.get("qwen3vl_4b_pass_rate") is not None
                               and x["qwen3vl_4b_pass_rate"] <= threshold),
                    num_proc=self.num_proc,
                    desc="Difficulty Filter (existing)",
                )
                print(f"  Difficulty (existing field): {pre} -> {len(dataset)} (removed {pre - len(dataset)})")
            elif self.difficulty_filter and self.difficulty_filter.small_model_client:
                def _compute_pr(example):
                    pr = self.difficulty_filter.compute_pass_rate(
                        example.get("question", ""),
                        example.get("image"),
                        example.get("answer", "")
                    )
                    example["qwen3vl_4b_pass_rate"] = pr
                    return example

                threshold = difficulty
                dataset = dataset.map(_compute_pr, desc="Compute Pass Rate")
                dataset = dataset.filter(
                    lambda x: x["qwen3vl_4b_pass_rate"] <= threshold,
                    num_proc=self.num_proc,
                )
                print(f"  Difficulty (computed): {pre} -> {len(dataset)} (removed {pre - len(dataset)})")
            else:
                print("  Skipping difficulty filter (no small model client and no existing field)")
        else:
            print("\n[5/5] Skipping difficulty filtering")

        retain_rate = (len(dataset) / initial_count * 100) if initial_count > 0 else 0
        print(f"\n  Total: {initial_count} -> {len(dataset)} (retain rate {retain_rate:.2f}%)")

        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            dataset.to_parquet(output_path)
            print(f"\n  Selection complete, saved to: {output_path}")

        return dataset
