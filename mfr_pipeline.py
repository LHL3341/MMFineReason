"""
MMFineReason Pipeline entry point.
Four-stage pipeline: Collection → Cleaning → Distillation → Selection
"""
import os
import argparse
import yaml
from typing import List, Dict, Any, Optional
from datasets import Dataset, load_dataset

from pipeline.collection import CollectionPipeline, DATASET_ID
from pipeline.cleaning import CleaningPipeline
from pipeline.distillation import DistillationPipeline
from pipeline.selection import (
    SelectionPipeline, ContentFilter, StructureFilter,
    QualityFilter, DifficultyFilter, DEFAULT_NUM_PROC,
)
from pipeline.model_client import create_model_client


def _make_client(cfg: Optional[Dict]) -> Optional[Any]:
    if not cfg or not isinstance(cfg, dict):
        return None
    try:
        return create_model_client(cfg)
    except Exception as e:
        print(f"  Failed to create model client: {e}")
        return None


class MMFineReasonPipeline:
    """MMFineReason four-stage pipeline."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.num_proc = config.get("num_proc", DEFAULT_NUM_PROC)
        self._init_stages()

    def _init_stages(self):
        cfg = self.config
        col_cfg = cfg.get("collection", {})
        cln_cfg = cfg.get("cleaning", {})
        dst_cfg = cfg.get("distillation", {})
        sel_cfg = cfg.get("selection", {})

        # Stage 1: Collection
        self.collection = CollectionPipeline(
            dataset_id=col_cfg.get("dataset_id", DATASET_ID),
            cache_dir=col_cfg.get("cache_dir"),
        )

        # Stage 2: Cleaning
        self.cleaning = CleaningPipeline(
            llm_client=_make_client(cln_cfg.get("llm_client")),
            max_image_size=cln_cfg.get("max_image_size", 2048),
        )

        # Stage 3: Distillation
        cot_cfg = dst_cfg.get("cot", {})
        cap_cfg = dst_cfg.get("caption", {})
        cot_client = _make_client(cot_cfg.get("client"))
        cap_client = _make_client(cap_cfg.get("client"))
        cot_params = {k: v for k, v in cot_cfg.items() if k != "client"}
        cap_params = {k: v for k, v in cap_cfg.items() if k != "client"}

        self.distillation = DistillationPipeline(
            cot_model_client=cot_client,
            caption_model_client=cap_client,
            cot_config=cot_params or None,
            caption_config=cap_params or None,
        )

        # Stage 4: Selection
        cf_cfg = sel_cfg.get("content_filter", {})
        sf_cfg = sel_cfg.get("structure_filter", {})
        qf_cfg = sel_cfg.get("quality_filter", {})
        df_cfg = sel_cfg.get("difficulty_filter", {})

        content_filter = ContentFilter(
            min_response_words=cf_cfg.get("min_response_words", 200),
            min_caption_words=cf_cfg.get("min_caption_words", 100),
            error_keyword=cf_cfg.get("error_keyword", "[ERROR]"),
        )
        structure_filter = StructureFilter(
            no_answer_check_datasets=sf_cfg.get("no_answer_check_datasets"),
        )
        quality_filter = QualityFilter(
            ngram_n=qf_cfg.get("ngram_n", 50),
            ngram_freq=qf_cfg.get("ngram_freq", 3),
            llm_client=_make_client(cln_cfg.get("llm_client")),
        )
        diff_client = _make_client(df_cfg.get("client"))
        difficulty_filter = DifficultyFilter(
            small_model_client=diff_client,
            num_samples=df_cfg.get("num_samples", 4),
        ) if diff_client else None

        self.selection = SelectionPipeline(
            content_filter=content_filter,
            structure_filter=structure_filter,
            quality_filter=quality_filter,
            difficulty_filter=difficulty_filter,
            num_proc=self.num_proc,
        )

    # ─── Run individual stages ─────────────────────────────

    def run_collection(self, output_dir: str, **kwargs) -> Dataset:
        col_cfg = self.config.get("collection", {})
        output_path = os.path.join(output_dir, "collected.parquet")
        return self.collection.run(
            subsets=kwargs.get("subsets", col_cfg.get("subsets")),
            max_samples_per_subset=kwargs.get("max_samples"),
            output_path=output_path,
        )

    def run_cleaning(self, input_path: str, output_dir: str, **kwargs) -> Dataset:
        cln_cfg = self.config.get("cleaning", {})
        dataset = load_dataset("parquet", data_files=input_path, split="train")
        output_path = os.path.join(output_dir, "cleaned.parquet")
        return self.cleaning.run(
            dataset, output_path=output_path,
            clean_images=kwargs.get("clean_images", cln_cfg.get("clean_images", True)),
        )

    def run_distillation(self, input_path: str, output_dir: str, **kwargs) -> Dataset:
        dataset = load_dataset("parquet", data_files=input_path, split="train")
        output_path = os.path.join(output_dir, "distilled.parquet")
        return self.distillation.run(
            dataset, output_path=output_path,
            distill_cot=kwargs.get("distill_cot", True),
            generate_caption=kwargs.get("generate_caption", True),
        )

    def run_selection(self, input_path: str, output_dir: str, **kwargs) -> Dataset:
        sel_cfg = self.config.get("selection", {})
        qf_cfg = sel_cfg.get("quality_filter", {})
        df_cfg = sel_cfg.get("difficulty_filter", {})
        dataset = load_dataset("parquet", data_files=input_path, split="train")
        output_path = os.path.join(output_dir, "selected.parquet")
        return self.selection.run(
            dataset, output_path=output_path,
            consistency=kwargs.get("consistency", qf_cfg.get("consistency", True)),
            difficulty=kwargs.get("difficulty", df_cfg.get("threshold")),
            use_existing_fields=kwargs.get("use_existing_fields", True),
        )

    # ─── Run full pipeline ─────────────────────────────────

    def run_full(self, output_dir: str = "output", **kwargs) -> Dict[str, str]:
        os.makedirs(output_dir, exist_ok=True)
        outputs = {}
        cln_cfg = self.config.get("cleaning", {})
        sel_cfg = self.config.get("selection", {})
        qf_cfg = sel_cfg.get("quality_filter", {})
        df_cfg = sel_cfg.get("difficulty_filter", {})

        # Stage 1
        ds = self.run_collection(output_dir, **kwargs)
        outputs["collected"] = os.path.join(output_dir, "collected.parquet")

        # Stage 2
        ds = self.cleaning.run(
            ds, output_path=os.path.join(output_dir, "cleaned.parquet"),
            clean_images=kwargs.get("clean_images", cln_cfg.get("clean_images", False)),
        )
        outputs["cleaned"] = os.path.join(output_dir, "cleaned.parquet")

        # Stage 3
        ds = self.distillation.run(
            ds, output_path=os.path.join(output_dir, "distilled.parquet"),
            distill_cot=kwargs.get("distill_cot", True),
            generate_caption=kwargs.get("generate_caption", True),
        )
        outputs["distilled"] = os.path.join(output_dir, "distilled.parquet")

        # Stage 4
        ds = self.selection.run(
            ds, output_path=os.path.join(output_dir, "selected.parquet"),
            consistency=kwargs.get("consistency", qf_cfg.get("consistency", True)),
            difficulty=kwargs.get("difficulty", df_cfg.get("threshold")),
            use_existing_fields=kwargs.get("use_existing_fields", True),
        )
        outputs["selected"] = os.path.join(output_dir, "selected.parquet")

        print("\n" + "=" * 70)
        print("Pipeline completed!")
        print("=" * 70)
        for stage, path in outputs.items():
            print(f"  {stage}: {path}")

        return outputs


# ─── Config loading ────────────────────────────────────────

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


# ─── CLI ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MMFineReason Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--step", type=str,
                        choices=["all", "collection", "cleaning", "distillation", "selection"],
                        default="all")
    parser.add_argument("--input", type=str, help="Input parquet path")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--subsets", type=str, nargs="+",
                        help="HF subsets to download (e.g. BMMR Euclid30K)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples per subset (for quick testing)")
    parser.add_argument("--num-proc", type=int, default=None,
                        help="Number of parallel workers")
    parser.add_argument("--difficulty", type=float, default=None,
                        help="Difficulty threshold (e.g. 0.0=hardest only, 0.5=moderately hard, unset=skip)")

    args = parser.parse_args()

    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        print(f"  Config file {args.config} not found, using defaults")
        config = {}

    if args.subsets:
        col = config.setdefault("collection", {})
        col["subsets"] = args.subsets
    if args.num_proc is not None:
        config["num_proc"] = args.num_proc

    pipeline = MMFineReasonPipeline(config)

    kwargs = {
        "max_samples": args.max_samples,
        "subsets": args.subsets,
        "difficulty": args.difficulty,
    }

    if args.step == "all":
        pipeline.run_full(args.output, **kwargs)

    elif args.step == "collection":
        pipeline.run_collection(args.output, **kwargs)

    elif args.step == "cleaning":
        if not args.input:
            args.input = os.path.join(args.output, "collected.parquet")
        pipeline.run_cleaning(args.input, args.output)

    elif args.step == "distillation":
        if not args.input:
            args.input = os.path.join(args.output, "cleaned.parquet")
        pipeline.run_distillation(args.input, args.output)

    elif args.step == "selection":
        if not args.input:
            args.input = os.path.join(args.output, "distilled.parquet")
        pipeline.run_selection(args.input, args.output, **kwargs)


if __name__ == "__main__":
    main()
