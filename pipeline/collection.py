"""
Stage 1: Collection — Data collection.
Downloads the pre-processed MMFineReason-Full dataset from HuggingFace.
"""
import os
from typing import List, Optional
from datasets import Dataset, load_dataset, concatenate_datasets, get_dataset_config_names


DATASET_ID = "OpenDataArena/MMFineReason-Full-2.3M-Qwen3-VL-235B-Thinking"


class CollectionPipeline:
    """Stage 1: Download dataset from HuggingFace."""

    def __init__(self, dataset_id: str = DATASET_ID,
                 cache_dir: Optional[str] = None):
        self.dataset_id = dataset_id
        self.cache_dir = cache_dir

    def _get_available_configs(self) -> List[str]:
        """Get all available config (subset) names for the dataset."""
        try:
            configs = get_dataset_config_names(self.dataset_id)
            return [c for c in configs if c != "default"]
        except Exception:
            return []

    def _load_single(self, config_name: Optional[str] = None,
                     max_samples: Optional[int] = None) -> Dataset:
        kwargs = dict(split="train", cache_dir=self.cache_dir)
        if config_name:
            ds = load_dataset(self.dataset_id, config_name, **kwargs)
        else:
            ds = load_dataset(self.dataset_id, **kwargs)
        if max_samples and len(ds) > max_samples:
            ds = ds.select(range(max_samples))
        return ds

    def run(self, subsets: Optional[List[str]] = None,
            max_samples_per_subset: Optional[int] = None,
            output_path: Optional[str] = None) -> Dataset:
        print("=" * 70)
        print("Stage 1: Collection")
        print("=" * 70)

        available_configs = self._get_available_configs()

        if subsets:
            targets = subsets
        elif available_configs:
            print(f"  Detected {len(available_configs)} subsets: {available_configs}")
            targets = available_configs
        else:
            targets = None

        if targets:
            datasets_list = []
            for name in targets:
                print(f"  Downloading subset: {name}")
                ds = self._load_single(name, max_samples_per_subset)
                print(f"    {len(ds)} samples")
                datasets_list.append(ds)
            dataset = concatenate_datasets(datasets_list)
        else:
            print(f"  Downloading dataset: {self.dataset_id}")
            dataset = self._load_single(None, max_samples_per_subset)

        print(f"  Done: {len(dataset)} samples, {len(dataset.column_names)} columns")
        print(f"  Fields: {dataset.column_names}")

        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            dataset.to_parquet(output_path)
            print(f"\n  Collection complete, saved to: {output_path}")

        return dataset
