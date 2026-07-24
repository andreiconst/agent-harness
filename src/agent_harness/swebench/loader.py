"""Pull SWE-bench instances from the Hugging Face dataset."""

from __future__ import annotations

from datasets import load_dataset

DATASET_NAME = "princeton-nlp/SWE-bench_Lite"


def load_instances(split: str = "test", dataset: str = DATASET_NAME):
    return load_dataset(dataset, split=split)


def get_instance(instance_id: str, split: str = "test", dataset: str = DATASET_NAME) -> dict:
    ds = load_instances(split=split, dataset=dataset)
    for row in ds:
        if row["instance_id"] == instance_id:
            return dict(row)
    raise KeyError(f"instance {instance_id!r} not found in {dataset}:{split}")
