from __future__ import annotations

import sys
import os
sys.path.insert(0, '/home/sonnet/projects/slam-datagen')

import json
from collections import Counter

import hydra
from omegaconf import DictConfig

from slam_datagen.datasets.synthetic_nli import (build_synthetic_nli_dataset,
                                                 write_synthetic_nli_dataset)
from slam_datagen.utils.common import get_config_path

CONFIG_NAME = "config_generate_synthetic_nli_dataset"


def generate_synthetic_nli_dataset(cfg: DictConfig) -> None:
    samples = build_synthetic_nli_dataset(cfg=cfg)
    print(samples[0])

    output_path = write_synthetic_nli_dataset(
        samples=samples, output_file=cfg.output_file
    )
    print(f"Dataset written to {output_path}")

    preview_count = min(cfg.preview_samples, len(samples))
    if preview_count:
        print("Preview:")
        for sample in samples[:preview_count]:
            print(
                json.dumps(
                    {
                    "id": sample.id,
                    "premise": sample.premise,
                    "hypothesis": sample.hypothesis,
                    "gold_label": sample.gold_label,
                    "split": sample.split,
                    "hop_difficulty": sample.hop_difficulty,
                    "distractor_difficulty": sample.distractor_difficulty,
                    "background_knowledge": sample.background_knowledge,
                    "world_seed": sample.world_seed,
                    "templates_version": sample.templates_version,
                    "ood_tag": sample.ood_tag,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )


if __name__ == "__main__":
    hydra.main(
        config_path=str(get_config_path()),
        config_name=CONFIG_NAME,
        version_base="1.3",
    )(generate_synthetic_nli_dataset)()
