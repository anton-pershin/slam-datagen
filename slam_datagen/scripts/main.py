from __future__ import annotations

import json
from collections import Counter

import hydra
from omegaconf import DictConfig

from slam_datagen.datasets.merge_quality import (build_merge_quality_dataset,
                                                 write_merge_quality_dataset)
from slam_datagen.personal_data import PersonalDataGenerator
from slam_datagen.utils.common import get_config_path

CONFIG_NAME = "config_main"


def main(cfg: DictConfig) -> None:
    generator = PersonalDataGenerator(seed=cfg.random_seed)
    samples = build_merge_quality_dataset(generator=generator, cfg=cfg)

    output_path = write_merge_quality_dataset(
        samples=samples, output_file=cfg.output_file
    )
    print(f"Dataset written to {output_path}")

    preview_count = min(cfg.preview_samples, len(samples))
    if preview_count:
        print("Preview:")
        for sample in samples[:preview_count]:
            chunk_counts = Counter(chunk.format for chunk in sample.chunks)
            print(
                json.dumps(
                    {
                        "name": sample.provided_identifiers["name"],
                        "ssn": sample.provided_identifiers["ssn"],
                        "chunk_counts": dict(chunk_counts),
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
    )(main)()
