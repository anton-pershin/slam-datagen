from __future__ import annotations

import hydra
from omegaconf import DictConfig

from slam_datagen.datasets.human_messages import (build_human_messages_dataset,
                                                  write_human_messages_dataset)
from slam_datagen.llm.message_generator import MessageGeneratorViaLlm
from slam_datagen.utils.common import get_config_path

CONFIG_NAME = "config_generate_human_messages"


def generate_human_messages(cfg: DictConfig) -> None:
    prompt_cfg = cfg.human_message_prompts
    model = hydra.utils.instantiate(cfg.llm)
    message_generator = MessageGeneratorViaLlm(
        model=model,
        system_prompt=prompt_cfg.system_prompt,
    )

    samples = build_human_messages_dataset(
        cfg=cfg,
        prompt_cfg=prompt_cfg,
        message_generator=message_generator,
    )

    output_path = write_human_messages_dataset(
        samples=samples,
        output_file=cfg.output_file,
    )
    print(f"Dataset written to {output_path}")

    preview_count = min(cfg.preview_samples, len(samples))
    if preview_count:
        print("Preview:")
        for sample in samples[:preview_count]:
            print(sample)


if __name__ == "__main__":
    hydra.main(
        config_path=str(get_config_path()),
        config_name=CONFIG_NAME,
        version_base="1.3",
    )(generate_human_messages)()
