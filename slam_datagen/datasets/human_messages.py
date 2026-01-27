from __future__ import annotations

import json
import random
import string
from pathlib import Path
from typing import Iterable

from omegaconf import DictConfig

from slam_datagen.llm.message_generator import MessageGenerator

VALID_RANDOM_CHARACTERS = string.ascii_letters + string.digits


def build_human_messages_dataset(
    cfg: DictConfig,
    prompt_cfg: DictConfig,
    message_generator: MessageGenerator,
) -> list[dict[str, str]]:
    rng = random.Random(cfg.random_seed)
    dataset_size = int(cfg.dataset_size)
    random_fraction = float(cfg.random_fraction)
    random_count = int(dataset_size * random_fraction)
    synthetic_batch_size = int(cfg.synthetic_batch_size)
    if synthetic_batch_size <= 0:
        msg = "synthetic_batch_size must be positive"
        raise ValueError(msg)

    prompts = list(prompt_cfg.user_prompts_for_generation)
    if not prompts:
        msg = "user_prompts_for_generation must contain at least one prompt"
        raise ValueError(msg)

    samples: list[dict[str, str]] = []

    for _ in range(random_count):
        samples.append(
            {
                "text": _generate_random_sequence(rng, cfg.random_length_range),
                "type": "random",
            }
        )

    synthetic_samples_target = dataset_size - random_count
    while synthetic_samples_target > 0:
        prompt = rng.choice(prompts)
        batch = message_generator.generate_many(prompt, synthetic_batch_size)
        for text in batch:
            samples.append({"text": text.strip(), "type": "synthetic"})
            synthetic_samples_target -= 1
            if synthetic_samples_target == 0:
                break

    rng.shuffle(samples)
    return samples[:dataset_size]


def write_human_messages_dataset(
    samples: Iterable[dict[str, str]],
    output_file: str | Path,
) -> Path:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    return output_path


def _generate_random_sequence(rng: random.Random, length_range: list[int]) -> str:
    if not length_range or len(length_range) != 2:
        msg = "random_length_range must be a two-element list"
        raise ValueError(msg)
    min_len, max_len = (int(length_range[0]), int(length_range[1]))
    if min_len <= 0 or max_len < min_len:
        msg = "random_length_range must contain positive ascending values"
        raise ValueError(msg)
    target_len = rng.randint(min_len, max_len)
    return "".join(rng.choice(VALID_RANDOM_CHARACTERS) for _ in range(target_len))
