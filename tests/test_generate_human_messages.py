"""Tests for the human message generation script."""

import math
import string
from collections import Counter
from unittest.mock import MagicMock, patch

from omegaconf import OmegaConf

from slam_datagen.datasets.human_messages import build_human_messages_dataset
from slam_datagen.llm.message_generator import MessageGeneratorViaLlm


def test_build_human_messages_dataset_batches_synthetic_messages() -> None:
    """Ensure dataset batches synthetic generation while honoring random fraction."""

    cfg = OmegaConf.create(
        {
            "dataset_size": 12,
            "random_fraction": 0.25,
            "random_length_range": [5, 8],
            "synthetic_batch_size": 3,
            "human_message_prompts": {
                "system_prompt": "System prompt",
                "user_prompts_for_generation": [
                    "Write a casual WhatsApp style check-in.",
                    "Share a short Telegram update about weekend plans.",
                    "Send a quick status ping between close friends.",
                ],
            },
            "random_seed": 7,
        }
    )

    synthetic_batches = [
        [
            f"synthetic message {idx}"
            for idx in range(start, start + cfg.synthetic_batch_size)
        ]
        for start in range(0, 20, cfg.synthetic_batch_size)
    ]
    used_prompts: list[str] = []
    batches_returned = 0

    with patch("slam_datagen.llm.message_generator.Agent") as MockAgent:
        mock_agent = MagicMock()
        MockAgent.return_value = mock_agent

        def _run_sync(prompt: str, *_, **__) -> list[str]:
            nonlocal batches_returned
            used_prompts.append(prompt)
            batch = synthetic_batches[batches_returned]
            batches_returned += 1
            return batch

        mock_agent.run_sync.side_effect = _run_sync

        message_generator = MessageGeneratorViaLlm(
            model=MagicMock(),
            system_prompt=cfg.human_message_prompts.system_prompt,
        )

        samples = build_human_messages_dataset(
            cfg=cfg,
            prompt_cfg=cfg.human_message_prompts,
            message_generator=message_generator,
        )

    assert len(samples) == cfg.dataset_size
    assert all(sample.keys() == {"text", "type"} for sample in samples)

    random_samples = [sample for sample in samples if sample["type"] == "random"]
    synthetic_samples = [sample for sample in samples if sample["type"] == "synthetic"]

    expected_random_count = int(cfg.dataset_size * cfg.random_fraction)
    assert len(random_samples) == expected_random_count
    assert len(synthetic_samples) == cfg.dataset_size - expected_random_count

    valid_alphabet = set(string.ascii_letters + string.digits)
    min_len, max_len = cfg.random_length_range
    for sample in random_samples:
        text = sample["text"]
        assert min_len <= len(text) <= max_len
        assert set(text).issubset(valid_alphabet)

    observed_synthetic_texts = [sample["text"] for sample in synthetic_samples]
    flat_expected = [msg for batch in synthetic_batches for msg in batch]
    assert Counter(observed_synthetic_texts) == Counter(
        flat_expected[: len(synthetic_samples)]
    )

    expected_batches = math.ceil(len(synthetic_samples) / cfg.synthetic_batch_size)
    assert batches_returned == expected_batches

    base_prompts = set(cfg.human_message_prompts.user_prompts_for_generation)
    expected_instruction = (
        f"Produce {cfg.synthetic_batch_size} distinct short chat messages as a JSON array of strings."
        " Avoid commentary."
    )
    for prompt in used_prompts:
        if "\n\n" in prompt:
            base, instruction = prompt.split("\n\n", 1)
        else:
            base, instruction = prompt, ""
        assert base in base_prompts
        assert instruction == expected_instruction
