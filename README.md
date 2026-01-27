# slam-datagen
Data generation library for LLMs and other foundational models

## Getting started

1. Create a virtual environment, e.g.
```bash
conda create -n myenv python=3.12
conda activate myenv
```
2. Install necessary packages
```bash
pip install -r requirements.txt
```
3. Set up `/config/user_settings/user_settings.yaml`
4. Run one of the scripts `/slam_datagen/scripts/XXX.py` and do not forget to modify the corresponding config file in `/config/config_XXX.yaml`
```bash
python slam_datagen/scripts/XXX.py
```

⚠️  DO NOT commit your `user_settings.yaml`

## Scripts

### `generate_merge_quality_dataset.py`

Generates the merge-quality dataset where every line is a persona with ground-truth attributes plus JSON/XML/Markdown chunks that mix the target record with distractors.

```bash
conda activate slam
python slam_datagen/scripts/generate_merge_quality_dataset.py
```

#### Configuration

1. In `config/user_settings/user_settings.yaml`, point Hydra to your workspace (do not commit secrets):
   ```yaml
   project_path: /abs/path/to/slam-datagen
   result_dir: ${user_settings.project_path}/outputs
   hydra_root: ${user_settings.project_path}/hydra
   hydra_dir: ${user_settings.hydra_root}/${now:%Y-%m-%d}/${now:%H-%M-%S}
   ```

2. Tune dataset behavior in `config/config_generate_merge_quality_dataset.yaml`:
   - `random_seed`: make generation reproducible
   - `dataset_size`: number of personas to emit
   - `chunk_formats`: subset of `json|xml|markdown`
   - `distractor_chunks_per_format`: distractor count per non-markdown format
   - `ground_truth_field_range`: `[min,max]` flattened attributes to keep per persona (drives sparsity)
   - `markdown_distractor_rows`: number of distractor rows each markdown chunk should contain
   - `markdown_chunks_per_person`: number of markdown chunks to emit per persona
   - `markdown_target_row_probability`: chance that a markdown chunk includes the target row
   - `output_file`: JSONL destination (defaults under Hydra run dir)
   - `preview_samples`: how many samples to summarize on stdout

#### Output

- Writes `${result_dir}/merge_quality_dataset.jsonl` (one JSON object per line with `ground_truth`, `provided_identifiers`, and chunk list)
- Prints a short preview with identifier and chunk counts so you can sanity-check the run immediately

### `generate_human_messages.py`

Synthesizes human-like chat snippets plus random alphanumeric strings. Uses configurable LLM prompts and writes `{text, type}` JSONL entries (type is `synthetic` or `random`).

```bash
conda activate slam
python slam_datagen/scripts/generate_human_messages.py
```

#### Configuration

1. Extend `config/user_settings/user_settings.yaml` with `project_path`, `result_dir`, and Hydra paths as shown above.
2. Adjust `config/config_generate_human_messages.yaml`:
   - `human_message_prompts`: selects a prompt pack from `config/human_message_prompts/*.yaml` (defaults to `en`; switch via `python ... human_message_prompts=ru`).
   - `dataset_size`: total number of messages to produce.
   - `random_fraction`: share of entries replaced with random alphanumeric sequences.
   - `random_length_range`: `[min,max]` length of random sequences.
   - `synthetic_batch_size`: number of chat snippets requested per LLM call (messages are still emitted individually in the final dataset, but batching improves diversity and throughput).
   - `random_seed`: keeps both LLM prompt selection and random strings reproducible.
   - `output_file`: destination JSONL (defaults under Hydra dir).
   - `preview_samples`: number of samples printed to stdout after generation.

   Each prompt pack contains a `system_prompt` and `user_prompts_for_generation`. To add a new language, drop another YAML file into `config/human_message_prompts/` and reference it via `human_message_prompts=<name>`.

#### Output

- Writes `${result_dir}/human_messages_dataset.jsonl`, each line `{"text": ..., "type": "synthetic"|"random"}`.
- Prints previews so you can verify both random strings and LLM outputs.
