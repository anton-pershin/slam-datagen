# slam-datagen
Data generationa library for LLMs and other foundational models

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
4. Run one of the scripts `/slam_datagen/scripts/XXX.py` and do not forget to modify the corresponding config file in `/config/config_XXX.yaml'
```bash
python slam_datagen/scripts/XXX.py
```

⚠️  DO NOT commit your `user_settings.yaml`

## Scripts

### `main.py`

Generates the merge-quality dataset where every line is a persona with ground-truth attributes plus JSON/XML/Markdown chunks that mix the target record with distractors.

```bash
conda activate slam
python slam_datagen/scripts/main.py
```

#### Configuration

1. In `config/user_settings/user_settings.yaml`, point Hydra to your workspace (do not commit secrets):
   ```yaml
   project_path: /abs/path/to/slam-datagen
   result_dir: ${user_settings.project_path}/outputs
   hydra_root: ${user_settings.project_path}/hydra
   hydra_dir: ${user_settings.hydra_root}/${now:%Y-%m-%d}/${now:%H-%M-%S}
   ```

2. Tune dataset behavior in `config/config_main.yaml`:
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
