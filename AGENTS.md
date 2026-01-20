# AGENTS.md

## Setup commands

- Activate the environment: `conda activate slam`
- Run the main script: `python slam_datagen/scripts/main.py`
- Run tests: `pytest`

## Main script configuration

- The main scripts is configured via hydra in `config/config_main.yaml`
- In general, all the user-specific fields (API tokens, paths etc.) should be stored in `config/user_settings/user_settings.yaml`. Do not commit `user_settings.yaml` without explicit permission

## Developer guide

- Before making any changes, create a local git branch named `YYYYMMDD_short_task_description` where `YYYYMMDD` stands for the current date. Checkout this local branch and make all the changes there. Do not forget to make occasional commits during your work to be able to roll back to previous versions of the code if necessary. NEVER commit `config/user_settings/user_settings.yaml`, keep its changes unstaged.
- This code is configured using hydra. Its configs can be found in `config/`. No matter what constants/literals are used, they should be taken from configs. Object construction can also be made via `hydra.utils.instantiate` but use it only with simple classes (i.e., not derived from some base class).
- Run the following linters before finishing the job:
  - black to ensure good formatting (note that it changes the code): `black slam_datagen/`
  - isort to ensure good import sorting (note that it changes the code): `isort slam_datagen/`
  - pylint to ensure typing: `pylint slam_datagen/`
  - mypy to ensure typing: `mypy slam_datagen/`
- Use type hints, their use is necessiated by linters
- Update `README.md` when new functionality is added or there is outdated information
