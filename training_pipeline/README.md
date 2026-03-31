# Training Pipeline

This folder contains the staged training workflow for the Swarm baseline:

1. build deterministic curriculum tasks
2. evaluate the privileged expert
3. collect datasets and labels
4. build and train the student
5. export deployable agents

The goal of this layout is to keep training code readable and auditable. The
ordered scripts live under [`stages`](./stages/README.md), shared code lives in
[`training_lib`](./training_lib/__init__.py), and deployable agents live in
[`final_agents`](./final_agents/README.md).

## Layout

- [`TRAINING_ARCHITECTURE.md`](./TRAINING_ARCHITECTURE.md): high-level design and constraints
- [`stages`](./stages/README.md): numbered build steps in execution order
- [`training_env.py`](./training_env.py): validator-aligned training wrapper
- [`training_lib`](./training_lib/__init__.py): reusable training helpers
- [`final_agents`](./final_agents/README.md): deployable agents and exports
- [`artifacts/README.md`](./artifacts/README.md): generated outputs, intentionally not part of the publish surface

## Typical Commands

```bash
python3 training_pipeline/stages/01_env_and_curriculum/prepare_env_and_curriculum.py
python3 training_pipeline/stages/02_privileged_expert/build_privileged_expert.py
python3 training_pipeline/stages/03_dataset_and_labels/collect_dataset_and_labels.py
```

For validator-like stage-02 evaluation:

```bash
python3 training_pipeline/stages/02_privileged_expert/run_full_expert_eval.py \
  --num-runs 20 \
  --num-workers 3
```

Generated run outputs stay under `training_pipeline/artifacts/`, but that tree
is ignored for publication by default.
