# Training Pipeline

This folder contains the staged training workflow for the Swarm baseline.

The pipeline has 9 ordered stages:

1. build deterministic curriculum tasks
2. build and evaluate privileged experts
3. collect rollout datasets and labels
4. build the deployable student model
5. behavior-clone the student
6. run DAgger relabeling and retraining
7. RL fine-tune the student
8. train residual last-meter corrections
9. export deployable final agents

The goal of this layout is to keep training code readable and auditable. The
ordered scripts live under [`stages`](./stages/README.md), shared code lives in
[`training_lib`](./training_lib/__init__.py), generated outputs live under
[`artifacts`](./artifacts/README.md), and deployable agents live in
[`final_agents`](./final_agents/README.md).

Current hard constraints:

- deployable actor input is exactly `{"depth", "state"}`
- `state.shape == (141,)`
- validator-aligned `SIM_DT = 1/50`
- privileged simulator truth is training-only
- moving-target success means contact, not stable landing

## Layout

- [`TRAINING_ARCHITECTURE.md`](./TRAINING_ARCHITECTURE.md): high-level design and constraints
- [`stages`](./stages/README.md): numbered build steps in execution order
- [`training_env.py`](./training_env.py): validator-aligned training wrapper
- [`training_lib`](./training_lib/__init__.py): reusable training helpers
- [`stages/02_privileged_expert/experts`](./stages/02_privileged_expert/experts/README.md): specialist expert registry, shared base config, and per-category overrides
- [`final_agents`](./final_agents/README.md): deployable agents and exports
- [`artifacts/README.md`](./artifacts/README.md): generated outputs, intentionally not part of the publish surface

## Current Flow

- Phase 1 creates deterministic `train/val/test` task manifests by explicit map category such as `open_static` or `city_dynamic`.
- Phase 2 evaluates either one shared expert or category-specialist experts selected from the registry.
- Phase 3 saves one `.npz` per rollout episode plus manifests, summaries, mode vocabulary, and weighted sampling manifests for later training.
- Phase 4 builds the student architecture and saves an initialized checkpoint plus a runtime preview.
- Phase 5 trains behavior cloning with recurrent sequence batching, live progress logging, and patience-based early stopping.

## Typical Commands

```bash
validator_env/bin/python training_pipeline/stages/01_env_and_curriculum/prepare_env_and_curriculum.py
validator_env/bin/python training_pipeline/stages/02_privileged_expert/build_privileged_expert.py
validator_env/bin/python training_pipeline/stages/03_dataset_and_labels/collect_dataset_and_labels.py
validator_env/bin/python training_pipeline/stages/04_student_model/build_student_model.py
validator_env/bin/python training_pipeline/stages/05_behavior_cloning/train_behavior_cloning.py
```

For specialist Phase-2 evaluation against a manifest:

```bash
validator_env/bin/python training_pipeline/stages/02_privileged_expert/build_privileged_expert.py \
  --curriculum-manifest training_pipeline/artifacts/01_env_and_curriculum/curriculum_manifest.json \
  --expert-registry training_pipeline/stages/02_privileged_expert/experts/registry.json \
  --output-dir training_pipeline/artifacts/02_privileged_expert_eval
```

For validator-like randomized Phase-2 evaluation:

```bash
validator_env/bin/python training_pipeline/stages/02_privileged_expert/run_full_expert_eval.py \
  --num-runs 20 \
  --num-workers 3
```

Generated run outputs stay under `training_pipeline/artifacts/`. That tree is
ignored for publication by default.
