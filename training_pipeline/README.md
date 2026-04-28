# Training Pipeline

This is a local scaffold for a Swarm training pipeline on top of `main`.

It is intentionally non-functional. The goal is to preserve the shape and the
main ideas from `V4.0.0-training_pipeline` without changing the live benchmark,
validator, miner, or CLI code on `main`.

The pipeline is organized into 9 ordered stages:

1. `01_env_and_curriculum`
2. `02_privileged_expert`
3. `03_dataset_and_labels`
4. `04_student_model`
5. `05_behavior_cloning`
6. `06_dagger`
7. `07_rl_finetune`
8. `08_residual_landing`
9. `09_final_agents`

Rules for this scaffold:

- Treat it as a design/workspace area, not production runtime code.
- Keep the deploy-time observation contract aligned with `{"depth", "state"}`.
- Keep training-only signals and helpers out of the final exported agent.
- Use the real `swarm` packaging and benchmark loop to validate final exports.

See:

- [`TRAINING_ARCHITECTURE.md`](./TRAINING_ARCHITECTURE.md)
- [`stages/README.md`](./stages/README.md)
- [`artifacts/README.md`](./artifacts/README.md)
- [`final_agents/README.md`](./final_agents/README.md)
