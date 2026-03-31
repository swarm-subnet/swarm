# Training Stages

This tree is the ordered build plan for the training pipeline.

Read and implement from low number to high number:

1. `01_env_and_curriculum`
2. `02_privileged_expert`
3. `03_dataset_and_labels`
4. `04_student_model`
5. `05_behavior_cloning`
6. `06_dagger`
7. `07_rl_finetune`
8. `08_residual_landing`
9. `09_final_agents`

The rule is simple:

- Lower folders create prerequisites for higher folders.
- Nothing in `final_agents` should depend on privileged labels.
- Promotion goes from training artifacts to deployable agent, never the other
  way around.
