# Training Stages

Read these folders in numeric order.

1. `01_env_and_curriculum`
2. `02_privileged_expert`
3. `03_dataset_and_labels`
4. `04_student_model`
5. `05_behavior_cloning`
6. `06_dagger`
7. `07_rl_finetune`
8. `08_residual_landing`
9. `09_final_agents`

Simple rule:

- lower-numbered stages create prerequisites for higher-numbered stages
- stages 1-3 define tasks and data
- stages 4-7 define the main learning loop
- stages 8-9 refine and export the deployable controller
