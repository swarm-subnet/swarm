# Training Pipeline Architecture

This document is the high-level plan.
The filesystem walkthrough lives in [`stages`](./stages/README.md).

## 1. Non-Negotiable Constraints

Keep these fixed throughout the project:

1. The deployed actor must only consume the real validator observation
   contract: `{"depth", "state"}`.
2. The training pipeline is pinned to validator `SIM_DT = 1/50`.
3. The deployable low-dimensional state contract is fixed at `141` floats.
   Model input dimensionality must not drift with `sim_dt`.
4. Privileged simulator truth is allowed only during training.
5. The student policy must be deployable without `info["privileged"]`.
6. RL does not start until the expert, dataset, and imitation pipeline exist.
7. `final_agents` contains only deployable agents.
8. The resultant codebase will be open-sourced, so there must be no obfuscated
   code, hidden logic, intentionally confusing abstractions, or opaque binary
   blobs used to hide core behavior.
9. Training and deployment code must stay readable and reviewable:
   names should be descriptive, module boundaries should be explicit, and each
   stage should make its inputs and outputs clear.
10. If a component is essential to reproducing results, it must be implemented
   or documented in-repo. Critical behavior should not depend on private manual
   steps or undocumented external conventions.
11. Final solutions should prefer clarity over cleverness. A slightly less
   compact implementation is acceptable if it is substantially easier to audit,
   explain, and extend.
12. The final submission format is restrictive: the packaged model is centered
    on top-level `drone_agent.py` plus allowed model artifacts and
    `requirements.txt`. Do not assume arbitrary training helper modules will be
    available in the final packaged submission.

Relevant code:

- [`swarm/core/moving_drone.py`](../swarm/core/moving_drone.py)
- [`swarm/utils/env_factory.py`](../swarm/utils/env_factory.py)
- [`swarm/validator/task_gen.py`](../swarm/validator/task_gen.py)
- [`training_env.py`](./training_env.py)
- [`final_agents/drone_agent.py`](./final_agents/drone_agent.py)

## 2. Ordered Build Plan

This is the correct order to build the training stack.

### Step 1: Environment And Curriculum

Build the training environment first.
Do not start with models.

You need:

- deterministic task generation by type
- static vs moving control
- train/validation/test splits
- held-out evaluation by map family

The actor still sees only `{"depth", "state"}`.
Training code may also inspect `info["privileged"]`.

Filesystem guide:

- [`01_env_and_curriculum`](./stages/01_env_and_curriculum/README.md)

### Step 2: Privileged Expert

Build a "godmode" expert that can see privileged information.

The current design is:

- one shared expert implementation
- one shared mode vocabulary
- one registry that assigns `map_category -> teacher_id`
- per-category config overrides and quality gates
- per-category reporting, because expert quality is uneven by map category

This keeps the action contract shared while allowing categories such as
`open_dynamic` or `mountain_static` to evolve independently.

Filesystem guide:

- [`02_privileged_expert`](./stages/02_privileged_expert/README.md)

### Step 3: Dataset And Labels

Once the expert exists, build the collector.

Every rollout should be able to produce:

- imitation data
- privileged teacher-state data
- perception labels

Record at least:

- deployable observation
- action
- reward
- termination flags
- privileged teacher state
- recurrent sequence boundaries

Current dataset policy:

- save one `.npz` per episode
- record `teacher_id`, `teacher_version`, and `map_category`
- export one shared `mode_vocabulary.json`
- build weighted sampling manifests for later BC and DAgger stages
- prefer success-only datasets for imitation training, with failures kept
  separately for debugging if needed

Filesystem guide:

- [`03_dataset_and_labels`](./stages/03_dataset_and_labels/README.md)

### Step 4: Student Model

Only now build the deployable student.

Recommended first version:

- depth encoder
- state MLP
- GRU fusion
- action head
- auxiliary perception heads

The student must remain deployable on the real observation contract.
This stage creates the model config, validates the contract, saves an initial
checkpoint, and exports a runtime preview.

Filesystem guide:

- [`04_student_model`](./stages/04_student_model/README.md)

### Step 5: Behavior Cloning

Use the expert dataset to pretrain the student.

Losses:

- action imitation loss
- auxiliary perception losses

This stage should already produce a student that can solve easy maps and expose
where pure imitation breaks.

Current implementation notes:

- recurrent sequence batching over Phase-3 episode files
- optional weighted dataset manifests from Phase 3
- live progress logging during training
- early stopping based on validation total loss
- train and validation both log total loss and action-only loss

Filesystem guide:

- [`05_behavior_cloning`](./stages/05_behavior_cloning/README.md)

### Step 6: DAgger

Run the student in the environment, ask the expert what it should have done on
those student-visited states, merge the new labels into the dataset, retrain,
and repeat.

This is the first major robustness stage.

The merge policy should stay weighted by dataset/category policy rather than
blind concatenation.

Filesystem guide:

- [`06_dagger`](./stages/06_dagger/README.md)

### Step 7: RL Fine-Tuning

Only after imitation and DAgger are working should you use RL.

Recommended default:

- recurrent PPO
- actor sees deployable observations only
- critic sees privileged labels too

This is the asymmetric-critic stage.

Filesystem guide:

- [`07_rl_finetune`](./stages/07_rl_finetune/README.md)

### Step 8: Residual Landing / Intercept

Residual RL is not the starting point.
Use it only when you already have a competent base policy.

Good targets for the residual:

- moving-platform intercept timing
- final static touchdown
- tight near-goal corrections

Filesystem guide:

- [`08_residual_landing`](./stages/08_residual_landing/README.md)

### Step 9: Final Agents

Promote only deployable artifacts into `final_agents`.

Anything that depends on privileged information stays outside.

Filesystem guide:

- [`09_final_agents`](./stages/09_final_agents/README.md)
- [`final_agents`](./final_agents/README.md)

## 3. Recommended Model Roles

### Student

The student is the policy you eventually deploy.

Input:

- depth image `(128, 128, 1)`
- low-dimensional state `(141,)`

Recommended architecture:

- depth CNN
- state MLP
- GRU
- action head
- optional auxiliary perception heads

### Teacher

The teacher is training-only.

It can use:

- true goal position
- current platform position
- platform velocity
- adjusted task start/goal
- challenge type
- moving/static flag

That information should come from [`training_env.py`](./training_env.py).

### Critic

The critic is training-only.

In the recommended setup:

- actor uses deployable observations only
- critic uses deployable observations plus privileged labels

## 4. Why This Order Matters

This order prevents the common mistakes:

- starting RL before you have a stable expert
- training on mixed random tasks before you have a curriculum
- leaking privileged labels into the deployed actor
- trying to learn platform detection only through sparse policy improvement
- using residual RL before there is a strong base policy to correct

## 5. Deployment Layout

`training_pipeline` now has two roles:

- training architecture and scaffolding
- deployable agents under `final_agents`

Top-level [`drone_agent.py`](./drone_agent.py) is now only a compatibility
entrypoint that imports from `final_agents` inside this repo workspace.

For real packaged submission, the deployable logic must still resolve from the
top-level `drone_agent.py` and any allowed model artifacts. That means the
final learned export should be either:

- a self-contained top-level `drone_agent.py`, or
- a top-level `drone_agent.py` that loads an allowed runtime artifact such as a
  scripted model or ONNX file.

## 6. Current Status

Implemented:

- [`training_env.py`](./training_env.py)
- cleaned hardcoded baseline in [`final_agents/drone_agent.py`](./final_agents/drone_agent.py)
- ordered build tree in [`stages`](./stages/README.md)
- specialist expert registry and per-category overrides
- Phase-3 dataset builder with weighted manifests and teacher provenance
- Phase-4 student model builder and runtime preview export
- Phase-5 behavior cloning with progress logs and early stopping

Partially implemented / still iterative:

- DAgger loop
- PPO fine-tuning
- residual landing refinement

## References

- DAgger:
  - https://proceedings.mlr.press/v15/ross11a.html
- Asymmetric actor-critic:
  - https://openai.com/index/asymmetric-actor-critic-for-image-based-robot-learning/
- PPO:
  - https://arxiv.org/abs/1707.06347
- Residual RL:
  - https://tore.tuhh.de/entities/publication/0f20de6e-0305-4e82-9cde-345323da5fd0
- Residual RL from demonstrations:
  - https://arxiv.org/abs/2106.08050
