# Training Pipeline Architecture

This file captures the high-level order and the boundaries for a Swarm training
stack derived from the old V4 pipeline design.

## Non-Negotiable Constraints

- The final submitted policy must still be packaged as a top-level
  `drone_agent.py` plus allowed artifacts.
- The actor must consume only deploy-time observations: `{"depth", "state"}`.
- The training environment should stay aligned with the real benchmark timestep,
  action format, and success conditions.
- Privileged labels are allowed in training, but not in the final exported
  controller.
- RL should refine a working policy, not replace basic data and imitation work.

## Ordered Build Plan

1. Environment and curriculum
2. Privileged expert
3. Dataset and labels
4. Student model
5. Behavior cloning
6. DAgger
7. RL fine-tuning
8. Residual landing/intercept refinement
9. Final agent export

## Why This Order Matters

- Early stages define tasks and data.
- Middle stages define the main policy learning loop.
- Final stages export a small deployable controller instead of a training
  workspace.

## What This Scaffold Does

- Documents the intended stages.
- Provides placeholder entry scripts for each stage.
- Leaves the real `main` codepath unchanged.

## What This Scaffold Does Not Do

- It does not run end-to-end.
- It does not modify `swarm`, `neurons`, `tests`, or CLI behavior.
- It does not claim that the V4 pipeline has been merged into `main`.
