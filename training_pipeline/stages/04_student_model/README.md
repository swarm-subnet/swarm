# 04 Student Model

Goal:

- Build the deployable policy architecture.
- Freeze the student contract before any imitation or RL training.

Build here:

- depth encoder
- state MLP
- recurrent fusion block
- action head
- optional auxiliary perception heads

Recommended first version:

- CNN over depth
- MLP over the 141-d state
- GRU over fused features

Done when:

- the model can consume only deployable observations
- the model can optionally emit auxiliary perception predictions during training
- the model config is saved
- an initialized checkpoint exists for later stages
- a runtime preview export passes a smoke test

Current outputs:

- `student_model_config.json`
- `student_init.pt`
- `student_runtime_preview.pt`
- `student_model_summary.json`
- `student_smoke_test.json`
