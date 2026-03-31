# 01 Env And Curriculum

Goal:

- Create the training environment and the curriculum scheduler.
- Freeze deterministic train/val/test tasks that every later stage will reuse.
- Keep the default-model contract pinned to validator `SIM_DT = 1/50` and
  `state.shape == (141,)`.

Build here:

- wrappers around [`../../training_env.py`](../../training_env.py)
- fixed stage definitions
- seed split logic
- evaluation helpers for held-out seeds
- manifest generation and validation
- environment smoke tests for generated tasks

Outputs:

- `make_training_env(...)`
- stage sampler
- train/val/test seed lists
- `curriculum_manifest.json`
- `curriculum_summary.json`
- `smoke_test_results.json`

Done when:

- you can generate deterministic tasks by type
- you can switch between static and moving objectives
- you can run held-out evaluations without touching later folders
- the manifest is regenerated identically on repeated runs
- every manifest task can be re-derived from the repo task generator
- the smoke-tested tasks boot in the simulator and return the expected training info
- any non-validator `sim_dt` is rejected by the default-model pipeline
