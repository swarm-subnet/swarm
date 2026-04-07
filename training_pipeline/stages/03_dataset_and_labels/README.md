# 03 Dataset And Labels

Goal:

- Turn expert rollouts into reusable training data.

Build here:

- trajectory recorder
- dataset format
- perception label generator
- split logic for train/val/test
- teacher provenance
- weighted sampling manifests for later merges

Record at least:

- `obs`
- `action`
- `reward`
- `done`
- `info["privileged"]`
- recurrent sequence boundaries

Perception labels should include:

- platform visible / not visible
- relative platform direction
- image location or heatmap
- approximate distance bucket

Done when:

- one expert run produces both imitation data and perception supervision
- each episode records `teacher_id`, `teacher_version`, and `map_category`
- merged datasets are sampled by weighting policy, not just raw episode count

Outputs:

- one `.npz` per episode under `split/map_category/teacher_id/`
- one `.json` metadata file next to each episode
- `dataset_manifest.json` with structured episode rows and split/category counts
- `dataset_summary.json` for quick inspection
- `mode_vocabulary.json` for the shared expert mode mapping
- `train_sampling_manifest.json` / `val_sampling_manifest.json` / `test_sampling_manifest.json` when episodes exist for that split
