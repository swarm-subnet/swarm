# 03 Dataset And Labels

Goal:

- Turn expert rollouts into reusable training data.
- Keep the dataset aligned with the specialist-expert registry and the fixed
  deployable observation contract.

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
- optional `rejected_episodes.json` when success-only export drops failed expert episodes

Current schema:

- `depth`: `(T, 128, 128, 1)`
- `state`: `(T, 141)`
- `action`: `(T, 5)`
- `teacher_state`: `(T, 15)`
- temporal labels such as `reward`, `terminated`, `truncated`, `mode_id`
- perception labels such as `visible`, `pixel_row`, `pixel_col`, `pixel_norm`, `distance_bucket`

Important collector options:

- `--expert-registry` to select experts by `map_category`
- `--only-success` to keep the BC dataset clean
- `--dataset-weighting-policy` to build weighted manifests for later stages
- `--stage-name`, `--map-seed`, and `--max-episodes-total` for smoke tests or targeted data refreshes
