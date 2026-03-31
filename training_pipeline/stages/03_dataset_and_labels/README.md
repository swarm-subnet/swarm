# 03 Dataset And Labels

Goal:

- Turn expert rollouts into reusable training data.

Build here:

- trajectory recorder
- dataset format
- perception label generator
- split logic for train/val/test

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
