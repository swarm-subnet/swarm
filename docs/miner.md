<a id="miner-top"></a>

# Swarm Miner Guide

Train an autonomous drone pilot, benchmark it against 1,100 procedurally generated worlds, and compete on the [leaderboard](https://swarm124.com/benchmark).

## What You Submit

A submission is a **model graph**: `model_graph.zip`, holding a `manifest.json` plus up to 16 ONNX networks under `models/`. The manifest wires the networks into one policy — which observation tensors feed which model, and which output is the action. Validators run it with the subnet-owned runner inside a locked container at 50 Hz.

You ship no code. No classes, no server, no dependency list — pure tensor graphs. Anything the manifest and the ONNX files cannot express does not run.

```
model_graph.zip
├── manifest.json        # the wiring: models, inputs, action
└── models/
    └── policy.onnx      # one or more networks, 50 MiB zip cap total
```

## System Requirements

Mining is extremely lightweight: your miner commits its submission to the chain (a GitHub URL for public families, or a hash plus a one-time artifact upload for the private families) and goes offline. Any machine with **Python 3.11+** and a network connection will do. Training hardware depends entirely on your approach — train with whatever you like, ship the result as ONNX.

## Installation

```bash
git clone https://github.com/swarm-subnet/swarm
cd swarm

chmod +x scripts/miner/install_dependencies.sh
./scripts/miner/install_dependencies.sh

chmod +x scripts/miner/setup.sh
./scripts/miner/setup.sh

source miner_env/bin/activate
pip install -e .
```

## Pick a Family

Five challenge families, each its own competition with its own champion and emissions. One hotkey competes in exactly one family.

| Family | ID | Track |
|--------|----|-------|
| [Autopilot](families/autopilot.md) | `cf_autopilot` | public |
| [Interceptor](families/interceptor.md) | `cf_interceptor` | public |
| [Swarm Autopilot](families/swarm_autopilot.md) | `cf_swarm_autopilot` | public |
| [Search and Rescue](families/search_and_rescue.md) | `cf_search_and_rescue` | private |
| [Swarm SAR](families/swarm_sar.md) | `cf_swarm_sar` | private |

Each family guide defines the exact contract: observation tensors (`obs.depth`, `obs.state`, and `obs.rgb` where the family has it), action shape and bounds, maps, episode rules, and scoring.

## Quick Start: Train a Baseline

The fastest path to a working artifact is the training starter for your family:

```bash
python3 RL/cf_autopilot/train.py --timesteps 500000
```

The starter trains a PPO policy in the real simulator, exports it to ONNX, writes the manifest, and packages everything into a validated `model_graph.zip`. It prints the artifact path when done. Swarm families take `--drones` to fix the training drone count.

Score it the way validators will — real Docker sandbox, real seeds:

```bash
python3 RL/test_RL.py --model RL/cf_autopilot/out/model_graph.zip \
  --family_id cf_autopilot --num-seeds 10
```

The baseline is a starting point, not a contender. Better architectures, better training, and better reward shaping are where you win.

## The Manifest

`manifest.json` declares the graph. The starter writes this for you; hand-build it when you go beyond a single network:

```json
{
  "model_graph_version": "model_graph.v1",
  "family_id": "cf_autopilot",
  "execution_profile": "swarm.onnx-neural.cpu.v1",
  "runner_abi": "graph_runner.v1",
  "models": [
    {"id": "policy", "file": "models/policy.onnx", "sha256": "<hash of the file>"}
  ],
  "nodes": [
    {"id": "policy", "model": "policy",
     "inputs": {"depth": "obs.depth", "state": "obs.state"}}
  ],
  "action": "policy.action"
}
```

- **models** — every ONNX file, pinned by SHA-256.
- **nodes** — execution steps. Inputs bind each model input to an observation (`obs.depth`), a memory slot (`memory.<name>`), or another node's output (`<node>.<output>`). A node can run every step or every N steps (`every_n_steps`, 1–256).
- **memory** — optional named slots for recurrent state: a node writes a tensor this step, another (or the same) reads it next step.
- **action** — exactly one node output, matching the family's action shape.

The full schema lives in `swarm/model_graph/model_graph.schema.json`.

## The Execution Profile

Every graph must pass admission against `swarm.onnx-neural.cpu.v1`:

| Rule | Value |
|------|-------|
| Format | ONNX opset 18, IR 10 |
| Dtype | float32 tensors only |
| Operators | allowlist of standard neural ops (Conv, Gemm, MatMul, activations, pooling, norms, RNN/LSTM/GRU, elementwise math) |
| Structural ops | Reshape, Slice, Squeeze, Concat and friends, with compile-time-constant shapes |
| Denied | control flow (If/Loop), comparisons, Where, ArgMax, Shape, Cast |
| Size | 50 MiB zip, 16 models max |

The exact lists live in `swarm/model_graph/execution_profile.v1.json`. The intent: your model is a neural network, not a program. Preprocessing like downsampling belongs inside the graph — the starters show the pattern with a constant-stride Slice.

## Package and Verify

Built your own source folder? Package and check it with the CLI:

```bash
swarm model package --source ./my_model --family-id cf_autopilot \
  --output model_graph.zip --overwrite
swarm model verify --model model_graph.zip
```

`package` zips `manifest.json` + `models/*.onnx` and runs the same admission validators run; invalid graphs are rejected on the spot with an `MG_` reason code. See the [CLI reference](CLI_readme.md) for every command and flag.

## Submit — Public Track

Public families live in a GitHub repo the backend can read:

```bash
swarm repo package --repo-root ./my_entry \
  --family-source cf_autopilot=./my_model
swarm repo verify --repo-root ./my_entry
```

This writes three things: `artifacts/cf_autopilot/model_graph.zip`, `submission_manifest.json` (family, artifact path, SHA-256, interface `model_graph.v1`), and `README.md` — a byte-exact copy of the benchmark template. **Do not edit the README.** Its hash is checked by the backend; any change means silent rejection. `swarm repo verify` catches this before you commit.

Push the repo to a public GitHub, then commit it on-chain:

```bash
python3 neurons/miner.py --github_url https://github.com/<you>/<repo> \
  --wallet.name <coldkey> --wallet.hotkey <hotkey>
```

## Submit — Private Track

Search and Rescue and Swarm SAR keep your weights off GitHub: the SHA-256 goes on-chain, the bytes go to the backend vault, and only trusted validators fetch them.

```bash
python3 neurons/miner.py --family_id cf_search_and_rescue \
  --artifact ./model_graph.zip \
  --backend_url https://api.swarm124.com \
  --wallet.name <coldkey> --wallet.hotkey <hotkey>
```

If the chain commit succeeded but the upload failed, rerun with `--upload_only` to (re)upload the same artifact.

## Rules of the Game

- **One hotkey, one submission.** A hotkey gets a single submission slot in a single family; the slot only frees up if the model expires at an epoch or version rollover. Competing again — after a failure or with a new model — means registering a fresh hotkey.
- **One family per repo.** A submission repo holds exactly one family's artifact.
- **Full benchmark.** Your model runs the family's complete 1,100-seed set. Validators evaluate seed batches; the backend recomputes the authoritative score from the seed records.
- **Champions defend weekly.** Seeds rotate every epoch and champions are re-evaluated on the fresh set. See [King of the Hill](king_of_the_hill.md) for emissions.

## Troubleshooting

Failures charged to your artifact carry an `MG_` reason code, visible on the website's model page:

| Code | Meaning | Fix |
|------|---------|-----|
| `MG_MANIFEST_MISSING` / `MG_MANIFEST_INVALID` | zip has no valid `manifest.json` | package with `swarm model package`, not by hand-zipping |
| `MG_PROFILE_DENIED_OP` | a model uses an op outside the allowlist | export simpler ops; check the profile JSON |
| `MG_SHAPE_CONTRACT_MISMATCH` | action or observation shapes don't match the family contract | compare against the family guide's tables |
| `MG_MODEL_FILE_INVALID` | an ONNX file is corrupt or malformed | re-export; verify the manifest SHA-256 matches the file |
| `MG_LOAD_FAILED` | the graph didn't load and serve within the startup window | shrink the model; test with `RL/test_RL.py` first |
| `MG_STEP_HARD_TIMEOUT` | an action step blew the per-step compute budget | reduce per-step FLOPs; the budget is baseline-equivalent, so slow hosts don't penalize you |

`swarm model verify` and `RL/test_RL.py` reproduce almost every rejection locally before you spend your weekly shot.

**Environment issues**: run `swarm doctor` to diagnose.

## Support

- **Discord**: [discord.gg/8dPqPDw7GC](https://discord.gg/8dPqPDw7GC) (ping @Miguelikk or @AliSaaf)
- **GitHub Issues**: open a ticket with logs & error trace
- **Website**: [swarm124.com](https://swarm124.com)
