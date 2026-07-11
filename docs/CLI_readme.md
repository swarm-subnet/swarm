# Swarm CLI

Command-line interface for benchmarking, packaging, and verifying Swarm drone-navigation submissions.

A submission is a **model graph**: a pure ONNX tensor graph (plus a `manifest.json`) that the subnet-owned runner loads and steps. Miners ship no code — no policy classes, no server, no requirements file. The CLI's job is to bundle your graph into the exact artifact the validator admits, and to check it locally before you publish.

---

## Install

```bash
pip install -e .
```

Or install from PyPI (the published release may lag this repo):

```bash
pip install swarm-sotapilot
```

Then run `swarm <command>` directly, or without installing:

```bash
python3 -m swarm <command>
```

---

## Challenge families

Family-aware commands (`swarm model package`, `swarm model test`, `swarm repo package`) take a `--family-id` from this set. Each family runs its own set of procedurally generated environment types:

| Family ID | Name | Environment types |
| --- | --- | --- |
| `cf_autopilot` | Autopilot / Navigation | City, Open, Mountain, Village, Warehouse, Forest |
| `cf_search_and_rescue` | Search and Rescue | City, Open, Mountain, Village, Warehouse, Forest |
| `cf_swarm_autopilot` | Swarm Autopilot | City, Open, Mountain, Village, Forest |
| `cf_swarm_sar` | Swarm Search and Rescue | City, Open, Mountain, Village, Forest |
| `cf_interceptor` | Interceptor | Open |

One hotkey competes in exactly one family, and a submission repo holds exactly one family.

---

## Where to start

The training starters under `RL/<family_id>/train.py` produce a packaged `model_graph.zip` directly, so you can go from a trained policy to an admittable artifact without hand-assembling the source folder. `RL/test_RL.py` runs validator-faithful local scoring (Docker + Cap'n Proto RPC) over a chosen number of seeds, so you can measure a model the way the subnet will before you submit.

---

## Source folder layout

`swarm model package` and `swarm model test` read a **source directory** that must contain:

```
<source>/
  manifest.json        # graph manifest (required)
  models/
    *.onnx             # one or more ONNX tensor graphs
```

`package` zips `manifest.json` and every `models/*.onnx` file into the output ZIP and nothing else.

---

## Commands

### `swarm doctor`

Checks that the local environment is ready to benchmark. Prints one line per check and exits `1` if any **required** check fails, otherwise `0`.

```bash
swarm doctor
```

Checks: Python 3.11+, the Docker binary and a reachable Docker daemon, the sandbox-lockdown binaries `nsenter` and `iptables` and whether the current user can apply network lockdown (root, or the right file capabilities), the Python modules `capnp`, `pybullet`, and `gym_pybullet_drones`, writable runtime and model directories, the submission-template files, and the benchmark-engine module. The sandbox-permission and binary-capability checks are marked optional; the rest are required.

### `swarm benchmark`

Runs a local benchmark against the default family's engine, evaluating a submission ZIP across its environment groups. There is no family flag; the command always drives the engine's default family. Exits `0` on success, `1` on any failure or interrupt.

```bash
# Default benchmark (3 seeds per environment group)
swarm benchmark --model Submission/model_graph.zip --workers 4

# Quick check (1 seed per environment group)
swarm benchmark --model Submission/model_graph.zip --seeds-per-group 1

# With options
swarm benchmark --model Submission/model_graph.zip --relax-timeouts --rpc-verbosity low
```

If `--model` is omitted, the CLI downloads the current champion (with SHA-256 integrity verification) and benchmarks it. If no champion is released, it exits `1`.

| Flag | Default | Purpose |
| --- | --- | --- |
| `--model <path>` | download champion | Submission ZIP to benchmark. |
| `--uid <int>` | infer from model name | Miner UID passed to the engine. |
| `--seeds-per-group <int>` | `3` | Seeds per environment group. |
| `--workers <int>` | fits available vCPUs, capped at 12 | Parallel Docker workers. |
| `--log-out <path>` | engine default (`/tmp/bench_full_eval.log`) | Benchmark log output path. |
| `--seed-file <path>` | discover seeds | Replay an exact seed set from JSON. |
| `--save-seed-file <path>` | — | Write the resolved seeds to JSON for later replay. |
| `--seed-search-rng <int>` | — | Seed for reproducible seed discovery. |
| `--summary-json-out <path>` | — | Write the benchmark summary as JSON. |
| `--relax-timeouts` | off | Slow-machine timeout overrides. |
| `--rpc-verbosity low\|mid\|high` | `mid` | RPC tracing verbosity. |

### `swarm model package`

Bundles a source folder into a submission ZIP, then runs graph admission on it. The source must contain `manifest.json` and `models/*.onnx` (see [Source folder layout](#source-folder-layout)); only those files go into the ZIP. After zipping, the command runs `admit_artifact`: an invalid graph (bad manifest, unsupported version, disallowed op, shape or output-contract violation, oversize archive, and so on) is **rejected** with an `MG_` reason code and the partial ZIP is deleted. It also checks the admitted graph's family matches `--family-id`.

Omit `--family-id` in a terminal and the command prints a menu of families and asks which one you trained for, so you never package the wrong one. Pass `--family-id` to skip the prompt; it is **required** for non-interactive runs (CI, piped input), which otherwise exit `1`.

```bash
# Interactive: pick the family from a menu
swarm model package --source ./my_model

# Explicit family (needed in scripts)
swarm model package --source ./my_model --family-id cf_search_and_rescue

# Custom output path, overwriting an existing zip
swarm model package --source ./my_model --family-id cf_autopilot \
  --output Submission/model_graph.zip --overwrite
```

| Flag | Default | Purpose |
| --- | --- | --- |
| `--source <dir>` | required | Source directory (`manifest.json` + `models/*.onnx`). |
| `--output <path>` | `Submission/model_graph.zip` | Output ZIP path. |
| `--overwrite` | off | Replace the output ZIP if it already exists. |
| `--family-id <id>` | prompt / required | Challenge family the artifact targets. |
| `--interface-version <ver>` | first supported for the family | Explicit policy interface version. |

On success it prints the output path, the number of files packed, the resolved family, and the interface version.

### `swarm model verify`

Validates an existing submission ZIP without repackaging. It runs graph admission (`admit_artifact`), then the policy-contract check, then a runtime smoke test that loads the graph and steps it. Prints the size, admission status and reason, the contract result and the graph's family/interface, and the smoke result. Exits `0` only if all three pass, otherwise `1`.

```bash
swarm model verify --model Submission/model_graph.zip
```

| Flag | Default | Purpose |
| --- | --- | --- |
| `--model <path>` | required | Submission ZIP to verify. |

### `swarm model test`

Checks a source folder's formatting and packaging readiness before you commit to an output path. It packages the source into a temporary ZIP, runs graph admission, and runs the runtime smoke probe, then prints an `OK`/`FAIL` line per check. Exits `1` if any required check fails.

```bash
swarm model test --source ./my_model --family-id cf_autopilot
```

| Flag | Default | Purpose |
| --- | --- | --- |
| `--source <dir>` | required | Source directory (`manifest.json` + `models/*.onnx`). |
| `--family-id <id>` | required | Challenge family the artifact targets. |
| `--interface-version <ver>` | first supported for the family | Explicit policy interface version. |

### `swarm repo package`

Builds or updates the public GitHub submission repo — the layout the backend reads when you commit your repo on-chain. It packages your source (same admission as `model package`) into `artifacts/<family_id>/model_graph.zip`, writes `submission_manifest.json`, and writes `README.md` as a byte-exact copy of the canonical template.

The `README.md` hash is enforced by the backend: if it does not match the template, the submission is rejected silently, so do not edit, reformat, or change the line endings of that file. The manifest records each artifact with `interface_version` `model_graph.v1`, `artifact_path` `artifacts/<family_id>/model_graph.zip`, and the artifact's SHA-256.

A repo holds exactly one family. Packaging a second family, or a family different from the one already in the manifest, is rejected — start a fresh repo to switch.

```bash
# Package your family
swarm repo package \
  --repo-root ./my_submission_repo \
  --family-source cf_autopilot=./autopilot_model

# Pin an interface version
swarm repo package \
  --repo-root ./my_submission_repo \
  --family-source cf_autopilot@model_graph.v1=./autopilot_model

# Single-family shortcut, updating the artifact later
swarm repo package \
  --repo-root ./my_submission_repo \
  --source ./autopilot_model_v2 --family-id cf_autopilot --overwrite
```

| Flag | Default | Purpose |
| --- | --- | --- |
| `--repo-root <dir>` | required | Repo root where `submission_manifest.json` and `artifacts/` live. |
| `--family-source <spec>` | — | `FAMILY_ID=PATH` or `FAMILY_ID@INTERFACE_VERSION=PATH`. One family per repo. |
| `--source <dir>` | — | Single-family source shortcut (use with `--family-id`). |
| `--family-id <id>` | — | Family for the `--source` shortcut. |
| `--interface-version <ver>` | first supported | Interface version for the `--source` shortcut. |
| `--overwrite` | off | Overwrite the targeted artifact ZIP if it exists. |

`--source` and `--family-id` must be given together. After packaging, run `swarm repo verify` before publishing.

### `swarm repo verify`

Validates a submission repo end to end: it loads `submission_manifest.json`, then for the published artifact runs the policy-contract check and a runtime smoke test, and confirms `README.md` is a byte-exact match of the template. Prints the manifest path, the README result, and per-artifact contract and smoke results. Exits `0` only if the manifest, the artifact, and the README all pass.

```bash
swarm repo verify --repo-root ./my_submission_repo
```

| Flag | Default | Purpose |
| --- | --- | --- |
| `--repo-root <dir>` | required | Repo root containing `submission_manifest.json`. |

### `swarm report`

Parses a benchmark log and prints its results block (or a short summary if the block is absent). Fails if the log has no parsable summary fields.

```bash
swarm report
swarm report --input /path/to/bench.log
```

| Flag | Default | Purpose |
| --- | --- | --- |
| `--input <path>` | `/tmp/bench_full_eval.log` | Benchmark log to summarize. |

### `swarm monitor`

Reads the validator runtime snapshot and events files and renders a terminal dashboard. Runs live by default; `--once` prints a single frame. Exits `0` on a clean exit or Ctrl-C, `1` on error.

```bash
swarm monitor

# One-shot snapshot, keep previous terminal content
swarm monitor --once --no-clear

# Override file paths
swarm monitor \
  --snapshot swarm/state/validator_runtime.json \
  --events swarm/state/validator_events.jsonl
```

| Flag | Default | Purpose |
| --- | --- | --- |
| `--snapshot <path>` | dashboard default | `validator_runtime.json` snapshot path. |
| `--events <path>` | dashboard default | `validator_events.jsonl` events path. |
| `--refresh-sec <float>` | `1.0` | Refresh interval for live mode. |
| `--max-events <int>` | `8` | Recent events to display. |
| `--once` | off | Render one frame and exit. |
| `--no-clear` | off | Do not clear the terminal between frames. |

If the snapshot and events files do not exist yet, start the validator first so telemetry can be written.

### `swarm champion`

Downloads the current champion model. Prints the champion UID, score, and per-map scores, then downloads the artifact with SHA-256 integrity verification against the hash the backend reports. If the champion is not released for download, it prints the score and exits `0` without downloading.

```bash
swarm champion
swarm champion --output my_champion.zip
```

| Flag | Default | Purpose |
| --- | --- | --- |
| `--output <path>` | `champion_UID_{uid}.zip` | Output file path. |
| `--backend-url <url>` | `SWARM_BACKEND_API_URL` or the public API | Backend API URL. |

## Tests

CLI behavior is covered in `tests/test_cli.py`: doctor, benchmark delegation, model verify/package/test, repo package/verify, and report parsing.
