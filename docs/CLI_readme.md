# Swarm CLI

Command-line interface for benchmarking, testing, and packaging drone navigation models.

---

## Install

```bash
pip install -e .
```

Or install from PyPI (the published release may lag this repo):

```bash
pip install swarm-sotapilot
```

Then use `swarm <command>` directly. Alternatively, run without installation:

```bash
python -m swarm <command>
```

---

## Challenge families

Family-aware commands (`swarm model package`, `swarm repo package`) take a `--family-id` from this set. Each family runs its own set of procedurally generated environment types:

| Family ID | Environment types |
| --- | --- |
| `cf_autopilot` | City, Open, Mountain, Village, Warehouse, Forest |
| `cf_search_and_rescue` | City, Open, Mountain, Village, Warehouse, Forest |
| `cf_swarm_autopilot` | City, Open, Mountain, Village, Forest |
| `cf_swarm_sar` | City, Open, Mountain, Village, Forest |
| `cf_interceptor` | Open |

---

## Commands

### `swarm doctor`

Checks your environment is ready for benchmarking.

```bash
swarm doctor
```

Verifies: Python version (3.11+), Docker (binary + daemon), sandbox lockdown binaries (`nsenter`, `iptables`) and their permissions, required Python modules (`capnp`, `pybullet`, `gym_pybullet_drones`), writable runtime directories, submission template files, and the benchmark engine module.

### `swarm benchmark`

Runs a local benchmark for the `cf_autopilot` family, evaluating a model across its 6 environment types (City, Open, Mountain, Village, Warehouse, Forest). This subcommand has no family flag; it always drives the engine's default family. The `--seeds-per-group` flag controls seeds per environment group (default: 3). Validators run 1,100 seeds per family per epoch.

```bash
# Default benchmark (3 seeds per environment group)
swarm benchmark --model Submission/submission.zip --family-id cf_swarm_sar --workers 4

# Quick test (1 seed per environment group)
swarm benchmark --model Submission/submission.zip --seeds-per-group 1

# With options
swarm benchmark --model Submission/submission.zip --workers 3 --relax-timeouts --rpc-verbosity low
```

If `--model` is omitted, the current champion (the default-family / cf_autopilot champion) is downloaded (with SHA-256 verification) and benchmarked instead.

Useful options:

- `--workers <n>`: parallel Docker workers (default: one worker per 2 vCPUs, capped at 12).
- `--seed-file <path>` / `--save-seed-file <path>`: replay an exact seed set / save the resolved seeds for later replay.
- `--summary-json-out <path>`: write the benchmark summary as JSON.
- `--log-out <path>`: benchmark log output path.
- `--relax-timeouts`: timeout overrides for slow machines.
- `--rpc-verbosity low|mid|high`: RPC tracing verbosity (default: mid).

### `swarm model verify`

Validates a submission ZIP against Swarm rules: checks structure, ZIP safety, and the 50 MiB uncompressed cap that intake enforces (the local `--max-uncompressed-mb` check defaults to 300), family policy-contract compatibility, and a local runtime smoke test that instantiates the family's entry-point controller and exercises its `reset`/`act` methods.

```bash
swarm model verify --model Submission/submission.zip
```

### `swarm model package`

Bundles a source folder into `Submission/submission.zip` (default path). Automatically includes `drone_agent.py`, `requirements.txt` (if present), model artifacts (`.pt`, `.pth`, `.onnx`, `.zip`, etc.), and a generated `swarm_policy_contract.json`.

Omit `--family-id` in a terminal and the command asks which family you trained for, so you never package the wrong one by accident. Pass `--family-id` to skip the prompt; it is required for non-interactive runs (CI, piped input).

```bash
# Interactive: pick the family from a menu
swarm model package --source ./my_agent

# Explicit family (skips the prompt, needed in scripts)
swarm model package --source ./my_agent --family-id cf_search_and_rescue

# Custom output path
swarm model package --source ./my_agent --family-id cf_autopilot --output Submission/submission.zip --overwrite
```

Options:

- `--family-id <id>`: challenge family implemented by this artifact (required for non-interactive runs; omit it in a terminal to pick from a menu). See the family table above for valid IDs.
- `--interface-version <version>`: explicit policy interface version. Defaults to the first supported version for the selected family.

### `swarm repo package`

Builds or updates a repo-root submission layout for your one family. This writes the artifact ZIP under `artifacts/<family_id>/submission.zip`, updates `submission_manifest.json`, and writes the canonical `README.md` (a byte-exact copy of the required template, so the backend accepts your repo).

```bash
# Package your family
swarm repo package \
  --repo-root ./my_submission_repo \
  --family-source cf_autopilot=./autopilot_agent

# Update the artifact later
swarm repo package \
  --repo-root ./my_submission_repo \
  --source ./autopilot_agent_v2 \
  --family-id cf_autopilot \
  --overwrite
```

`--family-source` takes `FAMILY_ID=PATH` or `FAMILY_ID@INTERFACE_VERSION=PATH`; a repo holds exactly one family, so passing two sources, or a family different from the one already in the manifest, is rejected. The `--source` + `--family-id` pair is a single-family shortcut for the same thing.

### `swarm repo verify`

Validates `submission_manifest.json`, the artifact hash/path, the family policy contract, a runtime smoke test, and the `README.md` hash for the published artifact in a repo layout. A `README.md` that was hand-edited or reformatted fails here, before you commit on-chain.

```bash
swarm repo verify --repo-root ./my_submission_repo --strict-manifest
```

### `swarm model test`

Validates a source folder before packaging: checks that `drone_agent.py` exists and compiles, `requirements.txt` has no blocked patterns, and estimated package size is within limits.

```bash
swarm model test --source ./my_agent
```

### `swarm report`

Parses benchmark log output and prints a summary. Default input: `/tmp/bench_full_eval.log`.

```bash
swarm report
swarm report --input /path/to/log
```

### `swarm monitor`

Reads the validator runtime snapshot/events files and renders a local terminal dashboard.

```bash
swarm monitor

# One-shot snapshot without screen clearing
swarm monitor --once --no-clear

# Override file paths
swarm monitor --snapshot swarm/state/validator_runtime.json --events swarm/state/validator_events.jsonl
```

Useful options:

- `--refresh-sec <seconds>`
  - Refresh interval for live mode.
- `--max-events <n>`
  - Number of recent events to render.
- `--once`
  - Print one frame and exit.
- `--no-clear`
  - Keep previous terminal content.

Expected data files:

- `swarm/state/validator_runtime.json`
- `swarm/state/validator_events.jsonl`

If those files do not exist yet, start the validator first so telemetry can be written.

### `swarm champion`

Downloads the current champion model.

```bash
# Download the champion
swarm champion

# Save to a specific path
swarm champion --output my_champion.zip
```

Options:

- `--output <path>`: output file path. Defaults to `champion_UID_{uid}.zip` in the current directory.
- `--backend-url <url>`: override the backend API URL (defaults to the public API).

The download includes SHA-256 integrity verification against the hash reported by the backend.

## Tests

CLI behavior is covered in `tests/test_cli.py`: doctor, benchmark delegation, model verify/package/test, and report parsing.
