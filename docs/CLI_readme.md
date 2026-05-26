# Swarm CLI

Command-line interface for benchmarking, testing, and packaging drone navigation models.

---

## Install

```bash
pip install -e .
```

Once published on PyPI:

```bash
pip install swarm-sotapilot
```

Then use `swarm <command>` directly. Alternatively, run without installation:

```bash
python -m swarm <command>
```

---

## Commands

### `swarm doctor`

Checks your environment is ready for benchmarking.

```bash
swarm doctor
```

Verifies: Python version, Docker (binary + daemon), required Python modules (`capnp`, `pybullet`, `gym_pybullet_drones`), writable runtime directories, submission template files, and benchmark script presence.

### `swarm benchmark`

Runs a local benchmark — evaluates a model across 6 procedurally generated environment types (City, Open, Mountain, Village, Warehouse, Forest). The `--seeds-per-group` flag controls seeds per environment type (default: 3). Validators run 1,000 seeds total.

```bash
# Default benchmark (3 seeds per environment group)
swarm benchmark --model Submission/submission.zip --workers 4

# Quick test (1 seed per environment type)
swarm benchmark --model Submission/submission.zip --seeds-per-group 1

# With options
swarm benchmark --model Submission/submission.zip --workers 3 --relax-timeouts --rpc-verbosity low
```

### `swarm model verify`

Validates a submission ZIP against Swarm rules — checks structure, size limits, path safety, family policy-contract compatibility, and a local runtime smoke test for `DroneFlightController.act()`.

```bash
swarm model verify --model Submission/submission.zip
```

### `swarm model package`

Bundles a source folder into `Submission/submission.zip` (default path). Automatically includes `drone_agent.py`, `requirements.txt` (if present), model artifacts (`.pt`, `.pth`, `.onnx`, `.zip`, etc.), and a generated `swarm_policy_contract.json`.

```bash
swarm model package --source ./my_agent

# Custom output path
swarm model package --source ./my_agent --output Submission/submission.zip --overwrite

# Explicit family selection
swarm model package --source ./my_agent --family-id cf_autopilot
```

### `swarm repo package`

Builds or updates a repo-root multi-family submission layout. This writes artifact ZIPs under `artifacts/<family_id>/submission.zip` and updates `submission_manifest.json`.

```bash
# Package two families at once
swarm repo package \
  --repo-root ./my_submission_repo \
  --family-source cf_search_and_rescue=./sar_agent \
  --family-source cf_autopilot=./autopilot_agent

# Update one family later without replacing the others
swarm repo package \
  --repo-root ./my_submission_repo \
  --source ./autopilot_agent_v2 \
  --family-id cf_autopilot \
  --overwrite
```

### `swarm repo verify`

Validates `submission_manifest.json`, artifact hashes/paths, family policy contracts, and runtime smoke tests for every published artifact in a repo layout.

```bash
swarm repo verify --repo-root ./my_submission_repo --strict-manifest
```

### `swarm model test`

Validates a source folder before packaging — checks that `drone_agent.py` exists and compiles, `requirements.txt` has no blocked patterns, and estimated package size is within limits.

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

- `--output <path>` — output file path. Defaults to `champion_UID_{uid}.zip` in the current directory.
- `--backend-url <url>` — override the backend API URL (defaults to the public API).

The download includes SHA-256 integrity verification against the hash reported by the backend.

## Tests

CLI behavior is covered in `tests/test_cli.py` — doctor, benchmark delegation, model verify/package/test, and report parsing.
