# Swarm CLI

Command-line interface for benchmarking, testing, and packaging drone navigation models.

---

## Install

```bash
pip install -e .
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

Runs a local benchmark — evaluates a model across procedurally generated environments. The `--seeds-per-group` flag controls seeds per environment type (default: 3). Validators run 1,000 seeds total.

```bash
# Default benchmark (3 seeds per environment group)
swarm benchmark --model Submission/submission.zip --workers 4

# Quick test (1 seed per environment type)
swarm benchmark --model Submission/submission.zip --seeds-per-group 1

# With options
swarm benchmark --model Submission/submission.zip --workers 3 --relax-timeouts --rpc-verbosity low
```

### `swarm model verify`

Validates a submission ZIP against Swarm rules — checks structure, size limits, path safety, and `drone_agent.py` compliance.

```bash
swarm model verify --model Submission/submission.zip
```

### `swarm model package`

Bundles a source folder into `Submission/submission.zip` (default path). Automatically includes `drone_agent.py`, `requirements.txt` (if present), and model artifacts (`.pt`, `.pth`, `.onnx`, `.zip`, etc.).

```bash
swarm model package --source ./my_agent

# Custom output path
swarm model package --source ./my_agent --output Submission/submission.zip --overwrite
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

---

`doctor`, `model verify`, `model test`, and `report` support `--json` for machine-readable output.

## Tests

CLI behavior is covered in `tests/test_cli.py` — doctor, benchmark delegation, model verify/package/test, and report parsing.
