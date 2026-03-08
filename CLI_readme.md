# Swarm CLI

This repository now includes a first-party CLI under the `swarm` command.

## Install / Use

If you are in the repo virtualenv:

```bash
source validator_env/bin/activate
pip install -e .
```

Then use:

```bash
swarm --help
```

You can also run without installation:

```bash
python -m swarm --help
```

## Commands

### `swarm doctor`

Checks local readiness for benchmarking:
- Python version
- Docker binary + daemon
- Required Python modules (`capnp`, `pybullet`, `gym_pybullet_drones`)
- Writable runtime directories (`state`, model directory)
- Submission template files
- Benchmark script presence
- Optional benchmark secret env var

Examples:

```bash
swarm doctor
swarm doctor --json
```

### `swarm benchmark`

Runs the full benchmark workflow by delegating to `debugging/bench_full_eval.py`.

Examples:

```bash
swarm benchmark --model model/UID_178.zip --workers 3 --relax-timeouts --rpc-verbosity low
swarm benchmark --full --model model/UID_178.zip --seeds-per-group 10 --workers 3
```

Notes:
- `--full` is accepted as a compatibility flag. Full benchmark is the default mode.
- UID inference behavior is handled by `bench_full_eval.py`.

### `swarm model verify`

Validates a submission zip against Swarm RPC submission rules.

Checks include:
- zip exists
- compressed size within limit
- zip safety checks (path traversal, decompressed size cap)
- `drone_agent.py` presence and structural compliance

Examples:

```bash
swarm model verify --model Submission/submission.zip
swarm model verify --model Submission/submission.zip --json
```

### `swarm model package`

Builds `submission.zip` from a source folder.

Requirements:
- source must contain `drone_agent.py`

Included files:
- `drone_agent.py`
- `requirements.txt` (if present)
- model artifacts with recognized extensions (`.pt`, `.pth`, `.onnx`, `.zip`, etc.)

Examples:

```bash
swarm model package --source ./my_agent --output Submission/submission.zip
swarm model package --source ./my_agent --output Submission/submission.zip --overwrite
```

### `swarm model test`

Validates a source folder before packaging/submission.

Checks include:
- `drone_agent.py` exists
- `drone_agent.py` compiles as Python
- `requirements.txt` has no blocked patterns (`-r`, direct URL/path refs, direct refs)
- estimated package size <= Swarm max model size

Examples:

```bash
swarm model test --source ./my_agent
swarm model test --source ./my_agent --json
```

### `swarm report`

Parses benchmark log output and prints a summary.

Default input is `/tmp/bench_full_eval.log`.

Examples:

```bash
swarm report
swarm report --input /tmp/bench_full_eval.log --json
```

## Test Coverage

CLI behavior is tested in:

- `tests/test_cli.py`

This includes coverage for:
- doctor success/failure behavior
- benchmark command delegation
- model verify pass/fail
- model package creation
- model test validation failures
- report parsing success/failure
