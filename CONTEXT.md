# Repository Context

## What This Repo Is

Swarm is a Python 3.11+ codebase for an autonomous drone navigation benchmark and the validator runtime that scores submitted agents on Bittensor Subnet 124.

At a high level, the repo supports two main workflows:

1. Miner workflow
   Build, package, verify, benchmark, and publish a drone policy.
2. Validator workflow
   Fetch submitted models, evaluate them in isolated Docker containers, score them, and report results to the backend.

Core benchmark assumptions:

- Models receive a `128x128` depth image plus a state vector.
- Models output 5D flight commands.
- Evaluation runs across procedurally generated environments such as city, mountain, village, warehouse, forest, and open terrain.

## Primary Entrypoints

- [`README.md`](README.md)
  Project overview, benchmark framing, and quick CLI examples.
- [`ARCHITECTURE.md`](ARCHITECTURE.md)
  Canonical code-organization and dependency-boundary guide.
- [`docs/miner.md`](docs/miner.md)
  Miner setup, submission structure, and GitHub publishing flow.
- [`docs/validator.md`](docs/validator.md)
  Validator setup, Docker requirements, telemetry, and runtime expectations.
- [`docs/CLI_readme.md`](docs/CLI_readme.md)
  CLI command reference.

## Important Packages

- [`swarm/`](swarm)
  Main Python package.
- [`swarm/benchmark/`](swarm/benchmark)
  Local benchmark orchestration.
- [`swarm/validator/`](swarm/validator)
  Validator pipeline, backend interaction, telemetry, and scoring flow.
- [`swarm/validator/docker/`](swarm/validator/docker)
  Docker-isolated model execution runtime.
- [`swarm/core/`](swarm/core)
  Drone simulation, map generation, and environment assembly.
- [`swarm/config/`](swarm/config)
  Runtime configuration and environment-backed settings.
- [`swarm/base/`](swarm/base)
  Shared neuron and validator base abstractions.
- [`neurons/`](neurons)
  Launch scripts for validator and miner processes.
- [`scripts/`](scripts)
  Thin operational utilities for benchmark runs, replay, visualization, and setup.
- [`tests/`](tests)
  Unit, integration, smoke, and selected end-to-end coverage.

## Key Files To Know

- [`swarm/cli.py`](swarm/cli.py)
  Public CLI entrypoint exposed as `swarm`.
- [`swarm/__init__.py`](swarm/__init__.py)
  Package version and protocol compatibility version.
- [`swarm/config/runtime.py`](swarm/config/runtime.py)
  Preferred place for runtime env parsing and shared settings.
- [`swarm/validator/forward.py`](swarm/validator/forward.py)
  Validator forward-loop entrypoint.
- [`swarm/validator/backend_api.py`](swarm/validator/backend_api.py)
  Backend HTTP client used by validator flows.
- [`swarm/validator/docker/docker_evaluator.py`](swarm/validator/docker/docker_evaluator.py)
  Public facade for secure model evaluation.
- [`swarm/core/moving_drone.py`](swarm/core/moving_drone.py)
  Main simulation environment.

## Development Shape

The architecture is intentionally split into thin public facades and focused implementation packages.

Preferred boundaries:

- Add benchmark logic under `swarm/benchmark/engine_parts/`.
- Add validator workflow logic under `swarm/validator/utils_parts/`.
- Add Docker runtime behavior under `swarm/validator/docker/docker_evaluator_parts/`.
- Add map-specific logic under `swarm/core/maps/` and the relevant `*_parts/` package.
- Add runtime env parsing in `swarm/config/runtime.py`.

Avoid:

- Putting reusable logic into `scripts/`.
- Mixing backend I/O, Docker control, and simulator logic in one module.
- Adding new monolithic `utils.py` style dumping grounds.

## Typical Commands

Local install:

```bash
pip install -e .
```

Common CLI workflow:

```bash
swarm doctor
swarm model test --source my_agent/
swarm model package --source my_agent/
swarm model verify --model Submission/submission.zip
swarm benchmark --model Submission/submission.zip --workers 4
swarm report
```

Validator monitoring:

```bash
swarm monitor
```

## Testing

- Default test runner: `pytest`
- Opt-in broader runtime/e2e coverage: `pytest --run-e2e`

There is substantial test coverage across CLI behavior, validator orchestration, Docker evaluation, benchmark scripts, map generation, telemetry, and environment determinism.

## Operational Notes

- Package metadata lives in [`pyproject.toml`](pyproject.toml).
- Runtime requirements are sourced from [`requirements.txt`](requirements.txt).
- Validator operation requires Docker.
- The repo includes large static assets under `swarm/assets/`.
- Generated runtime state commonly lands under `swarm/state/` and should not be treated as source architecture.

## Quick Orientation

If you are new to the repo, start in this order:

1. [`README.md`](README.md)
2. [`ARCHITECTURE.md`](ARCHITECTURE.md)
3. The relevant workflow doc in [`docs/`](docs)
4. The specific package entrypoint you intend to change
