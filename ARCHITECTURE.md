# Architecture

## Purpose
This repository has two closely related responsibilities:

1. Run a benchmark for autonomous drone navigation.
2. Run validator-side evaluation and scoring for submitted agents.

The codebase is organized so benchmark orchestration, validator orchestration, map generation, and Docker-isolated model execution are separate concerns.

## Design Principles
- Each module should have one clear job.
- Dependencies should point inward toward core logic.
- Business logic should be separate from CLI, filesystem, Docker, and backend I/O.
- Large domains should be packages, not monolithic files.
- Public entrypoints can be thin facades, but implementation should live in focused submodules.

## Top-Level Layout
- [swarm/](/home/miguel/subnets/swarm/swarm_subnet/swarm): main Python package
- [scripts/](/home/miguel/subnets/swarm/swarm_subnet/scripts): operational scripts and local tooling entrypoints
- [tests/](/home/miguel/subnets/swarm/swarm_subnet/tests): unit, integration, and opt-in e2e coverage
- [docs/](/home/miguel/subnets/swarm/swarm_subnet/docs): user and validator documentation
- [model/](/home/miguel/subnets/swarm/swarm_subnet/model): local benchmark/test model artifacts

## Main Package Structure
### Benchmark
- [swarm/benchmark/](/home/miguel/subnets/swarm/swarm_subnet/swarm/benchmark): benchmark orchestration
- [engine.py](/home/miguel/subnets/swarm/swarm_subnet/swarm/benchmark/engine.py): thin public facade
- [engine_parts/](/home/miguel/subnets/swarm/swarm_subnet/swarm/benchmark/engine_parts): implementation split by concern

`engine_parts/` responsibilities:
- `config.py`: run options and runtime configuration assembly
- `seeds.py`: seed loading, saving, grouping, and selection
- `dispatch.py`: scheduling policy and worker dispatch decisions
- `workers.py`: process workers, watchdogs, and batch execution
- `reporting.py`: result tables, summaries, and benchmark artifacts
- `entry.py`: high-level benchmark run entrypoint

### Validator
- [swarm/validator/](/home/miguel/subnets/swarm/swarm_subnet/swarm/validator): validator workflow and scoring
- [utils.py](/home/miguel/subnets/swarm/swarm_subnet/swarm/validator/utils.py): thin compatibility facade
- [utils_parts/](/home/miguel/subnets/swarm/swarm_subnet/swarm/validator/utils_parts): validator workflow split by domain
- [forward.py](/home/miguel/subnets/swarm/swarm_subnet/swarm/validator/forward.py): validator forward loop
- [backend_api.py](/home/miguel/subnets/swarm/swarm_subnet/swarm/validator/backend_api.py): backend HTTP client

`utils_parts/` responsibilities:
- `model_fetch.py`: fetch/download model artifacts
- `detection.py`: detect model changes and queue candidates
- `queue_worker.py`: queue processing and benchmark execution flow
- `backend_submission.py`: backend submission/publication calls
- `evaluation.py`: benchmark result aggregation and scoring summaries
- `heartbeat.py`: backend heartbeat coordination
- `state.py`: local validator state files and persistence
- `weights.py`: weight calculation helpers

### Docker Evaluation Runtime
- [swarm/validator/docker/](/home/miguel/subnets/swarm/swarm_subnet/swarm/validator/docker): secure model execution layer
- [docker_evaluator.py](/home/miguel/subnets/swarm/swarm_subnet/swarm/validator/docker/docker_evaluator.py): public facade and compatibility surface
- [docker_evaluator_parts/](/home/miguel/subnets/swarm/swarm_subnet/swarm/validator/docker/docker_evaluator_parts): implementation split by responsibility

`docker_evaluator_parts/` responsibilities:
- `lifecycle.py`: Docker image/container lifecycle and runtime setup
- `submission.py`: submission validation and preparation
- `networking.py`: Docker networking and isolation helpers
- `rpc.py`: Cap'n Proto RPC client interaction and timeout handling
- `batch.py`: per-seed and per-batch evaluation execution
- `parallel.py`: process-based parallel scheduling for validator evaluation

### Configuration
- [swarm/config/](/home/miguel/subnets/swarm/swarm_subnet/swarm/config): typed runtime settings
- [runtime.py](/home/miguel/subnets/swarm/swarm_subnet/swarm/config/runtime.py): environment-backed settings for benchmark, validator, Docker, and backend runtime

This is the preferred place for new runtime env parsing. Avoid scattering new `os.getenv()` calls across unrelated modules.

## Core Simulation and Maps
### Core Simulation
- [swarm/core/](/home/miguel/subnets/swarm/swarm_subnet/swarm/core): simulation environment, generators, and map-building logic
- [moving_drone.py](/home/miguel/subnets/swarm/swarm_subnet/swarm/core/moving_drone.py): main drone simulation environment
- [env_builder/](/home/miguel/subnets/swarm/swarm_subnet/swarm/core/env_builder): world assembly, generation, and cache helpers

### Map Family Entry Points
- [swarm/core/maps/](/home/miguel/subnets/swarm/swarm_subnet/swarm/core/maps): canonical map-family entrypoints
- [city/](/home/miguel/subnets/swarm/swarm_subnet/swarm/core/maps/city)
- [open/](/home/miguel/subnets/swarm/swarm_subnet/swarm/core/maps/open)
- [village/](/home/miguel/subnets/swarm/swarm_subnet/swarm/core/maps/village)
- [mountain/](/home/miguel/subnets/swarm/swarm_subnet/swarm/core/maps/mountain)
- [forest/](/home/miguel/subnets/swarm/swarm_subnet/swarm/core/maps/forest)
- [warehouse/](/home/miguel/subnets/swarm/swarm_subnet/swarm/core/maps/warehouse)

These packages are the preferred boundary for map-family-specific logic. If you are adding or changing a map family, start here.

### Large Generator Packages
Legacy public modules still exist as compatibility facades, but their implementations now live in focused packages:
- [city_generator_parts/](/home/miguel/subnets/swarm/swarm_subnet/swarm/core/city_generator_parts)
- [mountain_generator_parts/](/home/miguel/subnets/swarm/swarm_subnet/swarm/core/mountain_generator_parts)
- [forest_generator_parts/](/home/miguel/subnets/swarm/swarm_subnet/swarm/core/forest_generator_parts)
- [warehouse/](/home/miguel/subnets/swarm/swarm_subnet/swarm/core/warehouse) with subpackages:
  - `factory_parts`
  - `helpers_parts`
  - `layout_parts`
  - `loading_parts`
  - `office_parts`
  - `operations_parts`
  - `storage_parts`
  - `structure_parts`

## CLI and Scripts
- [swarm/cli.py](/home/miguel/subnets/swarm/swarm_subnet/swarm/cli.py): public CLI entrypoint exposed as `swarm`
- [scripts/bench_full_eval.py](/home/miguel/subnets/swarm/swarm_subnet/scripts/bench_full_eval.py): benchmark script entrypoint
- [scripts/README.md](/home/miguel/subnets/swarm/swarm_subnet/scripts/README.md): script-level usage notes

Rule:
- reusable logic belongs in `swarm/...`
- `scripts/...` should stay thin and delegate into package code

## Dependency Direction
Preferred dependency flow:
1. CLI/scripts depend on benchmark or validator packages.
2. Benchmark and validator orchestration depend on core simulation and typed config.
3. Docker runtime implements execution details used by validator/benchmark layers.
4. Map generation and simulation core should not depend on CLI or backend code.

Avoid the reverse flow. In particular:
- core simulation should not import CLI code
- map generators should not know about backend APIs
- validator orchestration should not contain low-level Docker implementation details

## State, Assets, and Generated Files
- [swarm/assets/](/home/miguel/subnets/swarm/swarm_subnet/swarm/assets): committed static assets
- [state/](/home/miguel/subnets/swarm/swarm_subnet/state): runtime state directory used by tooling
- [swarm/state/](/home/miguel/subnets/swarm/swarm_subnet/swarm/state): local generated cache/state under the package tree in some flows
- [bench_logs/](/home/miguel/subnets/swarm/swarm_subnet/bench_logs): local benchmark outputs

Rules:
- generated caches, state files, and logs should not become part of the source architecture
- new persistent runtime state should go through the validator/benchmark state abstractions, not arbitrary ad hoc files

## Testing Strategy
- `pytest`: fast default suite
- `pytest --run-e2e` or `SWARM_RUN_E2E=1 pytest`: opt-in e2e/runtime suite

Testing layers:
- unit and integration tests validate split modules and orchestration logic
- opt-in e2e tests validate simulator, Docker, and forward-loop behavior
- benchmark smoke runs validate the real benchmark path after major refactors

## How To Add New Code
### If you are adding benchmark behavior
Start in:
- [swarm/benchmark/engine_parts/](/home/miguel/subnets/swarm/swarm_subnet/swarm/benchmark/engine_parts)

### If you are adding validator workflow behavior
Start in:
- [swarm/validator/utils_parts/](/home/miguel/subnets/swarm/swarm_subnet/swarm/validator/utils_parts)
- [swarm/validator/forward.py](/home/miguel/subnets/swarm/swarm_subnet/swarm/validator/forward.py)

### If you are changing secure model execution
Start in:
- [swarm/validator/docker/docker_evaluator_parts/](/home/miguel/subnets/swarm/swarm_subnet/swarm/validator/docker/docker_evaluator_parts)

### If you are adding/changing map generation
Start in:
- [swarm/core/maps/](/home/miguel/subnets/swarm/swarm_subnet/swarm/core/maps)
- then the relevant `*_parts/` implementation package

### If you need environment/config flags
Start in:
- [swarm/config/runtime.py](/home/miguel/subnets/swarm/swarm_subnet/swarm/config/runtime.py)

Do not add new scattered env parsing unless there is a very strong reason.

## Anti-Patterns To Avoid
- adding new giant `utils.py` or `helpers.py` dumping grounds
- mixing backend I/O, Docker control, and core simulation logic in one module
- introducing new star-import facades when explicit imports are practical
- placing reusable library logic in `scripts/`
- putting map-family-specific logic outside `swarm/core/maps/` and its implementation packages
- adding new files that are only differentiated by size, not by responsibility

## Current Architectural Intent
This repo is moving toward:
- thin public facades
- focused implementation packages
- map-family boundaries that match the problem domain
- validator and benchmark runtimes that share execution ideas without collapsing into one monolith
- typed runtime configuration instead of scattered env reads

That structure is what new contributions should preserve.
