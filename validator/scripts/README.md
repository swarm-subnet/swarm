# Scripts

This directory contains repository-level executable utilities.

## Python scripts

- `bench_full_eval.py`
  - Full benchmark entrypoint used by the CLI and manual benchmark runs.
- `visualize_map.py`
  - Interactive live-render visualizer for manually flying a seed/map with the keyboard. Also reachable as `swarm visualize`, which resolves `--seed`/`--summary-json --failed` into the right `--type`/`--family-id` for you. Accepts `--family-id` (default `cf_autopilot`) so any challenge family renders with its own world-building logic, e.g. `python3 validator/scripts/visualize_map.py --type 1 --family-id cf_search_and_rescue`.
- `generate_video.py`
  - Renders mp4 flight videos for a model + seed, or a saved benchmark seed file. Also reachable as `swarm video`. Accepts `--family-id` (default `cf_autopilot`) and `--backend {local,benchmark}` (local fast replay vs. the exact benchmark Docker/RPC replay), e.g. `python3 validator/scripts/generate_video.py --model Submission/submission.zip --seed 42 --type 7 --family-id cf_interceptor_office --mode chase`.
- `stress_benchmark_compare.py`
  - Repeats validator-style benchmark samples for one model, saves per-run artifacts, and emits a comparison report with average-score variance across runs.
  - Example smoke test:
    - `python3 validator/scripts/stress_benchmark_compare.py --model Submission/submission.zip --seed-count 100 --repetitions 5 --workers 6 --relax-timeouts --rpc-verbosity low`
  - Example full run:
    - `python3 validator/scripts/stress_benchmark_compare.py --model Submission/submission.zip --seed-count 1000 --repetitions 5 --workers 6 --relax-timeouts --rpc-verbosity low`
- `test_timings.py`
  - Local timing breakdown tool for simulator step costs.
- `sar_spawn_audit.py`
  - SAR spawn-pipeline audit: runs many seeds per environment type and asserts the failure rate stays below the target threshold. Intended as the nightly audit.
- `sar_baseline_audit.py`
  - Records a baseline policy's success/failure profile on SAR seeds so the network has a "minimum competence" reference point.
- `sar_horizon_audit.py`
  - Sweeps episode horizons and reports per-environment confirm rates; used to validate the chosen horizon.
- `prebake_mannequin_parts.py`
  - One-shot mannequin prebake — splits a MakeHuman raw OBJ/MTL into the per-material parts the runtime loads. Run when adding a new character asset.

## Shell scripts

- `main/setup.sh`
- `main/install_dependencies.sh`
- `update/update_deploy.sh`
- `update/auto_update_deploy.sh`

These are operational setup scripts for validator environments. The miner's own
scripts live in `../../miner/src/scripts/`.
