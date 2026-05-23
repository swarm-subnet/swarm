# Scripts

This directory contains repository-level executable utilities.

## Python scripts

- `bench_full_eval.py`
  - Full benchmark entrypoint used by the CLI and manual benchmark runs.
- `stress_benchmark_compare.py`
  - Repeats validator-style benchmark samples for one model, saves per-run artifacts, and emits a comparison report with average-score variance across runs.
  - Example smoke test:
    - `python3 scripts/stress_benchmark_compare.py --model model/UID_178.zip --seed-count 100 --repetitions 5 --workers 6 --relax-timeouts --rpc-verbosity low`
  - Example full run:
    - `python3 scripts/stress_benchmark_compare.py --model model/UID_178.zip --seed-count 1000 --repetitions 5 --workers 6 --relax-timeouts --rpc-verbosity low`
- `test_timings.py`
  - Local timing breakdown tool for simulator step costs.
- `sar_spawn_audit.py`
  - SAR spawn-pipeline audit: runs many seeds per map family and asserts the failure rate stays below the target threshold. Intended as the nightly audit.
- `sar_baseline_audit.py`
  - Records a baseline policy's success/failure profile on SAR seeds so the network has a "minimum competence" reference point.
- `sar_horizon_audit.py`
  - Sweeps episode horizons and reports per-map confirm rates; used to validate the chosen horizon.
- `prebake_mannequin_parts.py`
  - One-shot mannequin prebake — splits a MakeHuman raw OBJ/MTL into the per-material parts the runtime loads. Run when adding a new character asset.

## Shell scripts

- `miner/setup.sh`
- `miner/install_dependencies.sh`

These are operational setup scripts for miner environments.
