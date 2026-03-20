# Scripts

This directory contains repository-level executable utilities.

## Python scripts

- `bench_full_eval.py`
  - Full benchmark entrypoint used by the CLI and manual benchmark runs.
- `benchmark_with_videos.py`
  - One-shot runner that executes benchmark, saves seeds + summary, renders exact benchmark replay videos, and fails if replayed results differ.
- `debug_idempotency_report.py`
  - Debug-only repeated-seed runner that replays one exact seed per map type multiple times and prints a grouped idempotency report.
- `generate_video.py`
  - Replay/video rendering utility for a model + seed, or a saved benchmark seed file.
- `gen_platform_images.py`
  - Scene image generator for documentation and visual inspection.
- `visualize_map.py`
  - Interactive live render visualizer for manually flying a seed/map with the keyboard.
- `test_timings.py`
  - Local timing breakdown tool for simulator step costs.

## Shell scripts

- `miner/setup.sh`
- `miner/install_dependencies.sh`

These are operational setup scripts for miner environments.
