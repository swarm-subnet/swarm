from __future__ import annotations

import json
import shutil
import statistics
import subprocess
from pathlib import Path

import pytest

from swarm import cli as swarm_cli
from swarm.benchmark.engine import main as benchmark_main
from swarm.benchmark.engine_parts._shared import BENCH_GROUP_ORDER

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "tests" / "default_model" / "default_model.zip"
FIXED_SEED_FILE = REPO_ROOT / "tests" / "fixtures" / "benchmark_default_model_fixed_100_seeds_v1.json"
FIXED_SEED_TOTAL = 100
BENCHMARK_WORKERS = 4
MIN_CLEAN_EXECUTION_COUNT = 95
EXPECTED_GROUP_COUNTS = {
    "type1_city": 17,
    "type2_open": 17,
    "type3_mountain": 17,
    "type4_village": 17,
    "type5_warehouse": 16,
    "type6_forest": 16,
}

# Calibrated from the fixed-seed benchmark run recorded on 2026-04-08.
# Observed baseline:
# - overall_avg_score = 0.2138604017157132
# - success_count = 24 / 100
#
# The envelopes allow mild simulator/runtime variation while still being tight
# enough to catch materially degraded installs.
EXPECTED_OVERALL_AVG_SCORE_RANGE: tuple[float, float] = (0.19, 0.24)
EXPECTED_SUCCESS_COUNT_RANGE: tuple[int, int] = (22, 26)


def _load_fixed_seed_groups() -> dict[str, list[int]]:
    return json.loads(FIXED_SEED_FILE.read_text())


def _overall_avg_score(summary: dict[str, object]) -> float:
    group_results = summary.get("group_results")
    assert isinstance(group_results, dict), "Benchmark summary JSON missing group_results."
    rows = [
        row
        for group in BENCH_GROUP_ORDER
        for row in list(group_results.get(group, []))
    ]
    assert rows, "Benchmark summary JSON did not contain any scored rows."
    return float(statistics.fmean(float(row["score"]) for row in rows))


def _success_count(summary: dict[str, object]) -> int:
    group_results = summary.get("group_results")
    assert isinstance(group_results, dict), "Benchmark summary JSON missing group_results."
    rows = [
        row
        for group in BENCH_GROUP_ORDER
        for row in list(group_results.get(group, []))
    ]
    assert rows, "Benchmark summary JSON did not contain any scored rows."
    return int(sum(1 for row in rows if bool(row["success"])))


def _clean_execution_count(summary: dict[str, object]) -> int:
    group_results = summary.get("group_results")
    assert isinstance(group_results, dict), "Benchmark summary JSON missing group_results."
    rows = [
        row
        for group in BENCH_GROUP_ORDER
        for row in list(group_results.get(group, []))
    ]
    assert rows, "Benchmark summary JSON did not contain any scored rows."
    return int(sum(1 for row in rows if bool(row.get("execution_ok", False))))


def _execution_failure_modes(summary: dict[str, object]) -> dict[str, int]:
    raw = summary.get("execution_status_counts", {})
    assert isinstance(raw, dict), "Benchmark summary JSON missing execution_status_counts."
    return {
        str(status): int(count)
        for status, count in raw.items()
        if str(status) not in {"completed", "seed_done"}
    }


def _require_docker() -> None:
    if shutil.which("docker") is None:
        pytest.skip("Docker binary not found in PATH.")
    info = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=20)
    if info.returncode != 0:
        detail = info.stderr.strip() or info.stdout.strip() or "unknown docker error"
        pytest.skip(f"Docker daemon unavailable: {detail}")


def _require_benchmark_permissions() -> None:
    check = swarm_cli._check_sandbox_lockdown_permissions()
    if not check.ok:
        pytest.skip(
            "Benchmark regression requires sandbox lockdown permissions for the current "
            "user (either root, or cap_sys_admin on nsenter plus cap_net_admin on "
            "iptables)."
        )


def test_default_model_fixed_seed_fixture_has_expected_shape() -> None:
    seed_groups = _load_fixed_seed_groups()

    assert list(seed_groups) == BENCH_GROUP_ORDER
    assert {group: len(seeds) for group, seeds in seed_groups.items()} == EXPECTED_GROUP_COUNTS
    assert sum(len(seeds) for seeds in seed_groups.values()) == FIXED_SEED_TOTAL


@pytest.mark.integration
@pytest.mark.slow
def test_default_model_fixed_100_seed_benchmark_regression(tmp_path: Path) -> None:
    """
    Fixed benchmark regression for the tracked default model.

    Seed provenance:
    - generated from the benchmark seed search path with `random.seed(20260407)`
    - started from `_find_seeds(17)` for each group
    - trimmed one seed from type5/type6 to land on exactly 100 total seeds
    """

    _require_docker()
    _require_benchmark_permissions()

    assert MODEL_PATH.exists(), f"Missing regression model: {MODEL_PATH}"
    assert FIXED_SEED_FILE.exists(), f"Missing fixed seed fixture: {FIXED_SEED_FILE}"

    summary_path = tmp_path / "default_model_fixed_100_summary.json"
    log_path = tmp_path / "default_model_fixed_100.log"

    benchmark_main(
        [
            "--model",
            str(MODEL_PATH),
            "--workers",
            str(BENCHMARK_WORKERS),
            "--seed-file",
            str(FIXED_SEED_FILE),
            "--summary-json-out",
            str(summary_path),
            "--log-out",
            str(log_path),
            "--relax-timeouts",
        ]
    )

    summary = json.loads(summary_path.read_text())
    observed_score = _overall_avg_score(summary)
    observed_success_count = _success_count(summary)
    observed_clean_execution_count = _clean_execution_count(summary)
    execution_failure_modes = _execution_failure_modes(summary)
    observed_seed_total = sum(
        len(list(summary.get("group_results", {}).get(group, [])))
        for group in BENCH_GROUP_ORDER
    )
    min_score, max_score = EXPECTED_OVERALL_AVG_SCORE_RANGE
    min_success_count, max_success_count = EXPECTED_SUCCESS_COUNT_RANGE

    assert observed_seed_total == FIXED_SEED_TOTAL
    assert observed_clean_execution_count >= MIN_CLEAN_EXECUTION_COUNT, (
        "Default-model fixed-seed benchmark had too many non-clean executions. "
        f"clean_execution_count={observed_clean_execution_count}/{FIXED_SEED_TOTAL}; "
        f"failure_modes={execution_failure_modes}. "
        "This indicates benchmark infrastructure instability (for example RPC "
        "disconnects or container issues), so score regression should not be "
        "interpreted as a model-only failure. "
        f"Summary: {summary_path}  Log: {log_path}"
    )
    assert min_success_count <= observed_success_count <= max_success_count, (
        f"Default-model fixed-seed benchmark success_count {observed_success_count} fell "
        f"outside the expected window [{min_success_count}, {max_success_count}]. "
        f"Summary: {summary_path}  Log: {log_path}"
    )
    assert min_score <= observed_score <= max_score, (
        f"Default-model fixed-seed benchmark overall_avg_score {observed_score:.4f} fell "
        f"outside the expected window [{min_score:.4f}, {max_score:.4f}]. "
        f"Summary: {summary_path}  Log: {log_path}"
    )
