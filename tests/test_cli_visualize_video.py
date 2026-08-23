"""The visualizer and the recorder were restored after an accidental deletion
(commit 04a21a0, "SAR v5.0.0") -- these tests lock down the CLI wiring and the
family-aware task construction so a future cleanup pass can't silently drop
them again without a test failing first.
"""
from __future__ import annotations

import json
import os

import pytest

import scripts.generate_video as generate_video
import scripts.visualize_map as visualize_map
from swarm import cli


# --------------------------------------------------------------------------
# swarm visualize
# --------------------------------------------------------------------------


def test_visualize_dispatches_with_resolved_type_and_family(monkeypatch):
    captured: dict = {}

    def _fake_main(argv):
        captured["argv"] = argv

    monkeypatch.setattr(visualize_map, "main", _fake_main)

    assert cli.main(["visualize", "--type", "2", "--family-id", "cf_interceptor"]) == 0
    assert captured["argv"][:4] == ["--type", "2", "--family-id", "cf_interceptor"]


def test_visualize_defaults_family_to_autopilot(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(visualize_map, "main", lambda argv: captured.setdefault("argv", argv))

    assert cli.main(["visualize", "--type", "1"]) == 0
    assert "--family-id" in captured["argv"]
    assert captured["argv"][captured["argv"].index("--family-id") + 1] == "cf_autopilot"


def test_visualize_requires_type_seed_or_summary(monkeypatch):
    monkeypatch.setattr(visualize_map, "main", lambda argv: pytest.fail("must not launch"))
    assert cli.main(["visualize"]) == 1


def test_visualize_rejects_mismatched_explicit_type(tmp_path, monkeypatch):
    seed_file = tmp_path / "seeds.json"
    seed_file.write_text(json.dumps({"type1_city": [42]}))
    monkeypatch.setattr(visualize_map, "main", lambda argv: pytest.fail("must not launch"))

    assert cli.main(
        ["visualize", "--seed", "42", "--seed-file", str(seed_file), "--type", "3"]
    ) == 1


def test_visualize_failed_lists_rows_without_index(tmp_path, monkeypatch, capsys):
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "group_results": {
                    "type1_city": [
                        {"seed": 11, "success": False, "score": 0.0, "sim_time": 4.2,
                         "execution_status": "collision"},
                    ],
                    "type2_open": [], "type3_mountain": [], "type4_village": [],
                    "type5_warehouse": [], "type6_forest": [],
                }
            }
        )
    )
    monkeypatch.setattr(visualize_map, "main", lambda argv: pytest.fail("must not launch"))

    assert cli.main(["visualize", "--summary-json", str(summary), "--failed"]) == 0
    out = capsys.readouterr().out
    assert "seed 11" in out
    assert "collision" in out


def test_visualize_failed_index_opens_the_chosen_seed(tmp_path, monkeypatch):
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "group_results": {
                    "type1_city": [], "type2_open": [], "type3_mountain": [],
                    "type4_village": [],
                    "type5_warehouse": [
                        {"seed": 77, "success": False, "score": 0.1, "sim_time": 9.9,
                         "execution_status": "timeout"},
                    ],
                    "type6_forest": [],
                }
            }
        )
    )
    captured: dict = {}
    monkeypatch.setattr(visualize_map, "main", lambda argv: captured.setdefault("argv", argv))

    assert cli.main(
        ["visualize", "--summary-json", str(summary), "--failed-index", "1"]
    ) == 0
    assert "--type" in captured["argv"]
    assert captured["argv"][captured["argv"].index("--type") + 1] == "5"
    assert "--seed" in captured["argv"]
    assert captured["argv"][captured["argv"].index("--seed") + 1] == "77"


# --------------------------------------------------------------------------
# swarm video
# --------------------------------------------------------------------------


def test_video_requires_model_to_exist(tmp_path):
    missing = tmp_path / "no_such_model.zip"
    assert cli.main(["video", "--model", str(missing), "--seed", "1", "--type", "1"]) == 1


def test_video_requires_seed_or_seed_file(tmp_path):
    model = tmp_path / "model.zip"
    model.write_bytes(b"not a real zip, existence is all that's checked here")
    assert cli.main(["video", "--model", str(model)]) == 1


def test_video_dispatches_with_family(tmp_path, monkeypatch):
    model = tmp_path / "model.zip"
    model.write_bytes(b"placeholder")
    captured: dict = {}
    monkeypatch.setattr(generate_video, "main", lambda argv: captured.setdefault("argv", argv))

    assert cli.main(
        ["video", "--model", str(model), "--seed", "5", "--type", "4",
         "--family-id", "cf_search_and_rescue"]
    ) == 0
    assert "--family-id" in captured["argv"]
    assert captured["argv"][captured["argv"].index("--family-id") + 1] == "cf_search_and_rescue"


# --------------------------------------------------------------------------
# family-aware task construction (scripts.generate_video.build_task)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "family_id,expected_drones",
    [
        ("cf_autopilot", 1),
        ("cf_interceptor", 1),
        ("cf_search_and_rescue", 1),
        ("cf_swarm_sar", 5),
        ("cf_swarm_autopilot", 5),
    ],
)
def test_build_task_is_family_aware(family_id, expected_drones):
    task = generate_video.build_task(12345, 1, family_id=family_id)
    assert task.family_id == family_id
    assert getattr(task, "num_drones", 1) == expected_drones


def test_build_task_defaults_to_autopilot():
    task = generate_video.build_task(999, 2)
    assert task.family_id == "cf_autopilot"


# --------------------------------------------------------------------------
# seed files written by `swarm benchmark --save-seed-file`
# --------------------------------------------------------------------------


def _write_real_seed_file(path, family_id="cf_autopilot"):
    """Write a seed file exactly as the benchmark does, envelope and all."""
    from swarm.benchmark.engine_parts.seeds import _save_type_seeds, family_bench_groups

    groups = {g: [1000 + i] for i, g in enumerate(family_bench_groups(family_id))}
    _save_type_seeds(path, groups, family_id=family_id)
    return groups


def test_seed_file_from_the_benchmark_is_readable(tmp_path):
    """`--save-seed-file` writes {schema_version, family_id, type_seeds}, not a bare
    group map -- reading it as a bare map silently yields zero jobs.
    """
    seed_file = tmp_path / "bench_seeds.json"
    groups = _write_real_seed_file(seed_file)

    jobs = generate_video._load_seed_jobs(seed_file, family_id="cf_autopilot")

    assert len(jobs) == sum(len(v) for v in groups.values())
    assert {j.seed for j in jobs} == {s for v in groups.values() for s in v}


def test_seed_lookup_finds_a_seed_in_a_real_seed_file(tmp_path):
    seed_file = tmp_path / "bench_seeds.json"
    groups = _write_real_seed_file(seed_file)
    warehouse_seed = groups["type5_warehouse"][0]

    assert cli._lookup_seed_type_in_seed_file(seed_file, warehouse_seed) == 5


def test_seed_file_family_mismatch_is_rejected(tmp_path):
    seed_file = tmp_path / "bench_seeds.json"
    _write_real_seed_file(seed_file, family_id="cf_autopilot")

    with pytest.raises(ValueError, match="family_id mismatch"):
        generate_video._load_seed_jobs(seed_file, family_id="cf_interceptor")


def test_video_rejects_seed_file_combined_with_seed(tmp_path, monkeypatch):
    model = tmp_path / "model.zip"
    model.write_bytes(b"placeholder")
    seed_file = tmp_path / "bench_seeds.json"
    _write_real_seed_file(seed_file)
    monkeypatch.setattr(generate_video, "main", lambda argv: pytest.fail("must not launch"))

    assert cli.main(
        ["video", "--model", str(model), "--seed-file", str(seed_file), "--seed", "42", "--type", "1"]
    ) == 1


@pytest.mark.parametrize("module", [generate_video, visualize_map])
def test_an_unknown_family_is_rejected_at_parse_time(module):
    """The task sampler accepts any string, and the failure only surfaces later when
    the env is built -- so the argument parser is what has to catch a typo.
    """
    parser = module._build_parser()
    argv = (
        ["--model", "x.zip", "--seed", "1", "--type", "1", "--family-id", "cf_typo"]
        if module is generate_video
        else ["--type", "1", "--family-id", "cf_typo"]
    )
    with pytest.raises(SystemExit):
        parser.parse_args(argv)


# --------------------------------------------------------------------------
# env construction (heavy: spins up a real PyBullet world; opt-in via --run-full)
# --------------------------------------------------------------------------


@pytest.mark.full
@pytest.mark.parametrize(
    "family_id,expected_sar_mode,expected_speed_limit",
    [
        ("cf_autopilot", False, "SPEED_LIMIT"),
        ("cf_search_and_rescue", True, "SPEED_LIMIT"),
        ("cf_interceptor", False, "INTERCEPTOR_MINER_SPEED"),
    ],
)
def test_visualizer_env_matches_family_runtime(
    family_id, expected_sar_mode, expected_speed_limit
):
    """sar_mode and the speed limit come from the family runtime, not a fixed
    default -- a hand-rolled env builder that skips this silently mis-renders
    search-and-rescue (no victim mode) and interceptor (wrong flight speed).
    """
    from swarm.constants import INTERCEPTOR_MINER_SPEED, SPEED_LIMIT

    task = generate_video.build_task(555, 1, family_id=family_id)
    env, _backend = visualize_map._build_visualizer_env(task, prefer_gpu=False)
    try:
        assert env.sar_mode is expected_sar_mode
        expected = INTERCEPTOR_MINER_SPEED if expected_speed_limit == "INTERCEPTOR_MINER_SPEED" else SPEED_LIMIT
        assert env.SPEED_LIMIT == expected
    finally:
        env.close()


# --------------------------------------------------------------------------
# `swarm report` finds the log `swarm benchmark` actually wrote
# --------------------------------------------------------------------------


def _write_bench_log(path, seeds=8):
    path.write_text(
        "=== BENCHMARK RESULTS ===\n"
        f"Seeds evaluated: {seeds}\n"
        "Workers used: 4\n"
        "Total wall-clock: 12.0s\n"
    )


def test_report_picks_up_the_per_run_log(tmp_path, monkeypatch, capsys):
    """The engine stamps uid+pid into the log name, so the fixed default never matched."""
    monkeypatch.setattr(cli, "DEFAULT_BENCH_LOG", tmp_path / "bench_full_eval.log")
    written = tmp_path / f"bench_full_eval_{os.getuid()}_4242.log"
    _write_bench_log(written)

    assert cli.main(["report"]) == 0
    assert str(written) in capsys.readouterr().out


def test_report_prefers_the_newest_run(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "DEFAULT_BENCH_LOG", tmp_path / "bench_full_eval.log")
    old = tmp_path / f"bench_full_eval_{os.getuid()}_1.log"
    new = tmp_path / f"bench_full_eval_{os.getuid()}_2.log"
    _write_bench_log(old, seeds=4)
    _write_bench_log(new, seeds=9)
    os.utime(old, (1_000_000, 1_000_000))
    os.utime(new, (2_000_000, 2_000_000))

    assert cli.main(["report"]) == 0
    assert str(new) in capsys.readouterr().out


def test_report_explicit_input_still_wins(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "DEFAULT_BENCH_LOG", tmp_path / "bench_full_eval.log")
    _write_bench_log(tmp_path / f"bench_full_eval_{os.getuid()}_9.log")
    chosen = tmp_path / "mine.log"
    _write_bench_log(chosen)

    assert cli.main(["report", "--input", str(chosen)]) == 0
    assert str(chosen) in capsys.readouterr().out


def test_report_ignores_another_users_log(tmp_path, monkeypatch, capsys):
    """/tmp is shared, and the uid in the filename is what keeps runs apart."""
    monkeypatch.setattr(cli, "DEFAULT_BENCH_LOG", tmp_path / "bench_full_eval.log")
    _write_bench_log(tmp_path / f"bench_full_eval_{os.getuid() + 1}_7.log")

    assert cli.main(["report"]) == 1
    assert "Run `swarm benchmark` first" in capsys.readouterr().err


def test_report_without_any_log_explains_itself(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "DEFAULT_BENCH_LOG", tmp_path / "bench_full_eval.log")
    assert cli.main(["report"]) == 1
    assert "Run `swarm benchmark` first" in capsys.readouterr().err
