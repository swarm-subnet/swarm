# The MIT License (MIT)
# Copyright © 2026 Swarm

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""The visualizer and the recorder were restored after an accidental deletion
(commit 04a21a0, "SAR v5.0.0") -- these tests lock down the CLI wiring and the
family-aware task construction so a future cleanup pass can't silently drop
them again without a test failing first.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import swarm.challenge_families as challenge_families
import validator.scripts.generate_video as generate_video
import validator.scripts.visualize_map as visualize_map
from swarm import cli
from swarm.challenge_families.solar_patrol import camera as solar_camera
from swarm.constants import SIM_DT

# --------------------------------------------------------------------------
# swarm visualize
# --------------------------------------------------------------------------


def test_visualize_dispatches_with_resolved_type_and_family(monkeypatch):
    """The resolved map type and family are handed to the viewer, at the front of the argv it is launched with."""
    captured: dict = {}

    def _fake_main(argv):
        """Keep the argv the viewer was launched with instead of opening a window."""
        captured["argv"] = argv

    monkeypatch.setattr(visualize_map, "main", _fake_main)

    assert cli.main(["visualize", "--type", "2", "--family-id", "cf_interceptor"]) == 0
    assert captured["argv"][:4] == ["--type", "2", "--family-id", "cf_interceptor"]


def test_visualize_defaults_family_to_search_and_rescue(monkeypatch):
    """With no family named on the command line, the viewer is still told a live one outright: cf_search_and_rescue."""
    captured: dict = {}
    monkeypatch.setattr(visualize_map, "main", lambda argv: captured.setdefault("argv", argv))

    assert cli.main(["visualize", "--type", "1"]) == 0
    assert "--family-id" in captured["argv"]
    assert captured["argv"][captured["argv"].index("--family-id") + 1] == "cf_search_and_rescue"


def test_visualize_requires_type_seed_or_summary(monkeypatch):
    """Nothing to open means a non-zero exit and no viewer launched, rather than an arbitrary map."""
    monkeypatch.setattr(visualize_map, "main", lambda argv: pytest.fail("must not launch"))
    assert cli.main(["visualize"]) == 1


def test_visualize_rejects_mismatched_explicit_type(tmp_path, monkeypatch):
    """A map type contradicting the one the seed was saved under is refused before the viewer starts."""
    seed_file = tmp_path / "seeds.json"
    seed_file.write_text(json.dumps({"type1_city": [42]}))
    monkeypatch.setattr(visualize_map, "main", lambda argv: pytest.fail("must not launch"))

    assert cli.main(
        ["visualize", "--seed", "42", "--seed-file", str(seed_file), "--type", "3"]
    ) == 1


def test_visualize_failed_lists_rows_without_index(tmp_path, monkeypatch, capsys):
    """Asking for the losing seeds with no index prints them and why each one lost, and opens nothing."""
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
    """Picking a losing seed by its printed number opens that seed on the map type it actually ran."""
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
    """A model path pointing at nothing stops the command before any rendering is attempted."""
    missing = tmp_path / "no_such_model.zip"
    assert cli.main(["video", "--model", str(missing), "--seed", "1", "--type", "1"]) == 1


def test_video_requires_seed_or_seed_file(tmp_path):
    """A model given nothing to fly is refused: either one seed and its type, or a whole seed file."""
    model = tmp_path / "model.zip"
    model.write_bytes(b"not a real zip, existence is all that's checked here")
    assert cli.main(["video", "--model", str(model)]) == 1


def test_video_dispatches_with_family(tmp_path, monkeypatch):
    """The recorder is launched with the family it was asked for, so the flight is filmed under those rules."""
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


def test_solar_parsers_accept_type_eight():
    """The public and script parsers accept type 8 for both video and visualization."""
    assert cli.build_parser().parse_args(["visualize", "--type", "8"]).type == 8
    assert cli.build_parser().parse_args(["video", "--model", "x.zip", "--type", "8"]).type == 8
    assert visualize_map._build_parser().parse_args(["--type", "8"]).type == 8
    assert generate_video._build_parser().parse_args(["--model", "x.zip", "--type", "8"]).type == 8


def test_visualize_without_a_family_opens_the_one_that_flies_the_map(tmp_path, monkeypatch):
    """With no family named, map 8 and a failed type-8 seed open Swarm Sentinel; a rescue map keeps the default."""
    opened = []
    monkeypatch.setattr(visualize_map, "main", lambda argv: opened.append(argv[argv.index("--family-id") + 1]))
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"group_results": {"type8_solar": [
        {"seed": 81, "success": False, "score": 0.0, "sim_time": 390.0}
    ]}}))

    assert cli.main(["visualize", "--type", "8"]) == 0
    assert cli.main(["visualize", "--summary-json", str(summary), "--failed-index", "1"]) == 0
    assert cli.main(["visualize", "--type", "2"]) == 0
    assert opened == ["cf_solar_patrol", "cf_solar_patrol", "cf_search_and_rescue"]


def test_video_without_a_family_films_the_one_that_flies_the_map(tmp_path, monkeypatch):
    """A type-8 seed or a Sentinel seed file is filmed as Swarm Sentinel when no family is named."""
    model = tmp_path / "model.zip"
    model.write_bytes(b"placeholder")
    seed_file = tmp_path / "seeds.json"
    seed_file.write_text(json.dumps({
        "schema_version": "challenge_family_seed_file.v1",
        "family_id": "cf_solar_patrol",
        "type_seeds": {"type8_solar": [1002]},
    }))
    filmed = []
    monkeypatch.setattr(generate_video, "main", lambda argv: filmed.append(argv[argv.index("--family-id") + 1]))

    assert cli.main(["video", "--model", str(model), "--seed", "5", "--type", "8"]) == 0
    assert cli.main(["video", "--model", str(model), "--seed-file", str(seed_file)]) == 0
    assert filmed == ["cf_solar_patrol", "cf_solar_patrol"]


def test_a_family_asked_for_a_map_it_never_flies_is_refused():
    """Swarm Sentinel on a mountain map, or a rescue family on the solar park, stops before any world is built."""
    with pytest.raises(ValueError, match="cf_solar_patrol"):
        generate_video.build_task(1002, 3, family_id="cf_solar_patrol")
    with pytest.raises(ValueError, match="cf_search_and_rescue"):
        generate_video.build_task(1002, 8, family_id="cf_search_and_rescue")


def test_the_video_script_loads_and_parses_without_the_simulator():
    """Opening the video script and reading its flags needs neither pybullet nor bittensor."""
    code = (
        "import sys; sys.modules['pybullet'] = None; sys.modules['bittensor'] = None; "
        "import validator.scripts.generate_video as g; "
        "print(g._build_parser().parse_args(['--model', 'x.zip', '--type', '8']).type)"
    )
    repo = Path(__file__).resolve().parents[2]
    result = subprocess.run([sys.executable, "-c", code], cwd=repo, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-800:]
    assert result.stdout.strip() == "8"


def test_solar_failed_summary_is_listed(tmp_path, capsys):
    """A failed type8_solar summary row appears in the visualizer's review list."""
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"group_results": {"type8_solar": [
        {"seed": 81, "success": False, "score": 0.0, "sim_time": 390.0}
    ]}}))
    assert cli.main(["visualize", "--summary-json", str(summary), "--failed"]) == 0
    assert "type 8 (solar)" in capsys.readouterr().out


# --------------------------------------------------------------------------
# family-aware task construction (validator.scripts.generate_video.build_task)
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
    """The sampled task keeps the family it was built for, and the swarm families bring five drones, not one."""
    task = generate_video.build_task(12345, 2, family_id=family_id)
    assert task.family_id == family_id
    assert getattr(task, "num_drones", 1) == expected_drones


def test_build_task_defaults_to_autopilot():
    """A task sampled with no family named comes back as an autopilot one, never with the field unset."""
    task = generate_video.build_task(999, 2)
    assert task.family_id == "cf_autopilot"


def test_solar_build_task_uses_family_builder(monkeypatch):
    """A type-8 patrol is requested from the runtime builder with the original seed and simulation dt."""
    captured = {}
    sentinel = object()

    def fake_builder(**kwargs):
        """Capture task builder arguments without constructing a park."""
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(challenge_families, "build_random_task", fake_builder)

    assert generate_video.build_task(123, 8, family_id="cf_solar_patrol") is sentinel
    assert captured == {"sim_dt": SIM_DT, "seed": 123, "family_id": "cf_solar_patrol"}


def test_solar_decision_clock_and_action_preserve_sticks():
    """A five-step patrol decision advances five ticks and keeps clipped sticks and all observation keys."""
    env = SimpleNamespace(
        family_runtime=SimpleNamespace(decision_steps=5),
        task=SimpleNamespace(family_id="cf_solar_patrol"),
        ACT_TYPE="velocity",
    )
    obs = {key: object() for key in ("state", "rgb", "thermal", "zoom", "depth")}
    seen = []

    class Agent:
        """Remember the exact observation the recorder hands to the policy."""
        def act(self, received):
            """Return sticks whose norm exceeds the legacy speed limit."""
            seen.append(received)
            return [1.0, 1.0, 1.0, 2.0]

    action = generate_video._agent_action(
        Agent(), obs, env, 1, 4,
        np.array([-1.0] * 4), np.array([1.0] * 4), "velocity", 0.5,
    )
    assert seen == [obs]
    np.testing.assert_array_equal(action, [1.0, 1.0, 1.0, 1.0])
    assert generate_video._decision_dt(env, 0.02) == pytest.approx(0.1)


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


def test_solar_cameras_draw_through_the_patrol_renderer(monkeypatch):
    """On a Sentinel env every video camera hands its eye and target to the patrol's colour renderer."""
    calls = []
    monkeypatch.setattr(generate_video, "_solar_frame", lambda env, eye, target, up, w, h, fov: calls.append(
        (env, tuple(np.round(target, 2)), w, h)) or np.zeros((h, w, 3), np.uint8))
    monkeypatch.setattr(solar_camera, "RAYCAST", True)
    env = SimpleNamespace(_solar=object())
    cam = generate_video.ChaseCamera.__new__(generate_video.ChaseCamera)
    generate_video._CameraBase.__init__(cam, 0, 64, 32, 60.0)
    cam._back, cam._up, cam._smooth_fwd = 6.0, 2.0, None
    generate_video._attach_family_renderer({"chase": cam}, env)
    frame = cam.capture(np.array([1.0, 2.0, 20.0]), np.array([0, 0, 0, 1.0]), np.eye(3), 0.04)
    assert frame.shape == (32, 64, 3)
    assert calls == [(env, (1.0, 2.0, 20.15), 64, 32)]


def test_a_decision_spanning_several_frame_slots_is_drawn_once():
    """Three frame slots in one 0.1 s decision draw each camera once, over the whole span, and write it three times."""
    drawn, written = [], {"chase": [], "fpv": []}

    class Camera:
        """Count the draws and the time span each one is asked to cover."""
        def capture(self, drone_pos, drone_quat, rot, dt):
            """Return a blank frame and remember the span it covers."""
            drawn.append(round(dt, 6))
            return np.zeros((2, 2, 3), np.uint8)

    class Writer(list):
        """Collect the frames written to one video."""
        append_data = list.append

    writers = {mode: Writer() for mode in written}
    next_t, count = generate_video._write_due_frames(
        {mode: Camera() for mode in written}, writers, None, None, None, 0.1, 0.0, 0.04)
    assert drawn == [0.12, 0.12]
    assert [len(w) for w in writers.values()] == [3, 3] and count == 3
    assert next_t == pytest.approx(0.12)
    assert generate_video._write_due_frames({}, {}, None, None, None, 0.1, 0.12, 0.04) == (0.12, 0)


def test_the_depth_video_of_a_patrol_uses_the_ray_caster(monkeypatch):
    """On a Sentinel env the depth camera asks for the patrol's ray caster; elsewhere it keeps TinyRenderer."""
    asked = []
    monkeypatch.setattr(generate_video, "_render_depth", lambda *args, raycast=False: asked.append(raycast) or np.ones(
        (generate_video.DEPTH_SENSOR_RES, generate_video.DEPTH_SENSOR_RES), np.float32))
    monkeypatch.setattr(solar_camera, "RAYCAST", True)
    cam = generate_video.DepthCamera(0, 64, 32)
    cam.capture(np.zeros(3), np.array([0, 0, 0, 1.0]), np.eye(3), 0.04)
    generate_video._attach_family_renderer({"depth": cam}, SimpleNamespace(_solar=object()))
    frame = cam.capture(np.zeros(3), np.array([0, 0, 0, 1.0]), np.eye(3), 0.04)
    assert asked == [False, True]
    assert frame.shape == (32, 64, 3)


def test_solar_seed_file_loads_type_eight_job(tmp_path):
    """A benchmark type8_solar seed file becomes a type-8 video job and resolves in visualize."""
    seed_file = tmp_path / "solar_seeds.json"
    seed_file.write_text(json.dumps({
        "schema_version": "challenge_family_seed_file.v1",
        "family_id": "cf_solar_patrol",
        "type_seeds": {"type8_solar": [2468]},
    }))
    assert generate_video._load_seed_jobs(seed_file, family_id="cf_solar_patrol") == [
        generate_video.VideoJob(seed=2468, challenge_type=8)
    ]
    assert cli._lookup_seed_type_in_seed_file(seed_file, 2468, family_id="cf_solar_patrol") == 8


def test_seed_lookup_finds_a_seed_in_a_real_seed_file(tmp_path):
    """A seed stored under the warehouse group resolves back to map type 5 through the file's envelope."""
    seed_file = tmp_path / "bench_seeds.json"
    groups = _write_real_seed_file(seed_file)
    warehouse_seed = groups["type5_warehouse"][0]

    assert cli._lookup_seed_type_in_seed_file(seed_file, warehouse_seed) == 5


def test_seed_file_family_mismatch_is_rejected(tmp_path):
    """Reading a seed file under the wrong family raises, rather than quietly yielding no jobs."""
    seed_file = tmp_path / "bench_seeds.json"
    _write_real_seed_file(seed_file, family_id="cf_autopilot")

    with pytest.raises(ValueError, match="family_id mismatch"):
        generate_video._load_seed_jobs(seed_file, family_id="cf_interceptor")


def test_video_rejects_seed_file_combined_with_seed(tmp_path, monkeypatch):
    """A whole seed file and a single seed together is contradictory, so the recorder is never launched."""
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

    task = generate_video.build_task(555, 2, family_id=family_id)
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
    """Write a log carrying the summary lines the report command parses its fields out of."""
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
    """With several of this user's logs on disk, the most recently written one is what gets summarized."""
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
    """A path named on the command line is read even when an auto-discovered log is sitting beside it."""
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
    """With nothing on disk to summarize, the command fails and says to run the benchmark first."""
    monkeypatch.setattr(cli, "DEFAULT_BENCH_LOG", tmp_path / "bench_full_eval.log")
    assert cli.main(["report"]) == 1
    assert "Run `swarm benchmark` first" in capsys.readouterr().err
