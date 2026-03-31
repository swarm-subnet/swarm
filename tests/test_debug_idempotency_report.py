from __future__ import annotations

import json

from scripts import debug_idempotency_report as debug_report


def _summary_for(seed: int, challenge_type: int, *, score_same: bool = True) -> dict:
    unique_scores = [0.01] if score_same else [0.01, 0.02]
    return {
        "uid": 178,
        "seed": seed,
        "challenge_type": challenge_type,
        "runs_requested": 5,
        "runs": [
            {
                "run": idx,
                "score": 0.01 if score_same or idx < 5 else 0.02,
                "success": False,
                "time_sec": 0.68,
                "wall_time_sec": 7.5,
            }
            for idx in range(1, 6)
        ],
        "unique_scores": unique_scores,
        "unique_success_values": [False],
        "unique_sim_times": [0.68],
        "unique_wall_times": [7.5],
        "idempotent_score": score_same,
        "idempotent_success": True,
        "idempotent_sim_time": True,
        "strict_idempotent": score_same,
    }


def test_resolve_group_seeds_uses_first_seed_from_each_group(monkeypatch):
    monkeypatch.setattr(
        debug_report,
        "_load_type_seeds",
        lambda path: {
            group: [1000 + idx, 2000 + idx]
            for idx, group in enumerate(debug_report.BENCH_GROUP_ORDER, start=1)
        },
    )

    resolved = debug_report._resolve_group_seeds(path := object())

    assert resolved == {
        group: 1000 + idx
        for idx, group in enumerate(debug_report.BENCH_GROUP_ORDER, start=1)
    }
    assert path is not None


def test_main_prints_report_and_writes_json(monkeypatch, tmp_path, capsys):
    model_path = tmp_path / "UID_178.zip"
    model_path.write_bytes(b"zip")
    json_out = tmp_path / "idempotency.json"
    seeds = {
        group: 3000 + idx for idx, group in enumerate(debug_report.BENCH_GROUP_ORDER, start=1)
    }

    monkeypatch.setattr(debug_report, "_resolve_group_seeds", lambda seed_file: seeds)
    monkeypatch.setattr(
        debug_report,
        "run_idempotency",
        lambda **kwargs: _summary_for(kwargs["seed"], kwargs["challenge_type"]),
    )

    rc = debug_report.main(
        [
            "--model",
            str(model_path),
            "--runs-per-map",
            "5",
            "--json-out",
            str(json_out),
        ]
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "=== RESULTS ===" in out
    assert "Group" in out
    assert "Run" in out
    assert "type1_city" in out
    assert "SCORE IDEMP" in out
    payload = json.loads(json_out.read_text())
    assert payload["group_seeds"] == seeds
    assert payload["runs_per_map"] == 5


def test_main_returns_nonzero_when_any_group_is_not_score_idempotent(monkeypatch, tmp_path):
    model_path = tmp_path / "UID_178.zip"
    model_path.write_bytes(b"zip")
    seeds = {
        group: 4000 + idx for idx, group in enumerate(debug_report.BENCH_GROUP_ORDER, start=1)
    }

    def _fake_run_idempotency(**kwargs):
        score_same = kwargs["challenge_type"] != 6
        return _summary_for(kwargs["seed"], kwargs["challenge_type"], score_same=score_same)

    monkeypatch.setattr(debug_report, "_resolve_group_seeds", lambda seed_file: seeds)
    monkeypatch.setattr(debug_report, "run_idempotency", _fake_run_idempotency)

    rc = debug_report.main(["--model", str(model_path)])

    assert rc == 1
