from __future__ import annotations

from types import SimpleNamespace

from swarm.validator import forward as forward_mod


def test_invalidate_local_state_for_regenerated_seeds(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(forward_mod, "clear_normal_model_queue", lambda: calls.append("queue"))
    monkeypatch.setattr(forward_mod, "clear_benchmark_cache", lambda: calls.append("cache"))

    validator = SimpleNamespace(
        seed_manager=SimpleNamespace(current_epoch_requires_state_invalidation=True)
    )

    forward_mod._invalidate_local_state_for_regenerated_seeds(validator)

    assert calls == ["queue", "cache"]
    assert validator.seed_manager.current_epoch_requires_state_invalidation is False


def test_invalidate_local_state_for_regenerated_seeds_noop_when_not_needed(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(forward_mod, "clear_normal_model_queue", lambda: calls.append("queue"))
    monkeypatch.setattr(forward_mod, "clear_benchmark_cache", lambda: calls.append("cache"))

    validator = SimpleNamespace(
        seed_manager=SimpleNamespace(current_epoch_requires_state_invalidation=False)
    )

    forward_mod._invalidate_local_state_for_regenerated_seeds(validator)

    assert calls == []


def test_partition_assigned_tasks_splits_by_phase() -> None:
    sync_data = {
        "assigned_tasks": [
            {
                "uid": 10,
                "phase": "REEVAL",
                "model_hash": "a" * 64,
                "github_url": "https://github.com/example/champ",
                "reeval_reason": "version_transition",
            },
            {
                "uid": 20,
                "phase": "SCREENING",
                "model_hash": "b" * 64,
                "github_url": "https://github.com/example/pending-s",
            },
            {
                "uid": 30,
                "phase": "BENCHMARK",
                "model_hash": "c" * 64,
                "github_url": "https://github.com/example/pending-b",
            },
        ]
    }

    reeval, pending = forward_mod._partition_assigned_tasks(sync_data)

    assert reeval == [{
        "uid": 10,
        "reason": "version_transition",
        "model_hash": "a" * 64,
        "github_url": "https://github.com/example/champ",
    }]
    assert pending == [
        {"uid": 20, "model_hash": "b" * 64, "github_url": "https://github.com/example/pending-s"},
        {"uid": 30, "model_hash": "c" * 64, "github_url": "https://github.com/example/pending-b"},
    ]


def test_partition_assigned_tasks_dedupes_pending_uid() -> None:
    sync_data = {
        "assigned_tasks": [
            {
                "uid": 42,
                "phase": "SCREENING",
                "model_hash": "a" * 64,
                "github_url": "https://github.com/example/first",
            },
            {
                "uid": 42,
                "phase": "BENCHMARK",
                "model_hash": "a" * 64,
                "github_url": "https://github.com/example/first",
            },
        ]
    }

    _, pending = forward_mod._partition_assigned_tasks(sync_data)

    assert len(pending) == 1
    assert pending[0]["uid"] == 42


def test_partition_assigned_tasks_defaults_missing_reeval_reason() -> None:
    sync_data = {
        "assigned_tasks": [
            {
                "uid": 11,
                "phase": "REEVAL",
                "model_hash": "a" * 64,
                "github_url": "https://github.com/example/x",
            },
        ]
    }

    reeval, _ = forward_mod._partition_assigned_tasks(sync_data)

    assert reeval[0]["reason"] == "version_transition"


def test_partition_assigned_tasks_skips_incomplete_entries() -> None:
    sync_data = {
        "assigned_tasks": [
            {"uid": None, "phase": "SCREENING", "model_hash": "x", "github_url": "y"},
            {"uid": 5, "phase": "SCREENING", "model_hash": "", "github_url": "y"},
            {"uid": 6, "phase": "SCREENING", "model_hash": "x", "github_url": ""},
            {"uid": 7, "phase": "UNKNOWN", "model_hash": "x", "github_url": "y"},
        ]
    }

    reeval, pending = forward_mod._partition_assigned_tasks(sync_data)

    assert reeval == []
    assert pending == []


def test_partition_assigned_tasks_handles_missing_key() -> None:
    reeval, pending = forward_mod._partition_assigned_tasks({})
    assert reeval == []
    assert pending == []


def test_hash_backend_models_returns_path_hash_url(monkeypatch, tmp_path) -> None:
    zip_path = tmp_path / "UID_42.zip"
    zip_path.write_bytes(b"payload")
    monkeypatch.setattr(forward_mod, "sha256sum", lambda _path: "deadbeef")

    result = forward_mod._hash_backend_models({
        42: (zip_path, "https://github.com/example/repo"),
    })

    assert result == {42: (zip_path, "deadbeef", "https://github.com/example/repo")}


def test_hash_backend_models_skips_failing_hash(monkeypatch, tmp_path) -> None:
    good_path = tmp_path / "UID_1.zip"
    good_path.write_bytes(b"ok")
    bad_path = tmp_path / "UID_2.zip"
    bad_path.write_bytes(b"bad")

    def _fake_hash(path):
        if path == bad_path:
            raise RuntimeError("boom")
        return "goodhash"

    monkeypatch.setattr(forward_mod, "sha256sum", _fake_hash)

    result = forward_mod._hash_backend_models({
        1: (good_path, "https://github.com/example/a"),
        2: (bad_path, "https://github.com/example/b"),
    })

    assert result == {1: (good_path, "goodhash", "https://github.com/example/a")}


def test_hash_backend_models_handles_empty_input() -> None:
    assert forward_mod._hash_backend_models({}) == {}
