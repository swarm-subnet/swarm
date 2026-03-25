from __future__ import annotations

import json
from datetime import timezone

import pytest


@pytest.fixture
def seed_manager_module(reload_module):
    return reload_module("swarm.validator.seed_manager")


def _patch_paths(monkeypatch, module, tmp_path):
    state_dir = tmp_path / "state"
    seeds_dir = state_dir / "epoch_seeds"
    monkeypatch.setattr(module, "STATE_DIR", state_dir)
    monkeypatch.setattr(module, "EPOCH_SEEDS_DIR", seeds_dir)
    return seeds_dir


def test_compute_raw_week_uses_epoch_anchor(seed_manager_module):
    m = seed_manager_module
    anchor = m.EPOCH_ANCHOR_UTC.timestamp()
    ts = anchor + (2 * m.EPOCH_DURATION_SECONDS) + 123
    assert m._compute_raw_week(ts) == 2


def test_manager_uses_raw_week_plus_one_for_epoch_number(seed_manager_module, monkeypatch, tmp_path):
    m = seed_manager_module
    _patch_paths(monkeypatch, m, tmp_path)
    monkeypatch.setattr(m, "BENCHMARK_TOTAL_SEED_COUNT", 2)
    monkeypatch.setattr(m, "BENCHMARK_SCREENING_SEED_COUNT", 1)
    monkeypatch.setattr(m, "_compute_raw_week", lambda ts=None: 11)

    manager = m.BenchmarkSeedManager()
    assert manager.epoch_number == 12


def test_generate_random_seeds_returns_correct_count(seed_manager_module):
    m = seed_manager_module
    seeds = m._generate_random_seeds(100)
    assert len(seeds) == 100
    assert all(0 <= s <= m._MAX_SEED for s in seeds)


def test_generate_random_seeds_are_not_identical_across_calls(seed_manager_module):
    m = seed_manager_module
    seeds_a = m._generate_random_seeds(50)
    seeds_b = m._generate_random_seeds(50)
    assert seeds_a != seeds_b


def test_manager_generates_and_splits_seeds(seed_manager_module, monkeypatch, tmp_path):
    m = seed_manager_module
    seeds_dir = _patch_paths(monkeypatch, m, tmp_path)
    monkeypatch.setattr(m, "BENCHMARK_TOTAL_SEED_COUNT", 6)
    monkeypatch.setattr(m, "BENCHMARK_SCREENING_SEED_COUNT", 2)
    monkeypatch.setattr(m, "_compute_raw_week", lambda ts=None: 100)

    manager = m.BenchmarkSeedManager()
    assert manager.epoch_number == 101
    assert len(manager.get_all_seeds()) == 6
    assert len(manager.get_screening_seeds()) == 2
    assert len(manager.get_benchmark_seeds()) == 4
    assert (seeds_dir / "epoch_101.json").exists()
    assert manager.current_epoch_requires_state_invalidation is True


def test_manager_loads_seeds_from_existing_file(seed_manager_module, monkeypatch, tmp_path):
    m = seed_manager_module
    seeds_dir = _patch_paths(monkeypatch, m, tmp_path)
    monkeypatch.setattr(m, "BENCHMARK_TOTAL_SEED_COUNT", 4)
    monkeypatch.setattr(m, "BENCHMARK_SCREENING_SEED_COUNT", 1)
    monkeypatch.setattr(m, "_compute_raw_week", lambda ts=None: 50)

    seeds_dir.mkdir(parents=True, exist_ok=True)
    saved_seeds = [111, 222, 333, 444]
    (seeds_dir / "epoch_51.json").write_text(json.dumps({
        "epoch_number": 51,
        "seeds": saved_seeds,
    }))

    manager = m.BenchmarkSeedManager()
    assert manager.get_all_seeds() == saved_seeds
    assert manager.current_epoch_requires_state_invalidation is False


def test_pending_publications_and_mark_published(seed_manager_module, monkeypatch, tmp_path):
    m = seed_manager_module
    seeds_dir = _patch_paths(monkeypatch, m, tmp_path)
    monkeypatch.setattr(m, "BENCHMARK_TOTAL_SEED_COUNT", 4)
    monkeypatch.setattr(m, "BENCHMARK_SCREENING_SEED_COUNT", 1)
    monkeypatch.setattr(m, "_compute_raw_week", lambda ts=None: 12)

    seeds_dir.mkdir(parents=True, exist_ok=True)
    old_epoch = seeds_dir / "epoch_1.json"
    old_epoch.write_text(json.dumps({"epoch_number": 1, "published": False}))

    manager = m.BenchmarkSeedManager()
    pending = manager.get_pending_publications()
    assert any(item.get("epoch_number") == 1 for item in pending)

    manager.mark_epoch_published(1)
    data = json.loads(old_epoch.read_text())
    assert data["published"] is True
    assert all(item.get("epoch_number") != 1 for item in manager.get_pending_publications())


def test_check_transition_and_advance_epoch(seed_manager_module, monkeypatch, tmp_path):
    m = seed_manager_module
    _patch_paths(monkeypatch, m, tmp_path)
    monkeypatch.setattr(m, "BENCHMARK_TOTAL_SEED_COUNT", 5)
    monkeypatch.setattr(m, "BENCHMARK_SCREENING_SEED_COUNT", 2)

    current_raw = {"value": 20}
    monkeypatch.setattr(m, "_compute_raw_week", lambda ts=None: current_raw["value"])

    manager = m.BenchmarkSeedManager()
    assert manager.epoch_number == 21
    assert manager.check_epoch_transition() is False

    current_raw["value"] = 21
    assert manager.check_epoch_transition() is True
    old_epoch = manager.advance_to_new_epoch()
    assert old_epoch == 21
    assert manager.epoch_number == 22
    assert any(item.get("epoch_number") == 21 for item in manager.get_pending_publications())


def test_epoch_time_range_returns_utc_datetimes(seed_manager_module, monkeypatch, tmp_path):
    m = seed_manager_module
    _patch_paths(monkeypatch, m, tmp_path)
    monkeypatch.setattr(m, "_compute_raw_week", lambda ts=None: 50)

    manager = m.BenchmarkSeedManager()
    start, end = manager.epoch_time_range(manager.epoch_number)
    assert start.tzinfo == timezone.utc
    assert end.tzinfo == timezone.utc
    assert (end - start).total_seconds() == m.EPOCH_DURATION_SECONDS


def test_align_to_epoch_switches_local_epoch_and_preserves_pending_publications(
    seed_manager_module,
    monkeypatch,
    tmp_path,
):
    m = seed_manager_module
    seeds_dir = _patch_paths(monkeypatch, m, tmp_path)
    monkeypatch.setattr(m, "BENCHMARK_TOTAL_SEED_COUNT", 4)
    monkeypatch.setattr(m, "BENCHMARK_SCREENING_SEED_COUNT", 1)
    monkeypatch.setattr(m, "_compute_raw_week", lambda ts=None: 10)

    manager = m.BenchmarkSeedManager()
    old_epoch = manager.epoch_number
    aligned_from = manager.align_to_epoch(15)

    assert aligned_from == old_epoch
    assert manager.epoch_number == 15
    assert (seeds_dir / "epoch_15.json").exists()
