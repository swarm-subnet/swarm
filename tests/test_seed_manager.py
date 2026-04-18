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


def _write_epoch_file(seeds_dir, epoch: int, seeds: list[int], *, published: bool = False) -> None:
    seeds_dir.mkdir(parents=True, exist_ok=True)
    (seeds_dir / f"epoch_{epoch}.json").write_text(json.dumps({
        "epoch_number": epoch,
        "seeds": seeds,
        "published": published,
    }))


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


def test_manager_cold_boot_starts_at_epoch_zero(seed_manager_module, monkeypatch, tmp_path):
    m = seed_manager_module
    _patch_paths(monkeypatch, m, tmp_path)

    manager = m.BenchmarkSeedManager()
    assert manager.epoch_number == 0
    assert manager.get_all_seeds() == []
    assert manager.get_pending_publications() == []


def test_manager_adopts_highest_local_epoch_on_boot(seed_manager_module, monkeypatch, tmp_path):
    m = seed_manager_module
    seeds_dir = _patch_paths(monkeypatch, m, tmp_path)
    monkeypatch.setattr(m, "BENCHMARK_TOTAL_SEED_COUNT", 4)
    monkeypatch.setattr(m, "BENCHMARK_SCREENING_SEED_COUNT", 1)

    _write_epoch_file(seeds_dir, 51, [111, 222, 333, 444], published=True)

    manager = m.BenchmarkSeedManager()
    assert manager.epoch_number == 51
    assert manager.get_all_seeds() == [111, 222, 333, 444]
    assert manager.current_epoch_requires_state_invalidation is False


def test_manager_picks_latest_epoch_when_multiple_files_present(seed_manager_module, monkeypatch, tmp_path):
    m = seed_manager_module
    seeds_dir = _patch_paths(monkeypatch, m, tmp_path)
    monkeypatch.setattr(m, "BENCHMARK_TOTAL_SEED_COUNT", 2)
    monkeypatch.setattr(m, "BENCHMARK_SCREENING_SEED_COUNT", 1)

    _write_epoch_file(seeds_dir, 3, [1, 2], published=True)
    _write_epoch_file(seeds_dir, 7, [9, 8], published=True)
    _write_epoch_file(seeds_dir, 5, [4, 5], published=True)

    manager = m.BenchmarkSeedManager()
    assert manager.epoch_number == 7
    assert manager.get_all_seeds() == [9, 8]


def test_manager_regenerates_seeds_when_file_is_corrupt(seed_manager_module, monkeypatch, tmp_path):
    m = seed_manager_module
    seeds_dir = _patch_paths(monkeypatch, m, tmp_path)
    monkeypatch.setattr(m, "BENCHMARK_TOTAL_SEED_COUNT", 3)
    monkeypatch.setattr(m, "BENCHMARK_SCREENING_SEED_COUNT", 1)

    seeds_dir.mkdir(parents=True, exist_ok=True)
    (seeds_dir / "epoch_8.json").write_text("{not-json")

    manager = m.BenchmarkSeedManager()
    assert manager.epoch_number == 8
    assert len(manager.get_all_seeds()) == 3
    assert manager.current_epoch_requires_state_invalidation is True


def test_pending_publications_and_mark_published(seed_manager_module, monkeypatch, tmp_path):
    m = seed_manager_module
    seeds_dir = _patch_paths(monkeypatch, m, tmp_path)
    monkeypatch.setattr(m, "BENCHMARK_TOTAL_SEED_COUNT", 4)
    monkeypatch.setattr(m, "BENCHMARK_SCREENING_SEED_COUNT", 1)

    _write_epoch_file(seeds_dir, 1, [10, 20, 30, 40], published=False)
    _write_epoch_file(seeds_dir, 12, [1, 2, 3, 4], published=True)

    manager = m.BenchmarkSeedManager()
    pending = manager.get_pending_publications()
    assert any(item.get("epoch_number") == 1 for item in pending)

    manager.mark_epoch_published(1)
    data = json.loads((seeds_dir / "epoch_1.json").read_text())
    assert data["published"] is True
    assert all(item.get("epoch_number") != 1 for item in manager.get_pending_publications())


def test_epoch_time_range_returns_utc_datetimes(seed_manager_module, monkeypatch, tmp_path):
    m = seed_manager_module
    _patch_paths(monkeypatch, m, tmp_path)

    manager = m.BenchmarkSeedManager()
    start, end = manager.epoch_time_range(42)
    assert start.tzinfo == timezone.utc
    assert end.tzinfo == timezone.utc
    assert (end - start).total_seconds() == m.EPOCH_DURATION_SECONDS


def test_align_to_epoch_from_cold_boot_sets_epoch_and_generates_seeds(
    seed_manager_module, monkeypatch, tmp_path
):
    m = seed_manager_module
    seeds_dir = _patch_paths(monkeypatch, m, tmp_path)
    monkeypatch.setattr(m, "BENCHMARK_TOTAL_SEED_COUNT", 3)
    monkeypatch.setattr(m, "BENCHMARK_SCREENING_SEED_COUNT", 1)

    manager = m.BenchmarkSeedManager()
    assert manager.epoch_number == 0

    aligned_from = manager.align_to_epoch(9)
    assert aligned_from == 0
    assert manager.epoch_number == 9
    assert len(manager.get_all_seeds()) == 3
    assert (seeds_dir / "epoch_9.json").exists()


def test_align_to_epoch_switches_local_epoch_and_preserves_pending_publications(
    seed_manager_module, monkeypatch, tmp_path
):
    m = seed_manager_module
    seeds_dir = _patch_paths(monkeypatch, m, tmp_path)
    monkeypatch.setattr(m, "BENCHMARK_TOTAL_SEED_COUNT", 4)
    monkeypatch.setattr(m, "BENCHMARK_SCREENING_SEED_COUNT", 1)

    _write_epoch_file(seeds_dir, 10, [1, 2, 3, 4], published=False)

    manager = m.BenchmarkSeedManager()
    old_epoch = manager.epoch_number
    aligned_from = manager.align_to_epoch(15)

    assert aligned_from == old_epoch
    assert manager.epoch_number == 15
    assert (seeds_dir / "epoch_15.json").exists()
    assert any(item.get("epoch_number") == 10 for item in manager.get_pending_publications())


def test_align_to_epoch_only_ignores_same_or_invalid_targets(seed_manager_module, monkeypatch, tmp_path):
    m = seed_manager_module
    seeds_dir = _patch_paths(monkeypatch, m, tmp_path)
    monkeypatch.setattr(m, "BENCHMARK_TOTAL_SEED_COUNT", 2)
    monkeypatch.setattr(m, "BENCHMARK_SCREENING_SEED_COUNT", 1)

    _write_epoch_file(seeds_dir, 20, [5, 6], published=True)

    manager = m.BenchmarkSeedManager()
    assert manager.epoch_number == 20
    assert manager.align_to_epoch(20) is None
    assert manager.align_to_epoch(0) is None
    assert manager.epoch_number == 20


def test_align_to_epoch_realigns_backward_when_local_is_ahead(seed_manager_module, monkeypatch, tmp_path):
    m = seed_manager_module
    seeds_dir = _patch_paths(monkeypatch, m, tmp_path)
    monkeypatch.setattr(m, "BENCHMARK_TOTAL_SEED_COUNT", 3)
    monkeypatch.setattr(m, "BENCHMARK_SCREENING_SEED_COUNT", 1)

    _write_epoch_file(seeds_dir, 9999, [1, 2, 3], published=True)

    manager = m.BenchmarkSeedManager()
    assert manager.epoch_number == 9999

    aligned_from = manager.align_to_epoch(40)
    assert aligned_from == 9999
    assert manager.epoch_number == 40
    assert (seeds_dir / "epoch_40.json").exists()
