"""The host speed factor is measured once and survives a restart."""
from __future__ import annotations

import time

import pytest

from swarm.validator.calibration import normalize_speed_factor
from swarm.validator.docker.docker_evaluator_parts import batch


VERSION = "cal-1"


@pytest.fixture
def cache_path(monkeypatch, tmp_path):
    path = tmp_path / "host_speed_factor.json"
    monkeypatch.setattr(batch, "_CALIBRATION_CACHE_PATH", path)
    monkeypatch.setattr(batch, "_HOST_SPEED_CALIBRATION", None)
    return path


def _calibration(*, local_p90_ms: float = 200.0, workers: int = 6, age_sec: float = 0.0):
    speed = normalize_speed_factor(local_p90_ms)
    return batch.HostSpeedCalibration(
        speed=speed,
        worker_count=workers,
        worker_speeds=(speed,) * workers,
        calibration_version=VERSION,
        computed_at=time.time() - age_sec,
    )


def _read(*, workers: int = 6, version: str = VERSION):
    return batch._read_calibration_cache(
        worker_count=workers, calibration_version=version,
    )


def test_a_measurement_survives_a_restart(cache_path):
    batch._write_calibration_cache(_calibration())

    loaded = _read()

    assert loaded is not None
    assert loaded.speed.factor == pytest.approx(_calibration().speed.factor)
    assert loaded.worker_count == 6


def test_a_measurement_older_than_the_window_is_ignored(cache_path):
    batch._write_calibration_cache(
        _calibration(age_sec=batch._CALIBRATION_MAX_AGE_SEC + 60)
    )

    assert _read() is None


def test_a_measurement_under_fewer_workers_is_ignored(cache_path):
    """Fewer concurrent workers means a kinder measurement than scoring will see."""
    batch._write_calibration_cache(_calibration(workers=4))

    assert _read(workers=8) is None


def test_a_measurement_from_another_baseline_is_ignored(cache_path):
    batch._write_calibration_cache(_calibration())

    assert _read(version="cal-2") is None


def test_a_measurement_from_another_host_is_ignored(cache_path, monkeypatch):
    batch._write_calibration_cache(_calibration())
    monkeypatch.setattr(batch, "_calibration_host_fingerprint", lambda: "other-host")

    assert _read() is None


def test_an_ineligible_host_is_never_cached(cache_path):
    """One bad reading must not sideline the host for the whole cache window."""
    ineligible = _calibration(local_p90_ms=260.8 * 10)
    assert not ineligible.speed.eligible

    batch._write_calibration_cache(ineligible)

    assert not cache_path.exists()


def test_a_corrupt_cache_is_ignored(cache_path):
    cache_path.write_text("{not json")

    assert _read() is None


def test_a_cached_measurement_means_no_stand_down(cache_path, monkeypatch):
    monkeypatch.setattr(
        batch, "load_baseline_manifest", lambda: {"calibration_version": VERSION},
    )
    batch._write_calibration_cache(_calibration())

    assert batch.host_speed_factor_is_fresh(6) is True
    assert batch._HOST_SPEED_CALIBRATION is not None


def test_an_empty_cache_means_the_host_must_measure(cache_path, monkeypatch):
    monkeypatch.setattr(
        batch, "load_baseline_manifest", lambda: {"calibration_version": VERSION},
    )

    assert batch.host_speed_factor_is_fresh(6) is False
