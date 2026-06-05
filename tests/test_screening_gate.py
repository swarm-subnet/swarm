import types

import pytest

from swarm.constants import BENCHMARK_VERSION, COPY_SD_MAX
from swarm.validator.utils_parts import screening_gate
from swarm.validator.utils_parts.screening_gate import (
    cache_screening_seed_scores,
    cannot_reach_bar,
    champion_seed_reference,
    copy_metrics,
    is_blatant_copy,
)

BAR = 0.888  # champion 0.873 + 0.015 margin
Z100 = 3.9
CHAMPION = [0.95 if i % 3 else 0.45 for i in range(120)]


@pytest.fixture(autouse=True)
def _tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(screening_gate, "_CACHE_FILE", tmp_path / "screening.json")


def _validator(current_top):
    return types.SimpleNamespace(
        backend_api=types.SimpleNamespace(current_top=current_top)
    )


def test_clear_loser_is_cut():
    assert cannot_reach_bar([0.5] * 100, BAR, Z100) is True


def test_contender_survives():
    assert cannot_reach_bar([0.85] * 100, BAR, Z100) is False


def test_borderline_below_bar_is_cut():
    assert cannot_reach_bar([0.80] * 100, BAR, Z100) is True


def test_high_variance_is_not_cut_on_wide_interval():
    unlucky = [1.0] * 85 + [0.0] * 15  # mean 0.85, wide spread
    assert cannot_reach_bar(unlucky, BAR, Z100) is False


def test_empty_scores_never_cut():
    assert cannot_reach_bar([], BAR, Z100) is False


def test_exact_copy_is_flagged():
    corr, sd_gap, mean_gap = copy_metrics(list(CHAMPION), CHAMPION)
    assert is_blatant_copy(corr, sd_gap, mean_gap, len(CHAMPION)) is True


def test_near_identical_copy_is_flagged():
    candidate = [c + (0.004 if i % 2 else -0.004) for i, c in enumerate(CHAMPION)]
    corr, sd_gap, mean_gap = copy_metrics(candidate, CHAMPION)
    assert is_blatant_copy(corr, sd_gap, mean_gap, len(candidate)) is True


def test_independent_model_is_not_a_copy():
    candidate = [0.45 if i % 3 else 0.95 for i in range(120)]
    corr, sd_gap, mean_gap = copy_metrics(candidate, CHAMPION)
    assert is_blatant_copy(corr, sd_gap, mean_gap, len(candidate)) is False


def test_half_identical_half_different_is_not_a_copy():
    candidate = [
        CHAMPION[i] if i < 60 else CHAMPION[i] + (0.3 if i % 2 else -0.3)
        for i in range(120)
    ]
    corr, sd_gap, mean_gap = copy_metrics(candidate, CHAMPION)
    assert sd_gap > COPY_SD_MAX
    assert is_blatant_copy(corr, sd_gap, mean_gap, len(candidate)) is False


def test_too_few_seeds_is_not_a_copy():
    corr, sd_gap, mean_gap = copy_metrics(CHAMPION[:50], CHAMPION[:50])
    assert is_blatant_copy(corr, sd_gap, mean_gap, 50) is False


def _record(model_hash, epoch, seeds, scores, family_id="cf_autopilot"):
    cache_screening_seed_scores(
        model_hash=model_hash, family_id=family_id, epoch=epoch,
        benchmark_version=BENCHMARK_VERSION, seeds=seeds, scores=scores,
    )


def test_reference_returns_scores_for_current_champion():
    _record("abc", 10, [11, 22, 33], [0.9, 0.8, 0.7])
    v = _validator({"family_id": "cf_autopilot", "model_hash": "abc"})
    assert champion_seed_reference(v, "cf_autopilot", 10, [11, 22, 33]) == {
        11: 0.9, 22: 0.8, 33: 0.7,
    }


def test_reference_skips_on_hash_mismatch():
    _record("OLD", 10, [11, 22], [0.9, 0.8])
    v = _validator({"family_id": "cf_autopilot", "model_hash": "NEW"})
    assert champion_seed_reference(v, "cf_autopilot", 10, [11, 22]) is None


def test_reference_skips_on_epoch_mismatch():
    _record("abc", 9, [11], [0.9])
    v = _validator({"family_id": "cf_autopilot", "model_hash": "abc"})
    assert champion_seed_reference(v, "cf_autopilot", 10, [11]) is None


def test_reference_skips_on_wrong_family():
    _record("abc", 10, [11], [0.9])
    v = _validator({"family_id": "cf_search_and_rescue", "model_hash": "abc"})
    assert champion_seed_reference(v, "cf_autopilot", 10, [11]) is None


def test_reference_none_without_cache():
    v = _validator({"family_id": "cf_autopilot", "model_hash": "abc"})
    assert champion_seed_reference(v, "cf_autopilot", 10, [11]) is None


def test_cache_survives_restart():
    # The cache is on disk and read fresh on each lookup, so a fresh validator
    # object (as after a restart) still resolves the scores.
    _record("abc", 10, [11, 22], [0.9, 0.8])
    fresh = _validator({"family_id": "cf_autopilot", "model_hash": "abc"})
    assert champion_seed_reference(fresh, "cf_autopilot", 10, [11, 22]) == {
        11: 0.9, 22: 0.8,
    }


def test_cache_merges_partial_coverage():
    _record("abc", 10, [11, 22], [0.9, 0.8])
    _record("abc", 10, [33], [0.7])
    v = _validator({"family_id": "cf_autopilot", "model_hash": "abc"})
    assert champion_seed_reference(v, "cf_autopilot", 10, [11, 22, 33]) == {
        11: 0.9, 22: 0.8, 33: 0.7,
    }


def test_cache_prunes_previous_epoch():
    _record("old", 9, [11], [0.9])
    _record("new", 10, [22], [0.8])
    v_old = _validator({"family_id": "cf_autopilot", "model_hash": "old"})
    assert champion_seed_reference(v_old, "cf_autopilot", 9, [11]) is None
    v_new = _validator({"family_id": "cf_autopilot", "model_hash": "new"})
    assert champion_seed_reference(v_new, "cf_autopilot", 10, [22]) == {22: 0.8}


def test_reference_safe_on_unparseable_cache():
    screening_gate._CACHE_FILE.write_text("not json {")
    v = _validator({"family_id": "cf_autopilot", "model_hash": "abc"})
    assert champion_seed_reference(v, "cf_autopilot", 10, [11]) is None


def test_reference_safe_on_wrong_shape_cache():
    screening_gate._CACHE_FILE.write_text("[]")
    v = _validator({"family_id": "cf_autopilot", "model_hash": "abc"})
    assert champion_seed_reference(v, "cf_autopilot", 10, [11]) is None
