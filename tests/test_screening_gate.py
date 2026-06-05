import types

from swarm.constants import BENCHMARK_VERSION, COPY_SD_MAX
from swarm.validator.utils_parts.screening_gate import (
    cannot_reach_bar,
    champion_reference,
    copy_metrics,
    is_blatant_copy,
    record_champion_seed_scores,
)

BAR = 0.888  # champion 0.873 + 0.015 margin
Z100 = 3.9
CHAMPION = [0.95 if i % 3 else 0.45 for i in range(120)]


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


def _validator(current_top):
    return types.SimpleNamespace(backend_api=types.SimpleNamespace(current_top=current_top))


def test_champion_reference_returns_scores_when_current():
    v = _validator({"family_id": "cf_autopilot", "model_hash": "abc"})
    record_champion_seed_scores(
        v, family_id="cf_autopilot", epoch=10, model_hash="abc",
        benchmark_version=BENCHMARK_VERSION, seeds=[11, 22, 33], scores=[0.9, 0.8, 0.7],
    )
    ref = champion_reference(v, "cf_autopilot", 10, [11, 22, 33])
    assert ref == {11: 0.9, 22: 0.8, 33: 0.7}


def test_champion_reference_skips_on_hash_mismatch():
    v = _validator({"family_id": "cf_autopilot", "model_hash": "NEW"})
    record_champion_seed_scores(
        v, family_id="cf_autopilot", epoch=10, model_hash="OLD",
        benchmark_version=BENCHMARK_VERSION, seeds=[11, 22], scores=[0.9, 0.8],
    )
    assert champion_reference(v, "cf_autopilot", 10, [11, 22]) is None


def test_champion_reference_skips_on_epoch_mismatch():
    v = _validator({"family_id": "cf_autopilot", "model_hash": "abc"})
    record_champion_seed_scores(
        v, family_id="cf_autopilot", epoch=9, model_hash="abc",
        benchmark_version=BENCHMARK_VERSION, seeds=[11], scores=[0.9],
    )
    assert champion_reference(v, "cf_autopilot", 10, [11]) is None


def test_champion_reference_none_without_cache():
    v = _validator({"family_id": "cf_autopilot", "model_hash": "abc"})
    assert champion_reference(v, "cf_autopilot", 10, [11]) is None


def test_record_merges_partial_reeval_coverage():
    v = _validator({"family_id": "cf_autopilot", "model_hash": "abc"})
    record_champion_seed_scores(
        v, family_id="cf_autopilot", epoch=10, model_hash="abc",
        benchmark_version=BENCHMARK_VERSION, seeds=[11, 22], scores=[0.9, 0.8],
    )
    record_champion_seed_scores(
        v, family_id="cf_autopilot", epoch=10, model_hash="abc",
        benchmark_version=BENCHMARK_VERSION, seeds=[33], scores=[0.7],
    )
    ref = champion_reference(v, "cf_autopilot", 10, [11, 22, 33])
    assert ref == {11: 0.9, 22: 0.8, 33: 0.7}
