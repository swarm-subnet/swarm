"""A late or cached sync payload must never overwrite newer backend weights."""

from swarm.validator.utils import accept_sync_version


class _Validator:
    pass


def test_stale_payload_is_ignored_and_newer_one_applies():
    validator = _Validator()

    assert accept_sync_version(validator, {"leaderboard_version": 7}) is True
    # A response that lost the race carries an older version.
    assert accept_sync_version(validator, {"leaderboard_version": 5}) is False
    assert accept_sync_version(validator, {"leaderboard_version": 8}) is True


def test_offline_fallback_does_not_reset_applied_weights():
    validator = _Validator()
    accept_sync_version(validator, {"leaderboard_version": 12})

    # The offline fallback reports version 0 while replaying cached kings.
    assert accept_sync_version(validator, {"leaderboard_version": 0}) is False
    assert validator._applied_leaderboard_version == 12


def test_missing_or_malformed_version_is_treated_as_zero():
    validator = _Validator()

    assert accept_sync_version(validator, {}) is True
    accept_sync_version(validator, {"leaderboard_version": 3})
    assert accept_sync_version(validator, {"leaderboard_version": "not-a-number"}) is False
