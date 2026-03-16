from __future__ import annotations

from datetime import datetime, timezone

from scripts.validator.health.check_validator_health import (
    DEFAULT_HOTKEY,
    DEFAULT_NETUID,
    HealthCheckResult,
    check_validator_health,
    collect_recent_hourly_health_checks,
    format_health_check_error,
    is_validator_healthy,
    parse_args,
    render_hourly_health_table,
    were_last_epochs_healthy,
)


class DummySubtensor:
    def __init__(self, current_block: int) -> None:
        self._current_block = current_block
        self.closed = False

    def get_current_block(self) -> int:
        return self._current_block

    def close(self) -> None:
        self.closed = True


def _install_mocks(
    monkeypatch,
    *,
    current_block: int,
    tempo: int,
    blocks_since: int,
    uid: int | None,
    last_updates: list[int],
):
    subtensor = DummySubtensor(current_block=current_block)

    monkeypatch.setattr(
        "scripts.validator.health.check_validator_health.build_subtensor",
        lambda target: subtensor,
    )
    monkeypatch.setattr(
        "scripts.validator.health.check_validator_health.get_subnet_epoch_state",
        lambda subtensor_obj, netuid, block: (tempo, blocks_since),
    )
    monkeypatch.setattr(
        "scripts.validator.health.check_validator_health.get_uid_for_hotkey",
        lambda subtensor_obj, netuid, hotkey, block: uid,
    )
    monkeypatch.setattr(
        "scripts.validator.health.check_validator_health.get_last_update_vector",
        lambda subtensor_obj, netuid, block: last_updates,
    )
    return subtensor


def test_latest_completed_epoch_is_ok_when_last_update_is_inside_epoch(monkeypatch):
    subtensor = _install_mocks(
        monkeypatch,
        current_block=7756745,
        tempo=360,
        blocks_since=64,
        uid=0,
        last_updates=[7756600],
    )

    result = check_validator_health(
        netuid=124,
        hotkey="hk",
        network="finney",
    )

    assert result.ok is True
    assert result.status == "OK"
    assert result.epoch_start == 7756320
    assert result.epoch_end == 7756680
    assert result.last_update_block == 7756600
    assert "OK: Validator is healthy." in result.message
    assert subtensor.closed is True


def test_latest_completed_epoch_is_error_when_last_update_is_stale(monkeypatch):
    subtensor = _install_mocks(
        monkeypatch,
        current_block=7756745,
        tempo=360,
        blocks_since=64,
        uid=0,
        last_updates=[7722054],
    )

    result = check_validator_health(
        netuid=124,
        hotkey="hk",
        network="finney",
    )

    assert result.ok is False
    assert result.status == "ERROR"
    assert result.epoch_start == 7756320
    assert result.epoch_end == 7756680
    assert result.last_update_block == 7722054
    assert "ERROR: No weights set in the latest completed epoch" in result.message
    assert subtensor.closed is True


def test_current_epoch_mode_checks_against_current_block(monkeypatch):
    subtensor = _install_mocks(
        monkeypatch,
        current_block=7756745,
        tempo=360,
        blocks_since=64,
        uid=0,
        last_updates=[7756730],
    )

    result = check_validator_health(
        netuid=124,
        hotkey="hk",
        network="finney",
        current_epoch=True,
    )

    assert result.ok is True
    assert result.status == "OK"
    assert result.epoch_start == 7756681
    assert result.epoch_end == 7756745
    assert result.last_update_block == 7756730
    assert "current epoch" in result.message
    assert subtensor.closed is True


def test_unregistered_validator_returns_error(monkeypatch):
    subtensor = _install_mocks(
        monkeypatch,
        current_block=100,
        tempo=10,
        blocks_since=3,
        uid=None,
        last_updates=[],
    )

    result = check_validator_health(
        netuid=124,
        hotkey="missing",
        network="finney",
    )

    assert result.ok is False
    assert result.status == "ERROR"
    assert result.validator_uid is None
    assert "not registered" in result.message
    assert subtensor.closed is True


def test_no_completed_epoch_returns_error(monkeypatch):
    subtensor = _install_mocks(
        monkeypatch,
        current_block=0,
        tempo=360,
        blocks_since=0,
        uid=0,
        last_updates=[0],
    )

    result = check_validator_health(
        netuid=124,
        hotkey="hk",
        network="finney",
    )

    assert result.ok is False
    assert result.status == "ERROR"
    assert result.epoch_start == -1
    assert result.epoch_end == -1
    assert "No completed epoch is available" in result.message
    assert subtensor.closed is True


def test_boolean_wrapper_matches_health_status(monkeypatch):
    _install_mocks(
        monkeypatch,
        current_block=7756745,
        tempo=360,
        blocks_since=64,
        uid=0,
        last_updates=[7756600],
    )

    assert (
        is_validator_healthy(
            netuid=124,
            hotkey="hk",
            network="finney",
        )
        is True
    )


def test_historical_block_error_message_suggests_archive():
    message = format_health_check_error(
        RuntimeError('UnknownBlock("State already discarded for 0xabc")')
    )

    assert "Use --network archive" in message


def test_parse_args_uses_hardcoded_defaults():
    args = parse_args([])

    assert args.netuid == DEFAULT_NETUID
    assert args.hotkey == DEFAULT_HOTKEY
    assert args.network == "archive"
    assert args.hours == 10
    assert args.last_epochs is None
    assert args.single_check is False


def test_render_hourly_health_table_includes_status_and_blocks():
    rows = [
        type(
            "Row",
            (),
            {
                "hours_ago": 1,
                "target_time_utc": datetime(2026, 3, 16, 9, 0, tzinfo=timezone.utc),
                "checked_at_block": 123,
                "checked_at_time_utc": datetime(2026, 3, 16, 9, 0, tzinfo=timezone.utc),
                "status": "OK",
                "epoch_start": 100,
                "epoch_end": 200,
                "last_update_block": 150,
                "last_update_time_utc": datetime(2026, 3, 16, 8, 59, tzinfo=timezone.utc),
            },
        )(),
        type(
            "Row",
            (),
            {
                "hours_ago": 0,
                "target_time_utc": datetime(2026, 3, 16, 10, 0, tzinfo=timezone.utc),
                "checked_at_block": 124,
                "checked_at_time_utc": datetime(2026, 3, 16, 10, 0, tzinfo=timezone.utc),
                "status": "ERROR",
                "epoch_start": 201,
                "epoch_end": 300,
                "last_update_block": 150,
                "last_update_time_utc": datetime(2026, 3, 16, 8, 59, tzinfo=timezone.utc),
            },
        )(),
    ]

    table = render_hourly_health_table(rows)

    assert "hrs_ago" in table
    assert "OK" in table
    assert "ERROR" in table
    assert "123" in table
    assert "201-300" in table


def test_collect_recent_hourly_health_checks_returns_requested_rows(monkeypatch):
    subtensor = DummySubtensor(current_block=500)

    monkeypatch.setattr(
        "scripts.validator.health.check_validator_health.build_subtensor",
        lambda target: subtensor,
    )
    monkeypatch.setattr(
        "scripts.validator.health.check_validator_health.find_block_for_target_time",
        lambda subtensor, target_time_utc, current_block: current_block
        - int(target_time_utc.minute / 10),
    )
    monkeypatch.setattr(
        "scripts.validator.health.check_validator_health.get_block_time_utc",
        lambda subtensor_obj, block: datetime(2026, 3, 16, 10, 0, tzinfo=timezone.utc),
    )

    def fake_check(subtensor, netuid, hotkey, checked_at_block, current_epoch):
        return HealthCheckResult(
            ok=True,
            status="OK",
            message="OK",
            netuid=netuid,
            hotkey=hotkey,
            validator_uid=0,
            checked_at_block=checked_at_block,
            epoch_start=100,
            epoch_end=200,
            last_update_block=150,
        )

    monkeypatch.setattr(
        "scripts.validator.health.check_validator_health._check_validator_health_with_subtensor",
        fake_check,
    )
    monkeypatch.setattr(
        "scripts.validator.health.check_validator_health.get_subnet_epoch_state",
        lambda subtensor_obj, netuid, block: (360, 64),
    )
    monkeypatch.setattr(
        "scripts.validator.health.check_validator_health.get_uid_for_hotkey",
        lambda subtensor_obj, netuid, hotkey, block: 0,
    )
    monkeypatch.setattr(
        "scripts.validator.health.check_validator_health.get_last_update_vector",
        lambda subtensor_obj, netuid, block: [7756600],
    )

    rows = collect_recent_hourly_health_checks(
        hours=3,
        now_utc=datetime(2026, 3, 16, 10, 0, tzinfo=timezone.utc),
    )

    assert len(rows) == 3
    assert rows[0].hours_ago == 2
    assert rows[-1].hours_ago == 0
    assert all(row.status == "OK" for row in rows)
    assert subtensor.closed is True


def test_were_last_epochs_healthy_returns_true_when_all_epochs_are_healthy(monkeypatch):
    subtensor = DummySubtensor(current_block=500)

    monkeypatch.setattr(
        "scripts.validator.health.check_validator_health.build_subtensor",
        lambda target: subtensor,
    )
    monkeypatch.setattr(
        "scripts.validator.health.check_validator_health.collect_history",
        lambda **kwargs: [
            type("Epoch", (), {"registered": True, "last_update_in_epoch": True})(),
            type("Epoch", (), {"registered": True, "last_update_in_epoch": True})(),
            type("Epoch", (), {"registered": True, "last_update_in_epoch": True})(),
        ],
    )

    assert were_last_epochs_healthy(epochs=3) is True
    assert subtensor.closed is True


def test_were_last_epochs_healthy_returns_false_when_any_epoch_is_stale(monkeypatch):
    subtensor = DummySubtensor(current_block=500)

    monkeypatch.setattr(
        "scripts.validator.health.check_validator_health.build_subtensor",
        lambda target: subtensor,
    )
    monkeypatch.setattr(
        "scripts.validator.health.check_validator_health.collect_history",
        lambda **kwargs: [
            type("Epoch", (), {"registered": True, "last_update_in_epoch": True})(),
            type("Epoch", (), {"registered": True, "last_update_in_epoch": False})(),
            type("Epoch", (), {"registered": True, "last_update_in_epoch": True})(),
        ],
    )

    assert were_last_epochs_healthy(epochs=3) is False
    assert subtensor.closed is True
