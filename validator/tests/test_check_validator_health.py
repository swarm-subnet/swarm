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

"""Judging a validator by whether it set weights inside an epoch, and the hourly and per-epoch reports built on it."""
from __future__ import annotations

from datetime import datetime, timezone

from validator.scripts.health.check_validator_health import (
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
    """A chain connection stand-in: one fixed block height, and a note of having been closed."""
    def __init__(self, current_block: int) -> None:
        """Hold the block height to report and start out open."""
        self._current_block = current_block
        self.closed = False

    def get_current_block(self) -> int:
        """Report the height this stand-in was built with."""
        return self._current_block

    def close(self) -> None:
        """Record that the caller released the connection."""
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
    """Answer all four chain queries the check makes with fixed values, and return the connection stub."""
    subtensor = DummySubtensor(current_block=current_block)

    monkeypatch.setattr(
        "validator.scripts.health.check_validator_health.build_subtensor",
        lambda target: subtensor,
    )
    monkeypatch.setattr(
        "validator.scripts.health.check_validator_health.get_subnet_epoch_state",
        lambda subtensor_obj, netuid, block: (tempo, blocks_since),
    )
    monkeypatch.setattr(
        "validator.scripts.health.check_validator_health.get_uid_for_hotkey",
        lambda subtensor_obj, netuid, hotkey, block: uid,
    )
    monkeypatch.setattr(
        "validator.scripts.health.check_validator_health.get_last_update_vector",
        lambda subtensor_obj, netuid, block: last_updates,
    )
    return subtensor


def test_latest_completed_epoch_is_ok_when_last_update_is_inside_epoch(monkeypatch):
    """Weights set inside the last finished epoch read as healthy, and that window is reported back with the verdict."""
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
    """A last update from before the window fails the check, however alive the box otherwise looks."""
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
    """Current-epoch mode judges the epoch still running, from its first block up to the one just read."""
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
    """A hotkey holding no UID on the subnet fails with no verdict on weights, and says it is not registered."""
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
    """Before any epoch has closed there is nothing to judge, and the window comes back as -1 rather than 0."""
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
    """The yes-or-no shortcut agrees with the full verdict it wraps, True for a healthy epoch."""
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
    """A node that has pruned the state it was asked for turns into advice to use an archive endpoint."""
    message = format_health_check_error(
        RuntimeError('UnknownBlock("State already discarded for 0xabc")')
    )

    assert "Use --network archive" in message


def test_parse_args_uses_hardcoded_defaults():
    """Run with no flags, the tool aims at subnet 124 through an archive node and reports the last ten hours."""
    args = parse_args([])

    assert args.netuid == DEFAULT_NETUID
    assert args.hotkey == DEFAULT_HOTKEY
    assert args.network == "archive"
    assert args.hours == 10
    assert args.last_epochs is None
    assert args.single_check is False


def test_render_hourly_health_table_includes_status_and_blocks():
    """Every row reaches the printed table with its verdict, sampled block and epoch range intact."""
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
    """Three hours asked for gives three rows, oldest first, over a single connection that is closed at the end."""
    subtensor = DummySubtensor(current_block=500)

    monkeypatch.setattr(
        "validator.scripts.health.check_validator_health.build_subtensor",
        lambda target: subtensor,
    )
    monkeypatch.setattr(
        "validator.scripts.health.check_validator_health.find_block_for_target_time",
        lambda subtensor, target_time_utc, current_block: current_block
        - int(target_time_utc.minute / 10),
    )
    monkeypatch.setattr(
        "validator.scripts.health.check_validator_health.get_block_time_utc",
        lambda subtensor_obj, block: datetime(2026, 3, 16, 10, 0, tzinfo=timezone.utc),
    )

    def fake_check(subtensor, netuid, hotkey, checked_at_block, current_epoch):
        """Report a healthy verdict for whichever block the hourly walk hands over."""
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
        "validator.scripts.health.check_validator_health._check_validator_health_with_subtensor",
        fake_check,
    )
    monkeypatch.setattr(
        "validator.scripts.health.check_validator_health.get_subnet_epoch_state",
        lambda subtensor_obj, netuid, block: (360, 64),
    )
    monkeypatch.setattr(
        "validator.scripts.health.check_validator_health.get_uid_for_hotkey",
        lambda subtensor_obj, netuid, hotkey, block: 0,
    )
    monkeypatch.setattr(
        "validator.scripts.health.check_validator_health.get_last_update_vector",
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
    """Registered and setting weights in every epoch of the run gives True, and the connection is released."""
    subtensor = DummySubtensor(current_block=500)

    monkeypatch.setattr(
        "validator.scripts.health.check_validator_health.build_subtensor",
        lambda target: subtensor,
    )
    monkeypatch.setattr(
        "validator.scripts.health.check_validator_health.collect_history",
        lambda **kwargs: [
            type("Epoch", (), {"registered": True, "last_update_in_epoch": True})(),
            type("Epoch", (), {"registered": True, "last_update_in_epoch": True})(),
            type("Epoch", (), {"registered": True, "last_update_in_epoch": True})(),
        ],
    )

    assert were_last_epochs_healthy(epochs=3) is True
    assert subtensor.closed is True


def test_were_last_epochs_healthy_returns_false_when_any_epoch_is_stale(monkeypatch):
    """A single epoch without weights sinks the whole run, whatever the epochs either side of it did."""
    subtensor = DummySubtensor(current_block=500)

    monkeypatch.setattr(
        "validator.scripts.health.check_validator_health.build_subtensor",
        lambda target: subtensor,
    )
    monkeypatch.setattr(
        "validator.scripts.health.check_validator_health.collect_history",
        lambda **kwargs: [
            type("Epoch", (), {"registered": True, "last_update_in_epoch": True})(),
            type("Epoch", (), {"registered": True, "last_update_in_epoch": False})(),
            type("Epoch", (), {"registered": True, "last_update_in_epoch": True})(),
        ],
    )

    assert were_last_epochs_healthy(epochs=3) is False
    assert subtensor.closed is True
