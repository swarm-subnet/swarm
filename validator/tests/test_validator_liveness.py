"""Tests for the validator's stall watchdog: the loop liveness clock and the exit it triggers."""

import asyncio
import threading
import time

import bittensor as bt
import pytest

from swarm.base import liveness
from swarm.base import validator as base_validator
from swarm.base.liveness import STALL_EXIT_CODE, LoopLiveness
from swarm.base.validator import BaseValidatorNeuron


class _FakeLoop:
    """A loop that queues callbacks until the test runs them, or refuses them once closed."""

    def __init__(self):
        """Start open with nothing queued."""
        self.callbacks = []
        self.closed = False

    def call_soon_threadsafe(self, callback):
        """Queue the callback, or raise as a closed asyncio loop does."""
        if self.closed:
            raise RuntimeError("Event loop is closed")
        self.callbacks.append(callback)

    def run_pending(self):
        """Run what the loop would have run on its next turn."""
        callbacks, self.callbacks = self.callbacks, []
        for callback in callbacks:
            callback()


@pytest.fixture
def clock():
    """A clock the test moves by hand."""
    now = [100.0]
    return now


def _liveness(loop, clock):
    """A liveness clock on the fake loop and the hand-moved clock."""
    return LoopLiveness(loop, clock=lambda: clock[0])


def test_an_answered_ping_resets_the_stall_clock(clock):
    """Time stops counting as stalled once the loop runs the ping."""
    loop = _FakeLoop()
    live = _liveness(loop, clock)
    clock[0] += 50
    live.ping()
    loop.run_pending()
    assert live.stalled_for() == 0


def test_an_unanswered_ping_keeps_counting_and_is_not_doubled(clock):
    """A loop that never turns keeps the clock running, and waiting pings do not pile up."""
    loop = _FakeLoop()
    live = _liveness(loop, clock)
    live.ping()
    clock[0] += 30
    live.ping()
    assert live.stalled_for() == 30
    assert len(loop.callbacks) == 1


def test_a_closed_loop_never_answers_and_keeps_counting_from_the_refused_ping(clock):
    """A closed loop raises nothing out of ping, and the refused ping stays the one the stall counts from."""
    loop = _FakeLoop()
    loop.closed = True
    live = _liveness(loop, clock)
    live.ping()
    clock[0] += 10
    live.ping()
    assert live.stalled_for() == 10


def test_nothing_counts_as_stalled_before_the_first_ping(clock):
    """A loop that was never pinged has nothing outstanding, however long ago the clock was made."""
    live = _liveness(_FakeLoop(), clock)
    clock[0] += 5_000
    assert live.stalled_for() == 0


def test_a_running_event_loop_answers_a_ping():
    """A real loop turning in another thread answers the ping before a callback queued after it."""
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        live = LoopLiveness(loop)
        live.ping()
        turned = threading.Event()
        loop.call_soon_threadsafe(turned.set)
        assert turned.wait(5)
        assert live.stalled_for() == 0
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(2)
        loop.close()


class _Validator(BaseValidatorNeuron):
    """A concrete validator for the watchdog check alone; it never forwards."""

    async def forward(self):
        """Never called by these tests."""


def test_thread_stacks_names_each_live_thread_and_where_it_stands():
    """The stall report holds a header and a stack for every live thread, a parked one included."""
    parked = threading.Event()
    thread = threading.Thread(target=parked.wait, name="parked_for_test", daemon=True)
    thread.start()
    try:
        report = liveness.thread_stacks()
    finally:
        parked.set()
        thread.join(2)
    assert f"--- thread {threading.current_thread().name}" in report
    assert "--- thread parked_for_test" in report
    assert "in wait" in report


def _validator(loop, clock, limit):
    """A validator holding only what the watchdog check reads."""
    validator = _Validator.__new__(_Validator)
    validator._liveness = _liveness(loop, clock)
    validator._stall_limit_sec = limit
    return validator


def test_a_validator_whose_loop_answers_keeps_running(clock):
    """A ping outstanding for less than the limit reports alive and never exits."""
    loop = _FakeLoop()
    validator = _validator(loop, clock, limit=60)
    exits = []
    assert validator.check_liveness(exit_process=exits.append)
    clock[0] += 59
    assert validator.check_liveness(exit_process=exits.append)
    assert exits == []


def test_a_slow_start_before_the_first_check_is_not_a_stall(clock):
    """Time spent building the validator, before any ping was sent, never counts against the limit."""
    validator = _validator(_FakeLoop(), clock, limit=1800)
    exits = []
    clock[0] += 1801
    assert validator.check_liveness(exit_process=exits.append)
    assert exits == []


def test_a_limit_shorter_than_the_poll_spares_a_loop_that_answers(clock):
    """A loop that answers every ping stays alive even when the checks come further apart than the limit."""
    loop = _FakeLoop()
    validator = _validator(loop, clock, limit=2)
    exits = []
    for _ in range(3):
        assert validator.check_liveness(exit_process=exits.append)
        loop.run_pending()
        clock[0] += 5
    assert exits == []


def _stalled_validator(clock, limit=60):
    """A validator whose loop took a ping and then sat past the limit without answering."""
    validator = _validator(_FakeLoop(), clock, limit=limit)
    assert validator.check_liveness(exit_process=pytest.fail)
    clock[0] += limit + 1
    return validator


def test_a_validator_stalled_past_the_limit_logs_every_stack_and_exits(clock, capsys):
    """Past the limit it prints the stall and each thread's stack, then exits with the restart code."""
    validator = _stalled_validator(clock)
    exits = []
    assert not validator.check_liveness(exit_process=exits.append)
    assert exits == [STALL_EXIT_CODE]
    report = capsys.readouterr().err
    assert "has not turned for 61s" in report
    assert f"--- thread {threading.current_thread().name}" in report


def test_a_limit_of_zero_turns_the_watchdog_off(clock):
    """With the limit at 0 nothing is pinged and nothing exits, however long the loop stalls."""
    loop = _FakeLoop()
    validator = _validator(loop, clock, limit=0)
    exits = []
    clock[0] += 10_000
    assert validator.check_liveness(exit_process=exits.append)
    assert exits == [] and loop.callbacks == []


def test_a_validator_still_exits_when_the_stacks_cannot_be_read(clock, capsys, monkeypatch):
    """A failure while dumping the stacks still logs the stall and still exits."""
    def broken_stacks():
        """Fail the way a frame walk on a dying interpreter can."""
        raise RuntimeError("frames gone")

    monkeypatch.setattr(base_validator, "thread_stacks", broken_stacks)
    validator = _stalled_validator(clock)
    exits = []
    assert not validator.check_liveness(exit_process=exits.append)
    assert exits == [STALL_EXIT_CODE]
    assert "frames gone" in capsys.readouterr().err


def test_a_validator_still_exits_while_the_stacks_are_still_being_read(clock, monkeypatch):
    """A stack walk that never returns, such as on a hung disk, is given up on at the deadline and the exit still comes."""
    release = threading.Event()

    def stuck_stacks():
        """Block the way a source-file read on a hung disk does."""
        release.wait()
        return ""

    monkeypatch.setattr(base_validator, "thread_stacks", stuck_stacks)
    monkeypatch.setattr(liveness, "REPORT_DEADLINE_SEC", 0.2)
    validator = _stalled_validator(clock)
    exits = []
    started = time.monotonic()
    try:
        assert not validator.check_liveness(exit_process=exits.append)
        assert time.monotonic() - started < 2
        assert exits == [STALL_EXIT_CODE]
    finally:
        release.set()


def test_a_validator_still_exits_when_the_logger_raises(clock, capsys, monkeypatch):
    """A logger that raises neither stops the stderr copy of the report nor the exit."""
    def broken_logger(message):
        """Fail the way a closed log handler does."""
        raise OSError("log closed")

    monkeypatch.setattr(bt.logging, "error", broken_logger)
    validator = _stalled_validator(clock)
    exits = []
    assert not validator.check_liveness(exit_process=exits.append)
    assert exits == [STALL_EXIT_CODE]
    assert "has not turned for 61s" in capsys.readouterr().err


def test_the_stderr_writer_prints_the_whole_report(capsys):
    """The stderr copy of the report carries the message unchanged."""
    liveness.write_stderr("stalled\nstacks")
    assert capsys.readouterr().err == "stalled\nstacks\n"


def test_a_failing_writer_does_not_stop_the_others():
    """One writer raising still lets every other writer receive the report."""
    received = []

    def broken(message):
        """Fail the way a broken stderr does."""
        raise OSError("broken pipe")

    liveness.report_stall(lambda: "stalled", (broken, received.append), deadline_sec=5)
    assert received == ["stalled"]


def test_a_report_that_cannot_be_built_still_reaches_the_writers():
    """A failing build still sends the writers a line saying why the report is missing."""
    received = []

    def broken_build():
        """Fail while building the report."""
        raise RuntimeError("no frames")

    liveness.report_stall(broken_build, (received.append,), deadline_sec=5)
    assert len(received) == 1 and "no frames" in received[0]


def test_a_blocked_writer_is_given_up_on_at_the_deadline():
    """A writer that never returns holds the report back no longer than the deadline."""
    release = threading.Event()
    started = time.monotonic()
    try:
        liveness.report_stall(lambda: "stalled", (lambda message: release.wait(),), deadline_sec=0.2)
        assert time.monotonic() - started < 2
    finally:
        release.set()


@pytest.mark.parametrize("raw, expected", [
    (None, liveness.DEFAULT_STALL_LIMIT_SEC), ("", liveness.DEFAULT_STALL_LIMIT_SEC), ("600", 600.0),
    ("0", 0.0), ("-5", -5.0), ("2", liveness.MIN_STALL_LIMIT_SEC), ("soon", liveness.DEFAULT_STALL_LIMIT_SEC), ("nan", liveness.DEFAULT_STALL_LIMIT_SEC),
    ("inf", liveness.DEFAULT_STALL_LIMIT_SEC), ("-inf", liveness.DEFAULT_STALL_LIMIT_SEC),
    ("1e999", liveness.DEFAULT_STALL_LIMIT_SEC),
])
def test_the_stall_limit_comes_from_the_environment(monkeypatch, raw, expected):
    """The limit comes from the environment, is raised to the minimum, falls back to the default when invalid or not finite, and 0 or less is kept as off."""
    if raw is None:
        monkeypatch.delenv(liveness.STALL_LIMIT_ENV, raising=False)
    else:
        monkeypatch.setenv(liveness.STALL_LIMIT_ENV, raw)
    assert liveness.stall_limit_sec() == expected
