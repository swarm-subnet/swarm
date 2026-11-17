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

"""The per-test time budget: a run with --max-test-seconds fails on every test over its budget and only on those."""

from __future__ import annotations

import pytest

PLUGIN = "validator.tests.time_budget"
# Sleeps never return early, so a 0.3 s test is always over 0.25 s; a test that does nothing stays far under it.
TESTS = """
import time
import pytest

def test_fast():
    pass

def test_slow():
    time.sleep(0.3)

@pytest.mark.time_budget(5)
def test_slow_with_room():
    time.sleep(0.3)

@pytest.mark.time_budget(0.1)
def test_slow_past_its_own_budget():
    time.sleep(0.3)
"""


def _run(pytester: pytest.Pytester, *args: str) -> pytest.RunResult:
    """Run the four sample tests in process with the plugin and the given options."""
    pytester.makepyfile(test_sample=TESTS)
    return pytester.runpytest_inprocess("-p", PLUGIN, "-p", "no:cacheprovider", "-p", "no:asyncio", *args)


def test_every_test_over_its_budget_fails_the_run_and_is_listed(pytester):
    """The unmarked slow test and the one past its own mark are listed and fail the run; the others pass."""
    result = _run(pytester, "--max-test-seconds=0.25")
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    result.assert_outcomes(passed=4)
    listed = {line.split()[-1]: line for line in result.outlines if "(allowed " in line}
    assert set(listed) == {"test_sample.py::test_slow", "test_sample.py::test_slow_past_its_own_budget"}
    assert "(allowed 0.25s)" in listed["test_sample.py::test_slow"]
    assert "(allowed 0.1s)" in listed["test_sample.py::test_slow_past_its_own_budget"]


def test_a_mark_with_room_lets_a_slow_test_pass(pytester):
    """A test slower than the default but within its own time_budget mark leaves the run green."""
    result = _run(pytester, "--max-test-seconds=0.25", "-k", "fast or with_room")
    assert result.ret == pytest.ExitCode.OK
    assert not any("(allowed " in line for line in result.outlines)


def test_a_test_run_twice_is_held_to_its_slowest_run(pytester):
    """The same test run twice in one session is timed by its slowest run, not by both runs added together."""
    pytester.makepyfile(test_sample=TESTS)
    result = pytester.runpytest_inprocess("-p", PLUGIN, "-p", "no:cacheprovider", "-p", "no:asyncio", "--keep-duplicates",
                                          "--max-test-seconds=0.5", "-k", "test_slow and not with_room and not own",
                                          "test_sample.py", "test_sample.py")
    result.assert_outcomes(passed=2)
    assert result.ret == pytest.ExitCode.OK


def test_without_the_option_no_test_is_held_to_a_budget(pytester):
    """Without --max-test-seconds a slow test passes and nothing is listed."""
    result = _run(pytester, "-k", "test_slow and not with_room and not own")
    assert result.ret == pytest.ExitCode.OK
    result.assert_outcomes(passed=1)
    assert not any("(allowed " in line for line in result.outlines)
