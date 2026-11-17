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

"""A time budget per test: with --max-test-seconds, the run fails when any test's own run takes longer than that,
unless it carries a time_budget mark of its own. Every test runs to the end and every one over its budget is listed, so
a slow test fails the pull request that adds it instead of slowing every later run. Setup is left out: a fixture shared
by a module is built by whichever of its tests runs first, and that changes with how the suite is split."""

from __future__ import annotations

from collections import defaultdict

import pytest

OPTION = "--max-test-seconds"
MARK = "time_budget"
_BUDGET_PROPERTY = "time_budget_s"


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register --max-test-seconds; without it no test is timed against a budget."""
    parser.addoption(OPTION, type=float, default=None, metavar="SECONDS",
                     help=f"Fail the run when a test's own run takes longer than SECONDS, unless it is marked "
                          f"{MARK}(seconds).")


def pytest_configure(config: pytest.Config) -> None:
    """Register the mark and, when a budget is set, the recorder that adds up each test's time."""
    config.addinivalue_line("markers", f"{MARK}(seconds): this test may take up to seconds under {OPTION}")
    if config.getoption(OPTION) is not None:
        config.pluginmanager.register(_Recorder(config.getoption(OPTION)), "time_budget_recorder")


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Carry a test's own budget on its reports, which reach the controller even when xdist runs the test."""
    mark = item.get_closest_marker(MARK)
    if mark is not None:
        item.user_properties.append((_BUDGET_PROPERTY, float(mark.args[0])))


class _Recorder:
    """Records each test's run time where the reports arrive, and fails the run on any overrun."""

    def __init__(self, default_s: float) -> None:
        """Start with no test timed, every unmarked test held to default_s."""
        self.default_s = default_s
        self.spent: dict[str, float] = defaultdict(float)
        self.budget: dict[str, float] = {}

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        """Take a test's run time from its call report, its slowest when it runs more than once, and note the test's
        own budget when it has one."""
        if report.when != "call":
            return
        self.spent[report.nodeid] = max(self.spent[report.nodeid], report.duration)
        for name, value in report.user_properties:
            if name == _BUDGET_PROPERTY:
                self.budget[report.nodeid] = value

    def overruns(self) -> list[tuple[str, float, float]]:
        """Each test over its budget, slowest first, as node id, seconds taken and seconds allowed."""
        over = [(node, spent, self.budget.get(node, self.default_s)) for node, spent in self.spent.items()
                if spent > self.budget.get(node, self.default_s)]
        return sorted(over, key=lambda row: -row[1])

    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(self, session: pytest.Session) -> None:
        """Fail a run that is otherwise green when a test went over its budget; an xdist worker leaves it to the
        controller, which sees every report."""
        if hasattr(session.config, "workerinput") or not self.overruns():
            return
        if session.exitstatus == pytest.ExitCode.OK:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED

    def pytest_terminal_summary(self, terminalreporter, exitstatus: int, config: pytest.Config) -> None:
        """List every test over its budget, with the mark that would allow it."""
        over = self.overruns()
        if not over:
            return
        terminalreporter.section(f"tests over their time budget ({OPTION}={self.default_s:g})", red=True)
        for node, spent, allowed in over:
            terminalreporter.line(f"{spent:7.1f}s  (allowed {allowed:g}s)  {node}")
        terminalreporter.line(f"Make the test faster, or give it @pytest.mark.{MARK}(seconds).")
