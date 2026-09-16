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

"""What the docstring check asks for, and what it deliberately stays quiet about.

The check is the only thing standing between the rule and nobody remembering it in review, so the
two ways it can be wrong are both worth pinning: letting an undocumented definition through, and
demanding prose from a change that documented nothing because it changed nothing.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "check_docstrings.py"

_spec = importlib.util.spec_from_file_location("check_docstrings", SCRIPT)
linter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(linter)

# Line 1 is the import, 4 the plain function, 8 the underscored one, 9 the function inside it,
# 14 the class and 15 its method. The tests below address those numbers directly.
SOURCE = '''import os


def normal(value):
    return os.fspath(value)


def _underscored(value):
    def inner(other):
        return other + 1
    return inner(value)


class Thing:
    def method(self):
        return 1
'''

DOCUMENTED = '''"""A module that already says what it is."""


def normal(value):
    """Return the value as a filesystem path."""
    return value
'''


def named(reports):
    """Return just the definition each report is about, dropping the file and line prefix."""
    return {report.split("no docstring on ")[1] for report in reports}


def test_all_three_kinds_of_function_are_reported():
    """A plain function, an underscore-named one and one written inside another all count."""
    found = named(linter.problems("pkg/thing.py", SOURCE, None))
    assert found == {"the module", "normal", "_underscored", "_underscored.inner", "Thing", "Thing.method"}


def test_a_change_inside_a_nested_helper_names_only_that_helper():
    """Touching a line in an inner function asks for a docstring there, not on the one around it."""
    assert named(linter.problems("pkg/thing.py", SOURCE, {10})) == {"_underscored.inner"}


def test_an_untouched_definition_is_left_alone():
    """A change in one function says nothing about the rest of the file."""
    assert named(linter.problems("pkg/thing.py", SOURCE, {5})) == {"normal"}


def test_a_change_outside_every_definition_asks_for_the_module_docstring():
    """A line at the top of the file belongs to the module, which needs a docstring like anything else."""
    assert named(linter.problems("pkg/thing.py", SOURCE, {1})) == {"the module"}


def test_code_that_is_already_documented_reports_nothing():
    """The check is about absence, so a file that carries its docstrings passes even under --all."""
    assert linter.problems("pkg/thing.py", DOCUMENTED, None) == []


def test_the_check_passes_its_own_source():
    """The script holds itself to the rule it enforces."""
    assert linter.problems("scripts/check_docstrings.py", SCRIPT.read_text(encoding="utf-8"), None) == []


def test_added_code_lines_are_numbered_in_the_new_file():
    """The lines a hunk adds are reported at the positions they occupy after the change."""
    diff = (
        "diff --git a/pkg/thing.py b/pkg/thing.py\n"
        "--- a/pkg/thing.py\n"
        "+++ b/pkg/thing.py\n"
        "@@ -4,0 +5,2 @@ def normal(value):\n"
        "+    value = value.strip()\n"
        "+    return value\n"
    )
    assert linter.added_lines(diff) == {"pkg/thing.py": {5, 6}}


def test_a_comment_or_a_blank_line_puts_nothing_in_scope():
    """Annotating or spacing out a file is not a reason to demand prose on what surrounds it."""
    diff = (
        "diff --git a/pkg/thing.py b/pkg/thing.py\n"
        "--- a/pkg/thing.py\n"
        "+++ b/pkg/thing.py\n"
        "@@ -3,0 +4,2 @@\n"
        "+    # the lock keeps the swap atomic\n"
        "+\n"
    )
    assert linter.added_lines(diff) == {"pkg/thing.py": set()}


def test_moving_a_file_adds_nothing():
    """A rename carries no new lines, so an old module does not have to be documented to be moved."""
    diff = (
        "diff --git a/pkg/old.py b/pkg/new.py\n"
        "similarity index 100%\n"
        "rename from pkg/old.py\n"
        "rename to pkg/new.py\n"
    )
    assert linter.added_lines(diff) == {}


def test_a_deleted_file_is_not_in_scope():
    """Nothing can be documented in a file the change removed."""
    diff = (
        "diff --git a/pkg/gone.py b/pkg/gone.py\n"
        "deleted file mode 100644\n"
        "--- a/pkg/gone.py\n"
        "+++ /dev/null\n"
        "@@ -1,2 +0,0 @@\n"
        "-import os\n"
        "-\n"
    )
    assert linter.added_lines(diff) == {}


def test_only_python_files_are_collected():
    """The check has nothing to say about a change to documentation or configuration."""
    diff = (
        "diff --git a/README.md b/README.md\n"
        "--- a/README.md\n"
        "+++ b/README.md\n"
        "@@ -1,0 +2 @@\n"
        "+a new sentence\n"
    )
    assert linter.added_lines(diff) == {}


def test_the_backend_mirror_is_out_of_scope():
    """A test pins that module to the backend copy, so a docstring there cannot land in one repo alone."""
    assert linter.excluded("swarm/submission_manifest/__init__.py")
    assert linter.excluded("swarm/assets/anything.py")
    assert not linter.excluded("swarm/validator/forward.py")
