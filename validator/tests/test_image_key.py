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
"""The CI images are named by .docker/image_key.sh, which both the Tests run and the
operator publish call, so the publish always builds on the image its commit tested."""

from __future__ import annotations

import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / ".docker" / "image_key.sh"


def _key(*args: str, cwd: Path = _REPO_ROOT) -> subprocess.CompletedProcess:
    """Runs the key script with the given arguments and returns the finished process."""
    return subprocess.run(["bash", str(_SCRIPT), *args], cwd=cwd, capture_output=True, text=True, check=False)


def test_base_key_is_stable_and_independent_of_the_working_directory(tmp_path):
    """Proves two calls on one checkout name the same image, wherever they run from."""
    first, second = _key("base"), _key("base", cwd=tmp_path)
    assert first.returncode == 0 and len(first.stdout.strip()) == 16
    assert first.stdout == second.stdout


def test_validator_key_follows_its_base_image():
    """Proves a different test image gives the validator image a different name."""
    one, two = _key("validator", "ghcr.io/a:base-1"), _key("validator", "ghcr.io/a:base-2")
    assert one.returncode == two.returncode == 0
    assert one.stdout != two.stdout


def test_missing_arguments_fail_instead_of_naming_an_image():
    """Proves a call without a kind or without a base ref prints no key."""
    assert _key().returncode == 2
    assert _key("validator").returncode == 2
