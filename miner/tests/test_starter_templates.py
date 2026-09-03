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

"""The starter controllers a miner copies before writing their own.

They ship as the first thing a miner runs, so they have to import on their own
and return an action of the shape their family declares.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "src" / "submission_template"


@pytest.mark.parametrize("name", ["drone_agent.py", "office_drone_agent.py"])
def test_starter_imports_on_its_own(name, tmp_path):
    """A miner copies the file out of the package, so it cannot rely on it.

    Run from elsewhere under -I, or the repo on sys.path answers the imports and
    a starter that reaches back into swarm passes anyway."""
    copied = tmp_path / name
    copied.write_bytes((TEMPLATE_DIR / name).read_bytes())
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PWD")}
    result = subprocess.run(
        [sys.executable, "-I", "-c",
         f"import importlib.util;"
         f"spec=importlib.util.spec_from_file_location('m', r'{copied}');"
         f"m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);"
         f"assert hasattr(m,'DroneFlightController')"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
    )
    assert result.returncode == 0, result.stderr[-1500:]


def _load(name):
    import importlib.util

    spec = importlib.util.spec_from_file_location("starter", TEMPLATE_DIR / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DroneFlightController()


def test_sar_starter_returns_the_six_element_action_its_family_declares():
    """[dir_x, dir_y, dir_z, speed, yaw, rgb_request], speed within [0, 1]."""
    controller = _load("drone_agent.py")
    action = np.asarray(
        controller.act({"state": np.zeros(64, dtype=np.float32)}), dtype=np.float32
    )
    assert action.shape == (6,)
    assert 0.0 <= action[3] <= 1.0


def test_office_starter_returns_the_four_stick_action_its_family_declares():
    controller = _load("office_drone_agent.py")
    action = np.asarray(
        controller.act({"state": np.zeros(64, dtype=np.float32)}), dtype=np.float32
    )
    assert action.shape == (4,)
