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

import subprocess
import sys


def test_gym_imports_when_pkg_resources_is_absent():
    code = """
import sys
class _Block:
    def find_spec(self, name, path=None, target=None):
        if name == "pkg_resources" or name.startswith("pkg_resources."):
            raise ModuleNotFoundError("No module named 'pkg_resources'")
        return None
sys.meta_path.insert(0, _Block())
for mod in list(sys.modules):
    if mod.startswith("pkg_resources"):
        del sys.modules[mod]
import swarm
from gym_pybullet_drones.control.BaseControl import BaseControl
import pkg_resources
path = pkg_resources.resource_filename("gym_pybullet_drones", "assets/cf2x.urdf")
import os
print("SHIM_OK", os.path.isfile(path))
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=180
    )
    assert result.returncode == 0, result.stderr
    assert "SHIM_OK True" in result.stdout
