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
