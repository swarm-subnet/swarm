"""The render threads' wait policy: importing swarm makes them sleep between pictures, unless the operator chose."""

import os
import subprocess
import sys

_PRINT_POLICY = "import swarm, os; print(os.environ.get('OMP_WAIT_POLICY'))"


def _policy_after_import(preset):
    """The wait policy a fresh interpreter holds after importing swarm, starting from preset (None for unset)."""
    env = {k: v for k, v in os.environ.items() if k != "OMP_WAIT_POLICY"}
    if preset is not None:
        env["OMP_WAIT_POLICY"] = preset
    return subprocess.check_output([sys.executable, "-c", _PRINT_POLICY], env=env, text=True).strip().splitlines()[-1]


def test_importing_swarm_makes_the_render_threads_wait_passively():
    """With no policy set, importing swarm sets the passive one."""
    assert _policy_after_import(None) == "PASSIVE"


def test_an_operator_policy_is_kept():
    """A policy the operator set before the import stays as it was."""
    assert _policy_after_import("ACTIVE") == "ACTIVE"
