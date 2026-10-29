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

"""Whether this process may cut a sandbox container off the network.

The evaluator enters a container's network namespace with nsenter and adds iptables
rules there. That needs root, or file capabilities on both binaries: nsenter's own
capabilities clear the ambient set across its exec, so iptables cannot inherit them.
"""

import os
import shutil
import subprocess


def binary_capabilities(path: str) -> set[str]:
    """The ``cap_*`` privileges getcap reports for a file; empty when it cannot be read."""
    getcap = shutil.which("getcap")
    if getcap is None:
        return set()
    try:
        result = subprocess.run(
            [getcap, path],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return set()
    if result.returncode != 0 or not result.stdout.strip():
        return set()
    _, _, caps_blob = result.stdout.partition(" ")
    caps: set[str] = set()
    for token in caps_blob.replace("=", ",").split(","):
        token = token.strip()
        if token.startswith("cap_"):
            caps.add(token)
    return caps


def lockdown_permission() -> tuple[bool, str]:
    """Whether the network lockdown is permitted here, with the reason or the fix.

    That means running as root, or nsenter holding cap_sys_admin and the binary iptables
    resolves to holding cap_net_admin.
    """
    nsenter_path = shutil.which("nsenter")
    iptables_path = shutil.which("iptables")
    if nsenter_path is None or iptables_path is None:
        missing = [
            name for name, path in (("nsenter", nsenter_path), ("iptables", iptables_path))
            if path is None
        ]
        return False, f"cannot assess without required binaries: {', '.join(missing)}"

    if hasattr(os, "geteuid") and os.geteuid() == 0:
        return True, "running as root; network lockdown should be permitted"

    resolved_iptables = os.path.realpath(iptables_path)
    nsenter_caps = binary_capabilities(nsenter_path)
    iptables_caps = binary_capabilities(resolved_iptables)
    if "cap_sys_admin" in nsenter_caps and "cap_net_admin" in iptables_caps:
        return True, (
            "binary capabilities detected "
            f"(nsenter={nsenter_path}, iptables={resolved_iptables})"
        )

    detail = (
        "current user is not root and sandbox network lockdown may fail; "
        f"run with sudo -E or grant cap_sys_admin to {nsenter_path} and "
        f"cap_net_admin to {resolved_iptables}"
    )
    if shutil.which("getcap") is None:
        detail += " (getcap unavailable, binary capabilities could not be inspected)"
    return False, detail
