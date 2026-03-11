import socket
import subprocess
from typing import Optional

import bittensor as bt


def _find_free_port(self) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

def _check_rpc_ready(self, container_name: str, timeout: float = 5.0) -> bool:
    """Check if the RPC server process is running inside the container."""
    try:
        result = subprocess.run(
            ["docker", "top", container_name],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0 and "main.py" in result.stdout
    except Exception:
        return False

def _get_docker_host_ip(self) -> str:
    """Get the Docker bridge gateway IP (host IP as seen from containers)"""
    try:
        result = subprocess.run(
            [
                "docker",
                "network",
                "inspect",
                "bridge",
                "-f",
                "{{range .IPAM.Config}}{{.Gateway}}{{end}}",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    return "172.17.0.1"

def _get_container_pid(self, container_name: str) -> Optional[int]:
    """Get the PID of a running container"""
    try:
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Pid}}", container_name],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            pid = int(result.stdout.strip())
            if pid > 0:
                return pid
    except Exception:
        pass
    return None

def _apply_network_lockdown(self, container_pid: int, validator_ip: str) -> bool:
    """Apply iptables rules in container's network namespace from HOST using nsenter"""
    try:
        rules = [
            [
                "nsenter",
                "-t",
                str(container_pid),
                "-n",
                "iptables",
                "-A",
                "OUTPUT",
                "-d",
                validator_ip,
                "-j",
                "ACCEPT",
            ],
            [
                "nsenter",
                "-t",
                str(container_pid),
                "-n",
                "iptables",
                "-A",
                "OUTPUT",
                "-d",
                "127.0.0.1",
                "-j",
                "ACCEPT",
            ],
            [
                "nsenter",
                "-t",
                str(container_pid),
                "-n",
                "iptables",
                "-A",
                "OUTPUT",
                "-m",
                "state",
                "--state",
                "ESTABLISHED,RELATED",
                "-j",
                "ACCEPT",
            ],
            [
                "nsenter",
                "-t",
                str(container_pid),
                "-n",
                "iptables",
                "-A",
                "OUTPUT",
                "-j",
                "DROP",
            ],
        ]
        for rule in rules:
            result = subprocess.run(
                rule, capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                bt.logging.warning(
                    f"Failed to apply iptables rule: {' '.join(rule)}"
                )
                return False
        return True
    except Exception as e:
        bt.logging.warning(f"Network lockdown failed: {e}")
        return False
