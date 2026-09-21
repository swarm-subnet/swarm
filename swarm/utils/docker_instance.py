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

"""Owner label for the Docker containers and images a process starts, so cleanup on a shared daemon touches no one else's."""

from __future__ import annotations

import os

INSTANCE_LABEL_KEY = "swarm.instance"
# Held in the environment so forked and spawned evaluation workers label their containers alike.
_INSTANCE_ID_ENV = "SWARM_DOCKER_INSTANCE_ID"
_DEFAULT_INSTANCE_ID = "local"


def set_instance_id(instance_id: str) -> None:
    """Name the owner of everything this process starts in Docker."""
    os.environ[_INSTANCE_ID_ENV] = str(instance_id)


def instance_id() -> str:
    """The owner name stamped on this process's containers and images."""
    return os.environ.get(_INSTANCE_ID_ENV) or _DEFAULT_INSTANCE_ID


def instance_label() -> str:
    """The key=value Docker label carrying the owner name."""
    return f"{INSTANCE_LABEL_KEY}={instance_id()}"


def is_unowned(owner: str) -> bool:
    """Whether a label value names nobody: absent, or the default a process carries before it is named."""
    return owner in ("", _DEFAULT_INSTANCE_ID)


def obs_shm_path(host_port: int) -> str:
    """The /dev/shm observation buffer for the container on this port, named after its owner."""
    return f"/dev/shm/swarm_obs_{instance_id()}_{host_port}.bin"
