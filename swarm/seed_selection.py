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

"""Select the challenge family for a map type or a benchmark seed file."""
from __future__ import annotations

import json
from pathlib import Path

from swarm.domain_model import (
    CHALLENGE_FAMILY_TO_ENVIRONMENT_TYPES,
    CHALLENGE_TYPE_TO_ENVIRONMENT_TYPE,
    DEFAULT_CLI_FAMILY_ID,
)

# What --type shows in a command's help: every map type the registry lists, with its world's name.
MAP_TYPE_HELP = "Map type (" + " ".join(f"{number}={name}" for number, name in CHALLENGE_TYPE_TO_ENVIRONMENT_TYPE.items()) + ")."


def family_for_type(challenge_type: int, family_id: str | None) -> str:
    """The family named, or else the one that flies this map type: the default when it does, otherwise the first that does."""
    if family_id is not None:
        return family_id
    environment = CHALLENGE_TYPE_TO_ENVIRONMENT_TYPE.get(int(challenge_type))
    flyers = [family for family, environments in CHALLENGE_FAMILY_TO_ENVIRONMENT_TYPES.items() if environment in environments]
    return DEFAULT_CLI_FAMILY_ID if DEFAULT_CLI_FAMILY_ID in flyers or not flyers else flyers[0]


def seed_file_family(seed_file: Path, family_id: str | None) -> str:
    """The family named, or else the one a seed file's envelope was saved for, or else the default."""
    if family_id is not None:
        return family_id
    saved = json.loads(seed_file.read_text())
    return str(saved.get("family_id") or DEFAULT_CLI_FAMILY_ID) if isinstance(saved, dict) else DEFAULT_CLI_FAMILY_ID
