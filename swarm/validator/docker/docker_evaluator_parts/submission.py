import re
from pathlib import Path

import bittensor as bt
import numpy as np

from swarm.constants import DOCKER_PIP_WHITELIST


def _normalize_package_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()

def _validate_requirements(self, requirements_path: Path, uid: int) -> bool:
    try:
        lines = requirements_path.read_text().splitlines()
    except Exception as e:
        bt.logging.warning(f"UID {uid}: Failed to read requirements.txt: {e}")
        return False

    rejected = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("-"):
            bt.logging.warning(f"UID {uid}: Pip option not allowed: {line}")
            return False

        if line.startswith(("git+", "http://", "https://", "file:", "./", "/")):
            bt.logging.warning(
                f"UID {uid}: Direct URL/path install not allowed: {line}"
            )
            return False

        if " @ " in line:
            bt.logging.warning(
                f"UID {uid}: PEP 508 direct reference not allowed: {line}"
            )
            return False

        line = line.split("#")[0].strip()
        if not line:
            continue

        line = line.split(";")[0].strip()
        name = re.split(r"[>=<!~\[]", line)[0].strip()
        if not name:
            continue

        normalized = self._normalize_package_name(name)
        if normalized not in DOCKER_PIP_WHITELIST:
            rejected.append(normalized)

    if rejected:
        bt.logging.warning(
            f"UID {uid}: Requirements rejected — packages not whitelisted: {', '.join(rejected)}"
        )
        return False

    return True

@staticmethod
def _serialize_observation(agent_capnp, obs):
    """Serialize a numpy observation dict into a Cap'n Proto Observation message."""
    message = agent_capnp.Observation.new_message()
    if isinstance(obs, dict):
        entries = message.init("entries", len(obs))
        for i, (key, value) in enumerate(obs.items()):
            arr = np.asarray(value, dtype=np.float32)
            entries[i].key = key
            entries[i].tensor.data = arr.tobytes()
            entries[i].tensor.shape = list(arr.shape)
            entries[i].tensor.dtype = str(arr.dtype)
    else:
        arr = np.asarray(obs, dtype=np.float32)
        entry = message.init("entries", 1)[0]
        entry.key = "__value__"
        entry.tensor.data = arr.tobytes()
        entry.tensor.shape = list(arr.shape)
        entry.tensor.dtype = str(arr.dtype)
    return message
