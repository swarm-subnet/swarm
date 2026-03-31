"""Public validator utils facade."""

import time

from swarm.validator.utils_parts import (
    backend_submission,
    detection,
    evaluation,
    heartbeat,
    model_fetch,
    queue_worker,
    state,
    weights,
)
from swarm.validator.utils_parts._shared import (
    CACHE_FILE,
    NORMAL_MODEL_QUEUE_FILE,
    NORMAL_MODEL_QUEUE_PROCESS_LIMIT,
    STATE_DIR,
)

for _module in (
    heartbeat,
    state,
    model_fetch,
    evaluation,
    detection,
    backend_submission,
    queue_worker,
    weights,
):
    for _name in dir(_module):
        if _name.startswith('__'):
            continue
        globals()[_name] = getattr(_module, _name)

del _module, _name

__all__ = [
    "CACHE_FILE",
    "NORMAL_MODEL_QUEUE_FILE",
    "NORMAL_MODEL_QUEUE_PROCESS_LIMIT",
    "STATE_DIR",
    "time",
]
for _module in (
    heartbeat,
    state,
    model_fetch,
    evaluation,
    detection,
    backend_submission,
    queue_worker,
    weights,
):
    for _name in dir(_module):
        if _name.startswith("__"):
            continue
        __all__.append(_name)
del _module, _name
