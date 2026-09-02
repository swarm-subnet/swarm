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

"""Public validator utils facade."""

import time

from swarm.validator.utils_parts import (
    backend_submission,
    detection,
    evaluation,
    heartbeat,
    model_fetch,
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
    weights,
):
    for _name in dir(_module):
        if _name.startswith("__"):
            continue
        __all__.append(_name)
del _module, _name
