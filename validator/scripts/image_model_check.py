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

"""Run the first-time model check on a known good model and exit non-zero unless it passes.

Meant to run inside the validator image through .docker/image_model_check.sh, where the model
cache sits at a path the host's Docker daemon does not have.
"""
import asyncio
import hashlib
import shutil
import sys
from pathlib import Path

from swarm.constants import MODEL_DIR
from swarm.core import model_verify


class _Recorder:
    """Pass every log call through to the real logger and keep its message."""

    def __init__(self, log):
        """Wrap the logger `log`."""
        self._log = log
        self.messages: list[str] = []

    def __getattr__(self, name):
        """Return the logger's method `name`, recording each message it is given."""
        method = getattr(self._log, name)

        def record(msg, *args, **kwargs):
            """Keep the message, then log it as usual."""
            self.messages.append(str(msg))
            return method(msg, *args, **kwargs)

        return record


def main(source: Path) -> int:
    """Copy `source` into the model cache, check it, and return 0 only when it passed."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model = MODEL_DIR / "UID_0.zip"
    shutil.copyfile(source, model)
    model_hash = hashlib.sha256(model.read_bytes()).hexdigest()

    recorder = _Recorder(model_verify._log)
    model_verify._log = recorder
    asyncio.run(model_verify.verify_new_model_with_docker(model, model_hash, "image-check", 0))

    for message in recorder.messages:
        print(message)
    passed = any("passed verification" in message for message in recorder.messages)
    blacklisted = model_hash in model_verify.load_blacklist()
    print(f"passed: {passed}  blacklisted: {blacklisted}")
    return 0 if passed and not blacklisted else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
