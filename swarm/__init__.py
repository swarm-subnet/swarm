# The MIT License (MIT)
# Copyright © 2025 Swarm

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

import sys
import warnings
from pathlib import Path

__version__ = "5.1.5.6"
version_split = __version__.split(".")
version_url = "https://raw.githubusercontent.com/swarm-subnet/swarm/refs/heads/main/swarm/__init__.py"

# Keep protocol compatibility keyed to the first three version components.
__spec_version__ = (
    (1000 * int(version_split[0]))
    + (10 * int(version_split[1]))
    + (1 * int(version_split[2]))
)

# Taking the deprecation here is what keeps it off the drone gym's own import of
# pkg_resources; the fallback below already covers the removal the warning announces.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    try:
        import pkg_resources  # noqa: F401
    except ImportError:
        # setuptools 82 removed pkg_resources; the drone gym still imports it
        import types
        from importlib.resources import files as _pkg_files

        _pkg_resources = types.ModuleType("pkg_resources")
        _pkg_resources.resource_filename = (
            lambda package, resource: str(_pkg_files(package).joinpath(resource))
        )
        sys.modules["pkg_resources"] = _pkg_resources
