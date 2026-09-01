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

"""Type 3 mountain challenge environment."""

from swarm.core import mountain_generator as _mountain_generator


def build_mountain_map(cli, seed, safe_zones, safe_zone_radius):
    return _mountain_generator.build_mountains(
        cli,
        seed,
        safe_zones,
        safe_zone_radius,
        forced_subtype=1,
    )


__all__ = list(getattr(_mountain_generator, "__all__", ())) + ["build_mountain_map"]

for _name in getattr(_mountain_generator, "__all__", ()):
    globals()[_name] = getattr(_mountain_generator, _name)

if "_name" in globals():
    del _name
