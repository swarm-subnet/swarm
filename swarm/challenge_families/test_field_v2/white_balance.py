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

"""The white balance and exposure of the field's colour camera, by the sun's elevation.

The engine draws the sunlit field warmer than the real camera streams it, more so under a low sun; the table takes
that cast out. The real camera also exposes the field to about the same brightness whatever the sun height, while
the engine draws it brighter as the sun rises, so each row also carries that exposure.
Sentinel's camera applies a family's table in linear light to every colour and zoom frame by day (solar's camera
module); the field's table is below. It was measured on the fixed field under Sentinel's light, so the patrols on the
areas, drawn under the light the areas were matched under, leave it off until it is measured again there.
"""

from __future__ import annotations

# Linear gains from the field's own colour path against the real camera, by sun elevation: the photos' soil colour,
# and their brightness at every sun height, as the camera's own exposure holds it.
WHITE_BALANCE = ((8, 0.937, 1.098, 1.167), (10, 0.841, 0.947, 0.998), (15, 0.732, 0.791, 0.803),
                 (20, 0.828, 0.871, 0.867), (25, 0.784, 0.804, 0.787), (30, 0.773, 0.773, 0.739),
                 (40, 0.769, 0.751, 0.708), (50, 0.763, 0.732, 0.673), (60, 0.753, 0.707, 0.642))
