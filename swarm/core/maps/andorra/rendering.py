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

"""The light the areas were colour-matched under: a grade on the daylight model's arguments, a graded sky photo, the
lowest sun and the far plane.

The areas' ground textures were matched to the aerial photo under this light, so an area looks right only in it. A
family turns it on with its runtime's area_light flag (see swarm.core.moving_drone).
"""

from __future__ import annotations

import hashlib
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image, ImageEnhance

EXPOSURE_SCALE = 0.85  # Preserve texture highlights on the converted terrain.
AMBIENT_SCALE = 0.50  # Shaded and sun-facing-away ground stays readable without washing out sunlit relief.
HAZE_DISTANCE_SCALE = 30.0  # Keep the distant mountain backdrop clear.
SHADOW_FLOOR = 0.50  # Shade stays dark but not black, so a person in it can be told apart.
SKY_SATURATION = 1.6  # Match the reference sky's colour strength.
SKY_CONTRAST = 1.2  # Match the reference sky's cloud separation.
# Each sky photo is first white balanced this far (0 none, 1 fully), so that a blue sky does not turn grey roofs navy.
SKY_BALANCE = 0.75
FAR_PLANE_M = 40000.0  # Include the converted area's distant mountain backdrop.
SUN_MIN_ELEVATION_DEG = 20.0  # Under it a valley lies in terrain shadow and a person there cannot be told apart.

# The converter's light block names these same values; the loader refuses an area converted under another grade.
LIGHT_GRADE = {"exposure_scale": EXPOSURE_SCALE, "ambient_scale": AMBIENT_SCALE, "haze_scale": HAZE_DISTANCE_SCALE,
               "shadow_floor": SHADOW_FLOOR, "sky_saturation": SKY_SATURATION, "sky_contrast": SKY_CONTRAST,
               "sky_balance": SKY_BALANCE}


def grade_daylight(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Grade the daylight model's renderer arguments in place and return them; a key the caller left out stays out."""
    for key, scale in (("exposure", EXPOSURE_SCALE), ("lightAmbientCoeff", AMBIENT_SCALE), ("hazeDistance", HAZE_DISTANCE_SCALE)):
        if key in kwargs:
            kwargs[key] *= scale
    if "shadowLightCoeff" in kwargs:
        kwargs["shadowLightCoeff"] = SHADOW_FLOOR
    return kwargs


@lru_cache(maxsize=None)
def _sky_folder() -> tempfile.TemporaryDirectory:
    """The folder graded sky photos are written to, made on first use and removed when the process exits."""
    return tempfile.TemporaryDirectory(prefix="swarm-area-skies-")


@lru_cache(maxsize=None)
def graded_sky_path(photo_path: str) -> str:
    """The graded copy of a sky photo, written once per process; texture ids stay local to each client."""
    name = hashlib.sha256(str(photo_path).encode()).hexdigest()
    target = Path(_sky_folder().name) / f"{name}.png"
    with Image.open(photo_path) as source:
        balanced = balance_sky(source.convert("RGB"), SKY_BALANCE)
        graded = ImageEnhance.Contrast(ImageEnhance.Color(balanced).enhance(SKY_SATURATION)).enhance(SKY_CONTRAST)
        graded.save(target)
    return str(target)


def sky_light_colour(photo: Image.Image) -> np.ndarray:
    """The colour an equirectangular sky photo casts on an up-facing surface: its upper half weighted by solid angle
    and incidence, in linear light, scaled to unit luminance."""
    stride = max(1, photo.height // 256)
    pixels = np.asarray(photo, dtype=np.float64)[::stride, ::stride] / 255.0
    linear = np.where(pixels <= 0.04045, pixels / 12.92, ((pixels + 0.055) / 1.055) ** 2.4)
    rows = linear.shape[0]
    elevation = (0.5 - (np.arange(rows) + 0.5) / rows) * np.pi
    upper = elevation > 0.0
    weight = np.cos(elevation[upper]) * np.sin(elevation[upper])
    colour = (linear[upper].mean(1) * weight[:, None]).sum(0)
    luminance = float(colour @ np.array([0.2126, 0.7152, 0.0722]))
    # A black or one-row photo casts no colour to correct.
    return colour / luminance if luminance > 0.0 and (colour > 0.0).all() else np.ones(3)


def balance_sky(photo: Image.Image, strength: float) -> Image.Image:
    """The photo with each channel scaled so its sky light moves `strength` of the way to neutral, at equal luminance.

    The gains are rounded to 1/1024 and applied as integer lookup tables, so every machine writes the same bytes."""
    if strength <= 0.0:
        return photo
    gains = [int(round(1024.0 * float(c) ** -strength)) for c in sky_light_colour(photo)]
    # A gain acts on linear light; through sRGB bytes it acts with the encoding's power, about 1 / 2.2.
    tables = []
    for gain in gains:
        scale = (gain / 1024.0) ** (1.0 / 2.2)
        step = int(round(scale * 1024.0))
        tables += [min(255, (value * step + 512) >> 10) for value in range(256)]
    return photo.point(tables)
