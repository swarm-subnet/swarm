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

"""Sensor noise and delay (task 16): the errors the real aircraft and its link carry, drawn from the seed.

- Position: RTK wanders. The reading drifts slowly around the truth, inside DJI's hover accuracy with RTK (0.1 m
  across the ground and 0.1 m up and down), and the height above take-off moves with it.
- The link: DJI's delay report gives 10 ms on the command link and 60 to 80 ms on the live video. A command acts one
  control step late, the shortest the 50 Hz simulator holds; everything the model sees, state and images together,
  is a snapshot 60 or 80 ms old, drawn per decision.
- Colour images look like the live stream: colour at half resolution and 8 x 8 blocks quantised, in whole-number
  arithmetic so every validator gets the same bytes. Thermal is left as the engine draws it, grain and all.

Each error draws from its own seed stream, so one never moves another's draws. The laser's range error belongs to
the laser part (task 12); ground distance has no error set.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Optional

import numpy as np
from scipy.signal import lfilter

from swarm.constants import SIM_DT

from .camera import View
from .contract import ACTION_DIM, DECISION_STEPS, HORIZON_S, STATE_SLICES, Command, decode_action
from .episode import SolarEpisode

RTK_HORIZONTAL_M = 0.1                 # DJI: M4TD hovering accuracy with RTK
RTK_VERTICAL_M = 0.1
RTK_CORRELATION_S = 30.0               # how slowly the reading wanders
COMMAND_DELAY_STEPS = 1                # DJI: sdr_cmd_delay 10 ms, under one 20 ms control step
DATA_DELAY_STEPS = (3, 4)              # DJI: liveview_delay_time 60 and 80 ms
VIDEO_QUALITY = 70                     # the stream's quantisation on the JPEG quality scale

DRIFT_SEED_STREAM = 0x5E16             # each error's own stream, so its draws never move another part's
LINK_SEED_STREAM = 0x5E17

# Before the first command arrives the aircraft holds still, with the camera as the patrol starts it.
_NOTHING = decode_action(np.zeros(ACTION_DIM, dtype=np.float32), None)

# The standard JPEG quantisation tables (ITU T.81 Annex K) for brightness and colour.
_LUMA_TABLE = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]], dtype=np.int64)
_CHROMA_TABLE = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99], [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99], [47, 66, 99, 99, 99, 99, 99, 99]] + [[99] * 8] * 4, dtype=np.int64)
_DCT_BITS = 16
# The 8-point DCT scaled to integers; no entry lies near a rounding tie, so every platform builds the same matrix.
_DCT = np.array([[round((math.sqrt(0.125) if k == 0 else 0.5) * math.cos((2 * n + 1) * k * math.pi / 16)
                        * (1 << _DCT_BITS)) for n in range(8)] for k in range(8)], dtype=np.int64)
# Every product and sum the transforms take is a whole number far under 2 ** 53, so float64 holds each exactly and the
# result is the same whatever order the machine adds in.
_DCT_F = _DCT.astype(np.float64)
_STEP_SCALE = float(1 << (2 * _DCT_BITS))
# JPEG's full-range YCbCr in 16-bit fixed point, and back from the two colour planes around 128.
_TO_YCC = ((19595, 38470, 7471), (-11059, -21709, 32768), (32768, -27439, -5329))
_FROM_YCC = np.array([[0, 91881], [-22554, -46802], [116130, 0]], dtype=np.float64)


def reset(env: Any, ep: SolarEpisode) -> None:
    """Draw this seed's position drift and link delays; nothing sent or delivered yet."""
    decisions = int(math.ceil(HORIZON_S / (SIM_DT * DECISION_STEPS))) + 2
    link = np.random.default_rng([LINK_SEED_STREAM, int(ep.seed)])
    ep.sensor_noise = {
        "drift": rtk_drift(ep.seed, int(round(HORIZON_S / SIM_DT)) + 1),
        "data_steps": link.choice(DATA_DELAY_STEPS, size=decisions),
        "commands": deque(),
        "held": None,
        "shown": {"feed": None, "zoom": None},
        "streamed": {},
    }


def rtk_drift(seed: int, steps: int) -> np.ndarray:
    """The position error for every control step of a patrol: east, north and up, in metres.

    Each axis is a Gauss-Markov walk with the DJI figure as two standard deviations, the horizontal pair scaled so
    its length stays inside the figure as often as the vertical one does. A walk that strays past the figure is
    folded back inside it, so the reading is never further off than DJI states and never rests on the limit. The walk
    runs one step after another in a fixed order, so every validator gets the same one.
    """
    keep = math.exp(-SIM_DT / RTK_CORRELATION_S)
    # A two-dimensional error's length passes 2.448 standard deviations as often as one axis passes 1.96.
    sigma = np.array([RTK_HORIZONTAL_M / 2.448, RTK_HORIZONTAL_M / 2.448, RTK_VERTICAL_M / 1.96])
    shocks = np.random.default_rng([DRIFT_SEED_STREAM, int(seed)]).standard_normal((steps, 3)) * sigma
    # The first step starts from the walk's settled spread, every later one keeps most of the last and adds a shock.
    shocks[1:] *= math.sqrt(1.0 - keep * keep)
    walk = lfilter([1.0], [1.0, -keep], shocks, axis=0)
    across = np.sqrt(walk[:, 0] * walk[:, 0] + walk[:, 1] * walk[:, 1])
    folded = RTK_HORIZONTAL_M - np.abs(np.mod(across, 2.0 * RTK_HORIZONTAL_M) - RTK_HORIZONTAL_M)
    walk[:, :2] *= (folded / np.maximum(across, 1e-12))[:, None]
    walk[:, 2] = RTK_VERTICAL_M - np.abs(np.mod(walk[:, 2] + RTK_VERTICAL_M, 4.0 * RTK_VERTICAL_M)
                                         - 2.0 * RTK_VERTICAL_M)
    return walk


def delay(env: Any, ep: SolarEpisode, command: Command) -> Command:
    """The command the aircraft acts on this control step: the one the link delivered, a step after it was sent."""
    queue = ep.sensor_noise["commands"]
    queue.append(command)
    return queue.popleft() if len(queue) > COMMAND_DELAY_STEPS else _NOTHING


def data_delay_steps(ep: SolarEpisode, decision_step: int) -> int:
    """How many control steps old the snapshot shown at this decision step is."""
    delays = ep.sensor_noise["data_steps"]
    return int(delays[min(decision_step // DECISION_STEPS, len(delays) - 1)])


def snapshot_due(ep: SolarEpisode) -> bool:
    """True on the control step whose state the next decision will be shown."""
    after = ep.step % DECISION_STEPS
    return after > 0 and after == DECISION_STEPS - data_delay_steps(ep, ep.step - after + DECISION_STEPS)


def hold(ep: SolarEpisode, snapshot: dict) -> None:
    """Keep a clean snapshot until the decision it is shown at."""
    ep.sensor_noise["held"] = snapshot


def delivered(ep: SolarEpisode) -> Optional[dict]:
    """The snapshot the link has delivered since the last observation, or None when none is on its way."""
    snapshot, ep.sensor_noise["held"] = ep.sensor_noise["held"], None
    return snapshot


def observe(env: Any, ep: SolarEpisode, snapshot: dict) -> dict:
    """What the model is shown from a clean snapshot: the position error on its state, the images as streamed.

    The snapshot carries its step, its state, the three images and the views of the feed and zoom frames.
    """
    noise = ep.sensor_noise
    state = snapshot["state"]
    error = noise["drift"][min(snapshot["step"], len(noise["drift"]) - 1)]
    state[STATE_SLICES["position_m"]] += error
    state[STATE_SLICES["height_above_takeoff_m"]] += error[2]
    noise["shown"] = {"feed": snapshot["feed_view"], "zoom": snapshot["zoom_view"]}
    return {"state": state, "rgb": _streamed(noise, "rgb", snapshot["rgb"]), "thermal": snapshot["thermal"],
            "zoom": _streamed(noise, "zoom", snapshot["zoom"])}


def shown_view(ep: SolarEpisode, image: str) -> Optional[View]:
    """The view of the feed or zoom frame the model was last shown, None before the first."""
    return ep.sensor_noise["shown"][image] if ep.sensor_noise else None


def _streamed(noise: dict, key: str, frame: np.ndarray) -> np.ndarray:
    """A colour image as the stream delivers it, worked out once per image; a blank image stays blank."""
    source, streamed = noise["streamed"].get(key, (None, None))
    if frame is not source:
        streamed = stream_look(frame) if frame.any() else frame
        noise["streamed"][key] = (frame, streamed)
    return streamed


def stream_look(frame: np.ndarray, quality: int = VIDEO_QUALITY) -> np.ndarray:
    """A float colour frame in 0..1 as a live video stream delivers it: colour at half resolution, 8 x 8 blocks
    quantised. Whole numbers throughout, each held exactly, so the bytes never depend on the machine."""
    # The colour sums stay under 2 ** 24, where float32 still holds every whole number exactly.
    rgb = np.rint(np.clip(frame, 0.0, 1.0) * np.float32(255.0))
    luma = _quantised(_ycc(rgb, _TO_YCC[0]) - 128.0, _scaled_steps(_LUMA_TABLE, quality))
    luma += 128.0
    # The stream keeps colour at half resolution: each 2 x 2 square averaged, and shown over the square again.
    rows = rgb[0::2] + rgb[1::2]
    half = rows[:, 0::2] + rows[:, 1::2]
    half += 2.0
    half *= 0.25
    np.floor(half, out=half)
    steps = _scaled_steps(_CHROMA_TABLE, quality)
    colour = np.stack([_quantised(_ycc(half, _TO_YCC[1]), steps), _quantised(_ycc(half, _TO_YCC[2]), steps)], axis=-1)
    shift = _fixed_point(colour @ _FROM_YCC.T).astype(np.float32).repeat(2, axis=0).repeat(2, axis=1)
    shift += luma.astype(np.float32)[..., None]
    np.clip(shift, 0.0, 255.0, out=shift)
    shift /= np.float32(255.0)
    return shift


def _ycc(rgb: np.ndarray, weights: tuple[int, int, int]) -> np.ndarray:
    """One YCbCr plane of a whole-number colour image, from its 16-bit fixed-point weights."""
    return _fixed_point(rgb[..., 0] * weights[0] + rgb[..., 1] * weights[1] + rgb[..., 2] * weights[2])


def _fixed_point(scaled: np.ndarray) -> np.ndarray:
    """A 16-bit fixed-point product, a fresh array, rounded back to whole numbers in place."""
    scaled += 32768.0
    scaled *= 1.0 / 65536.0
    return np.floor(scaled, out=scaled)


def _scaled_steps(table: np.ndarray, quality: int) -> np.ndarray:
    """A quantisation table at a quality from 1 to 100 by libjpeg's rule, shaped (row, 1, column) to scale a plane's
    blocks in place."""
    scale = 5000 // quality if quality < 50 else 200 - 2 * quality
    return np.clip((table * scale + 50) // 100, 1, 255).astype(np.float64)[:, None, :]


def _quantised(plane: np.ndarray, steps: np.ndarray) -> np.ndarray:
    """A plane centred on zero through the 8 x 8 DCT, rounded to the table's steps and back, as float64.

    The transforms work on whole numbers, and IEEE fixes every product and rounding, so no machine differs. Each block
    is transformed along its rows, then down its columns: the same whole numbers as the 2-D transform in one go.
    """
    h, w = plane.shape
    levels = _DCT_F @ (plane.astype(np.float64).reshape(-1, 8) @ _DCT_F.T).reshape(h // 8, 8, w)
    blocks = levels.reshape(h // 8, 8, w // 8, 8)
    blocks *= 1.0 / (_STEP_SCALE * steps)
    np.rint(blocks, out=blocks)
    blocks *= steps
    pixels = _DCT_F.T @ (levels.reshape(-1, 8) @ _DCT_F).reshape(h // 8, 8, w)
    pixels *= 1.0 / _STEP_SCALE
    np.rint(pixels, out=pixels)
    return pixels.reshape(h, w)
