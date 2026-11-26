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

"""The 1 m survey grids of an area (its spots.npz): ground height, top-of-objects height, land kind, and the
open-ground and dock layers, with the helpers between grid cells and area metres.

Rows run north to south and columns west to east, one cell a metre, each cell's centre on half metres. Local x is
metres east of the area's centre and y metres north. Heights are centimetres above the survey's base_height_m, kept as
the integers the file holds so a decision made on them is the same on every machine; 65535 marks a cell with no data.
The open-ground layer marks dry ground with no traffic, a slope of at most 20 degrees and nothing standing within
1.5 m; the person layer (PERSON_KEY), in packages that have it, marks every cell a person may stand on: land of a kind
people use, clear of crowns and of anything solid; the dock layer, in packages that have it, marks cells with a clear
column above them for a dock's lift-out.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

NO_DATA_CM = 65535
PERSON_KEY = "person_bits"


def _layer(bits: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """A packed bit layer (little bit order, row by row) as a boolean grid."""
    return np.unpackbits(bits, bitorder="little")[:math.prod(shape)].reshape(shape).astype(bool)


class Survey:
    """One area's grids, read from its spots.npz without pickle."""

    def __init__(self, path: Union[str, Path]):
        """Read and unpack every grid the file holds."""
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        self.metadata: Dict[str, Any] = json.loads(str(arrays["metadata"]))
        self.shape: Tuple[int, int] = tuple(int(v) for v in arrays["shape"])
        self.origin_en = np.asarray(arrays["origin_en"], dtype=np.float64)
        self.base_height_m = float(arrays["base_height_m"])
        self.ground_cm = np.asarray(arrays["ground_cm"]).reshape(self.shape)
        self.ceiling_cm = np.asarray(arrays["ceiling_cm"]).reshape(self.shape)
        self.kind = np.asarray(arrays["kind"]).reshape(self.shape)
        self.open = _layer(arrays["start_bits"], self.shape)
        # Packages from before the person or the dock layer leave it None.
        self.person: Optional[np.ndarray] = _layer(arrays[PERSON_KEY], self.shape) if PERSON_KEY in arrays else None
        self.dock: Optional[np.ndarray] = _layer(arrays["dock_bits"], self.shape) if "dock_bits" in arrays else None

    @property
    def ground_m(self) -> np.ndarray:
        """Ground height in metres above the survey base, NaN where there is no data."""
        return np.where(self.ground_cm == NO_DATA_CM, np.nan, self.ground_cm / 100.0)

    @property
    def ceiling_m(self) -> np.ndarray:
        """Height of the ground or the highest object on it in metres above the survey base, infinite where there is
        no data."""
        return np.where(self.ceiling_cm == NO_DATA_CM, np.inf, self.ceiling_cm / 100.0)

    def cell(self, x: Any, y: Any) -> Tuple[Any, Any]:
        """The (row, column) of the cell holding local metres (x east, y north); not clipped to the grid."""
        rows, cols = self.shape
        return (np.floor(rows / 2.0 - np.asarray(y, dtype=float)).astype(int),
                np.floor(np.asarray(x, dtype=float) + cols / 2.0).astype(int))

    def centre(self, row: Any, col: Any) -> Tuple[Any, Any]:
        """Local metres (x east, y north) of a cell's centre."""
        rows, cols = self.shape
        return np.asarray(col, dtype=float) + 0.5 - cols / 2.0, rows / 2.0 - np.asarray(row, dtype=float) - 0.5

    def z_shift(self, frame: Dict[str, Any]) -> float:
        """What turns a survey height into a world z under an area's frame; a frame from another origin is refused."""
        if not np.allclose(frame["origin_en"], self.origin_en, atol=0.01, rtol=0.0):
            raise ValueError(f"survey origin {self.origin_en.tolist()} is not the frame's {list(frame['origin_en'])}")
        return self.base_height_m - float(frame["base_height_m"]) + float(frame["z_offset_m"])
