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

"""Stand-ins for the Test Field v2 tests: the stand-in area package flown as the family's only area, a square outline,
and the people package.

The real areas are not in the released swarm-worlds package the tests run on, so the family's tests write the
stand-in area package (area_stand_in) into a folder, point the package root at it and give it a size table that
lists it for every field size: the site then cuts each seed's field from it and builds it through the real loader. Its
few trees hold no wooded field, so the wooded table lists nothing and every seed flies an open field there.
The people package stands in the same way (people_stand_in), until the installed release ships it.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from swarm.challenge_families.test_field_v2 import animal_skins, field_cut, field_sizes, figures, moves, site
from swarm.core.maps import andorra
from validator.tests import animal_stand_in, area_stand_in, moves_stand_in, people_stand_in

HALF_M = 30.0                          # a square outline 60 m across, for tests that need no world
OUTLINE = [[-HALF_M, -HALF_M], [HALF_M, -HALF_M], [HALF_M, HALF_M], [-HALF_M, HALF_M]]


def use_area(folder: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write the stand-in area package into a folder and make it the only area Test Field v2 flies, for every field
    size; returns the package root."""
    root = Path(folder) / "andorra"
    area_stand_in.write_area(root)
    monkeypatch.setenv(andorra.ROOT_ENV, str(root))
    monkeypatch.setattr(field_sizes, "ROWS", {area_stand_in.SLUG: "1" * field_cut.SIZE_BINS})
    monkeypatch.setattr(field_sizes, "WOODED_ROWS", {})
    monkeypatch.setattr(animal_skins, "ANIMALS_DIR", animal_stand_in.write_package(Path(folder) / "animals"))
    use_people(Path(folder) / "people", monkeypatch)
    site.area_tables.cache_clear()
    site.cut_area.cache_clear()
    return root


def use_people(folder: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Make the people package the field reads the installed one, or else a stand-in written into a folder, and the
    people's motion library likewise; returns the package's folder."""
    use_moves(Path(folder).parent / (Path(folder).name + "_moves"), monkeypatch)
    if os.path.isfile(os.path.join(figures.PEOPLE_DIR, "people.json")):
        return figures.PEOPLE_DIR
    package = people_stand_in.write_package(Path(folder))
    monkeypatch.setattr(figures, "PEOPLE_DIR", package)
    return package


def use_moves(folder: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Make the everyday motion library the field reads the installed one, or else a stand-in written into a folder;
    returns the library's folder."""
    if os.path.isfile(os.path.join(moves.MOVES_DIR, "motions.json")):
        return moves.MOVES_DIR
    library = moves_stand_in.write_library(Path(folder))
    monkeypatch.setattr(moves, "MOVES_DIR", library)
    return library
