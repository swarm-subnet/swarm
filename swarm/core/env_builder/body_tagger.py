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

from __future__ import annotations

from typing import Dict, Iterable

import pybullet as p

from .sar_types import BodyCategory


class BodyTagger:
    def __init__(self, cli: int) -> None:
        self.cli = cli
        self._tags: Dict[int, str] = {}

    @property
    def body_tags(self) -> Dict[int, str]:
        return self._tags

    def _store(self, uid: int, category) -> None:
        if isinstance(category, BodyCategory):
            value = category.value
        else:
            value = str(category)
        if uid is None or uid < 0:
            return
        self._tags[int(uid)] = value

    def create_body(self, category, **kwargs) -> int:
        kwargs.setdefault("physicsClientId", self.cli)
        uid = p.createMultiBody(**kwargs)
        self._store(uid, category)
        return uid

    def load_urdf(self, category, fileName: str, **kwargs) -> int:
        kwargs.setdefault("physicsClientId", self.cli)
        uid = p.loadURDF(fileName, **kwargs)
        self._store(uid, category)
        return uid

    def tag_existing(self, uid: int, category) -> None:
        self._store(uid, category)

    def tag_body_group(self, category, uids: Iterable[int]) -> None:
        for uid in uids:
            self._store(uid, category)
