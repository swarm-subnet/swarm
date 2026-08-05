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
