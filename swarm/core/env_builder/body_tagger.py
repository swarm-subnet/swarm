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

    def load_obj(self, category, fileName: str, **kwargs) -> int:
        cli = kwargs.pop("physicsClientId", self.cli)
        mesh_scale = kwargs.pop("meshScale", [1.0, 1.0, 1.0])
        base_position = kwargs.pop("basePosition", [0.0, 0.0, 0.0])
        flags = kwargs.pop("collision_flags", None)
        col_kwargs = {"physicsClientId": cli}
        if flags is not None:
            col_kwargs["flags"] = flags
        col_id = p.createCollisionShape(
            p.GEOM_MESH, fileName=fileName, meshScale=mesh_scale, **col_kwargs,
        )
        vis_kwargs = {"physicsClientId": cli}
        if "rgbaColor" in kwargs:
            vis_kwargs["rgbaColor"] = kwargs.pop("rgbaColor")
        vis_id = p.createVisualShape(
            p.GEOM_MESH, fileName=fileName, meshScale=mesh_scale, **vis_kwargs,
        )
        uid = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=base_position,
            physicsClientId=cli,
            **kwargs,
        )
        self._store(uid, category)
        return uid

    def tag_existing(self, uid: int, category) -> None:
        self._store(uid, category)

    def tag_body_group(self, category, uids: Iterable[int]) -> None:
        for uid in uids:
            self._store(uid, category)
