from __future__ import annotations

import os
import tempfile

import pybullet as p
import pytest

from swarm.core.env_builder.body_tagger import BodyTagger
from swarm.core.env_builder.sar_types import BodyCategory


_MINIMAL_URDF = """<?xml version='1.0'?>
<robot name='tagger_test'>
  <link name='base_link'>
    <inertial>
      <origin xyz='0 0 0' rpy='0 0 0'/>
      <mass value='0.0'/>
      <inertia ixx='0' ixy='0' ixz='0' iyy='0' iyz='0' izz='0'/>
    </inertial>
    <visual>
      <geometry>
        <box size='1 1 1'/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size='1 1 1'/>
      </geometry>
    </collision>
  </link>
</robot>
"""


@pytest.fixture
def urdf_path():
    fd, path = tempfile.mkstemp(suffix=".urdf")
    with os.fdopen(fd, "w") as f:
        f.write(_MINIMAL_URDF)
    yield path
    os.unlink(path)


def test_three_creation_paths(sar_pybullet, urdf_path):
    tagger = BodyTagger(sar_pybullet)

    col = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5], physicsClientId=sar_pybullet
    )
    uid_create = tagger.create_body(
        BodyCategory.SUPPORT_FLOOR,
        baseMass=0.0,
        baseCollisionShapeIndex=col,
        basePosition=[0, 0, 0.5],
    )
    assert uid_create >= 0
    assert tagger.body_tags[uid_create] == "SUPPORT_FLOOR"

    uid_urdf = tagger.load_urdf(
        BodyCategory.OBSTACLE_OTHER,
        urdf_path,
        basePosition=[2.0, 0.0, 0.5],
    )
    assert uid_urdf >= 0
    assert tagger.body_tags[uid_urdf] == "OBSTACLE_OTHER"

    existing_col = p.createCollisionShape(
        p.GEOM_SPHERE, radius=0.2, physicsClientId=sar_pybullet
    )
    raw_uid = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=existing_col,
        basePosition=[4.0, 0.0, 0.5],
        physicsClientId=sar_pybullet,
    )
    tagger.tag_existing(raw_uid, BodyCategory.OBSTACLE_CLUTTER)
    assert tagger.body_tags[raw_uid] == "OBSTACLE_CLUTTER"


def test_tag_body_group(sar_pybullet):
    tagger = BodyTagger(sar_pybullet)
    col = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1], physicsClientId=sar_pybullet
    )
    sub_uids = []
    for i in range(4):
        u = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=col if i == 0 else -1,
            basePosition=[i * 0.5, 0, 0.1],
            physicsClientId=sar_pybullet,
        )
        sub_uids.append(u)
    tagger.tag_body_group(BodyCategory.VICTIM, sub_uids)
    for u in sub_uids:
        assert tagger.body_tags[u] == "VICTIM"
