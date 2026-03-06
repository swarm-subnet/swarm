from __future__ import annotations

from swarm.protocol import MapTask, PolicyChunk, PolicyRef, PolicySynapse, ValidationResult


def test_map_task_pack_round_trip():
    task = MapTask(
        map_seed=123,
        start=(1.0, 2.0, 3.0),
        goal=(4.0, 5.0, 6.0),
        sim_dt=0.02,
        horizon=60.0,
        challenge_type=5,
        search_radius=7.5,
        moving_platform=True,
        version="1",
    )
    blob = task.pack()
    restored = MapTask.unpack(blob)
    assert restored.map_seed == task.map_seed
    assert tuple(restored.start) == task.start
    assert tuple(restored.goal) == task.goal
    assert restored.challenge_type == task.challenge_type
    assert restored.moving_platform == task.moving_platform


def test_policy_ref_and_chunk_as_dict():
    ref = PolicyRef(
        sha256="abc",
        entrypoint="main.py",
        framework="torch",
        size_bytes=42,
    )
    chunk = PolicyChunk(sha256="abc", data="deadbeef")

    assert ref.as_dict()["sha256"] == "abc"
    assert chunk.as_dict() == {"sha256": "abc", "data": "deadbeef"}


def test_policy_synapse_request_builders():
    ref_req = PolicySynapse.request_ref()
    blob_req = PolicySynapse.request_blob()
    assert ref_req.need_blob is None
    assert blob_req.need_blob is True


def test_policy_synapse_accessors_for_ref_chunk_result():
    ref = PolicyRef(sha256="h", entrypoint="e", framework="f", size_bytes=1)
    chunk = PolicyChunk(sha256="h", data="d")
    result = ValidationResult(uid=7, success=True, time_sec=1.2, score=0.8)

    syn_ref = PolicySynapse.from_ref(ref)
    syn_chunk = PolicySynapse.from_chunk(chunk)
    syn_result = PolicySynapse.from_result(result)

    assert syn_ref.policy_ref == ref
    assert syn_chunk.policy_chunk == chunk
    assert syn_result.validation_result == result


def test_policy_synapse_deserialize_returns_self():
    syn = PolicySynapse(need_blob=True)
    assert syn.deserialize() is syn
