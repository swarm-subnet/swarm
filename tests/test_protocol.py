from __future__ import annotations

import msgpack

from swarm.protocol import MapTask, PolicyRef, PolicySynapse, ValidationResult


def test_map_task_pack_round_trip():
    task = MapTask(
        map_seed=123,
        start=(1.0, 2.0, 3.0),
        goal=(4.0, 5.0, 6.0),
        sim_dt=0.02,
        horizon=60.0,
        challenge_type=5,
    )
    blob = task.pack()
    restored = MapTask.unpack(blob)
    assert restored.map_seed == task.map_seed
    assert tuple(restored.start) == task.start
    assert tuple(restored.goal) == task.goal
    assert restored.challenge_type == task.challenge_type
    assert restored.family_id == task.family_id
    assert restored.version == task.version


def test_map_task_unpack_infers_family_id_for_legacy_payloads():
    sar_blob = msgpack.packb(
        {
            "map_seed": 1,
            "start": (0.0, 0.0, 1.0),
            "goal": (1.0, 1.0, 1.0),
            "sim_dt": 0.02,
            "horizon": 60.0,
            "challenge_type": 2,
            "version": "5.0.0",
        },
        use_bin_type=True,
    )
    autopilot_blob = msgpack.packb(
        {
            "map_seed": 2,
            "start": (0.0, 0.0, 1.0),
            "goal": (2.0, 2.0, 2.0),
            "sim_dt": 0.02,
            "horizon": 45.0,
            "challenge_type": 1,
            "version": "4.2.0",
        },
        use_bin_type=True,
    )

    assert MapTask.unpack(sar_blob).family_id == "cf_search_and_rescue"
    assert MapTask.unpack(autopilot_blob).family_id == "cf_autopilot"


def test_policy_ref_as_dict_includes_github_url():
    ref = PolicyRef(
        sha256="abc",
        entrypoint="main.py",
        framework="torch",
        size_bytes=42,
        github_url="https://github.com/user/repo",
    )
    d = ref.as_dict()
    assert d["sha256"] == "abc"
    assert d["github_url"] == "https://github.com/user/repo"


def test_policy_synapse_request_ref():
    syn = PolicySynapse.request_ref()
    assert syn.ref is None
    assert syn.result is None


def test_policy_synapse_accessors_for_ref_and_result():
    ref = PolicyRef(
        sha256="h",
        entrypoint="e",
        framework="f",
        size_bytes=1,
        github_url="https://github.com/a/b",
    )
    result = ValidationResult(uid=7, success=True, time_sec=1.2, score=0.8)

    syn_ref = PolicySynapse.from_ref(ref)
    syn_result = PolicySynapse.from_result(result)

    assert syn_ref.policy_ref == ref
    assert syn_result.validation_result == result


def test_policy_synapse_deserialize_returns_self():
    syn = PolicySynapse()
    assert syn.deserialize() is syn
