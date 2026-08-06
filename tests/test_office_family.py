"""Office interceptor family: RC action contract, conversion math, and env behavior."""

import math

import numpy as np
import pybullet as p
import pytest

from swarm.constants import (
    OFFICE_CHALLENGE_TYPE,
    OFFICE_MAX_START_DISTANCE_M,
    OFFICE_MIN_START_DISTANCE_M,
    OFFICE_RC_DEAD_ZONE,
    OFFICE_RC_SLEW_PER_SEC,
    OFFICE_DET_PERIOD_STEPS,
    OFFICE_TARGET_ALT_MAX_M,
    OFFICE_TARGET_ALT_MIN_M,
    OFFICE_TELEM_DELAY_STEPS,
    OFFICE_TELEM_PERIOD_STEPS,
)
from swarm.core.maps.office import OFFICE_CEILING_M, OFFICE_X_RANGE, OFFICE_Y_RANGE
from swarm.core.moving_drone import rc_sticks_to_world_velocity
from swarm.domain_model import get_policy_interface_contract
from swarm.utils.env_factory import make_env
from swarm.validator import task_gen


@pytest.fixture(scope="module")
def office_env():
    task = task_gen.screening_task(
        1 / 50, 3, challenge_type=OFFICE_CHALLENGE_TYPE,
        distance_range=(OFFICE_MIN_START_DISTANCE_M, OFFICE_MAX_START_DISTANCE_M),
        family_id="cf_office_interceptor",
    )
    env = make_env(task)
    yield env
    env.close()


def test_office_contract_action_space():
    contract = get_policy_interface_contract("cf_office_interceptor", "submission_zip.v1")
    action = contract["action_space"]
    assert action["shape"] == [4]
    assert action["component_names"] == ["lr", "fb", "ud", "yaw"]
    assert action["lower_bound"] == [-1.0, -1.0, -1.0, -1.0]
    assert action["upper_bound"] == [1.0, 1.0, 1.0, 1.0]


def test_rc_body_to_world_math():
    # Facing +X: forward stick moves +X, right stick moves -Y.
    v = rc_sticks_to_world_velocity([0.0, 1.0, 0.0], 0.0, 3.0)
    assert np.allclose(v, [3.0, 0.0, 0.0], atol=1e-9)
    v = rc_sticks_to_world_velocity([1.0, 0.0, 0.0], 0.0, 3.0)
    assert np.allclose(v, [0.0, -3.0, 0.0], atol=1e-9)
    # Facing +Y (yaw 90 deg): forward moves +Y, right moves +X.
    v = rc_sticks_to_world_velocity([0.0, 1.0, 0.0], math.pi / 2, 3.0)
    assert np.allclose(v, [0.0, 3.0, 0.0], atol=1e-9)
    v = rc_sticks_to_world_velocity([1.0, 0.0, 0.0], math.pi / 2, 3.0)
    assert np.allclose(v, [3.0, 0.0, 0.0], atol=1e-9)
    # Up stick is yaw-independent.
    v = rc_sticks_to_world_velocity([0.0, 0.0, 0.5], 1.234, 3.0)
    assert np.allclose(v, [0.0, 0.0, 1.5], atol=1e-9)


def test_office_env_action_space_and_state(office_env):
    env = office_env
    assert env.action_space.shape == (1, 4)
    assert np.all(env.action_space.low == -1.0)
    assert np.all(env.action_space.high == 1.0)
    obs, _ = env.reset(seed=env.task.map_seed)
    contract = get_policy_interface_contract("cf_office_interceptor", "submission_zip.v1")
    assert obs["state"].shape == tuple(contract["smoke_test_observation"]["state"]["shape"])
    assert obs["rgb"].shape == (256, 256, 3)
    assert obs["rgb"].dtype == np.float32
    assert 0.0 <= float(obs["rgb"].min()) and float(obs["rgb"].max()) <= 1.0
    assert all(a.shape == (1, 4) for a in env.action_buffer)


def test_office_zero_action_hovers(office_env):
    env = office_env
    env.reset(seed=env.task.map_seed)
    start = env.pos[0].copy()
    for _ in range(50):
        env.step(np.zeros((1, 4), dtype=np.float32))
    drift = float(np.linalg.norm(env.pos[0][:2] - start[:2]))
    assert drift < 0.15, f"hover drifted {drift:.3f} m"


def test_office_dead_zone_and_slew(office_env):
    env = office_env
    env.reset(seed=env.task.map_seed)
    tiny = np.full((1, 4), OFFICE_RC_DEAD_ZONE * 0.5, dtype=np.float32)
    env.step(tiny)
    assert np.allclose(env._rc_command, 0.0)
    env.step(np.ones((1, 4), dtype=np.float32))
    max_step = OFFICE_RC_SLEW_PER_SEC * env.CTRL_TIMESTEP
    assert np.all(np.abs(env._rc_command) <= max_step + 1e-9)


def test_office_forward_matches_heading(office_env):
    env = office_env
    env.reset(seed=env.task.map_seed)
    # Climb clear of ground effect, gently enough to stay under the 3 m ceiling.
    for _ in range(60):
        env.step(np.array([[0.0, 0.0, 0.3, 0.0]], dtype=np.float32))
    start = env.pos[0].copy()
    for _ in range(40):
        env.step(np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32))
    delta = env.pos[0] - start
    heading = math.atan2(delta[1], delta[0])
    # Spawn yaw is 0, so forward stick must move the drone along +X.
    assert abs(delta[0]) > 0.3
    assert abs(heading) < math.radians(20), f"moved at {math.degrees(heading):.1f} deg"


def test_office_contract_matches_the_rig():
    contract = get_policy_interface_contract("cf_office_interceptor", "submission_zip.v1")
    assert contract["observation_assembly"]["state"] == [
        "tello_attitude", "tello_velocity", "tello_acceleration",
        "tello_altitude", "tello_detection", "action_history",
    ]
    channels = contract["observation_space"]["fields"]["state"]["semantic_channels"]
    # No ground truth in the state vector: everything must exist on the real rig.
    assert "position_xyz" not in channels
    assert "angular_velocity_xyz" not in channels
    assert "search_clue_offset_xy" not in channels
    assert contract["smoke_test_observation"]["state"]["shape"] == [127]
    # Vision is the real drone's camera: RGB, no depth sensor anywhere.
    assert contract["observation_space"]["required_keys"] == ["rgb", "state"]
    assert contract["observation_assembly"]["rgb"] == ["rgb_camera_office"]
    assert contract["smoke_test_observation"]["rgb"]["shape"] == [256, 256, 3]
    assert "depth" not in contract["observation_space"]["fields"]


def test_office_telemetry_cadence_and_age(office_env):
    env = office_env
    obs, _ = env.reset(seed=env.task.map_seed)
    telem = obs["state"][:15]
    assert np.all(telem[:13] == 0.0)
    assert telem[14] == 0.0, "telemetry must be invalid before the first packet"
    hover = np.zeros((1, 4), dtype=np.float32)
    dt = env.CTRL_TIMESTEP
    for i in range(1, OFFICE_TELEM_PERIOD_STEPS):
        obs, *_ = env.step(hover)
        assert obs["state"][14] == 0.0, f"no packet should arrive at step {i}"
    obs, *_ = env.step(hover)
    telem = obs["state"][:15]
    assert telem[14] == 1.0, "first packet must arrive on the period boundary"
    assert telem[13] == pytest.approx(OFFICE_TELEM_DELAY_STEPS * dt)
    # Between packets the readings hold while only the age advances.
    held = telem[:13].copy()
    obs, *_ = env.step(hover)
    assert np.array_equal(obs["state"][:13], held)
    assert obs["state"][13] == pytest.approx(telem[13] + dt)


def test_office_telemetry_tracks_flight(office_env):
    env = office_env
    env.reset(seed=env.task.map_seed)
    # Gentle climb: a full-stick climb would reach the 3 m ceiling and tumble.
    for _ in range(60):
        obs, *_ = env.step(np.array([[0.0, 0.0, 0.3, 0.0]], dtype=np.float32))
    telem = obs["state"][:15]
    assert telem[2] ** 2 + telem[3] ** 2 == pytest.approx(1.0, abs=1e-5)
    # SDK body frame is z-down: climbing shows as NEGATIVE vertical velocity.
    assert telem[6] < -0.2, "climb must show as negative body down-velocity"
    assert telem[10] > 0.5, "ToF must grow as the drone leaves the floor"
    assert abs(telem[11] - (env.pos[0][2] - 0.05)) < 0.3, "fused height must track altitude"
    for _ in range(40):
        obs, *_ = env.step(np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32))
    telem = obs["state"][:15]
    assert telem[4] > 0.5, "forward flight must show as positive body forward-velocity"
    assert abs(telem[5]) < 0.5, "sideways velocity must stay near zero"


def test_office_telemetry_matches_calibration(office_env):
    """Sim telemetry must stay inside the ranges measured on the SecureLink dataset
    (docs/families/securelink_calibration_summary.json)."""
    env = office_env
    env.reset(seed=env.task.map_seed)
    hover = np.zeros((1, 4), dtype=np.float32)
    packets, prev = [], None
    for _ in range(600):
        obs, *_ = env.step(hover)
        t = obs["state"][:15]
        if prev is None or not np.array_equal(t[:13], prev):
            packets.append(t.copy())
            prev = t[:13].copy()
    packets = np.stack(packets)
    expected = 600 // OFFICE_TELEM_PERIOD_STEPS
    loss = 1.0 - len(packets) / expected
    assert 0.0 <= loss < 0.12, f"packet loss {loss:.2%} outside the measured range"
    agz = packets[:, 9]
    assert -10.5 < agz.mean() < -9.1, "hover must read ~-1g on the down axis like the real IMU"
    baro_std = float(np.std(packets[:, 12]))
    assert 0.08 < baro_std < 0.30, f"baro std {baro_std:.3f} outside the measured 0.12-0.16 band"
    assert float(np.abs(packets[:, 4:7]).max()) < 0.5, "hover velocities must stay near zero"


def test_office_telemetry_deterministic():
    task = task_gen.random_task(1 / 50, 42, family_id="cf_office_interceptor")
    streams = []
    for _ in range(2):
        env = make_env(task)
        env.reset(seed=task.map_seed)
        states = []
        for i in range(3 * OFFICE_TELEM_PERIOD_STEPS):
            obs, *_ = env.step(np.array([[0.2, 0.5, 0.3, 0.1]], dtype=np.float32))
            states.append(obs["state"].copy())
        env.close()
        streams.append(np.stack(states))
    assert np.array_equal(streams[0], streams[1])


def test_office_target_spawns_in_band(office_env):
    env = office_env
    env.reset(seed=env.task.map_seed)
    uid = env._office_target_uid
    assert uid is not None and uid != int(env.DRONE_IDS[0])
    pos = env._office_target_pos
    assert OFFICE_TARGET_ALT_MIN_M <= pos[2] <= OFFICE_TARGET_ALT_MAX_M
    assert OFFICE_X_RANGE[0] < pos[0] < OFFICE_X_RANGE[1]
    assert OFFICE_Y_RANGE[0] < pos[1] < OFFICE_Y_RANGE[1]
    assert uid in env._collision_exempt_uids
    assert uid in env.family_runtime.protected_body_uids(env)


def test_office_target_flight_deterministic_and_clear():
    task = task_gen.random_task(1 / 50, 55, family_id="cf_office_interceptor")
    trajs = []
    for _ in range(2):
        env = make_env(task)
        env.reset(seed=task.map_seed)
        hover = np.zeros((1, 4), dtype=np.float32)
        pts = []
        for _ in range(400):
            env.step(hover)
            pts.append(env._office_target_pos.copy())
        assert not env._office_target_crashed
        trajs.append(np.stack(pts))
        env.close()
    assert np.array_equal(trajs[0], trajs[1])
    moved = float(np.linalg.norm(np.diff(trajs[0], axis=0), axis=1).sum())
    assert moved > 3.0, "target must actually fly its legs"
    assert trajs[0][:, 2].min() > 0.5, "target must stay well above the floor"


def test_office_target_profile_deterministic_and_varied():
    from swarm.challenge_families.office_interceptor import office_target_profile

    from swarm.constants import OFFICE_TARGET_FLEE_MAX, OFFICE_TARGET_FLEE_MIN

    assert office_target_profile(55) == office_target_profile(55)
    profiles = [office_target_profile(s) for s in range(40)]
    cruises = [prof["cruise"] for prof in profiles]
    assert max(cruises) - min(cruises) > 0.6, "seeds must deal a real speed spread"
    assert any(prof["react_range"] == 0.0 for prof in profiles), "some stay oblivious"
    assert any(prof["react_range"] > 0.0 for prof in profiles), "some get spooked"
    pauses = [prof["pause_prob"] for prof in profiles]
    assert max(pauses) - min(pauses) > 0.5
    for prof in profiles:
        assert OFFICE_TARGET_FLEE_MIN <= prof["flee_frac"] <= OFFICE_TARGET_FLEE_MAX


def test_office_target_flees_when_approached(office_env):
    from swarm.constants import OFFICE_TARGET_DODGE_REPLAN_STEPS

    env = office_env
    env.reset(seed=env.task.map_seed)
    env._office_target_profile = dict(env._office_target_profile,
                                      react_range=4.0, flee_frac=0.65, pause_prob=1.0)
    hover = np.zeros((1, 4), dtype=np.float32)
    for _ in range(30):
        env.step(hover)  # let the flight settle before the scare
    # Park the chaser body right next to the target, inside the spook range.
    tpos = env._office_target_pos.copy()
    p.resetBasePositionAndOrientation(int(env.DRONE_IDS[0]),
                                      (tpos + np.array([0.8, 0.0, 0.0])).tolist(),
                                      [0, 0, 0, 1], physicsClientId=env.CLIENT)
    fled = False
    for _ in range(2 * OFFICE_TARGET_DODGE_REPLAN_STEPS):
        env.step(hover)
        wp = env._office_target_wp
        cpos = np.asarray(env.pos[0], dtype=float)
        tpos = env._office_target_pos
        if wp is not None and float(np.dot(wp[:2] - tpos[:2], tpos[:2] - cpos[:2])) > 0.0:
            fled = True
            break
    assert fled, "a spooked target must pick a waypoint away from the chaser"


def test_office_target_oblivious_ignores_chaser(office_env):
    env = office_env
    env.reset(seed=env.task.map_seed)
    env._office_target_profile = dict(env._office_target_profile, react_range=0.0)
    hover = np.zeros((1, 4), dtype=np.float32)
    for _ in range(30):
        env.step(hover)
    tpos = env._office_target_pos.copy()
    p.resetBasePositionAndOrientation(int(env.DRONE_IDS[0]),
                                      (tpos + np.array([0.8, 0.0, 0.0])).tolist(),
                                      [0, 0, 0, 1], physicsClientId=env.CLIENT)
    before = env._office_target_last_dodge
    for _ in range(60):
        env.step(hover)
    assert env._office_target_last_dodge == before, "oblivious targets never dodge"


def test_office_target_pause_prob_respected(office_env):
    env = office_env
    env.reset(seed=env.task.map_seed)
    env._office_target_profile = dict(env._office_target_profile,
                                      pause_prob=0.0, react_range=0.0)
    hover = np.zeros((1, 4), dtype=np.float32)
    paused = 0
    for _ in range(600):
        env.step(hover)
        if env._office_target_pause > 0.0:
            paused += 1
    assert paused == 0, "pause_prob 0 must never hover at a waypoint"
    env.reset(seed=env.task.map_seed)
    env._office_target_profile = dict(env._office_target_profile,
                                      pause_prob=1.0, react_range=0.0)
    paused = 0
    for _ in range(600):
        env.step(hover)
        if env._office_target_pause > 0.0:
            paused += 1
    assert paused > 0, "pause_prob 1 must pause at reached waypoints"


def test_office_target_brake_guard_scales():
    """The production guard must cover the physical stopping distance with the
    full safety factor at every legal speed, cadence travel included."""
    from swarm.challenge_families.office_interceptor import _brake_guard_range
    from swarm.constants import (
        OFFICE_TARGET_BRAKE_DECEL, OFFICE_TARGET_GUARD_SAFETY, SIM_DT,
    )

    for v in (0.7, 1.2, 1.8, 1.95):
        guard = _brake_guard_range(v, SIM_DT)
        stop = v * v / (2.0 * OFFICE_TARGET_BRAKE_DECEL)
        cadence_travel = v * 2 * SIM_DT
        assert guard - cadence_travel >= OFFICE_TARGET_GUARD_SAFETY * stop - 1e-9


def test_office_spawn_rejects_sealed_room_and_floor_objects(office_env):
    """The two placement traps found in batch testing: the sealed corridor room
    (no flight path in or out) and flat floor objects under a spawn point."""
    env = office_env
    env.reset(seed=env.task.map_seed)
    fam = env.family_runtime
    # Sealed corridor along the north wall vs the open middle of the room.
    assert not fam._nav_in_main(env, np.array([8.0, 6.2, 1.4]))
    assert fam._nav_in_main(env, np.array([8.0, 3.0, 1.4]))
    # A floor point on top of a flat object: only the narrowphase probe sees it.
    assert not fam._point_is_clear(env, np.array([8.23, 4.15, 0.05]), floor=True)


def test_office_catch_is_physical_contact(office_env):
    """A scripted pursuit must end with a real ram: success, no chaser collision."""
    from swarm.core.moving_drone import world_to_body

    env = office_env
    env.reset(seed=env.task.map_seed)
    # Fixed route around the tall cabinet on this seed's line, then home in.
    # Full sticks (3.0 m/s) outrun every legal target profile (flee tops at 1.95).
    route = [np.array([4.8, 2.6, 1.8]), np.array([8.0, 2.6, 1.8]), np.array([10.0, 3.5, 1.8])]
    leg = 0
    for _ in range(3000):
        cpos = env.pos[0].copy()
        if cpos[2] < 1.2 and env._time_alive < 2.0:
            act = [0.0, 0.0, 0.5, 0.0]
        else:
            if leg < len(route) and np.linalg.norm(cpos - route[leg]) < 0.45:
                leg += 1
            aim = route[leg] if leg < len(route) else env._office_target_pos
            rel = aim - cpos
            yaw = float(env.rpy[0][2])
            f, r, _ = world_to_body(rel, yaw)
            act = [np.clip(r, -1.0, 1.0), np.clip(f, -1.0, 1.0), np.clip(rel[2] * 1.5, -0.6, 0.5), 0.0]
        _, _, term, trunc, _ = env.step(np.array([act], dtype=np.float32))
        if term or trunc:
            break
    assert env._success, "the pursuit must end in a catch"
    assert not env._collision, "ramming the target must not count as a chaser crash"
    assert env._t_to_goal is not None and env._t_to_goal < 30.0
    metrics = env.family_runtime.normalize_rollout_metrics(
        task=env.task,
        metrics=env.family_runtime.build_rollout_metrics(
            task=env.task, success=True, t=env._t_to_goal, horizon=env.EP_LEN_SEC,
            min_clearance=env._min_clearance_episode, collision=False,
            failure_reason=env._failure_reason,
        ),
    )
    assert metrics["success_term"] == 1.0
    assert metrics["final_score"] > 0.6


def test_office_detector_no_sight_no_boxes(office_env):
    """Facing away from the target, the emulator must stay silent and go stale."""
    env = office_env
    env.reset(seed=env.task.map_seed)
    # Target spawns toward +x from the chaser on this seed; turn the camera away.
    for _ in range(45):
        env.step(np.array([[0.0, 0.0, 0.3, 1.0]], dtype=np.float32))
    away = np.zeros((1, 4), dtype=np.float32)
    boxes_seen = 0
    for _ in range(60):
        obs, *_ = env.step(away)
        rel = env._office_target_pos - env.pos[0]
        yaw = math.atan2(rel[1], rel[0]) - env.rpy[0][2]
        # Only count frames where the target truly sits behind the camera.
        if abs(math.remainder(yaw, 2 * math.pi)) > 2.0 and obs["state"][15] > 0:
            boxes_seen += 1
    fp_budget = 3  # rare false positives are allowed, sightings are not
    assert boxes_seen <= fp_budget


def test_office_detector_sees_target(office_env):
    env = office_env
    obs, _ = env.reset(seed=env.task.map_seed)
    hits = 0
    for _ in range(100):
        obs, *_ = env.step(np.zeros((1, 4), dtype=np.float32))
        if obs["state"][15] > 0:
            hits += 1
    # Target starts inside the camera frame on this seed: recall must show up.
    assert hits > 10
    d = obs["state"][15:27]
    assert 0.0 <= d[2] <= 1.0 and 0.0 <= d[3] <= 1.0
    assert d[6] >= 0.25 or d[0] == 0


def test_office_detector_recall_emerges(office_env):
    """The documented ~0.956 marginal recall must emerge in good conditions,
    streaks included (this catches the streak-chain bias that once capped it)."""
    env = office_env
    env.reset(seed=env.task.map_seed)
    frames = hits = 0
    for i in range(3000):
        obs, *_ = env.step(np.zeros((1, 4), dtype=np.float32))
        if env._office_telem_step % OFFICE_DET_PERIOD_STEPS == 0:
            truth = env._office_det_pending
            if (truth is not None and truth["vis"] >= 0.8 and truth["dist"] <= 5.0
                    and min(truth["px"], 256 - truth["px"],
                            truth["py"], 256 - truth["py"]) / 256.0 >= 0.1):
                frames += 1
                if obs["state"][15] > 0:
                    hits += 1
    assert frames > 100, "test setup must produce clearly-visible frames"
    recall = hits / frames
    assert 0.90 < recall <= 1.0, f"marginal recall {recall:.3f} off the ~0.956 spec"


def test_office_detector_confidence_not_an_oracle():
    """Real and ghost confidences must overlap: no single threshold may separate
    them, or 'tracking is the policy's job' dies."""
    from swarm.challenge_families.office_interceptor import _det_fp_box, _det_true_conf

    rng = np.random.default_rng(0)
    # obs-camera scale (focal ~128 px): ~3 px = far target, ~15 px = close target
    far_real = [_det_true_conf(rng, 0.9, 3.0) for _ in range(400)]
    near_real = [_det_true_conf(rng, 1.0, 15.0) for _ in range(400)]
    fps = [_det_fp_box(rng, [(128.0, 128.0)], 128.0)[4] for _ in range(400)]
    assert min(far_real) < max(fps), "far real boxes must dip into the FP range"
    assert max(fps) > 0.5, "ghosts must sometimes look confident"
    assert max(near_real) > 0.9, "close clean boxes must still look strong"


def test_office_detector_occluded_means_silent(office_env):
    """A target parked behind the column must be invisible to the emulator."""

    env = office_env
    env.reset(seed=env.task.map_seed)
    cpos = np.array([12.5, 5.96, 1.5])
    p.resetBasePositionAndOrientation(int(env.DRONE_IDS[0]), cpos.tolist(),
                                      [0, 0, 0, 1], physicsClientId=env.CLIENT)
    tpos = np.array([15.5, 5.96, 1.5])  # the column at x=14.065 stands between
    p.resetBasePositionAndOrientation(env._office_target_uid, tpos.tolist(),
                                      [0, 0, 0, 1], physicsClientId=env.CLIENT)
    env._office_target_pos = tpos.copy()
    env._updateAndStoreKinematicInformation()
    truth = env.family_runtime._detector_capture(env)
    assert truth is None, "fully occluded target must produce no capture at all"


def test_office_detector_deterministic():
    task = task_gen.random_task(1 / 50, 88, family_id="cf_office_interceptor")
    streams = []
    for _ in range(2):
        env = make_env(task)
        env.reset(seed=task.map_seed)
        rows = []
        for _ in range(150):
            obs, *_ = env.step(np.array([[0.1, 0.0, 0.2, 0.3]], dtype=np.float32))
            rows.append(obs["state"][15:27].copy())
        env.close()
        streams.append(np.stack(rows))
    assert np.array_equal(streams[0], streams[1])


def test_office_rgb_deterministic_and_isolated():
    """Same seed => bit-identical frames across envs AND across resets."""
    task = task_gen.random_task(1 / 50, 91, family_id="cf_office_interceptor")
    streams = []
    for _ in range(2):
        env = make_env(task)
        obs, _ = env.reset(seed=task.map_seed)
        rows = [obs["rgb"].copy()]
        for _ in range(10):
            obs, *_ = env.step(np.array([[0.0, 0.0, 0.3, 0.2]], dtype=np.float32))
            rows.append(obs["rgb"].copy())
        # a second reset must reproduce frame 0 exactly (held-frame cache cleared)
        obs2, _ = env.reset(seed=task.map_seed)
        assert np.array_equal(obs2["rgb"], rows[0])
        env.close()
        streams.append(np.stack(rows))
    assert np.array_equal(streams[0], streams[1])


def test_office_rgb_appearance_varies_by_seed():
    frames = []
    for seed in (91, 92):
        task = task_gen.random_task(1 / 50, seed, family_id="cf_office_interceptor")
        env = make_env(task)
        obs, _ = env.reset(seed=task.map_seed)
        frames.append(obs["rgb"].copy())
        env.close()
    diff = float(np.abs(frames[0] - frames[1]).mean())
    assert diff > 0.02, f"different seeds must look different (mean diff {diff:.4f})"


def test_office_rgb_stream_cadence_and_noise(office_env):
    """Held frames repeat exactly; fresh captures differ (sensor noise at least)."""
    env = office_env
    obs, _ = env.reset(seed=env.task.map_seed)
    hover = np.zeros((1, 4), dtype=np.float32)
    frames = [obs["rgb"].copy()]
    for _ in range(4):
        obs, *_ = env.step(hover)
        frames.append(obs["rgb"].copy())
    # capture layout with period 2: [reset, s1, s2] share capture 0; s3 fresh; s4 holds s3
    assert np.array_equal(frames[0], frames[1]), "reset and step 1 share the first capture"
    assert np.array_equal(frames[1], frames[2]), "held frame must repeat byte-exact"
    assert not np.array_equal(frames[2], frames[3]), "fresh capture must differ"
    assert np.array_equal(frames[3], frames[4]), "the fresh capture is then held"


def test_office_world_is_randomized(office_env):

    env = office_env
    env.reset(seed=env.task.map_seed)
    plane = getattr(env, "PLANE_ID", None)
    assert plane is not None
    vis = p.getVisualShapeData(plane, physicsClientId=env.CLIENT)
    assert vis and vis[0][7][3] == 0.0, "the gym plane visual must be hidden"
    tinted = 0
    for uid in range(p.getNumBodies(physicsClientId=env.CLIENT)):
        body = p.getBodyUniqueId(uid, physicsClientId=env.CLIENT)
        for shape in p.getVisualShapeData(body, physicsClientId=env.CLIENT):
            rgba = shape[7]
            if rgba[3] == 1.0 and any(abs(c - 1.0) > 0.01 for c in rgba[:3]):
                tinted += 1
                break
    assert tinted >= 5, f"map bodies must carry per-episode tints (found {tinted})"


def test_office_crash_pays_participation(office_env):
    from swarm.challenge_families import evaluate_rollout
    from swarm.protocol import FailureReason

    env = office_env
    env.reset(seed=env.task.map_seed)
    # Full forward from the floor plows into furniture within a few seconds.
    for _ in range(400):
        _, _, term, trunc, _ = env.step(np.array([[0.0, 1.0, 0.3, 0.0]], dtype=np.float32))
        if term or trunc:
            break
    assert env._collision
    assert env._failure_reason == FailureReason.OBSTACLE_COLLISION.value
    res = evaluate_rollout(task=env.task, success=env._success, t=env._time_alive,
                           horizon=env.EP_LEN_SEC, min_clearance=env._min_clearance_episode,
                           collision=env._collision, failure_reason=env._failure_reason)
    assert res.score == pytest.approx(0.01)


def test_office_task_generation_deterministic():
    a = task_gen.random_task(1 / 50, 77, family_id="cf_office_interceptor")
    b = task_gen.random_task(1 / 50, 77, family_id="cf_office_interceptor")
    assert a == b
    assert a.challenge_type == OFFICE_CHALLENGE_TYPE
    assert OFFICE_X_RANGE[0] < a.start[0] < OFFICE_X_RANGE[1]
    assert OFFICE_Y_RANGE[0] < a.start[1] < OFFICE_Y_RANGE[1]
    assert 0.0 < a.goal[2] < OFFICE_CEILING_M
