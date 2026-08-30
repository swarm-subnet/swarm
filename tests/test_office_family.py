"""Office interceptor family: RC action contract, conversion math, and env behavior."""

import math

import numpy as np
import pybullet as p
import pytest

from swarm.constants import (
    OFFICE_ACTUATOR_JITTER,
    OFFICE_CATCH_HOLD_STEPS,
    OFFICE_CATCH_LEVEL_M,
    OFFICE_CATCH_RADIUS_M,
    OFFICE_CHALLENGE_TYPE,
    OFFICE_MAX_START_DISTANCE_M,
    OFFICE_MIN_START_DISTANCE_M,
    OFFICE_RC_DEAD_ZONE,
    OFFICE_RC_SPEED,
    OFFICE_SCALE_JITTER_MAX,
    OFFICE_SCALE_JITTER_MIN,
    OFFICE_TARGET_SIZE_JITTER,
    OFFICE_RC_SLEW_PER_SEC,
    OFFICE_DET_PERIOD_STEPS,
    OFFICE_TARGET_ALT_MAX_M,
    OFFICE_TARGET_ALT_MIN_M,
    OFFICE_TELEM_DELAY_STEPS,
    OFFICE_TELEM_PERIOD_STEPS,
)
from swarm.challenge_families.office_interceptor import OfficeInterceptorChallengeFamily
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
        family_id="cf_interceptor_office",
    )
    env = make_env(task)
    yield env
    env.close()


def test_office_contract_action_space():
    contract = get_policy_interface_contract("cf_interceptor_office", "submission_zip.v1")
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
    contract = get_policy_interface_contract("cf_interceptor_office", "submission_zip.v1")
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
    """Dead zone and slew still bound the sticks; both are this episode's values,
    not the nominal ones, because the airframe is dealt per episode."""
    env = office_env
    env.reset(seed=env.task.map_seed)
    tiny = np.full((1, 4), env._rc_dead_zone * 0.5, dtype=np.float32)
    env.step(tiny)
    assert np.allclose(env._rc_command, 0.0)
    env.step(np.ones((1, 4), dtype=np.float32))
    assert np.all(np.abs(env._rc_command) <= env._rc_max_step + 1e-9)
    nominal = OFFICE_RC_SLEW_PER_SEC * env.CTRL_TIMESTEP
    assert abs(env._rc_max_step - nominal) / nominal <= OFFICE_ACTUATOR_JITTER + 1e-9


def test_office_physics_rate_motor_lag_and_scope(office_env):
    """The realism layer's invariants: office ticks physics at 5x control with a
    hover-reset ZOH motor lag on the chaser only; other families stay at 1x."""
    import math as _math

    from swarm.constants import OFFICE_MOTOR_TAU_SEC

    env = office_env
    env.reset(seed=env.task.map_seed)
    assert env.PYB_FREQ == 5 * env.CTRL_FREQ and env.PYB_STEPS_PER_CTRL == 5
    nominal_alpha = 1.0 - _math.exp(-env.PYB_TIMESTEP / OFFICE_MOTOR_TAU_SEC)
    tau = -env.PYB_TIMESTEP / _math.log(1.0 - env._office_motor_alpha)
    nominal_tau = -env.PYB_TIMESTEP / _math.log(1.0 - nominal_alpha)
    assert abs(tau - nominal_tau) / nominal_tau <= OFFICE_ACTUATOR_JITTER + 1e-9, (
        "motor lag is dealt per episode, but only inside the airframe band")
    assert env._office_rpm.shape == (env.NUM_DRONES, 4), "lag state is chaser-side only"
    assert np.allclose(env._office_rpm, env.HOVER_RPM), "lag must reset to hover RPM"
    # Exact five-substep ZOH: rpm = hover + (1 - (1-a)^5)(command - hover).
    rpm0 = env._office_rpm.copy()
    env.step(np.array([[0, 0, 1, 0]], dtype=np.float32))
    alpha = env._office_motor_alpha
    cmd = np.asarray(env.last_clipped_action, dtype=float)
    expect = rpm0 + (1.0 - (1.0 - alpha) ** 5) * (cmd - rpm0)
    assert np.allclose(env._office_rpm, expect, rtol=1e-9), \
        "the lag filter must run once per physics substep, five per control step"
    other = task_gen.random_task(1 / 50, 3, family_id="cf_search_and_rescue")
    other_env = make_env(other)
    try:
        assert other_env.PYB_FREQ == other_env.CTRL_FREQ, "only office runs substeps"
    finally:
        other_env.close()


def test_office_substep_refresh_scoped_to_office(office_env):
    """Office refreshes cached kinematics inside the substep loop (five plus the
    end-of-step one); a plain-PYB env with substeps but no office flag must not."""
    env = office_env
    env.reset(seed=env.task.map_seed)
    calls = {"n": 0}
    orig = env._updateAndStoreKinematicInformation

    def counting():
        calls["n"] += 1
        return orig()

    env._updateAndStoreKinematicInformation = counting
    try:
        env.step(np.zeros((1, 4), dtype=np.float32))
    finally:
        env._updateAndStoreKinematicInformation = orig
    assert calls["n"] == env.PYB_STEPS_PER_CTRL + 1

    # A real non-office env forced to the same 5x substeps must NOT refresh in-loop.
    from swarm.core.moving_drone import MovingDroneAviary
    from swarm.utils.env_factory import runtime_profile_for_task
    from gym_pybullet_drones.utils.enums import ActionType, ObservationType

    task = task_gen.random_task(1 / 50, 3, family_id="cf_search_and_rescue")
    prof = runtime_profile_for_task(task)
    env2 = MovingDroneAviary(task, act=ActionType.VEL, gui=False, record=False,
                             obs=ObservationType.RGB, ctrl_freq=50, pyb_freq=250,
                             **dict(prof.env_bootstrap))
    env2.SPEED_LIMIT, env2.MAX_YAW_RATE, env2.ACT_TYPE = 3.0, 3.141, ActionType.VEL
    env2.reset(seed=task.map_seed)
    calls2 = {"n": 0}
    orig2 = env2._updateAndStoreKinematicInformation

    def counting2():
        calls2["n"] += 1
        return orig2()

    env2._updateAndStoreKinematicInformation = counting2
    try:
        env2.step(np.zeros(env2.action_space.shape, dtype=np.float32))
    finally:
        env2._updateAndStoreKinematicInformation = orig2
        env2.close()
    assert calls2["n"] == 1, "plain PYB without the office flag must keep library behavior"


def test_office_aero_forces_hit_chaser_only(office_env):
    """Drag and ground effect apply to the chaser base; the target body receives
    only its own vertical rotor thrust — its speeds are behavioral spec."""
    env = office_env
    env.reset(seed=env.task.map_seed)
    for _ in range(30):
        env.step(np.array([[0, 1, 0.4, 0]], dtype=np.float32))  # moving: drag active
    calls = []
    orig = p.applyExternalForce

    def spy(uid, link, force, pos, flags, physicsClientId=0):
        calls.append((int(uid), int(link), tuple(float(f) for f in force)))
        return orig(uid, link, force, pos, flags, physicsClientId=physicsClientId)

    p.applyExternalForce = spy
    try:
        env.family_runtime.apply_world_physics(env)
    finally:
        p.applyExternalForce = orig
    chaser, target = int(env.DRONE_IDS[0]), int(env._office_target_uid)
    target_calls = [c for c in calls if c[0] == target]
    assert target_calls and all(
        c[1] in (0, 1, 2, 3) and c[2][0] == 0.0 and c[2][1] == 0.0 for c in target_calls
    ), "the target gets rotor thrust only, never drag or ground effect"
    chaser_base = [c for c in calls if c[0] == chaser and c[1] == -1]
    assert len(chaser_base) >= 2, "chaser base must receive drift plus aero forces"


def test_office_never_scans_clearance(office_env):
    """The clearance scan segfaults pybullet against the office meshes (seed 53)
    and office scoring never reads it — it must stay gated off."""
    env = office_env
    env.reset(seed=env.task.map_seed)
    for _ in range(10):
        env.step(np.zeros((1, 4), dtype=np.float32))
    assert env._min_clearance_episode is None


def test_office_cannot_escape_the_room(office_env):
    """A full-stick climb tunnels the thin ceiling sheet between substeps; the
    out-of-bounds failsafe must end the episode as a collision, not an escape."""
    env = office_env
    env.reset(seed=env.task.map_seed)
    for i in range(900):
        _, _, term, trunc, _ = env.step(np.array([[0, 0, 1, 0]], dtype=np.float32))
        if term or trunc:
            break
    assert env._collision and not env._success
    assert env.pos[0][2] < OFFICE_CEILING_M + 0.7, "must be caught right past the shell"


def test_office_malformed_actions_canonicalize_to_hover(office_env):
    """The contract promises hover for garbage input, never an exception."""
    env = office_env
    env.reset(seed=env.task.map_seed)
    for a in (np.full((1, 4), np.nan, dtype=np.float32),
              np.full((1, 4), np.inf, dtype=np.float32),
              np.zeros((1, 8), dtype=np.float32),
              np.zeros((1, 2), dtype=np.float32),
              np.zeros((3,), dtype=np.float32)):
        obs, *_ = env.step(a)
        assert np.all(np.isfinite(obs["state"]))
        assert np.allclose(env._rc_command, 0.0), "garbage input must command hover"


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
    # Forward stick must move the drone along its own seeded spawn heading.
    off = abs(math.remainder(heading - env._office_spawn_yaw, 2 * math.pi))
    assert float(np.hypot(delta[0], delta[1])) > 0.3
    assert off < math.radians(20), f"moved {math.degrees(off):.1f} deg off heading"


def test_office_body_accel_ignores_spawn_heading(office_env):
    """Body-frame acceleration must rotate with the TRUE heading: during a pure
    forward burst the forward axis dominates, whatever the seeded spawn yaw."""
    env = office_env
    env.reset(seed=env.task.map_seed)
    hover = np.zeros((1, 4), dtype=np.float32)
    for _ in range(80):
        env.step(np.array([[0, 0, 0.45, 0]], dtype=np.float32)
                 if env.pos[0][2] < 1.5 else hover)
    afs, ars = [], []
    for _ in range(30):
        obs, *_ = env.step(np.array([[0, 1, 0, 0]], dtype=np.float32))
        if obs["state"][14] > 0:  # valid packet
            afs.append(float(obs["state"][7]))
            ars.append(float(obs["state"][8]))
    assert afs, "the burst must deliver telemetry packets"
    assert abs(np.mean(afs)) > abs(np.mean(ars)), \
        "forward acceleration must land on the body-forward axis, not smear sideways"
    assert np.mean(afs) > 0.3, "accelerating forward must read as positive af"


def test_office_spawn_heading_random_and_yaw_relative(office_env):
    """The IMU has no world compass: headings vary by seed and reported yaw
    starts at zero wherever the drone happens to face."""
    env = office_env
    env.reset(seed=env.task.map_seed)
    assert abs(env._office_spawn_yaw) > 1e-6, "seeded heading must not be the old fixed 0"
    hover = np.zeros((1, 4), dtype=np.float32)
    for _ in range(20 * OFFICE_TELEM_PERIOD_STEPS):
        obs, *_ = env.step(hover)
        # A real packet carries a unit sin/cos pair; the zero state does not.
        if float(obs["state"][2]) ** 2 + float(obs["state"][3]) ** 2 > 0.5:
            break
    else:
        raise AssertionError("no telemetry packet survived the drop chance")
    rel = math.atan2(float(obs["state"][2]), float(obs["state"][3]))
    assert abs(rel) < math.radians(8), "reported yaw must start near zero, not world yaw"
    headings = set()
    for seed in (11, 12, 13):
        task = task_gen.random_task(1 / 50, seed, family_id="cf_interceptor_office")
        e2 = make_env(task)
        e2.reset(seed=task.map_seed)
        headings.add(round(float(e2._office_spawn_yaw), 3))
        e2.close()
    assert len(headings) == 3, "each seed must deal its own heading"


def test_office_contract_matches_the_rig():
    contract = get_policy_interface_contract("cf_interceptor_office", "submission_zip.v1")
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
    task = task_gen.random_task(1 / 50, 42, family_id="cf_interceptor_office")
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
    task = task_gen.random_task(1 / 50, 55, family_id="cf_interceptor_office")
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


def test_office_target_invisible_in_rgb(office_env):
    """The detector must stay the only sighting channel: moving the target
    through the camera's view cannot change a single rendered pixel."""
    env = office_env
    env.reset(seed=env.task.map_seed)
    uid = int(env._office_target_uid)
    cpos = env.pos[0].copy()
    rot = np.array(p.getMatrixFromQuaternion(env.quat[0])).reshape(3, 3)
    fwd = rot[:, 0]
    away = (cpos - fwd * 5.0).tolist()
    for dist in (0.5, 1.5):  # large on screen and mid-range, wall behind both
        p.resetBasePositionAndOrientation(uid, (cpos + fwd * dist).tolist(),
                                          [0, 0, 0, 1], physicsClientId=env.CLIENT)
        with_target, _, _ = env._getDroneImages(0)
        p.resetBasePositionAndOrientation(uid, away, [0, 0, 0, 1],
                                          physicsClientId=env.CLIENT)
        without_target, _, _ = env._getDroneImages(0)
        assert np.array_equal(with_target, without_target), f"target visible at {dist} m"
    # Positive control: an opaque body at the same spot MUST change the render.
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1] * 3,
                              rgbaColor=[1, 0, 0, 1], physicsClientId=env.CLIENT)
    box = p.createMultiBody(baseVisualShapeIndex=vis,
                            basePosition=(cpos + fwd * 1.5).tolist(),
                            physicsClientId=env.CLIENT)
    control, _, _ = env._getDroneImages(0)
    p.removeBody(box, physicsClientId=env.CLIENT)
    assert not np.array_equal(without_target, control), "the check cannot fail"


def test_office_tof_never_ranges_target(office_env):
    """A target under the chaser must not echo in the ToF, upright or tilted
    with the ray clipping its edge (the case a naive recast re-hits)."""
    env = office_env
    env.reset(seed=env.task.map_seed)
    fam = env.family_runtime
    uid = int(env._office_target_uid)
    cpos = env.pos[0].copy()
    p.resetBasePositionAndOrientation(int(env.DRONE_IDS[0]),
                                      [cpos[0], cpos[1], 2.5], [0, 0, 0, 1],
                                      physicsClientId=env.CLIENT)
    env._updateAndStoreKinematicInformation()
    hover = env.pos[0]
    away = [cpos[0] + 4.0, cpos[1], 1.5]
    tilt = p.getQuaternionFromEuler([0.52, 0.0, 0.0])  # ~30 degrees
    cases = (([hover[0], hover[1], 1.4], [0, 0, 0, 1]),
             ([hover[0] + 0.06, hover[1], 1.4], tilt))  # ray clips the edge
    # No target in the ray: bit-identical to the generic helper, or every
    # episode's telemetry silently changes.
    p.resetBasePositionAndOrientation(uid, away, [0, 0, 0, 1],
                                      physicsClientId=env.CLIENT)
    assert fam._office_tof(env, 0) == env._get_altitude_distance(0)
    baseline = fam._snapshot(env, 0, env.CTRL_TIMESTEP)[9]
    for tpos, tquat in cases:
        p.resetBasePositionAndOrientation(uid, tpos, tquat, physicsClientId=env.CLIENT)
        raw = p.rayTest([hover[0], hover[1], hover[2] - 0.03],
                        [hover[0], hover[1], hover[2] - 20.0],
                        physicsClientId=env.CLIENT)[0]
        assert int(raw[0]) == uid, "case must actually put the target in the ray"
        # Through the real telemetry path: a reversion to the unfiltered helper
        # in _snapshot must fail here, not only in the unit call.
        with_target = fam._snapshot(env, 0, env.CTRL_TIMESTEP)[9]
        assert with_target == pytest.approx(baseline, abs=1e-6), \
            "ToF must read the floor, not the target"


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
    # Start level with the target on a clear line and close the gap. The route is
    # arranged from the episode's own geometry: the map is no longer a fixed shape
    # a memorised waypoint list could cross.
    assert _place_with_clear_view(env, dist=2.0), "no clear approach to arrange"
    for _ in range(3000):
        cpos = env.pos[0].copy()
        rel = env._office_target_pos - cpos
        yaw = float(env.rpy[0][2])
        f, r, _ = world_to_body(rel, yaw)
        act = [np.clip(r, -1.0, 1.0), np.clip(f, -1.0, 1.0),
               np.clip(rel[2] * 1.5, -0.6, 0.5), 0.0]
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
    # Point the camera squarely away from the target: spawn heading is seeded.
    cpos = env.pos[0]
    rel = env._office_target_pos - cpos
    back = math.atan2(float(rel[1]), float(rel[0])) + math.pi
    p.resetBasePositionAndOrientation(int(env.DRONE_IDS[0]), cpos.tolist(),
                                      p.getQuaternionFromEuler([0, 0, back]),
                                      physicsClientId=env.CLIENT)
    env._rc_target_yaw[:] = back
    env._updateAndStoreKinematicInformation()
    away = np.zeros((1, 4), dtype=np.float32)
    boxes_seen = 0
    away_frames = 0
    for _ in range(60):
        obs, *_ = env.step(away)
        rel = env._office_target_pos - env.pos[0]
        yaw = math.atan2(rel[1], rel[0]) - env.rpy[0][2]
        # Only count frames where the target truly sits behind the camera.
        if abs(math.remainder(yaw, 2 * math.pi)) > 2.0:
            away_frames += 1
            if obs["state"][15] > 0:
                boxes_seen += 1
    assert away_frames >= 30, "the flip must actually put the target behind the camera"
    fp_budget = 3  # rare false positives are allowed, sightings are not
    assert boxes_seen <= fp_budget


def _aim_at_target(env):
    """Face the camera at the target: spawn heading is seeded, not aligned."""
    cpos = env.pos[0]
    rel = env._office_target_pos - cpos
    yaw = math.atan2(float(rel[1]), float(rel[0]))
    p.resetBasePositionAndOrientation(int(env.DRONE_IDS[0]), cpos.tolist(),
                                      p.getQuaternionFromEuler([0, 0, yaw]),
                                      physicsClientId=env.CLIENT)
    env._rc_target_yaw[:] = yaw
    env._updateAndStoreKinematicInformation()


def _place_with_clear_view(env, dist=2.5):
    """Put the chaser `dist` from the target, level with it and looking at it, on a
    direction with an unobstructed line of sight. Spawns are drawn over free space
    now, so a test that needs a sighting has to arrange one rather than assume the
    seed provides it."""
    from swarm.constants import CAMERA_EYE_UP_M, OFFICE_CAMERA_EYE_FWD_M

    tgt = np.array(env._office_target_pos, dtype=float)
    ignore = {int(env.DRONE_IDS[0]), int(env._office_target_uid), -1}
    for k in range(72):
        a = k * (2.0 * math.pi / 72.0)
        pos = tgt + np.array([math.cos(a) * dist, math.sin(a) * dist, 0.0])
        yaw = math.atan2(tgt[1] - pos[1], tgt[0] - pos[0])
        eye = pos + np.array([math.cos(yaw), math.sin(yaw), 0.0]) * OFFICE_CAMERA_EYE_FWD_M
        eye[2] += CAMERA_EYE_UP_M
        hit = p.rayTest(eye.tolist(), tgt.tolist(), physicsClientId=env.CLIENT)[0]
        if int(hit[0]) in ignore:
            p.resetBasePositionAndOrientation(
                int(env.DRONE_IDS[0]), pos.tolist(),
                p.getQuaternionFromEuler([0, 0, yaw]), physicsClientId=env.CLIENT)
            p.resetBaseVelocity(int(env.DRONE_IDS[0]), [0, 0, 0], [0, 0, 0],
                                physicsClientId=env.CLIENT)
            env._rc_target_yaw[:] = yaw
            env._updateAndStoreKinematicInformation()
            return True
    return False


def test_office_detector_sees_target(office_env):
    env = office_env
    obs, _ = env.reset(seed=env.task.map_seed)
    assert _place_with_clear_view(env), "no clear line of sight to arrange"
    hits = 0
    for _ in range(100):
        obs, *_ = env.step(np.zeros((1, 4), dtype=np.float32))
        if obs["state"][15] > 0:
            hits += 1
    # The target is in frame and unobstructed: recall must show up.
    assert hits > 10
    d = obs["state"][15:27]
    assert 0.0 <= d[2] <= 1.0 and 0.0 <= d[3] <= 1.0
    assert d[6] >= 0.25 or d[0] == 0


def test_office_detector_recall_emerges(office_env):
    """The documented ~0.956 marginal recall must emerge in good conditions,
    streaks included (this catches the streak-chain bias that once capped it)."""
    env = office_env
    env.reset(seed=env.task.map_seed)
    assert _place_with_clear_view(env, dist=3.0), "no clear line of sight to arrange"
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
    task = task_gen.random_task(1 / 50, 88, family_id="cf_interceptor_office")
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
    task = task_gen.random_task(1 / 50, 91, family_id="cf_interceptor_office")
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
        task = task_gen.random_task(1 / 50, seed, family_id="cf_interceptor_office")
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
    a = task_gen.random_task(1 / 50, 77, family_id="cf_interceptor_office")
    b = task_gen.random_task(1 / 50, 77, family_id="cf_interceptor_office")
    assert a == b
    assert a.challenge_type == OFFICE_CHALLENGE_TYPE
    assert OFFICE_X_RANGE[0] < a.start[0] < OFFICE_X_RANGE[1]
    assert OFFICE_Y_RANGE[0] < a.start[1] < OFFICE_Y_RANGE[1]
    assert 0.0 < a.goal[2] < OFFICE_CEILING_M


def test_plain_appearance_is_inspection_only():
    """Evaluation must always get the per-seed skin, whatever the visualizer did."""
    from swarm.challenge_families import office_interceptor as oi

    assert oi._PLAIN_APPEARANCE is False, "randomized appearance is the default"
    try:
        oi.use_plain_appearance(True)
        assert oi._PLAIN_APPEARANCE is True
        oi.use_plain_appearance(False)
        assert oi._PLAIN_APPEARANCE is False
    finally:
        oi.use_plain_appearance(False)


def _place_pair(env, offset):
    """Put the target in open air and the chaser at `offset` from it, then let the
    family run one catch check."""
    fam = OfficeInterceptorChallengeFamily()
    cli = env.CLIENT
    tgt = int(env._office_target_uid)
    base = np.array([float(env.pos[0][0]), float(env.pos[0][1]), 1.60])
    p.resetBasePositionAndOrientation(tgt, base.tolist(), [0, 0, 0, 1], physicsClientId=cli)
    p.resetBasePositionAndOrientation(int(env.DRONE_IDS[0]), (base + offset).tolist(),
                                      [0, 0, 0, 1], physicsClientId=cli)
    env._updateAndStoreKinematicInformation()
    p.performCollisionDetection(physicsClientId=cli)
    return fam


def test_office_level_intercept_registers(office_env):
    """A level pass inside the catch radius scores once it is held, not by luck."""
    env = office_env
    env.reset(seed=env.task.map_seed)
    fam = _place_pair(env, np.array([OFFICE_CATCH_RADIUS_M - 0.01, 0.0, 0.0]))
    env._success = False
    env._office_catch_hold = 0
    for i in range(OFFICE_CATCH_HOLD_STEPS):
        assert not env._success, f"caught after only {i} steps inside the box"
        fam._update_target(env)
    assert env._success, "a held level intercept must register"


def test_office_overhead_is_not_a_catch(office_env):
    """Hovering above the target is not an interception: the hull is flat, so a
    centre-distance test alone would pay for downwashing it out of the air."""
    env = office_env
    env.reset(seed=env.task.map_seed)
    for dz in (0.09, 0.11, 0.14):
        fam = _place_pair(env, np.array([0.0, 0.0, dz]))
        env._success = False
        env._office_catch_hold = 0
        for _ in range(3 * OFFICE_CATCH_HOLD_STEPS):
            fam._update_target(env)
        assert not env._success, f"hovering {dz:.2f} m overhead scored a catch"


def test_office_airframe_differs_per_episode():
    """Every episode deals its own airframe, so a stick-to-motion constant cannot
    be inverted once and reused."""
    seen = set()
    for seed in (11, 12, 13):
        task = task_gen.random_task(1 / 50, seed, family_id="cf_interceptor_office")
        env = make_env(task)
        env.reset(seed=task.map_seed)
        seen.add((round(env.SPEED_LIMIT, 6), round(env._rc_dead_zone, 6),
                  round(env._rc_max_step, 6), round(env._office_motor_alpha, 6)))
        assert abs(env.SPEED_LIMIT - OFFICE_RC_SPEED) / OFFICE_RC_SPEED <= OFFICE_ACTUATOR_JITTER + 1e-9
        env.close()
    assert len(seen) == 3, "each seed must deal its own airframe"


def test_office_airframe_is_deterministic():
    """Two envs on the same seed must agree, or validators would disagree."""
    draws = []
    for _ in range(2):
        task = task_gen.random_task(1 / 50, 21, family_id="cf_interceptor_office")
        env = make_env(task)
        env.reset(seed=task.map_seed)
        draws.append((env.SPEED_LIMIT, env._rc_dead_zone, env._rc_max_step,
                      env._office_motor_alpha))
        env.close()
    assert draws[0] == draws[1]


def test_office_target_size_varies_per_episode():
    """The silhouette is dealt per episode, so box size cannot be inverted into
    distance with a memorised constant."""
    sizes = set()
    for seed in (31, 32, 33):
        task = task_gen.random_task(1 / 50, seed, family_id="cf_interceptor_office")
        env = make_env(task)
        env.reset(seed=task.map_seed)
        w, h = env._office_target_w_m, env._office_target_h_m
        sizes.add((round(w, 6), round(h, 6)))
        assert abs(w - 0.18) / 0.18 <= OFFICE_TARGET_SIZE_JITTER + 1e-9
        assert abs(h - 0.08) / 0.08 <= OFFICE_TARGET_SIZE_JITTER + 1e-9
        # and the boxes the policy sees are built from THIS episode's silhouette
        fam = OfficeInterceptorChallengeFamily()
        if _place_with_clear_view(env, dist=2.0):
            truth = fam._detector_capture(env)
            assert truth is not None, "arranged a clear view but got no detection"
            focal = env._office_det_focal
            assert truth["w"] == pytest.approx(focal * w / truth["dist"], rel=1e-6)
            assert truth["h"] == pytest.approx(focal * h / truth["dist"], rel=1e-6)
        env.close()
    assert len(sizes) == 3, "each seed must deal its own silhouette"


def test_office_spawns_spread_over_the_floor():
    """Spawns are drawn over free space, not nudged from one suggested point, so
    no spatial prior over spawn location is worth memorising. Every spawn must
    still be valid: clear of geometry and inside the separation band."""
    starts, goals = [], []
    for seed in range(40, 64):
        task = task_gen.random_task(1 / 50, seed, family_id="cf_interceptor_office")
        env = make_env(task)
        env.reset(seed=task.map_seed)
        s = np.array(env.task.start, dtype=float)
        g = np.array(env.task.goal, dtype=float)
        gap = float(np.linalg.norm(s[:2] - g[:2]))
        assert OFFICE_MIN_START_DISTANCE_M - 1e-6 <= gap <= OFFICE_MAX_START_DISTANCE_M + 1e-6, (
            f"seed {seed}: separation {gap:.2f} m is outside the band")
        fam = OfficeInterceptorChallengeFamily()
        assert fam._point_is_clear(env, s, floor=True), f"seed {seed}: start is not clear"
        assert fam._point_is_clear(env, g, floor=False), f"seed {seed}: target is not clear"
        probe = np.array([s[0], s[1], 1.4 * env._office_scale[2]])
        assert fam._nav_in_main(env, probe), f"seed {seed}: start is sealed off"
        assert fam._nav_in_main(env, g), f"seed {seed}: target is sealed off"
        xr, yr = env._office_x_range, env._office_y_range
        assert xr[0] < s[0] < xr[1] and yr[0] < s[1] < yr[1], "start outside this room"
        starts.append(s[:2]); goals.append(g[:2])
        env.close()
    starts = np.array(starts)
    # Spread, not a ring: both axes must use a real share of the room.
    span_x = starts[:, 0].max() - starts[:, 0].min()
    span_y = starts[:, 1].max() - starts[:, 1].min()
    assert span_x > 0.5 * (OFFICE_X_RANGE[1] - OFFICE_X_RANGE[0]), f"x span only {span_x:.1f} m"
    assert span_y > 0.4 * (OFFICE_Y_RANGE[1] - OFFICE_Y_RANGE[0]), f"y span only {span_y:.1f} m"
    assert len({tuple(np.round(s, 3)) for s in starts}) == len(starts), "spawns repeat"


def test_office_room_size_varies_per_episode():
    """The room is dealt per episode, so a metric grid fitted to one floorplan does
    not line up with the next. Every axis still moves by a real, bounded amount."""
    from swarm.core.maps.office.builder import office_scale
    seen = set()
    for seed in (51, 52, 53, 54):
        sx, sy, sz = office_scale(seed)
        seen.add((round(sx, 6), round(sy, 6), round(sz, 6)))
        for a in (sx, sy, sz):
            assert OFFICE_SCALE_JITTER_MIN - 1e-9 <= abs(a - 1.0) <= OFFICE_SCALE_JITTER_MAX + 1e-9
    assert len(seen) == 4, "each seed must deal its own room"
    assert office_scale(51) == office_scale(51), "the room must be deterministic"


def test_office_spawns_follow_the_room(office_env):
    """Spawn bounds track the episode's room, so nobody is placed outside it."""
    env = office_env
    env.reset(seed=env.task.map_seed)
    sx, sy, _ = env._office_scale
    assert env._office_x_range == (OFFICE_X_RANGE[0] * sx, OFFICE_X_RANGE[1] * sx)
    assert env._office_y_range == (OFFICE_Y_RANGE[0] * sy, OFFICE_Y_RANGE[1] * sy)
    for pt in (np.array(env.task.start, dtype=float), np.array(env.task.goal, dtype=float)):
        assert env._office_x_range[0] < pt[0] < env._office_x_range[1]
        assert env._office_y_range[0] < pt[1] < env._office_y_range[1]


def test_office_room_size_varies_and_bounds_follow():
    """The room is dealt a size per episode, and everything that reads the room
    reads this episode's size — otherwise a stretched map would score against
    yesterday's walls."""
    from swarm.core.maps.office.builder import OFFICE_CEILING_M as NOMINAL_CEILING

    rooms = set()
    for seed in (101, 102, 103):
        task = task_gen.random_task(1 / 50, seed, family_id="cf_interceptor_office")
        env = make_env(task)
        env.reset(seed=task.map_seed)
        sx, sy, sz = env._office_scale
        for axis in (sx, sy, sz):
            assert 0.02 - 1e-9 <= abs(axis - 1.0) <= 0.05 + 1e-9, f"scale {axis} off spec"
        # the bounds the family flies by are this episode's, not the drawing's
        assert env._office_x_range == pytest.approx(
            (OFFICE_X_RANGE[0] * sx, OFFICE_X_RANGE[1] * sx))
        assert env._office_y_range == pytest.approx(
            (OFFICE_Y_RANGE[0] * sy, OFFICE_Y_RANGE[1] * sy))
        assert env._office_ceiling_m == pytest.approx(NOMINAL_CEILING * sz)
        # the target's flight band must still fit under the shrunk ceiling
        assert OFFICE_TARGET_ALT_MAX_M < env._office_ceiling_m
        rooms.add((round(sx, 6), round(sy, 6), round(sz, 6)))
        env.close()
    assert len(rooms) == 3, "each seed must deal its own room"


def test_office_room_size_is_deterministic():
    """Two builds of one seed must agree, or validators would disagree."""
    from swarm.core.maps.office.builder import office_scale
    assert office_scale(77) == office_scale(77)
    assert office_scale(77) != office_scale(78)


def test_office_fast_level_pass_still_registers(office_env):
    """A quick level pass must score. Physics runs five substeps per control step, so
    a graze can exist and separate between contact queries; the held proximity
    envelope is what makes a genuine fast interception count anyway."""
    env = office_env
    env.reset(seed=env.task.map_seed)
    fam = OfficeInterceptorChallengeFamily()
    cli = env.CLIENT
    tgt = int(env._office_target_uid)
    base = np.array([float(env.pos[0][0]), float(env.pos[0][1]), 1.60])
    p.resetBasePositionAndOrientation(tgt, base.tolist(), [0, 0, 0, 1], physicsClientId=cli)
    # sweep straight through the target at full stick speed, sampled at control cadence
    speed, dt = env.SPEED_LIMIT, 1.0 / 50.0
    env._success = False
    env._office_catch_hold = 0
    steps_inside = 0
    for k in range(-40, 41):
        pos = base + np.array([k * speed * dt, 0.0, 0.0])
        p.resetBasePositionAndOrientation(int(env.DRONE_IDS[0]), pos.tolist(),
                                          [0, 0, 0, 1], physicsClientId=cli)
        env._updateAndStoreKinematicInformation()
        p.performCollisionDetection(physicsClientId=cli)
        if abs(k * speed * dt) <= OFFICE_CATCH_RADIUS_M:
            steps_inside += 1
        fam._update_target(env)
    assert steps_inside >= OFFICE_CATCH_HOLD_STEPS, (
        f"a {speed:.2f} m/s pass only spends {steps_inside} steps inside the envelope; "
        "the hold requirement would make genuine fast intercepts unscoreable")
    assert env._success, "a full-speed level pass through the target must register"
