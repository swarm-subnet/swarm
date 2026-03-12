from __future__ import annotations

import math

import numpy as np

from scripts import visualize_map as vis_mod


def test_parser_accepts_all_map_types() -> None:
    args = vis_mod._build_parser().parse_args(["--type", "6", "--seed", "431623"])
    assert args.type == 6
    assert args.seed == 431623


def test_motion_from_pressed_keys_maps_forward_and_boosted() -> None:
    translation, yaw = vis_mod._motion_from_pressed_keys(
        {"w", "shift"}, speed=4.0, boost=2.0
    )

    assert np.allclose(translation, np.array([8.0, 0.0, 0.0], dtype=np.float32))
    assert yaw == 0.0


def test_motion_from_pressed_keys_maps_arrow_vertical() -> None:
    translation, yaw = vis_mod._motion_from_pressed_keys(
        {"up"}, speed=3.0, boost=2.0
    )

    assert np.allclose(translation, np.array([0.0, 0.0, 3.0], dtype=np.float32))
    assert yaw == 0.0


def test_normalise_keysym_maps_special_keys() -> None:
    assert vis_mod._normalise_keysym("Shift_L") == "shift"
    assert vis_mod._normalise_keysym("Up") == "up"
    assert vis_mod._normalise_keysym("Escape") == "escape"
    assert vis_mod._normalise_keysym("w") == "w"


def test_compute_render_size_scales_and_clamps() -> None:
    assert vis_mod._compute_render_size(960, 540, 0.75) == (720, 405)
    assert vis_mod._compute_render_size(100, 100, 0.1) == (320, 180)


def test_default_visual_profile_uses_map_specific_values() -> None:
    city = vis_mod._default_visual_profile(1)
    warehouse = vis_mod._default_visual_profile(5)

    assert city.render_distance == 100.0
    assert warehouse.render_scale == 0.60
    assert warehouse.render_fps == 8.0


def test_resolve_visual_profile_honors_explicit_overrides() -> None:
    args = vis_mod._build_parser().parse_args(
        [
            "--type",
            "1",
            "--seed",
            "123",
            "--render-scale",
            "0.8",
            "--render-distance",
            "44",
            "--render-fps",
            "9",
        ]
    )

    profile = vis_mod._resolve_visual_profile(args)
    assert profile.render_scale == 0.8
    assert profile.render_distance == 44.0
    assert profile.render_fps == 9.0


def test_should_render_frame_respects_motion_and_idle_rates() -> None:
    assert vis_mod._should_render_frame(
        now=1.0,
        last_render_at=None,
        has_frame=False,
        moving=False,
        pose_changed=False,
        target_fps=8.0,
        idle_fps=2.0,
    )
    assert not vis_mod._should_render_frame(
        now=1.05,
        last_render_at=1.0,
        has_frame=True,
        moving=True,
        pose_changed=True,
        target_fps=8.0,
        idle_fps=2.0,
    )
    assert vis_mod._should_render_frame(
        now=1.2,
        last_render_at=1.0,
        has_frame=True,
        moving=True,
        pose_changed=True,
        target_fps=8.0,
        idle_fps=2.0,
    )
    assert vis_mod._should_render_frame(
        now=1.6,
        last_render_at=1.0,
        has_frame=True,
        moving=False,
        pose_changed=False,
        target_fps=8.0,
        idle_fps=2.0,
    )


def test_pose_changed_detects_motion_and_yaw() -> None:
    previous = (0.0, 0.0, 1.0, 0.0)
    assert not vis_mod._pose_changed(previous, (0.01, 0.01, 1.0, 0.01))
    assert vis_mod._pose_changed(previous, (0.10, 0.0, 1.0, 0.0))
    assert vis_mod._pose_changed(previous, (0.0, 0.0, 1.0, 0.10))


def test_apply_visualizer_cull_hides_and_restores() -> None:
    class _DummyBullet:
        def __init__(self) -> None:
            self.visual_calls: list[tuple[int, tuple[float, ...]]] = []
            self.collision_calls: list[tuple[int, int, int]] = []

        def getBasePositionAndOrientation(self, _body_id, physicsClientId=None):
            _ = physicsClientId
            return ([0.0, 0.0, 1.0], (0.0, 0.0, 0.0, 1.0))

        def changeVisualShape(self, uid, _link, rgbaColor, physicsClientId=None):
            _ = physicsClientId
            self.visual_calls.append((uid, tuple(rgbaColor)))

        def setCollisionFilterGroupMask(
            self, uid, _link, group, mask, physicsClientId=None
        ):
            _ = physicsClientId
            self.collision_calls.append((uid, group, mask))

    env = type(
        "DummyEnv",
        (),
        {
            "DRONE_IDS": [7],
            "getPyBulletClient": lambda self: 99,
            "_cull_targets": [
                (1, 80.0, 0.0, 1.0, [1.0, 0.0, 0.0, 1.0]),
                (2, 10.0, 0.0, 1.0, [0.0, 1.0, 0.0, 1.0]),
            ],
            "_cull_vis_hidden": set(),
            "_cull_phys_disabled": set(),
        },
    )()
    bullet = _DummyBullet()

    vis_mod._apply_visualizer_cull(env, bullet, visual_radius=50.0, physics_radius=60.0)

    assert (1, (0.0, 0.0, 0.0, 0.0)) in bullet.visual_calls
    assert (1, 0, 0) in bullet.collision_calls

    env._cull_vis_hidden.add(2)
    env._cull_phys_disabled.add(2)
    vis_mod._apply_visualizer_cull(env, bullet, visual_radius=50.0, physics_radius=60.0)

    assert (2, (0.0, 1.0, 0.0, 1.0)) in bullet.visual_calls
    assert (2, 1, 0xFF) in bullet.collision_calls


def test_advance_free_fly_moves_drone_and_updates_yaw() -> None:
    class _DummyBullet:
        def __init__(self) -> None:
            self.position = [1.0, 2.0, 0.5]
            self.quat = (0.0, 0.0, 0.0, 1.0)
            self.reset_calls: list[tuple[list[float], list[float]]] = []

        def getBasePositionAndOrientation(self, _body_id, physicsClientId=None):
            _ = physicsClientId
            return self.position, self.quat

        def getEulerFromQuaternion(self, _quat):
            return (0.0, 0.0, 0.0)

        def getQuaternionFromEuler(self, euler):
            return tuple(euler)

        def resetBasePositionAndOrientation(
            self, _body_id, pos, quat, physicsClientId=None
        ):
            _ = physicsClientId
            self.position = list(pos)
            self.quat = tuple(quat)
            self.reset_calls.append((list(pos), list(quat)))

        def resetBaseVelocity(
            self,
            _body_id,
            linearVelocity,
            angularVelocity,
            physicsClientId=None,
        ):
            _ = linearVelocity, angularVelocity, physicsClientId

    env = type(
        "DummyEnv",
        (),
        {"DRONE_IDS": [7], "getPyBulletClient": lambda self: 99},
    )()
    bullet = _DummyBullet()

    vis_mod._advance_free_fly(
        env,
        bullet,
        np.array([2.0, 0.0, 1.0], dtype=np.float32),
        yaw_input=1.0,
        dt=0.5,
    )

    assert bullet.reset_calls
    moved_pos, moved_quat = bullet.reset_calls[-1]
    assert moved_pos[0] > 1.0
    assert moved_pos[2] > 0.5
    assert moved_quat[2] > 0.0


def test_advance_free_fly_pose_keeps_hover_when_no_motion() -> None:
    position = np.array([1.0, 2.0, 0.5], dtype=np.float32)
    next_position, next_yaw = vis_mod._advance_free_fly_pose(
        position.copy(),
        yaw=0.3,
        translation=np.zeros(3, dtype=np.float32),
        yaw_input=0.0,
        dt=0.5,
    )

    assert np.allclose(next_position, position)
    assert next_yaw == 0.3


def test_camera_eye_and_target_follow_is_close_to_drone() -> None:
    class _DummyBullet:
        def getBasePositionAndOrientation(self, _body_id, physicsClientId=None):
            _ = physicsClientId
            return ([10.0, 0.0, 1.0], (0.0, 0.0, 0.0, 1.0))

        def getEulerFromQuaternion(self, _quat):
            return (0.0, 0.0, 0.0)

    env = type(
        "DummyEnv",
        (),
        {"DRONE_IDS": [7], "getPyBulletClient": lambda self: 99},
    )()

    eye, target = vis_mod._camera_eye_and_target(env, _DummyBullet(), "follow")

    assert eye[0] < 9.0
    assert target[0] > 10.0
    assert eye[2] > 1.0


def test_fps_tracker_reports_after_interval(monkeypatch) -> None:
    times = iter([10.0, 10.2, 11.3])
    monkeypatch.setattr(vis_mod.time, "perf_counter", lambda: next(times))

    tracker = vis_mod._FpsTracker(report_every_sec=1.0)
    assert tracker.tick() is None
    msg = tracker.tick()

    assert msg is not None
    assert "FPS" in msg
    assert math.isclose(tracker.last_fps, 2.0 / 1.3, rel_tol=1e-6)

