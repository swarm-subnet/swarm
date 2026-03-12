#!/usr/bin/env python3
"""
Interactive Swarm map visualizer.

This viewer uses the same off-screen CPU renderer as ``gen_platform_images.py``
and ``generate_video.py`` so the scene matches benchmark media output instead of
relying on PyBullet's live GUI rendering.

Controls
--------
W / S        forward / backward
A / D        strafe left / right
Arrow Up     climb
Arrow Down   descend
Q / E        yaw left / right
Shift+WASD   boosted movement
R            reset to the task start
Esc          quit
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@dataclass(frozen=True)
class _MapVisualProfile:
    render_scale: float
    render_distance: float
    render_fps: float


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Open a Swarm map in a live render window and manually fly the drone.",
    )
    parser.add_argument(
        "--type",
        type=int,
        required=True,
        choices=[1, 2, 3, 4, 5, 6],
        help="Challenge type (1=City 2=Open 3=Mountain 4=Village 5=Warehouse 6=Forest).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Map seed for deterministic generation.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=4.0,
        help="Base flight speed in metres per second (default: 4.0).",
    )
    parser.add_argument(
        "--boost",
        type=float,
        default=2.0,
        help="Multiplier for shifted movement (default: 2.0).",
    )
    parser.add_argument(
        "--camera",
        choices=["follow", "fixed"],
        default="follow",
        help="Viewer camera mode (default: follow).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=960,
        help="Window width (default: 960).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=540,
        help="Window height (default: 540).",
    )
    parser.add_argument(
        "--render-scale",
        type=float,
        default=None,
        help="Internal render scale relative to window size. Defaults depend on map type.",
    )
    parser.add_argument(
        "--render-distance",
        type=float,
        default=None,
        help="Maximum camera/render distance in metres. Defaults depend on map type.",
    )
    parser.add_argument(
        "--render-fps",
        type=float,
        default=None,
        help="Maximum render FPS. Defaults depend on map type.",
    )
    return parser


def _ensure_local_ansible_temp() -> None:
    ansible_tmp = Path(os.environ.get("ANSIBLE_LOCAL_TEMP", "/tmp/swarm_ansible"))
    ansible_tmp.mkdir(parents=True, exist_ok=True)
    os.environ["ANSIBLE_LOCAL_TEMP"] = str(ansible_tmp)


def _motion_from_pressed_keys(
    pressed: Iterable[str], speed: float, boost: float
) -> Tuple[np.ndarray, float]:
    pressed_set = set(pressed)
    translation = np.zeros(3, dtype=np.float32)
    yaw = 0.0
    boosted = "shift" in pressed_set

    key_speed = float(speed * (boost if boosted else 1.0))

    if "w" in pressed_set:
        translation[0] = key_speed
    elif "s" in pressed_set:
        translation[0] = -key_speed
    if "a" in pressed_set:
        translation[1] = -key_speed
    elif "d" in pressed_set:
        translation[1] = key_speed
    if "up" in pressed_set:
        translation[2] = key_speed
    elif "down" in pressed_set:
        translation[2] = -key_speed
    if "q" in pressed_set:
        yaw = -1.0 * (boost if boosted else 1.0)
    elif "e" in pressed_set:
        yaw = 1.0 * (boost if boosted else 1.0)

    return translation, yaw


def _normalise_keysym(keysym: str) -> str:
    lowered = (keysym or "").lower()
    if lowered in {"shift_l", "shift_r"}:
        return "shift"
    if lowered in {"up", "down", "left", "right", "escape"}:
        return lowered
    if len(lowered) == 1:
        return lowered
    return lowered


def _compute_render_size(width: int, height: int, scale: float) -> Tuple[int, int]:
    render_w = max(320, int(round(width * scale)))
    render_h = max(180, int(round(height * scale)))
    return render_w, render_h


def _default_visual_profile(challenge_type: int) -> _MapVisualProfile:
    profiles = {
        1: _MapVisualProfile(render_scale=0.65, render_distance=100.0, render_fps=8.0),
        2: _MapVisualProfile(render_scale=0.72, render_distance=100.0, render_fps=8.0),
        3: _MapVisualProfile(render_scale=0.68, render_distance=100.0, render_fps=6.0),
        4: _MapVisualProfile(render_scale=0.66, render_distance=100.0, render_fps=8.0),
        5: _MapVisualProfile(render_scale=0.60, render_distance=100.0, render_fps=8.0),
        6: _MapVisualProfile(render_scale=0.65, render_distance=100.0, render_fps=7.0),
    }
    return profiles[int(challenge_type)]


def _resolve_visual_profile(args) -> _MapVisualProfile:
    defaults = _default_visual_profile(int(args.type))
    return _MapVisualProfile(
        render_scale=float(args.render_scale)
        if args.render_scale is not None
        else defaults.render_scale,
        render_distance=float(args.render_distance)
        if args.render_distance is not None
        else defaults.render_distance,
        render_fps=float(args.render_fps)
        if args.render_fps is not None
        else defaults.render_fps,
    )


def _get_drone_pose(env, pybullet_module) -> Tuple[np.ndarray, float]:
    drone_id = int(env.DRONE_IDS[0])
    pos, quat = pybullet_module.getBasePositionAndOrientation(
        drone_id, physicsClientId=env.getPyBulletClient()
    )
    yaw = pybullet_module.getEulerFromQuaternion(quat)[2]
    return np.asarray(pos, dtype=np.float32), float(yaw)


def _set_drone_pose(env, pybullet_module, position: np.ndarray, yaw: float) -> None:
    drone_id = int(env.DRONE_IDS[0])
    quat = pybullet_module.getQuaternionFromEuler([0.0, 0.0, yaw])
    pybullet_module.resetBasePositionAndOrientation(
        drone_id,
        position.tolist(),
        quat,
        physicsClientId=env.getPyBulletClient(),
    )
    pybullet_module.resetBaseVelocity(
        drone_id,
        linearVelocity=[0.0, 0.0, 0.0],
        angularVelocity=[0.0, 0.0, 0.0],
        physicsClientId=env.getPyBulletClient(),
    )


def _advance_free_fly(
    env,
    pybullet_module,
    translation: np.ndarray,
    yaw_input: float,
    dt: float,
) -> None:
    position, yaw = _get_drone_pose(env, pybullet_module)
    yaw += yaw_input * 1.8 * dt

    cos_yaw = float(np.cos(yaw))
    sin_yaw = float(np.sin(yaw))
    forward = np.array([cos_yaw, sin_yaw, 0.0], dtype=np.float32)
    right = np.array([-sin_yaw, cos_yaw, 0.0], dtype=np.float32)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    world_delta = (
        forward * float(translation[0])
        + right * float(translation[1])
        + up * float(translation[2])
    ) * float(dt)

    next_position = position + world_delta
    next_position[2] = max(0.15, float(next_position[2]))
    _set_drone_pose(env, pybullet_module, next_position, yaw)


def _camera_eye_and_target(
    env, pybullet_module, mode: str
) -> Tuple[list[float], list[float]]:
    position, yaw = _get_drone_pose(env, pybullet_module)
    cos_yaw = float(np.cos(yaw))
    sin_yaw = float(np.sin(yaw))
    forward = np.array([cos_yaw, sin_yaw, 0.0], dtype=np.float32)

    if mode == "fixed":
        target = position + np.array([0.0, 0.0, 0.5], dtype=np.float32)
        eye = target + np.array([-12.0, -12.0, 7.0], dtype=np.float32)
        return eye.tolist(), target.tolist()

    target = position + forward * 1.25 + np.array([0.0, 0.0, 0.20], dtype=np.float32)
    eye = position - forward * 0.55 + np.array([0.0, 0.0, 0.28], dtype=np.float32)
    return eye.tolist(), target.tolist()


def _render_frame(
    env,
    pybullet_module,
    mode: str,
    width: int,
    height: int,
    render_distance: float,
) -> np.ndarray:
    cli = env.getPyBulletClient()
    eye, target = _camera_eye_and_target(env, pybullet_module, mode)
    view = pybullet_module.computeViewMatrix(
        cameraEyePosition=eye,
        cameraTargetPosition=target,
        cameraUpVector=[0.0, 0.0, 1.0],
    )
    projection = pybullet_module.computeProjectionMatrixFOV(
        fov=52.0,
        aspect=width / height,
        nearVal=0.05,
        farVal=max(10.0, float(render_distance)),
    )
    _, _, rgba, _, _ = pybullet_module.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view,
        projectionMatrix=projection,
        renderer=pybullet_module.ER_TINY_RENDERER,
        shadow=0,
        lightDirection=[0.4, 0.4, 1.0],
        physicsClientId=cli,
    )
    return np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]


def _apply_visualizer_cull(
    env,
    pybullet_module,
    visual_radius: float,
    physics_radius: float,
) -> None:
    if not hasattr(env, "_cull_targets"):
        return

    cli = env.getPyBulletClient()
    drone_pos = pybullet_module.getBasePositionAndOrientation(
        int(env.DRONE_IDS[0]), physicsClientId=cli
    )[0]
    dx, dy = float(drone_pos[0]), float(drone_pos[1])
    vis_hidden = getattr(env, "_cull_vis_hidden", set())
    phys_disabled = getattr(env, "_cull_phys_disabled", set())

    for uid, cx, cy, hs, rgba in getattr(env, "_cull_targets", ()):
        dist = float(np.hypot(cx - dx, cy - dy))
        surface_dist = dist - hs

        if surface_dist > visual_radius:
            if uid not in vis_hidden:
                pybullet_module.changeVisualShape(
                    uid, -1, rgbaColor=[0, 0, 0, 0], physicsClientId=cli
                )
                vis_hidden.add(uid)
        elif uid in vis_hidden:
            pybullet_module.changeVisualShape(
                uid, -1, rgbaColor=rgba, physicsClientId=cli
            )
            vis_hidden.discard(uid)

        if surface_dist > physics_radius:
            if uid not in phys_disabled:
                pybullet_module.setCollisionFilterGroupMask(
                    uid, -1, 0, 0, physicsClientId=cli
                )
                phys_disabled.add(uid)
        elif uid in phys_disabled:
            pybullet_module.setCollisionFilterGroupMask(
                uid, -1, 1, 0xFF, physicsClientId=cli
            )
            phys_disabled.discard(uid)


def _capture_pose_snapshot(env, pybullet_module) -> Tuple[float, float, float, float]:
    position, yaw = _get_drone_pose(env, pybullet_module)
    return float(position[0]), float(position[1]), float(position[2]), float(yaw)


def _pose_changed(
    previous: Tuple[float, float, float, float] | None,
    current: Tuple[float, float, float, float],
    position_epsilon: float = 0.04,
    yaw_epsilon: float = 0.04,
) -> bool:
    if previous is None:
        return True

    dx = current[0] - previous[0]
    dy = current[1] - previous[1]
    dz = current[2] - previous[2]
    position_delta = float(np.sqrt(dx * dx + dy * dy + dz * dz))
    yaw_delta = abs(current[3] - previous[3])
    if yaw_delta > np.pi:
        yaw_delta = abs((2.0 * np.pi) - yaw_delta)
    return position_delta >= position_epsilon or yaw_delta >= yaw_epsilon


def _should_render_frame(
    now: float,
    last_render_at: float | None,
    has_frame: bool,
    moving: bool,
    pose_changed: bool,
    target_fps: float,
    idle_fps: float,
    force: bool = False,
) -> bool:
    if force or not has_frame or last_render_at is None:
        return True

    elapsed = now - last_render_at
    if moving or pose_changed:
        return elapsed >= (1.0 / max(target_fps, 0.1))
    return elapsed >= (1.0 / max(idle_fps, 0.1))


def _print_controls(type_label: str, seed: int) -> None:
    print("=" * 72)
    print(" Swarm Map Visualizer")
    print("=" * 72)
    print(f" Map type: {type_label}")
    print(f" Seed:     {seed}")
    print(" Controls:")
    print("   W/S        forward/backward")
    print("   A/D        strafe left/right")
    print("   Up/Down    climb/descend")
    print("   Q/E        yaw left/right")
    print("   Shift+key  boosted motion")
    print("   R          reset to start")
    print("   Esc        quit")
    print("=" * 72)


class _FpsTracker:
    def __init__(self, report_every_sec: float = 1.0) -> None:
        self._report_every_sec = float(report_every_sec)
        self._window_start = time.perf_counter()
        self._last_report = self._window_start
        self._frames = 0
        self.last_fps = 0.0

    def tick(self) -> str | None:
        self._frames += 1
        now = time.perf_counter()
        elapsed = now - self._last_report
        if elapsed < self._report_every_sec:
            return None
        self.last_fps = self._frames / max(elapsed, 1e-6)
        total_elapsed = now - self._window_start
        message = f"[visualizer] {self.last_fps:.1f} FPS | elapsed {total_elapsed:.1f}s"
        self._last_report = now
        self._frames = 0
        return message


class _TkViewer:
    def __init__(self, width: int, height: int, title: str):
        import tkinter as tk
        from PIL import Image, ImageTk

        self._image_mod = Image
        self._imagetk_mod = ImageTk
        self._pressed: set[str] = set()
        self.reset_requested = False
        self.quit_requested = False

        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry(f"{width}x{height}")
        self.root.minsize(640, 360)
        self.root.protocol("WM_DELETE_WINDOW", self._request_quit)
        self.root.bind("<KeyPress>", self._on_press)
        self.root.bind("<KeyRelease>", self._on_release)

        self.label = tk.Label(self.root)
        self.label.pack(fill="both", expand=True)
        self.label.focus_set()
        self._photo = None

    @property
    def pressed(self) -> set[str]:
        return set(self._pressed)

    def _request_quit(self) -> None:
        self.quit_requested = True

    def _on_press(self, event) -> None:
        key = _normalise_keysym(getattr(event, "keysym", ""))
        if not key:
            return
        if key == "escape":
            self.quit_requested = True
            return
        if key == "r":
            self.reset_requested = True
            return
        self._pressed.add(key)

    def _on_release(self, event) -> None:
        key = _normalise_keysym(getattr(event, "keysym", ""))
        if not key:
            return
        self._pressed.discard(key)

    def draw(self, frame: np.ndarray) -> None:
        image = self._image_mod.fromarray(frame)
        current_w = max(1, self.label.winfo_width())
        current_h = max(1, self.label.winfo_height())
        if (current_w, current_h) != image.size:
            image = image.resize(
                (current_w, current_h), self._image_mod.Resampling.BILINEAR
            )
        self._photo = self._imagetk_mod.PhotoImage(image=image)
        self.label.configure(image=self._photo)

    def pump(self) -> None:
        self.root.update_idletasks()
        self.root.update()

    def consume_reset(self) -> bool:
        if self.reset_requested:
            self.reset_requested = False
            return True
        return False

    def close(self) -> None:
        try:
            self.root.destroy()
        except Exception:
            pass

    def set_title(self, title: str) -> None:
        try:
            self.root.title(title)
        except Exception:
            pass


def main() -> None:
    args = _build_parser().parse_args()

    import pybullet as p

    from scripts.generate_video import TYPE_LABELS, build_task
    from swarm.utils.env_factory import make_env

    _ensure_local_ansible_temp()
    task = build_task(seed=args.seed, challenge_type=args.type)
    env = make_env(task, gui=False)
    profile = _resolve_visual_profile(args)
    render_width, render_height = _compute_render_size(
        args.width, args.height, profile.render_scale
    )
    visual_cull_radius = max(15.0, float(profile.render_distance))
    physics_cull_radius = max(
        visual_cull_radius + 10.0, float(profile.render_distance) + 10.0
    )
    render_fps = max(2.0, float(profile.render_fps))
    idle_render_fps = max(1.0, min(2.0, render_fps / 3.0))

    if hasattr(env, "_restore_culled_bodies"):
        try:
            env._restore_culled_bodies()
        except Exception:
            pass

    type_label = TYPE_LABELS.get(args.type, f"type{args.type}")
    window_name = f"Swarm Visualizer - {type_label} - seed {args.seed}"
    _print_controls(type_label, args.seed)
    print(
        f" Window: {args.width}x{args.height} | Render: {render_width}x{render_height} | Renderer: cpu-tiny | Distance: {profile.render_distance:.0f}m | Target FPS: {render_fps:.1f}",
        flush=True,
    )
    viewer = _TkViewer(args.width, args.height, window_name)
    fps = _FpsTracker()
    cached_frame: np.ndarray | None = None
    last_render_at: float | None = None
    last_render_pose: Tuple[float, float, float, float] | None = None
    last_cull_update_at: float | None = None
    last_tick_at = time.perf_counter()
    sim_accumulator = 0.0

    try:
        while True:
            viewer.pump()
            loop_now = time.perf_counter()
            loop_dt = min(0.1, max(0.0, loop_now - last_tick_at))
            last_tick_at = loop_now
            sim_accumulator += loop_dt

            if viewer.quit_requested:
                break

            if viewer.consume_reset():
                env.reset(seed=task.map_seed)
                if hasattr(env, "_cull_enabled"):
                    env._cull_enabled = False
                if hasattr(env, "_restore_culled_bodies"):
                    try:
                        env._restore_culled_bodies()
                    except Exception:
                        pass
                cached_frame = None
                last_render_at = None
                last_render_pose = None
                last_cull_update_at = None
                sim_accumulator = 0.0
                continue

            translation, yaw_input = _motion_from_pressed_keys(
                viewer.pressed, args.speed, args.boost
            )
            moving = bool(np.any(translation) or yaw_input)

            while sim_accumulator >= task.sim_dt:
                if moving:
                    _advance_free_fly(env, p, translation, yaw_input, task.sim_dt)
                p.stepSimulation(physicsClientId=env.getPyBulletClient())
                sim_accumulator -= task.sim_dt

            current_pose = _capture_pose_snapshot(env, p)
            pose_changed = _pose_changed(last_render_pose, current_pose)
            should_render = _should_render_frame(
                now=loop_now,
                last_render_at=last_render_at,
                has_frame=cached_frame is not None,
                moving=moving,
                pose_changed=pose_changed,
                target_fps=render_fps,
                idle_fps=idle_render_fps,
            )

            if should_render:
                if (
                    last_cull_update_at is None
                    or moving
                    or (loop_now - last_cull_update_at) >= 0.15
                ):
                    _apply_visualizer_cull(
                        env,
                        p,
                        visual_radius=visual_cull_radius,
                        physics_radius=physics_cull_radius,
                    )
                    last_cull_update_at = loop_now

                cached_frame = _render_frame(
                    env,
                    p,
                    args.camera,
                    render_width,
                    render_height,
                    profile.render_distance,
                )
                viewer.draw(cached_frame)
                last_render_at = time.perf_counter()
                last_render_pose = current_pose

                fps_msg = fps.tick()
                if fps_msg is not None:
                    print(fps_msg, flush=True)
                    viewer.set_title(f"{window_name} | {fps.last_fps:.1f} FPS")

            time.sleep(0.002 if moving else 0.005)
    finally:
        viewer.close()
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
