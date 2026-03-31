"""Live GUI visualizer for the stage-02 privileged expert on one repeated seed.

This is a standalone script for understanding expert behavior in real time.
It:
- builds the exact validator-style task from one seed via ``random_task()``
- launches the PyBullet GUI
- runs the privileged expert live on that task
- overlays current mode / distances / outcome in the GUI
- repeats the same seed forever unless a repeat count is provided

Intended use:
    python training_pipeline/stages/02_privileged_expert/visualize_expert_gui.py --seed 12345
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pybullet as p

CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]
for root in (CURRENT_DIR, DEFAULT_MODEL_ROOT, REPO_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from training_env import DEFAULT_SIM_DT, make_training_env
from training_lib.experts import (
    PrivilegedExpertConfig,
    load_expert_config,
    make_expert_policy,
)
from swarm.core.drone import track_drone
from swarm.validator.task_gen import random_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, help="Single validator-style seed to replay repeatedly.")
    parser.add_argument("--expert-config", type=Path, default=None, help="Optional JSON expert config saved by stage 02.")
    parser.add_argument("--repeat-count", type=int, default=0, help="Number of episodes to run. 0 means loop forever.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional per-episode step cap.")
    parser.add_argument("--playback-speed", type=float, default=1.0, help="1.0 is real time, 2.0 is 2x, etc.")
    parser.add_argument("--reset-delay-sec", type=float, default=1.5, help="Pause after each episode before replaying the same seed.")
    parser.add_argument("--camera", choices=("follow", "fixed"), default="follow")
    parser.add_argument("--hud-every-steps", type=int, default=2, help="Refresh HUD/debug lines every N steps.")
    parser.add_argument(
        "--disable-depth-obstacle-checks",
        action="store_true",
        default=None,
        help="Disable depth-based obstacle avoidance inside the expert.",
    )
    parser.add_argument("--privileged-raycast-stride", type=int, default=1, help="Refresh expensive privileged ray-casts every N steps.")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PrivilegedExpertConfig:
    if args.expert_config is not None:
        config = load_expert_config(args.expert_config)
    else:
        config = PrivilegedExpertConfig()
    if args.disable_depth_obstacle_checks is not None:
        config.use_depth_obstacle_checks = not bool(args.disable_depth_obstacle_checks)
    return config


def _action_to_env_shape(env, action: np.ndarray) -> np.ndarray:
    expected_shape = tuple(getattr(env.action_space, "shape", ()))
    action = np.asarray(action, dtype=np.float32)
    if expected_shape and action.shape != expected_shape:
        action = action.reshape(expected_shape)
    return action


def _set_fixed_camera(env, task) -> None:
    cli = env.getPyBulletClient()
    start = np.asarray(task.start, dtype=np.float32)
    goal = np.asarray(task.goal, dtype=np.float32)
    mid = ((start + goal) / 2.0).tolist()
    dist = float(max(4.0, np.linalg.norm(goal[:2] - start[:2]) * 0.75 + 4.0))
    p.resetDebugVisualizerCamera(
        cameraDistance=dist,
        cameraYaw=35.0,
        cameraPitch=-35.0,
        cameraTargetPosition=[mid[0], mid[1], max(start[2], goal[2]) + 1.0],
        physicsClientId=cli,
    )


class GuiHud:
    def __init__(self, cli: int):
        self.cli = cli
        self._text_ids: dict[str, int] = {}
        self._line_ids: dict[str, int] = {}

    def text(
        self,
        key: str,
        text: str,
        position: list[float],
        *,
        color: tuple[float, float, float] = (1.0, 1.0, 1.0),
        parent_body: int | None = None,
        text_size: float = 1.2,
    ) -> None:
        kwargs: dict[str, Any] = {
            "textPosition": position,
            "textColorRGB": list(color),
            "textSize": text_size,
            "replaceItemUniqueId": int(self._text_ids.get(key, -1)),
            "physicsClientId": self.cli,
        }
        if parent_body is not None:
            kwargs["parentObjectUniqueId"] = int(parent_body)
            kwargs["parentLinkIndex"] = -1
        self._text_ids[key] = int(p.addUserDebugText(text, **kwargs))

    def line(
        self,
        key: str,
        start: list[float],
        end: list[float],
        *,
        color: tuple[float, float, float] = (0.0, 1.0, 0.0),
        width: float = 2.0,
    ) -> None:
        self._line_ids[key] = int(
            p.addUserDebugLine(
                start,
                end,
                lineColorRGB=list(color),
                lineWidth=width,
                replaceItemUniqueId=int(self._line_ids.get(key, -1)),
                physicsClientId=self.cli,
            )
        )


def _challenge_label(challenge_type: int) -> str:
    return {
        1: "city",
        2: "open",
        3: "mountain",
        4: "village",
        5: "warehouse",
        6: "forest",
    }.get(int(challenge_type), f"type_{challenge_type}")


def _keyboard_requested(keys: dict[int, int], code: int) -> bool:
    state = int(keys.get(code, 0))
    return bool(state & getattr(p, "KEY_WAS_TRIGGERED", 0x1))


def _handle_keyboard() -> tuple[bool, bool]:
    keys = p.getKeyboardEvents()
    escape_code = getattr(p, "B3G_ESCAPE", 27)
    quit_requested = _keyboard_requested(keys, escape_code)
    reset_requested = _keyboard_requested(keys, ord("r")) or _keyboard_requested(keys, ord("R"))
    return quit_requested, reset_requested


def update_gui_overlay(
    *,
    env,
    hud: GuiHud,
    observation: dict[str, Any],
    info: dict[str, Any],
    metadata: dict[str, Any],
    episode_index: int,
    step_index: int,
    task,
) -> None:
    privileged = dict(info.get("privileged", {}))
    state = np.asarray(observation["state"], dtype=np.float32).reshape(-1)
    drone_pos = state[0:3].astype(np.float32)
    platform_pos = np.asarray(privileged.get("platform_position", task.goal), dtype=np.float32).reshape(3)
    target_world = np.asarray(metadata.get("target_world", platform_pos), dtype=np.float32).reshape(3)

    drone_id = int(env.DRONE_IDS[0])
    hud.text(
        "header",
        (
            f"episode={episode_index} step={step_index} "
            f"type={_challenge_label(int(privileged.get('challenge_type', task.challenge_type)))} "
            f"{'moving' if bool(privileged.get('moving_platform', task.moving_platform)) else 'static'}"
        ),
        [0.0, 0.0, 1.0],
        color=(1.0, 1.0, 0.2),
        parent_body=drone_id,
        text_size=1.35,
    )
    hud.text(
        "mode",
        (
            f"mode={metadata.get('expert_mode', 'unknown')} "
            f"dist={float(privileged.get('distance_to_platform', 0.0)):.2f}m "
            f"xy={float(privileged.get('xy_distance_to_platform', 0.0)):.2f}m "
            f"zerr={float(privileged.get('z_error_to_platform', 0.0)):.2f}m"
        ),
        [0.0, 0.0, 0.8],
        color=(0.8, 1.0, 0.8),
        parent_body=drone_id,
    )
    hud.text(
        "planner",
        (
            f"los={bool(metadata.get('line_of_sight_to_platform', False))} "
            f"cruise_z={float(metadata.get('planner_cruise_z', target_world[2])):.2f} "
            f"success={bool(info.get('success', False))} "
            f"collision={bool(info.get('collision', False))}"
        ),
        [0.0, 0.0, 0.6],
        color=(0.7, 0.9, 1.0),
        parent_body=drone_id,
    )

    hud.line("to_platform", drone_pos.tolist(), platform_pos.tolist(), color=(0.1, 0.95, 0.1), width=2.2)
    hud.line("to_target", drone_pos.tolist(), target_world.tolist(), color=(0.2, 0.4, 1.0), width=2.2)

    platform_marker = platform_pos.copy()
    platform_marker[2] += 0.25
    hud.text(
        "platform",
        "platform",
        platform_marker.tolist(),
        color=(0.0, 1.0, 0.0),
        text_size=1.1,
    )

    target_marker = target_world.copy()
    target_marker[2] += 0.18
    hud.text(
        "target",
        f"target:{metadata.get('expert_mode', 'unknown')}",
        target_marker.tolist(),
        color=(0.2, 0.4, 1.0),
        text_size=1.0,
    )


def run_episode(
    *,
    env,
    policy,
    task,
    episode_index: int,
    args: argparse.Namespace,
    hud: GuiHud,
) -> tuple[str, int]:
    observation, info = env.reset(seed=task.map_seed)
    policy.reset()

    cli = env.getPyBulletClient()
    drone_id = int(env.DRONE_IDS[0])
    if args.camera == "fixed":
        _set_fixed_camera(env, task)

    step_index = 0
    terminated = False
    truncated = False
    reason = "other"

    while not (terminated or truncated):
        if args.max_steps is not None and step_index >= args.max_steps:
            reason = "max_steps"
            break

        quit_requested, reset_requested = _handle_keyboard()
        if quit_requested:
            raise KeyboardInterrupt
        if reset_requested:
            reason = "manual_reset"
            break

        action = _action_to_env_shape(env, policy.act(observation, info))
        metadata = dict(policy.get_last_metadata())

        if args.camera == "follow":
            track_drone(cli, drone_id)
        if args.hud_every_steps > 0 and (step_index % args.hud_every_steps == 0):
            update_gui_overlay(
                env=env,
                hud=hud,
                observation=observation,
                info=info,
                metadata=metadata,
                episode_index=episode_index,
                step_index=step_index,
                task=task,
            )

        next_observation, _reward, terminated, truncated, next_info = env.step(action)
        observation = next_observation
        info = next_info
        step_index += 1

        if args.playback_speed > 0.0:
            time.sleep(float(task.sim_dt) / max(float(args.playback_speed), 1e-6))

    if bool(info.get("success", False)):
        reason = "success"
    elif bool(info.get("collision", False)):
        reason = "collision"
    elif truncated and reason == "other":
        reason = "timeout"

    metadata = dict(policy.get_last_metadata())
    update_gui_overlay(
        env=env,
        hud=hud,
        observation=observation,
        info=info,
        metadata=metadata,
        episode_index=episode_index,
        step_index=step_index,
        task=task,
    )
    return reason, step_index


def main() -> None:
    args = parse_args()
    config = build_config(args)
    policy = make_expert_policy(config)
    task = random_task(sim_dt=DEFAULT_SIM_DT, seed=int(args.seed))

    print("Expert GUI viewer")
    print(f"seed={args.seed}")
    print(f"task={asdict(task)}")
    print("expert=strong")
    print("Controls: R = reset episode, Esc = quit")

    env = make_training_env(
        task,
        gui=True,
        privileged=True,
        raycast_stride_steps=args.privileged_raycast_stride,
    )
    hud = GuiHud(env.getPyBulletClient())

    episode_index = 0
    try:
        while args.repeat_count <= 0 or episode_index < args.repeat_count:
            reason, steps = run_episode(
                env=env,
                policy=policy,
                task=task,
                episode_index=episode_index,
                args=args,
                hud=hud,
            )
            print(f"episode={episode_index} finished reason={reason} steps={steps}")
            episode_index += 1
            if args.reset_delay_sec > 0.0:
                time.sleep(float(args.reset_delay_sec))
    except KeyboardInterrupt:
        print("Stopping expert GUI viewer.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
