"""Privileged expert controller used by data-collection and DAgger."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from swarm.constants import (
    PLATFORM_DELAY_MAX,
    PLATFORM_DELAY_MIN,
    PLATFORM_LINEAR_DIRECTIONS,
    PLATFORM_MOVEMENT_PATTERNS,
    PLATFORM_RADIUS_MAX,
    PLATFORM_RADIUS_MIN,
    PLATFORM_SPEED_MAX,
    PLATFORM_SPEED_MIN,
    PLATFORM_TRANSITION_MAX,
    PLATFORM_TRANSITION_MIN,
)

from .common import load_json, save_json
from .geometry import (
    action_from_target_vector,
    choose_lateral_detour,
    front_depth_m,
    normalize,
    relative_vector_to_pixel,
    yaw_from_direction,
)


CHALLENGE_TYPE_LABELS = {
    1: "city",
    2: "open",
    3: "mountain",
    4: "village",
    5: "warehouse",
    6: "forest",
}
EXPERT_ACTION_DIM = 5
EXPERT_MODE_VOCAB_VERSION = "1"
EXPERT_MODE_ORDER = (
    "idle",
    "climb_to_cruise",
    "transit_to_hover",
    "direct_hover",
    "hover_align",
    "avoid_obstacle",
    "recover_stall",
    "recover_tilt",
    "descend_static",
    "sink_static",
    "landing_commit_static",
    "touchdown_static",
    "hold_contact_static",
    "settle_static",
    "match_velocity_moving",
    "descend_moving",
    "commit_contact_moving",
)
EXPERT_MODE_TO_ID = {mode: idx for idx, mode in enumerate(EXPERT_MODE_ORDER)}


def challenge_type_label(challenge_type: int) -> str:
    return CHALLENGE_TYPE_LABELS.get(int(challenge_type), f"type_{challenge_type}")


def map_category_label(challenge_type: int, moving_platform: bool) -> str:
    motion = "dynamic" if bool(moving_platform) else "static"
    return f"{challenge_type_label(int(challenge_type))}_{motion}"


def expert_mode_id(mode_name: str) -> int:
    return int(EXPERT_MODE_TO_ID.get(str(mode_name), EXPERT_MODE_TO_ID["idle"]))


def default_mode_vocabulary_payload() -> dict[str, Any]:
    return {
        "version": EXPERT_MODE_VOCAB_VERSION,
        "action_dim": EXPERT_ACTION_DIM,
        "modes": [
            {
                "id": int(mode_id),
                "name": mode_name,
            }
            for mode_name, mode_id in EXPERT_MODE_TO_ID.items()
        ],
    }


@lru_cache(maxsize=1024)
def _platform_motion_params_from_seed(seed: int) -> tuple[str, float, float, float, float, float, str, float]:
    seed = int(seed)
    selector_rng = np.random.RandomState(seed & 0xFFFFFFFF)
    for _ in range(4):
        selector_rng.rand()
    pattern_idx = int(selector_rng.randint(0, len(PLATFORM_MOVEMENT_PATTERNS)))
    pattern = str(PLATFORM_MOVEMENT_PATTERNS[pattern_idx])

    param_rng = np.random.RandomState((seed + 77777) & 0xFFFFFFFF)
    speed = float(param_rng.uniform(PLATFORM_SPEED_MIN, PLATFORM_SPEED_MAX))
    radius = float(param_rng.uniform(PLATFORM_RADIUS_MIN, PLATFORM_RADIUS_MAX))
    delay = float(param_rng.uniform(PLATFORM_DELAY_MIN, PLATFORM_DELAY_MAX))
    transition = float(param_rng.uniform(PLATFORM_TRANSITION_MIN, PLATFORM_TRANSITION_MAX))
    phase = float(param_rng.uniform(0.0, 2.0 * math.pi))
    linear_dir = str(PLATFORM_LINEAR_DIRECTIONS[int(param_rng.randint(0, len(PLATFORM_LINEAR_DIRECTIONS)))])
    linear_angle = float(param_rng.uniform(0.0, 2.0 * math.pi))
    return pattern, speed, radius, delay, transition, phase, linear_dir, linear_angle


def _platform_nominal_orbit_state(
    center: np.ndarray,
    *,
    pattern: str,
    speed: float,
    radius: float,
    phase: float,
    linear_dir: str,
    linear_angle: float,
    t_eff: float,
) -> tuple[np.ndarray, np.ndarray]:
    center = np.asarray(center, dtype=np.float32).reshape(3)
    omega = float(speed) * (0.5 if pattern == "linear" else 0.3)

    if pattern == "circular":
        angle = float(t_eff) * float(speed) * 0.3 + float(phase)
        position = np.array(
            [
                float(center[0]) + float(radius) * math.cos(angle),
                float(center[1]) + float(radius) * math.sin(angle),
                float(center[2]),
            ],
            dtype=np.float32,
        )
        velocity = np.array(
            [
                -float(radius) * math.sin(angle) * (float(speed) * 0.3),
                float(radius) * math.cos(angle) * (float(speed) * 0.3),
                0.0,
            ],
            dtype=np.float32,
        )
        return position, velocity

    if pattern == "linear":
        angle = float(t_eff) * float(speed) * 0.5 + float(phase)
        offset = float(radius) * math.sin(angle)
        offset_rate = float(radius) * math.cos(angle) * omega
        if linear_dir == "x":
            direction = np.array([1.0, 0.0], dtype=np.float32)
        elif linear_dir == "y":
            direction = np.array([0.0, 1.0], dtype=np.float32)
        else:
            direction = np.array([math.cos(linear_angle), math.sin(linear_angle)], dtype=np.float32)
        position = center.copy()
        position[0:2] += direction * offset
        velocity = np.zeros(3, dtype=np.float32)
        velocity[0:2] = direction * offset_rate
        return position, velocity

    if pattern == "figure8":
        angle = float(t_eff) * float(speed) * 0.3 + float(phase)
        position = np.array(
            [
                float(center[0]) + float(radius) * math.sin(angle),
                float(center[1]) + float(radius) * math.sin(2.0 * angle) * 0.5,
                float(center[2]),
            ],
            dtype=np.float32,
        )
        velocity = np.array(
            [
                float(radius) * math.cos(angle) * (float(speed) * 0.3),
                float(radius) * math.cos(2.0 * angle) * (float(speed) * 0.3),
                0.0,
            ],
            dtype=np.float32,
        )
        return position, velocity

    return center.copy(), np.zeros(3, dtype=np.float32)


def _platform_nominal_state(center: np.ndarray, seed: int, time_alive: float) -> tuple[np.ndarray, np.ndarray, str]:
    pattern, speed, radius, delay, transition, phase, linear_dir, linear_angle = _platform_motion_params_from_seed(seed)
    center = np.asarray(center, dtype=np.float32).reshape(3)
    t = max(0.0, float(time_alive))

    if t < delay:
        return center.copy(), np.zeros(3, dtype=np.float32), pattern

    orbit_start_pos, _ = _platform_nominal_orbit_state(
        center,
        pattern=pattern,
        speed=speed,
        radius=radius,
        phase=phase,
        linear_dir=linear_dir,
        linear_angle=linear_angle,
        t_eff=0.0,
    )
    if t < delay + transition:
        ratio = (t - delay) / max(transition, 1e-6)
        smooth = ratio * ratio * (3.0 - 2.0 * ratio)
        smooth_rate = (6.0 * ratio * (1.0 - ratio)) / max(transition, 1e-6)
        displacement = orbit_start_pos - center
        position = center + smooth * displacement
        velocity = displacement * smooth_rate
        return position.astype(np.float32), velocity.astype(np.float32), pattern

    orbit_pos, orbit_vel = _platform_nominal_orbit_state(
        center,
        pattern=pattern,
        speed=speed,
        radius=radius,
        phase=phase,
        linear_dir=linear_dir,
        linear_angle=linear_angle,
        t_eff=t - delay - transition,
    )
    return orbit_pos.astype(np.float32), orbit_vel.astype(np.float32), pattern


@dataclass(frozen=True)
class ExpertIdentity:
    teacher_id: str = "expert_shared"
    teacher_version: str = "v0"
    map_category: str = "global"
    parent_teacher_id: str | None = None


@dataclass(frozen=True)
class ExpertQualityGate:
    min_success_rate: float = 0.0
    min_mean_score: float = 0.0
    min_num_episodes: int = 1


@dataclass(frozen=True)
class SpecialistExpertSpec:
    map_category: str
    teacher_id: str
    teacher_version: str
    config_override_path: str | None = None
    eval_summary_path: str | None = None
    quality_gate_path: str | None = None
    dataset_weight: float = 1.0
    enabled: bool = True
    parent_teacher_id: str | None = None
    notes: str = ""


def _resolve_relative_path(root_path: str | Path, relative_path: str | None) -> Path | None:
    if relative_path is None:
        return None
    relative = Path(relative_path)
    if relative.is_absolute():
        return relative
    return Path(root_path).resolve().parent / relative


def load_quality_gate(path: str | Path) -> ExpertQualityGate:
    return ExpertQualityGate(**load_json(path))


def load_specialist_registry(path: str | Path) -> dict[str, Any]:
    payload = load_json(path)
    if "shared" not in payload or "experts" not in payload:
        raise ValueError(f"Invalid specialist registry at {path}: expected 'shared' and 'experts'")
    mode_vocab_path = _resolve_relative_path(path, payload["shared"].get("mode_vocabulary_path"))
    if mode_vocab_path is None:
        raise ValueError(f"Specialist registry at {path} is missing shared.mode_vocabulary_path")
    mode_vocab = load_json(mode_vocab_path)
    expected_mode_vocab = default_mode_vocabulary_payload()
    if mode_vocab.get("action_dim") != expected_mode_vocab["action_dim"]:
        raise ValueError(
            f"Mode vocabulary at {mode_vocab_path} has action_dim={mode_vocab.get('action_dim')}, "
            f"expected {expected_mode_vocab['action_dim']}"
        )
    if [row.get("name") for row in mode_vocab.get("modes", [])] != list(EXPERT_MODE_ORDER):
        raise ValueError(
            f"Mode vocabulary at {mode_vocab_path} does not match the shared expert mode order."
        )
    return payload


def iter_specialist_specs(registry_payload: Mapping[str, Any]) -> list[SpecialistExpertSpec]:
    return [SpecialistExpertSpec(**row) for row in registry_payload.get("experts", [])]


def resolve_specialist_spec(registry_payload: Mapping[str, Any], map_category: str) -> SpecialistExpertSpec:
    matches = [
        spec
        for spec in iter_specialist_specs(registry_payload)
        if spec.enabled and str(spec.map_category) == str(map_category)
    ]
    if not matches:
        raise KeyError(f"No enabled specialist expert registered for map_category={map_category!r}")
    if len(matches) > 1:
        raise ValueError(f"Multiple specialist experts registered for map_category={map_category!r}")
    return matches[0]


def compose_expert_config(
    *,
    base_config_path: str | Path,
    config_override_path: str | Path | None = None,
) -> PrivilegedExpertConfig:
    payload = load_json(base_config_path)
    payload.pop("expert_kind", None)
    if config_override_path is not None:
        overrides = load_json(config_override_path)
        overrides.pop("expert_kind", None)
        payload.update(overrides)
    return PrivilegedExpertConfig(**payload)


def build_specialist_policy(
    *,
    registry_path: str | Path,
    map_category: str,
) -> tuple["PrivilegedExpertPolicy", SpecialistExpertSpec]:
    registry = load_specialist_registry(registry_path)
    spec = resolve_specialist_spec(registry, map_category)
    shared = dict(registry["shared"])
    base_config_path = _resolve_relative_path(registry_path, shared.get("base_config_path"))
    if base_config_path is None:
        raise ValueError(f"Specialist registry at {registry_path} is missing shared.base_config_path")
    override_path = _resolve_relative_path(registry_path, spec.config_override_path)
    config = compose_expert_config(
        base_config_path=base_config_path,
        config_override_path=override_path,
    )
    identity = ExpertIdentity(
        teacher_id=spec.teacher_id,
        teacher_version=spec.teacher_version,
        map_category=spec.map_category,
        parent_teacher_id=spec.parent_teacher_id,
    )
    return PrivilegedExpertPolicy(config=config, identity=identity), spec


def evaluate_quality_gate(
    *,
    summary: Mapping[str, Any],
    map_category: str,
    quality_gate: ExpertQualityGate,
    teacher_id: str,
    teacher_version: str,
) -> dict[str, Any]:
    metrics = dict(summary.get("by_map_category", {}).get(str(map_category), {}))
    if not metrics:
        raise KeyError(f"Summary does not contain metrics for map_category={map_category!r}")
    num_episodes = int(metrics.get("num_episodes", 0))
    success_rate = float(metrics.get("success_rate", 0.0))
    mean_score = float(metrics.get("mean_score", 0.0))
    accepted = (
        num_episodes >= int(quality_gate.min_num_episodes)
        and success_rate >= float(quality_gate.min_success_rate)
        and mean_score >= float(quality_gate.min_mean_score)
    )
    return {
        "teacher_id": str(teacher_id),
        "teacher_version": str(teacher_version),
        "map_category": str(map_category),
        "accepted": bool(accepted),
        "quality_gate": asdict(quality_gate),
        "observed_metrics": {
            "num_episodes": num_episodes,
            "success_rate": success_rate,
            "mean_score": mean_score,
        },
    }


def _clip_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm <= max_norm or norm <= 1e-8:
        return vector.astype(np.float32)
    return (vector * (max_norm / norm)).astype(np.float32)


def _dominant_target_yaw(direction: np.ndarray, fallback_yaw_norm: float = 0.0) -> float:
    direction = np.asarray(direction, dtype=np.float32).reshape(3)
    if np.linalg.norm(direction[:2]) < 1e-6:
        return float(fallback_yaw_norm)
    return float(yaw_from_direction(direction))


@dataclass
class PrivilegedExpertConfig:
    """Readable, tunable parameters for the privileged teacher."""

    cruise_speed_m_s: float = 1.8
    climb_speed_m_s: float = 1.0
    align_speed_m_s: float = 0.6
    descend_speed_m_s: float = 0.35
    touchdown_speed_m_s: float = 0.18
    recovery_speed_m_s: float = 0.7

    obstacle_distance_m: float = 1.8
    use_depth_obstacle_checks: bool = True
    speed_limit_m_s: float = 3.0

    static_hover_height_m: float = 1.0
    moving_hover_height_m: float = 1.25
    static_descend_xy_radius_m: float = 0.80
    moving_descend_xy_radius_m: float = 1.20
    static_touchdown_xy_radius_m: float = 0.80
    moving_touchdown_xy_radius_m: float = 1.00
    hover_z_tolerance_m: float = 0.35
    static_sink_xy_radius_m: float = 0.80
    static_sink_height_m: float = 0.08
    sink_speed_m_s: float = 0.65
    static_landing_commit_xy_radius_m: float = 0.30
    static_landing_commit_height_m: float = 0.38
    static_settle_xy_radius_m: float = 0.60
    static_settle_release_xy_radius_m: float = 0.90
    static_settle_height_m: float = 0.03
    static_settle_speed_m_s: float = 0.22
    static_settle_damping: float = 1.10

    static_rel_xy_speed_gate_m_s: float = 0.35
    moving_rel_xy_speed_gate_m_s: float = 0.85
    static_vertical_speed_gate_m_s: float = 0.25

    position_damping: float = 0.55
    touchdown_damping: float = 0.85
    vertical_bias_gain: float = 0.25

    moving_intercept_base_sec: float = 0.35
    moving_intercept_gain_sec_per_m: float = 0.10
    moving_intercept_horizon_cap_sec: float = 1.4
    moving_terminal_lead_sec: float = 0.12
    moving_transit_speed_cap_city_m_s: float = 0.90
    moving_transit_speed_cap_non_city_m_s: float = 1.20
    moving_match_xy_radius_m: float = 8.0
    moving_match_release_xy_radius_m: float = 10.0
    moving_match_xy_radius_non_open_m: float = 4.5
    moving_match_release_xy_radius_non_open_m: float = 6.0
    moving_match_height_m: float = 0.22
    moving_match_rel_xy_speed_gate_m_s: float = 0.90
    moving_match_damping: float = 1.05
    moving_terminal_hover_speed_m_s: float = 0.35
    moving_hover_align_speed_cap_m_s: float = 0.35
    moving_commit_xy_radius_m: float = 1.10
    moving_commit_release_xy_radius_m: float = 3.00
    moving_commit_height_m: float = 0.40
    moving_commit_rel_xy_speed_gate_m_s: float = 1.50
    moving_commit_speed_m_s: float = 0.32
    moving_contact_speed_m_s: float = 0.34
    moving_contact_target_z_offset_m: float = 0.06
    moving_contact_guard_height_m: float = 0.00
    moving_below_platform_recover_bias: float = 0.18
    moving_terminal_near_target_speed_floor_m_s: float = 0.55
    moving_terminal_near_target_speed_gain_m_s_per_m: float = 0.45
    use_pattern_terminal_match: bool = True

    min_progress_m: float = 0.18
    stall_timeout_sec: float = 2.5
    recovery_hold_sec: float = 0.8
    recovery_lateral_m: float = 1.0
    recovery_forward_m: float = 2.0
    recovery_climb_extra_m: float = 1.0
    max_recovery_bias_m: float = 4.0
    recovery_bias_block_height_m: float = 7.0

    max_speed_delta_norm_per_step: float = 0.03
    max_yaw_delta_norm_per_step: float = 0.06
    direction_blend: float = 0.25
    near_target_speed_radius_m: float = 4.0
    near_target_speed_floor_m_s: float = 0.30
    near_target_speed_gain_m_s_per_m: float = 0.25
    aggressive_lateral_fraction: float = 0.65
    max_aggressive_lateral_speed_m_s: float = 0.85
    max_vertical_speed_m_s: float = 0.75
    tilt_slowdown_start_rad: float = 0.20
    tilt_slowdown_full_rad: float = 0.45
    tilt_min_speed_scale: float = 0.15
    tilt_recover_start_rad: float = 0.55
    tilt_recover_release_rad: float = 0.18
    tilt_recover_speed_m_s: float = 0.20
    tilt_recover_climb_m: float = 0.80
    tilt_recover_damping: float = 1.20

    transit_hover_clearance_m: float = 0.35
    direct_hover_max_abs_z_error_m: float = 1.25
    static_direct_hover_clear_path_max_abs_z_error_m: float = 2.0
    moving_direct_hover_clear_path_max_abs_z_error_m: float = 60.0
    static_direct_hover_distance_limit_m: float = 12.0
    moving_direct_hover_distance_limit_m: float = 80.0
    moving_direct_hover_min_xy_m: float = 20.0
    moving_match_speed_radius_m: float = 6.0
    moving_match_speed_m_s: float = 0.45
    close_xy_skip_cruise_radius_m: float = 1.25
    close_xy_skip_cruise_moving_radius_m: float = 2.00
    close_xy_recovery_bias_cap_m: float = 0.35
    recovery_bias_decay_m: float = 0.35
    forced_hover_xy_radius_m: float = 1.00
    forced_hover_descent_clearance_m: float = 0.50
    transit_progress_vertical_weight: float = 0.20
    xy_progress_only_height_gap_m: float = 1.00
    high_altitude_transit_speed_m_s: float = 1.05
    static_touchdown_enter_height_m: float = 1.00
    static_touchdown_release_height_m: float = 0.85
    static_contact_hold_z_offset_m: float = -0.08
    static_contact_hold_speed_m_s: float = 0.18

class PrivilegedExpertPolicy:
    """Strong privileged teacher built as a deterministic finite-state controller.

    The actor uses simulator truth aggressively but cleanly:
    - exact platform pose and velocity
    - exact line-of-sight and route hints from ``info["privileged"]["planner"]``
    - explicit hover, descent, touchdown, and recovery modes
    """

    def __init__(
        self,
        config: PrivilegedExpertConfig | None = None,
        *,
        identity: ExpertIdentity | None = None,
    ):
        self.config = config or PrivilegedExpertConfig()
        self.identity = identity or ExpertIdentity()
        self.last_mode = "idle"
        self.last_target_vector = np.zeros(3, dtype=np.float32)
        self.last_target_world = np.zeros(3, dtype=np.float32)
        self.last_action = np.zeros(5, dtype=np.float32)
        self.progress_reference = np.inf
        self.stall_time_sec = 0.0
        self.recovery_end_time = 0.0
        self.recovery_bias_m = 0.0
        self.recovery_index = 0
        self.static_settle_active = False
        self.static_touchdown_active = False
        self.moving_match_active = False
        self.moving_contact_active = False
        self.tilt_recover_active = False
        self.last_metadata: dict[str, Any] = {}

    def reset(self) -> None:
        self.last_mode = "idle"
        self.last_target_vector = np.zeros(3, dtype=np.float32)
        self.last_target_world = np.zeros(3, dtype=np.float32)
        self.last_action = np.zeros(5, dtype=np.float32)
        self.progress_reference = np.inf
        self.stall_time_sec = 0.0
        self.recovery_end_time = 0.0
        self.recovery_bias_m = 0.0
        self.recovery_index = 0
        self.static_settle_active = False
        self.static_touchdown_active = False
        self.moving_match_active = False
        self.moving_contact_active = False
        self.tilt_recover_active = False
        self.last_metadata = {}

    def get_last_metadata(self) -> dict[str, Any]:
        return {
            "expert_mode": self.last_mode,
            "expert_mode_id": expert_mode_id(self.last_mode),
            "target_vector": self.last_target_vector.astype(np.float32),
            "target_world": self.last_target_world.astype(np.float32),
            "stall_time_sec": float(self.stall_time_sec),
            "recovery_bias_m": float(self.recovery_bias_m),
            "teacher_id": self.identity.teacher_id,
            "teacher_version": self.identity.teacher_version,
            "map_category": self.identity.map_category,
            "parent_teacher_id": self.identity.parent_teacher_id,
            "expert_action_dim": EXPERT_ACTION_DIM,
            "expert_mode_vocab_version": EXPERT_MODE_VOCAB_VERSION,
            **self.last_metadata,
        }

    def _drone_state(self, observation: dict[str, Any], privileged: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        state = np.asarray(observation["state"], dtype=np.float32).reshape(-1)
        if state.shape[0] >= 9:
            return state[0:3], state[3:6], state[6:9]
        return (
            np.asarray(privileged["drone_position"], dtype=np.float32).reshape(3),
            np.asarray(privileged["drone_rpy"], dtype=np.float32).reshape(3),
            np.asarray(privileged["drone_velocity"], dtype=np.float32).reshape(3),
        )

    def _intercept_target(self, privileged: dict[str, Any]) -> np.ndarray:
        platform_pos = np.asarray(privileged["platform_position"], dtype=np.float32).reshape(3)
        moving = bool(privileged.get("moving_platform", False))
        challenge_type = int(privileged.get("challenge_type", 0))
        if not moving:
            return platform_pos.copy()
        dist_xy = float(privileged.get("xy_distance_to_platform", np.linalg.norm(privileged["relative_platform"][0:2])))
        lead_time = self.config.moving_intercept_base_sec + self.config.moving_intercept_gain_sec_per_m * dist_xy
        horizon_cap = 2.4 if challenge_type == 3 else self.config.moving_intercept_horizon_cap_sec
        lead_time = float(np.clip(lead_time, 0.0, horizon_cap))
        if dist_xy <= self.config.moving_match_xy_radius_m:
            lead_time = min(lead_time, self.config.moving_terminal_lead_sec)
        predicted_pos, _, _ = self._predict_platform_state(privileged, lead_time=lead_time)
        return predicted_pos

    def _predict_platform_state(
        self,
        privileged: dict[str, Any],
        *,
        lead_time: float,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        platform_pos = np.asarray(privileged["platform_position"], dtype=np.float32).reshape(3)
        platform_vel = np.asarray(privileged["platform_velocity"], dtype=np.float32).reshape(3)
        if not bool(privileged.get("moving_platform", False)):
            return platform_pos.copy(), platform_vel.copy(), "static"

        seed = int(privileged.get("map_seed", 0))
        center = np.asarray(privileged.get("goal_position", platform_pos), dtype=np.float32).reshape(3)
        time_alive = float(privileged.get("time_alive", 0.0))
        nominal_now_pos, nominal_now_vel, pattern = _platform_nominal_state(center, seed, time_alive)
        nominal_future_pos, nominal_future_vel, _ = _platform_nominal_state(
            center,
            seed,
            time_alive + max(0.0, float(lead_time)),
        )

        # Anchor the model-predicted displacement to the actual current platform
        # state so small runtime deviations still track correctly.
        predicted_pos = platform_pos + (nominal_future_pos - nominal_now_pos)
        predicted_vel = platform_vel + (nominal_future_vel - nominal_now_vel)
        predicted_pos[2] = float(platform_pos[2]) + float(nominal_future_pos[2] - nominal_now_pos[2])
        predicted_vel[2] = float(platform_vel[2]) + float(nominal_future_vel[2] - nominal_now_vel[2])
        return predicted_pos.astype(np.float32), predicted_vel.astype(np.float32), pattern

    def _platform_motion_pattern(self, privileged: dict[str, Any]) -> str:
        if not bool(privileged.get("moving_platform", False)):
            return "static"
        seed = int(privileged.get("map_seed", 0))
        return _platform_motion_params_from_seed(seed)[0]

    def _fallback_detour_direction(self, rel_platform: np.ndarray) -> np.ndarray:
        rel_platform = np.asarray(rel_platform, dtype=np.float32).reshape(3)
        lateral = np.array([-rel_platform[1], rel_platform[0], 0.0], dtype=np.float32)
        if np.linalg.norm(lateral[:2]) < 1e-6:
            lateral = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        return normalize(lateral)

    def _hover_target(self, target_abs: np.ndarray, moving_platform: bool) -> np.ndarray:
        hover = np.asarray(target_abs, dtype=np.float32).copy()
        hover[2] += self.config.moving_hover_height_m if moving_platform else self.config.static_hover_height_m
        return hover

    def _route_targets(self, drone_pos: np.ndarray, privileged: dict[str, Any], target_abs: np.ndarray) -> dict[str, np.ndarray]:
        planner = dict(privileged.get("planner", {}))
        hover_pos = self._hover_target(target_abs, bool(privileged.get("moving_platform", False)))
        planner_hover = np.asarray(planner.get("hover_position", hover_pos), dtype=np.float32).reshape(3)
        hover_pos = planner_hover.copy()
        planner_cruise_z = float(
            planner.get(
                "cruise_z",
                max(
                    float(drone_pos[2]),
                    float(hover_pos[2]) + self.config.transit_hover_clearance_m,
                ),
            )
        )
        # Keep transit legs from descending below the current altitude.
        # For easy static curricula this is the main stabilizer: move laterally
        # while high, then descend above the platform instead of diving
        # diagonally toward a low hover point.
        cruise_z = max(
            float(drone_pos[2]),
            planner_cruise_z,
            float(hover_pos[2]) + self.config.transit_hover_clearance_m,
        )
        if self.recovery_bias_m > 0.0:
            cruise_z += self.recovery_bias_m
        planner_cruise = np.asarray(planner.get("cruise_position", [drone_pos[0], drone_pos[1], cruise_z]), dtype=np.float32).reshape(3)
        planner_transit = np.asarray(planner.get("transit_position", [target_abs[0], target_abs[1], cruise_z]), dtype=np.float32).reshape(3)
        cruise_pos = planner_cruise.copy()
        cruise_pos[2] = cruise_z
        transit_pos = planner_transit.copy()
        transit_pos[2] = cruise_z
        return {
            "cruise": cruise_pos,
            "transit": transit_pos,
            "hover": hover_pos.astype(np.float32),
        }


    def _progress_metric(
        self,
        drone_pos: np.ndarray,
        target_abs: np.ndarray,
        hover_abs: np.ndarray,
        transit_abs: np.ndarray,
        mode: str,
    ) -> float:
        if mode in {
            "descend_static",
            "touchdown_static",
            "hold_contact_static",
            "settle_static",
            "descend_moving",
            "match_velocity_moving",
            "commit_contact_moving",
        }:
            return float(np.linalg.norm(target_abs - drone_pos))
        hover_delta = np.asarray(hover_abs, dtype=np.float32) - np.asarray(drone_pos, dtype=np.float32)
        hover_xy_dist = float(np.linalg.norm(hover_delta[0:2]))
        if hover_xy_dist <= self.config.close_xy_skip_cruise_moving_radius_m:
            return float(np.linalg.norm(hover_delta))
        if float(drone_pos[2] - hover_abs[2]) >= self.config.xy_progress_only_height_gap_m:
            return float(np.linalg.norm((transit_abs - drone_pos)[0:2]))
        return float(
            np.linalg.norm(hover_delta[0:2])
            + self.config.transit_progress_vertical_weight * abs(float(hover_delta[2]))
        )

    def _update_progress(self, progress_metric: float, time_alive: float, *, height_above_hover: float = 0.0) -> bool:
        if progress_metric < self.progress_reference - self.config.min_progress_m:
            self.progress_reference = progress_metric
            self.stall_time_sec = 0.0
            self.recovery_bias_m = max(0.0, float(self.recovery_bias_m - self.config.recovery_bias_decay_m))
            return False
        self.stall_time_sec += 0.02
        if self.stall_time_sec >= self.config.stall_timeout_sec:
            self.stall_time_sec = 0.0
            self.progress_reference = progress_metric
            if height_above_hover >= self.config.recovery_bias_block_height_m:
                self.recovery_bias_m = max(0.0, float(self.recovery_bias_m - self.config.recovery_bias_decay_m))
            else:
                self.recovery_bias_m = float(
                    min(self.config.max_recovery_bias_m, self.recovery_bias_m + self.config.recovery_climb_extra_m)
                )
            self.recovery_end_time = time_alive + self.config.recovery_hold_sec
            self.recovery_index += 1
            return True
        return False

    def _recovery_target(self, drone_pos: np.ndarray, cruise_pos: np.ndarray, transit_pos: np.ndarray) -> np.ndarray:
        lateral_cycle = (
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
            np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, -1.0, 0.0], dtype=np.float32),
        )
        offset = lateral_cycle[self.recovery_index % len(lateral_cycle)] * self.config.recovery_lateral_m
        forward = np.asarray(transit_pos, dtype=np.float32) - np.asarray(drone_pos, dtype=np.float32)
        forward[2] = 0.0
        forward = normalize(forward) * self.config.recovery_forward_m if np.linalg.norm(forward[:2]) > 1e-6 else 0.0
        target = np.asarray(cruise_pos, dtype=np.float32).copy()
        target[0:2] = drone_pos[0:2] + offset[0:2] + np.asarray(forward, dtype=np.float32)[0:2]
        target[2] = max(float(target[2]), float(drone_pos[2]) + 0.8)
        return target

    def _make_action(
        self,
        *,
        drone_pos: np.ndarray,
        drone_rpy: np.ndarray,
        drone_vel: np.ndarray,
        target_abs: np.ndarray,
        target_velocity: np.ndarray,
        speed_m_s: float,
        damping: float,
        vertical_bias: float = 0.0,
        near_target_speed_floor_m_s: float | None = None,
        near_target_speed_gain_m_s_per_m: float | None = None,
    ) -> np.ndarray:
        position_error = np.asarray(target_abs, dtype=np.float32) - np.asarray(drone_pos, dtype=np.float32)
        relative_velocity = np.asarray(drone_vel, dtype=np.float32) - np.asarray(target_velocity, dtype=np.float32)
        command_vec = position_error - damping * relative_velocity
        command_vec[2] += vertical_bias
        if np.linalg.norm(command_vec) < 1e-6:
            command_vec = position_error
        requested_speed = self._shape_speed(
            base_speed_m_s=float(np.clip(speed_m_s, 0.0, self.config.speed_limit_m_s)),
            command_vec=command_vec,
            position_error=position_error,
            drone_rpy=drone_rpy,
            near_target_speed_floor_m_s=near_target_speed_floor_m_s,
            near_target_speed_gain_m_s_per_m=near_target_speed_gain_m_s_per_m,
        )
        action = action_from_target_vector(
            command_vec,
            speed_m_s=requested_speed,
            speed_limit_m_s=self.config.speed_limit_m_s,
        )
        return self._smooth_action(action, command_vec)

    def _shape_speed(
        self,
        *,
        base_speed_m_s: float,
        command_vec: np.ndarray,
        position_error: np.ndarray,
        drone_rpy: np.ndarray,
        near_target_speed_floor_m_s: float | None = None,
        near_target_speed_gain_m_s_per_m: float | None = None,
    ) -> float:
        speed = float(np.clip(base_speed_m_s, 0.0, self.config.speed_limit_m_s))
        distance = float(np.linalg.norm(position_error))
        near_target_floor = (
            self.config.near_target_speed_floor_m_s
            if near_target_speed_floor_m_s is None
            else float(near_target_speed_floor_m_s)
        )
        near_target_gain = (
            self.config.near_target_speed_gain_m_s_per_m
            if near_target_speed_gain_m_s_per_m is None
            else float(near_target_speed_gain_m_s_per_m)
        )
        if distance <= self.config.near_target_speed_radius_m:
            near_cap = near_target_floor + near_target_gain * distance
            speed = min(speed, float(np.clip(near_cap, near_target_floor, self.config.speed_limit_m_s)))

        command_norm = float(np.linalg.norm(command_vec))
        lateral_norm = float(np.linalg.norm(np.asarray(command_vec, dtype=np.float32)[0:2]))
        lateral_fraction = lateral_norm / max(command_norm, 1e-6)
        if lateral_fraction >= self.config.aggressive_lateral_fraction:
            speed = min(speed, self.config.max_aggressive_lateral_speed_m_s)
        if abs(float(command_vec[2])) >= lateral_norm:
            speed = min(speed, self.config.max_vertical_speed_m_s)

        tilt_mag = max(abs(float(drone_rpy[0])), abs(float(drone_rpy[1])))
        if tilt_mag >= self.config.tilt_slowdown_start_rad:
            alpha = (tilt_mag - self.config.tilt_slowdown_start_rad) / max(
                self.config.tilt_slowdown_full_rad - self.config.tilt_slowdown_start_rad,
                1e-6,
            )
            alpha = float(np.clip(alpha, 0.0, 1.0))
            scale = 1.0 - alpha * (1.0 - self.config.tilt_min_speed_scale)
            speed *= scale

        return float(np.clip(speed, 0.0, self.config.speed_limit_m_s))

    def _smooth_action(self, action: np.ndarray, command_vec: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).reshape(5)
        smoothed = action.copy()

        prev_dir = np.asarray(self.last_action[0:3], dtype=np.float32)
        new_dir = np.asarray(action[0:3], dtype=np.float32)
        if np.linalg.norm(prev_dir) > 1e-6:
            blended_dir = normalize(
                self.config.direction_blend * new_dir + (1.0 - self.config.direction_blend) * prev_dir
            )
            smoothed[0:3] = blended_dir

        speed_delta = float(action[3] - self.last_action[3])
        speed_delta = float(
            np.clip(
                speed_delta,
                -self.config.max_speed_delta_norm_per_step,
                self.config.max_speed_delta_norm_per_step,
            )
        )
        smoothed[3] = float(np.clip(self.last_action[3] + speed_delta, 0.0, 1.0))

        desired_yaw = _dominant_target_yaw(command_vec, fallback_yaw_norm=float(self.last_action[4]))
        yaw_delta = float(desired_yaw - self.last_action[4])
        yaw_delta = float(
            np.clip(
                yaw_delta,
                -self.config.max_yaw_delta_norm_per_step,
                self.config.max_yaw_delta_norm_per_step,
            )
        )
        smoothed[4] = float(np.clip(self.last_action[4] + yaw_delta, -1.0, 1.0))

        self.last_action = smoothed.astype(np.float32)
        return self.last_action.copy()

    def _choose_mode(
        self,
        *,
        observation: dict[str, Any],
        privileged: dict[str, Any],
        drone_pos: np.ndarray,
        drone_rpy: np.ndarray,
        drone_vel: np.ndarray,
        target_abs: np.ndarray,
        cruise_pos: np.ndarray,
        transit_pos: np.ndarray,
        hover_pos: np.ndarray,
        time_alive: float,
    ) -> tuple[str, np.ndarray, np.ndarray, float, float]:
        moving_platform = bool(privileged.get("moving_platform", False))
        challenge_type = int(privileged.get("challenge_type", 0))
        motion_pattern = self._platform_motion_pattern(privileged)
        platform_pos = np.asarray(privileged["platform_position"], dtype=np.float32).reshape(3)
        platform_vel = np.asarray(privileged["platform_velocity"], dtype=np.float32).reshape(3)
        rel_platform = platform_pos - drone_pos
        xy_dist = float(np.linalg.norm(rel_platform[:2]))
        z_error = float(rel_platform[2])
        altitude_above_platform = float(drone_pos[2] - platform_pos[2])
        rel_vxy = float(np.linalg.norm(drone_vel[:2] - platform_vel[:2]))
        vz = float(abs(drone_vel[2]))
        landing_stable_time = float(privileged.get("landing_stable_time", 0.0))
        planner = dict(privileged.get("planner", {}))
        direct_to_hover_clear = bool(planner.get("direct_to_hover_clear", False))
        planner_blocked_count = int(planner.get("blocked_count", 0))
        planner_descent_blocked = bool(planner.get("descent_blocked", False))
        line_of_sight = bool(privileged.get("line_of_sight_to_platform", False))
        hover_xy_dist = float(np.linalg.norm((hover_pos - drone_pos)[0:2]))
        use_depth_obstacle_checks = bool(self.config.use_depth_obstacle_checks)
        depth = np.asarray(observation.get("depth", 0.0), dtype=np.float32)
        depth_available = bool(depth.size > 0 and np.any(depth))
        if use_depth_obstacle_checks and depth_available:
            front_blocked = front_depth_m(depth) < self.config.obstacle_distance_m
            target_vector_world = np.asarray(
                hover_pos - drone_pos if hover_xy_dist <= 6.0 else transit_pos - drone_pos,
                dtype=np.float32,
            )
            depth_2d = np.squeeze(depth)
            target_pixel = None
            if depth_2d.ndim == 2:
                target_pixel = relative_vector_to_pixel(
                    target_vector_world,
                    roll=float(drone_rpy[0]),
                    pitch=float(drone_rpy[1]),
                    yaw=float(drone_rpy[2]),
                    image_height=int(depth_2d.shape[0]),
                    image_width=int(depth_2d.shape[1]),
                )
            avoid_body = choose_lateral_detour(
                depth,
                target_pixel_col=(None if target_pixel is None else int(target_pixel[1])),
                blocked_distance_m=self.config.obstacle_distance_m,
            )
            yaw = float(drone_rpy[2])
            cy, sy = math.cos(yaw), math.sin(yaw)
            avoid_direction = normalize(
                np.array(
                    [
                        avoid_body[0] * cy - avoid_body[1] * sy,
                        avoid_body[0] * sy + avoid_body[1] * cy,
                        avoid_body[2],
                    ],
                    dtype=np.float32,
                )
            )
        else:
            front_blocked = not (line_of_sight and direct_to_hover_clear)
            avoid_direction = self._fallback_detour_direction(rel_platform)
        cruise_altitude_reached = float(drone_pos[2]) >= float(cruise_pos[2]) - 0.25
        close_xy_skip_cruise_radius = (
            self.config.close_xy_skip_cruise_moving_radius_m
            if moving_platform
            else self.config.close_xy_skip_cruise_radius_m
        )
        close_xy_for_approach = hover_xy_dist <= close_xy_skip_cruise_radius
        hover_path_clear = direct_to_hover_clear or (line_of_sight and planner_blocked_count == 0)
        near_hover = (
            hover_xy_dist <= (self.config.moving_descend_xy_radius_m if moving_platform else self.config.static_descend_xy_radius_m)
            and abs(float(hover_pos[2] - drone_pos[2])) <= self.config.hover_z_tolerance_m
        )
        moving_match_radius = (
            12.0
            if challenge_type == 3
            else (
                self.config.moving_match_xy_radius_m
                if challenge_type == 2
                else self.config.moving_match_xy_radius_non_open_m
            )
        )
        moving_match_release_radius = (
            14.0
            if challenge_type == 3
            else (
                self.config.moving_match_release_xy_radius_m
                if challenge_type == 2
                else self.config.moving_match_release_xy_radius_non_open_m
            )
        )
        direct_hover_allowed = (
            hover_path_clear
            and xy_dist
            <= (
                self.config.moving_direct_hover_distance_limit_m
                if moving_platform
                else self.config.static_direct_hover_distance_limit_m
            )
            and abs(float(hover_pos[2] - drone_pos[2]))
            <= (
                (
                    self.config.moving_direct_hover_clear_path_max_abs_z_error_m
                    if moving_platform
                    else self.config.static_direct_hover_clear_path_max_abs_z_error_m
                )
                if direct_to_hover_clear
                else self.config.direct_hover_max_abs_z_error_m
            )
            and (
                (
                    moving_platform
                    and planner_blocked_count == 0
                    and challenge_type == 3
                    and xy_dist >= moving_match_release_radius
                )
                or ((not moving_platform) and (challenge_type in {1, 2, 6} or planner_blocked_count == 0))
            )
        )

        if close_xy_for_approach and hover_path_clear:
            self.recovery_bias_m = min(float(self.recovery_bias_m), self.config.close_xy_recovery_bias_cap_m)
            self.recovery_end_time = min(float(self.recovery_end_time), time_alive)

        progress_mode = "transit"
        if moving_platform:
            moving_touch_xy_radius = max(0.30, float(self.config.moving_touchdown_xy_radius_m))
            forced_contact_xy_radius = min(0.35, moving_touch_xy_radius)
            moving_commit_height_gate = self.config.moving_commit_height_m + 0.35
            self.static_settle_active = False
            self.static_touchdown_active = False
            if self.moving_contact_active:
                if xy_dist > self.config.moving_commit_release_xy_radius_m or altitude_above_platform > 1.10:
                    self.moving_contact_active = False
            elif (
                xy_dist <= self.config.moving_commit_xy_radius_m
                and altitude_above_platform <= moving_commit_height_gate
                and (
                    rel_vxy <= self.config.moving_commit_rel_xy_speed_gate_m_s
                    or xy_dist <= forced_contact_xy_radius
                )
            ):
                self.moving_contact_active = True

            if self.moving_match_active:
                if xy_dist > moving_match_release_radius:
                    self.moving_match_active = False
            elif (
                xy_dist <= moving_match_radius
                or (
                    xy_dist <= self.config.moving_commit_xy_radius_m + 0.35
                    and altitude_above_platform <= moving_commit_height_gate + 0.25
                )
            ):
                self.moving_match_active = True

            if self.moving_contact_active:
                progress_mode = "commit_contact_moving"
            elif self.moving_match_active:
                if (
                    (xy_dist <= moving_touch_xy_radius and altitude_above_platform <= moving_commit_height_gate)
                    or (
                        rel_vxy <= self.config.moving_match_rel_xy_speed_gate_m_s
                        and altitude_above_platform <= self.config.moving_hover_height_m + 0.35
                    )
                ):
                    progress_mode = "descend_moving"
                else:
                    progress_mode = "match_velocity_moving"
            elif near_hover and (
                rel_vxy <= self.config.moving_rel_xy_speed_gate_m_s
                or (xy_dist <= self.config.moving_commit_xy_radius_m and altitude_above_platform <= moving_commit_height_gate)
            ):
                progress_mode = "descend_moving"
        else:
            self.moving_match_active = False
            self.moving_contact_active = False
            if self.static_touchdown_active:
                if xy_dist > self.config.static_settle_release_xy_radius_m or altitude_above_platform > self.config.static_touchdown_release_height_m:
                    self.static_touchdown_active = False
            elif (
                landing_stable_time > 0.0
                or (xy_dist <= self.config.static_touchdown_xy_radius_m and altitude_above_platform <= self.config.static_touchdown_enter_height_m)
                or (line_of_sight and xy_dist <= 0.15 and altitude_above_platform <= 1.0)
            ):
                self.static_touchdown_active = True

            if self.static_settle_active:
                if xy_dist > self.config.static_settle_release_xy_radius_m or altitude_above_platform > 0.80:
                    self.static_settle_active = False
            elif (
                line_of_sight
                and xy_dist <= self.config.static_settle_xy_radius_m
                and altitude_above_platform <= self.config.static_landing_commit_height_m + 0.20
            ):
                self.static_settle_active = True

            if self.static_touchdown_active:
                progress_mode = "hold_contact_static" if landing_stable_time > 0.0 else "touchdown_static"
            elif self.static_settle_active:
                progress_mode = "settle_static"
            elif (
                line_of_sight
                and xy_dist <= self.config.static_landing_commit_xy_radius_m
                and altitude_above_platform <= self.config.static_landing_commit_height_m
            ):
                progress_mode = "landing_commit_static"
            elif (
                line_of_sight
                and xy_dist <= self.config.static_sink_xy_radius_m
            ):
                progress_mode = "sink_static"
            elif xy_dist <= self.config.static_descend_xy_radius_m and float(drone_pos[2] - platform_pos[2]) <= self.config.static_hover_height_m + 0.25:
                progress_mode = "descend_static"

        progress_metric = self._progress_metric(drone_pos, target_abs, hover_pos, transit_pos, progress_mode)
        allow_recovery_stall = progress_mode in {"transit", "commit_contact_moving"} and (
            (not moving_platform) or (not hover_path_clear)
        )
        if allow_recovery_stall and self._update_progress(
            progress_metric,
            time_alive,
            height_above_hover=max(0.0, float(drone_pos[2] - hover_pos[2])),
        ):
            progress_mode = "recover_stall"

        if self.recovery_end_time > time_alive:
            return "recover_stall", self._recovery_target(drone_pos, cruise_pos, transit_pos), np.zeros(3, dtype=np.float32), self.config.recovery_speed_m_s, self.config.position_damping

        if front_blocked and not near_hover and not (line_of_sight and direct_to_hover_clear):
            avoid_xy_step = max(self.config.recovery_forward_m, self.config.recovery_lateral_m)
            avoid_target = np.asarray(drone_pos, dtype=np.float32).copy()
            avoid_target[0:2] += avoid_direction[0:2] * avoid_xy_step
            avoid_target[2] = max(
                float(drone_pos[2]) + max(0.5, self.config.recovery_climb_extra_m),
                float(drone_pos[2]) + float(avoid_direction[2]) * self.config.recovery_climb_extra_m,
            )
            return "avoid_obstacle", avoid_target.astype(np.float32), np.zeros(3, dtype=np.float32), self.config.recovery_speed_m_s, self.config.position_damping

        if progress_mode == "recover_stall":
            return "recover_stall", self._recovery_target(drone_pos, cruise_pos, transit_pos), np.zeros(3, dtype=np.float32), self.config.recovery_speed_m_s, self.config.position_damping

        if moving_platform:
            match_target_pos, _, _ = self._predict_platform_state(
                privileged,
                lead_time=self.config.moving_terminal_lead_sec,
            )
            use_pattern_terminal_match = bool(self.config.use_pattern_terminal_match) and motion_pattern in {"circular", "figure8"}
            terminal_lead = platform_vel.astype(np.float32).copy()
            terminal_lead[2] = 0.0
            terminal_lead *= float(self.config.moving_terminal_lead_sec)
            if progress_mode == "commit_contact_moving":
                touchdown = match_target_pos.copy() if use_pattern_terminal_match else platform_pos.copy()
                if not use_pattern_terminal_match:
                    touchdown[0:2] += terminal_lead[0:2]
                touchdown[2] = float(platform_pos[2]) + max(0.0, self.config.moving_contact_target_z_offset_m)
                contact_speed = max(self.config.moving_contact_speed_m_s, self.config.moving_commit_speed_m_s)
                return "commit_contact_moving", touchdown, platform_vel, contact_speed, self.config.moving_match_damping
            if progress_mode == "match_velocity_moving":
                if use_pattern_terminal_match:
                    match = match_target_pos.copy()
                    match[2] = float(match_target_pos[2]) + self.config.moving_match_height_m
                    return "match_velocity_moving", match, platform_vel, self.config.moving_terminal_hover_speed_m_s, self.config.moving_match_damping
                match = platform_pos.copy()
                match[0:2] += terminal_lead[0:2]
                match[2] = float(platform_pos[2]) + self.config.moving_match_height_m
                return "match_velocity_moving", match, platform_vel, self.config.moving_terminal_hover_speed_m_s, self.config.moving_match_damping
            if progress_mode == "descend_moving":
                descend = match_target_pos.copy() if use_pattern_terminal_match else platform_pos.copy()
                if not use_pattern_terminal_match:
                    descend[0:2] += terminal_lead[0:2]
                descend[2] = float(platform_pos[2]) + self.config.moving_commit_height_m
                return "descend_moving", descend, platform_vel, self.config.moving_commit_speed_m_s, self.config.moving_match_damping
            if hover_path_clear and xy_dist <= self.config.moving_match_speed_radius_m:
                return "hover_align", hover_pos, platform_vel, min(self.config.moving_match_speed_m_s, self.config.moving_hover_align_speed_cap_m_s), self.config.position_damping
        else:
            if progress_mode == "hold_contact_static":
                hold = platform_pos.copy()
                hold[2] = float(platform_pos[2]) + self.config.static_contact_hold_z_offset_m
                return "hold_contact_static", hold, np.zeros(3, dtype=np.float32), max(self.config.static_contact_hold_speed_m_s, 0.12), self.config.touchdown_damping
            if progress_mode == "touchdown_static":
                touchdown = platform_pos.copy()
                touchdown[2] = float(platform_pos[2]) - 0.28
                return "touchdown_static", touchdown, np.zeros(3, dtype=np.float32), max(self.config.touchdown_speed_m_s, 0.30), self.config.touchdown_damping
            if progress_mode == "settle_static":
                settle = platform_pos.copy()
                if landing_stable_time > 0.0 or (rel_vxy <= 0.55 and vz <= 0.40 and xy_dist <= 0.30):
                    settle[2] = float(platform_pos[2]) - 0.05
                    settle_speed = self.config.touchdown_speed_m_s
                else:
                    settle[2] = float(platform_pos[2]) + self.config.static_settle_height_m
                    settle_speed = self.config.static_settle_speed_m_s
                return "settle_static", settle, np.zeros(3, dtype=np.float32), settle_speed, self.config.static_settle_damping
            if progress_mode == "landing_commit_static":
                touchdown = drone_pos.copy()
                if xy_dist > 0.05:
                    touchdown[0:2] = platform_pos[0:2]
                touchdown[2] = float(platform_pos[2]) - 0.06
                return "landing_commit_static", touchdown, np.zeros(3, dtype=np.float32), self.config.touchdown_speed_m_s, self.config.touchdown_damping
            if progress_mode == "sink_static":
                sink = platform_pos.copy()
                sink[2] = float(platform_pos[2]) + self.config.static_sink_height_m
                return "sink_static", sink, np.zeros(3, dtype=np.float32), self.config.sink_speed_m_s, self.config.touchdown_damping
            if progress_mode == "descend_static":
                descend = platform_pos.copy()
                descend[2] = float(platform_pos[2]) - 0.03
                return "descend_static", descend, np.zeros(3, dtype=np.float32), self.config.descend_speed_m_s, self.config.touchdown_damping

        transit_speed_m_s = self.config.cruise_speed_m_s
        if moving_platform:
            transit_speed_m_s = min(
                transit_speed_m_s,
                self.config.moving_transit_speed_cap_city_m_s if challenge_type == 1 else self.config.moving_transit_speed_cap_non_city_m_s,
            )
        if (not moving_platform) and float(drone_pos[2] - hover_pos[2]) >= self.config.xy_progress_only_height_gap_m:
            transit_speed_m_s = min(transit_speed_m_s, self.config.high_altitude_transit_speed_m_s)

        if direct_hover_allowed and float(np.linalg.norm(hover_pos - drone_pos)) > 1.0:
            return "direct_hover", hover_pos, platform_vel if moving_platform else np.zeros(3, dtype=np.float32), transit_speed_m_s, self.config.position_damping

        if (
            (not moving_platform)
            and hover_xy_dist <= self.config.forced_hover_xy_radius_m
            and float(drone_pos[2] - hover_pos[2]) >= self.config.forced_hover_descent_clearance_m
            and (not planner_descent_blocked)
        ):
            return "hover_align", hover_pos, np.zeros(3, dtype=np.float32), self.config.align_speed_m_s, self.config.position_damping

        if close_xy_for_approach and hover_path_clear and abs(float(hover_pos[2] - drone_pos[2])) > self.config.hover_z_tolerance_m:
            hover_speed = min(transit_speed_m_s, self.config.moving_hover_align_speed_cap_m_s) if moving_platform else transit_speed_m_s
            return "hover_align", hover_pos, platform_vel if moving_platform else np.zeros(3, dtype=np.float32), hover_speed, self.config.position_damping

        if not cruise_altitude_reached:
            return "climb_to_cruise", cruise_pos, np.zeros(3, dtype=np.float32), self.config.climb_speed_m_s, self.config.position_damping

        if float(np.linalg.norm((transit_pos - drone_pos)[0:2])) > 0.75:
            return "transit_to_hover", transit_pos, platform_vel if moving_platform else np.zeros(3, dtype=np.float32), transit_speed_m_s, self.config.position_damping

        final_hover_speed = min(self.config.align_speed_m_s, self.config.moving_hover_align_speed_cap_m_s) if moving_platform else self.config.align_speed_m_s
        return "hover_align", hover_pos, platform_vel if moving_platform else np.zeros(3, dtype=np.float32), final_hover_speed, self.config.position_damping

    def act(self, observation: dict[str, Any], info: dict[str, Any]) -> np.ndarray:
        privileged = dict(info.get("privileged", {}))
        if not privileged:
            raise ValueError("PrivilegedExpertPolicy requires info['privileged']")

        drone_pos, drone_rpy, drone_vel = self._drone_state(observation, privileged)
        moving_platform = bool(privileged.get("moving_platform", False))
        challenge_type = int(privileged.get("challenge_type", 0))
        motion_pattern = self._platform_motion_pattern(privileged)
        target_abs = self._intercept_target(privileged)
        route_targets = self._route_targets(drone_pos, privileged, target_abs)
        cruise_pos = route_targets["cruise"]
        transit_pos = route_targets["transit"]
        hover_pos = route_targets["hover"]
        time_alive = float(privileged.get("time_alive", 0.0))
        tilt_mag = max(abs(float(drone_rpy[0])), abs(float(drone_rpy[1])))

        moving_tilt_recover_start = min(self.config.tilt_recover_start_rad, 0.45) if moving_platform else self.config.tilt_recover_start_rad

        if self.tilt_recover_active:
            if tilt_mag <= self.config.tilt_recover_release_rad:
                self.tilt_recover_active = False
        elif tilt_mag >= moving_tilt_recover_start:
            self.tilt_recover_active = True

        if self.tilt_recover_active:
            recover_target = np.asarray(drone_pos, dtype=np.float32).copy()
            recover_target[2] = float(drone_pos[2]) + self.config.tilt_recover_climb_m
            action = self._make_action(
                drone_pos=drone_pos,
                drone_rpy=drone_rpy,
                drone_vel=drone_vel,
                target_abs=recover_target,
                target_velocity=np.zeros(3, dtype=np.float32),
                speed_m_s=self.config.tilt_recover_speed_m_s,
                damping=self.config.tilt_recover_damping,
                vertical_bias=0.0,
            )
            self.last_mode = "recover_tilt"
            self.last_target_world = recover_target.astype(np.float32)
            self.last_target_vector = (self.last_target_world - drone_pos).astype(np.float32)
            self.last_metadata = {
                "moving_platform": moving_platform,
                "distance_to_platform": float(privileged.get("distance_to_platform", np.linalg.norm(privileged["relative_platform"]))),
                "xy_distance_to_platform": float(privileged.get("xy_distance_to_platform", np.linalg.norm(privileged["relative_platform"][0:2]))),
                "planner_cruise_z": float(privileged.get("planner", {}).get("cruise_z", recover_target[2])),
                "line_of_sight_to_platform": bool(privileged.get("line_of_sight_to_platform", False)),
                "landing_stable_time": float(privileged.get("landing_stable_time", 0.0)),
                "static_settle_active": bool(self.static_settle_active),
                "static_touchdown_active": bool(self.static_touchdown_active),
                "motion_pattern": motion_pattern,
                "tilt_mag_rad": float(tilt_mag),
                "tilt_recover_active": True,
            }
            return action

        mode, commanded_target, target_velocity, speed_m_s, damping = self._choose_mode(
            observation=observation,
            privileged=privileged,
            drone_pos=drone_pos,
            drone_rpy=drone_rpy,
            drone_vel=drone_vel,
            target_abs=target_abs,
            cruise_pos=cruise_pos,
            transit_pos=transit_pos,
            hover_pos=hover_pos,
            time_alive=time_alive,
        )
        vertical_bias = 0.0
        near_target_speed_floor_m_s = None
        near_target_speed_gain_m_s_per_m = None
        if mode in {
            "descend_static",
            "touchdown_static",
            "hold_contact_static",
            "descend_moving",
            "commit_contact_moving",
            "sink_static",
            "landing_commit_static",
            "settle_static",
        }:
            vertical_bias = -self.config.vertical_bias_gain
        if mode == "descend_moving":
            vertical_bias = -max(self.config.vertical_bias_gain, 0.35)
            near_target_speed_floor_m_s = self.config.moving_terminal_near_target_speed_floor_m_s
            near_target_speed_gain_m_s_per_m = self.config.moving_terminal_near_target_speed_gain_m_s_per_m
        elif mode == "commit_contact_moving":
            vertical_bias = -max(self.config.vertical_bias_gain, 0.45)
            near_target_speed_floor_m_s = self.config.moving_terminal_near_target_speed_floor_m_s
            near_target_speed_gain_m_s_per_m = self.config.moving_terminal_near_target_speed_gain_m_s_per_m

        commanded_target = np.asarray(commanded_target, dtype=np.float32).reshape(3).copy()
        if moving_platform and mode in {"descend_moving", "commit_contact_moving"}:
            platform_pos = np.asarray(privileged["platform_position"], dtype=np.float32).reshape(3)
            top_contact_z = float(platform_pos[2]) + max(0.0, self.config.moving_contact_target_z_offset_m)
            commanded_target[2] = max(float(commanded_target[2]), top_contact_z)
            if float(drone_pos[2]) <= float(platform_pos[2]):
                vertical_bias = max(vertical_bias, self.config.moving_below_platform_recover_bias)
        action = self._make_action(
            drone_pos=drone_pos,
            drone_rpy=drone_rpy,
            drone_vel=drone_vel,
            target_abs=commanded_target,
            target_velocity=target_velocity,
            speed_m_s=speed_m_s,
            damping=damping,
            vertical_bias=vertical_bias,
            near_target_speed_floor_m_s=near_target_speed_floor_m_s,
            near_target_speed_gain_m_s_per_m=near_target_speed_gain_m_s_per_m,
        )

        self.last_mode = mode
        self.last_target_world = np.asarray(commanded_target, dtype=np.float32).reshape(3)
        self.last_target_vector = (self.last_target_world - drone_pos).astype(np.float32)
        self.last_metadata = {
            "moving_platform": moving_platform,
            "distance_to_platform": float(privileged.get("distance_to_platform", np.linalg.norm(privileged["relative_platform"]))),
            "xy_distance_to_platform": float(privileged.get("xy_distance_to_platform", np.linalg.norm(privileged["relative_platform"][0:2]))),
            "planner_cruise_z": float(privileged.get("planner", {}).get("cruise_z", commanded_target[2])),
            "line_of_sight_to_platform": bool(privileged.get("line_of_sight_to_platform", False)),
            "landing_stable_time": float(privileged.get("landing_stable_time", 0.0)),
            "static_settle_active": bool(self.static_settle_active),
            "static_touchdown_active": bool(self.static_touchdown_active),
            "moving_match_active": bool(self.moving_match_active),
            "moving_contact_active": bool(self.moving_contact_active),
            "motion_pattern": motion_pattern,
            "tilt_mag_rad": float(tilt_mag),
            "tilt_recover_active": bool(self.tilt_recover_active),
        }
        return action

def save_expert_config(path: str | Path, config: PrivilegedExpertConfig) -> None:
    save_json(path, asdict(config))


def save_mode_vocabulary(path: str | Path) -> None:
    save_json(path, default_mode_vocabulary_payload())


def load_expert_config(path: str | Path) -> PrivilegedExpertConfig:
    payload = load_json(path)
    payload.pop("expert_kind", None)
    return PrivilegedExpertConfig(**payload)


def make_expert_policy(
    config: PrivilegedExpertConfig,
    *,
    identity: ExpertIdentity | None = None,
) -> PrivilegedExpertPolicy:
    return PrivilegedExpertPolicy(config, identity=identity)
