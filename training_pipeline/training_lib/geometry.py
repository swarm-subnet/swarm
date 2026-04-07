"""Geometry helpers shared by experts, dataset builders, and models."""

from __future__ import annotations

import math

import numpy as np

DEPTH_MIN_M = 0.5
DEPTH_MAX_M = 20.0
DEFAULT_CAMERA_FOV_RAD = 0.5 * math.pi
DEFAULT_SPEED_LIMIT_M_S = 3.0


def normalize(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm < eps:
        return np.zeros_like(vector, dtype=np.float32)
    return (vector / norm).astype(np.float32)


def depth_to_meters(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    return DEPTH_MIN_M + (DEPTH_MAX_M - DEPTH_MIN_M) * np.clip(depth, 0.0, 1.0)


def front_depth_m(depth: np.ndarray, patch_radius_px: int = 4) -> float:
    depth = np.squeeze(np.asarray(depth, dtype=np.float32))
    if depth.ndim != 2:
        return DEPTH_MAX_M
    height, width = depth.shape
    row = height // 2
    c0 = max(0, width // 2 - patch_radius_px)
    c1 = min(width, width // 2 + patch_radius_px + 1)
    return float(np.min(depth_to_meters(depth[row : row + 1, c0:c1])))


def choose_lateral_detour(
    depth: np.ndarray,
    *,
    target_pixel_col: int | None = None,
    blocked_distance_m: float = 1.8,
    hold_altitude: bool = True,
    num_sectors: int = 5,
) -> np.ndarray:
    depth = np.squeeze(np.asarray(depth, dtype=np.float32))
    if depth.ndim != 2:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    depth_m = depth_to_meters(depth)
    _, width = depth_m.shape
    target_pixel_col = int(np.clip(width // 2 if target_pixel_col is None else target_pixel_col, 0, width - 1))
    sector_edges = np.linspace(0, width, max(2, num_sectors) + 1, dtype=int)
    best_score = -float("inf")
    best_theta = 0.0

    for idx in range(len(sector_edges) - 1):
        c0 = int(sector_edges[idx])
        c1 = int(max(c0 + 1, sector_edges[idx + 1]))
        sector = depth_m[:, c0:c1]
        if sector.size == 0:
            continue
        mean_clearance = float(np.mean(sector))
        p20_clearance = float(np.percentile(sector, 20.0))
        center_col = 0.5 * (c0 + c1 - 1)
        target_penalty = abs(center_col - target_pixel_col) / max(width * 0.5, 1.0)
        blocked_penalty = max(0.0, blocked_distance_m - p20_clearance)
        score = mean_clearance - 0.65 * target_penalty - 0.75 * blocked_penalty
        if score > best_score:
            best_score = score
            col_norm = (center_col - (width - 1) * 0.5) / max((width - 1) * 0.5, 1.0)
            best_theta = float(np.clip(col_norm, -1.0, 1.0)) * (DEFAULT_CAMERA_FOV_RAD * 0.42)

    body_forward = max(0.45, math.cos(best_theta))
    body_left = -math.sin(best_theta)
    body_up = 0.0 if hold_altitude else 0.05
    return normalize(np.array([body_forward, body_left, body_up], dtype=np.float32))


def yaw_from_direction(direction: np.ndarray) -> float:
    direction = np.asarray(direction, dtype=np.float32).reshape(3)
    if np.linalg.norm(direction[:2]) < 1e-8:
        return 0.0
    return float(np.clip(np.arctan2(direction[1], direction[0]) / np.pi, -1.0, 1.0))


def action_from_target_vector(
    relative_target: np.ndarray,
    *,
    speed_m_s: float,
    speed_limit_m_s: float = DEFAULT_SPEED_LIMIT_M_S,
) -> np.ndarray:
    direction = normalize(relative_target)
    speed_norm = float(np.clip(speed_m_s / speed_limit_m_s, 0.0, 1.0))
    action = np.zeros(5, dtype=np.float32)
    action[:3] = direction[:3]
    action[3] = speed_norm
    action[4] = yaw_from_direction(direction)
    return action


def relative_vector_to_pixel(
    relative_vector_world: np.ndarray,
    *,
    roll: float,
    pitch: float,
    yaw: float,
    image_height: int,
    image_width: int,
    camera_fov_rad: float = DEFAULT_CAMERA_FOV_RAD,
) -> tuple[int, int] | None:
    """Project a world-frame relative vector into image pixel coordinates."""

    vec = normalize(relative_vector_world)
    if np.linalg.norm(vec) < 1e-8:
        return None

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float32)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float32)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float32)
    rotation = rz @ ry @ rx
    body_vector = (rotation.T @ vec).astype(np.float32)

    if body_vector[0] <= 1e-6:
        return None

    half_fov = camera_fov_rad / 2.0
    horizontal = float(np.arctan2(-body_vector[1], body_vector[0]))
    vertical = float(np.arctan2(-body_vector[2], body_vector[0]))

    col = (image_width - 1) / 2.0 + (image_width / 2.0) * (horizontal / half_fov)
    row = (image_height - 1) / 2.0 + (image_height / 2.0) * (vertical / half_fov)
    row = int(np.clip(row, 0, image_height - 1))
    col = int(np.clip(col, 0, image_width - 1))
    return row, col
