"""Dataset builders for expert rollouts and perception labels."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .common import EpisodeSummary, ensure_dir, save_json
from .geometry import DEFAULT_CAMERA_FOV_RAD, relative_vector_to_pixel

MODE_TO_ID = {
    "idle": 0,
    "go_search_center": 1,
    "avoid_obstacle": 2,
    "intercept_platform": 3,
    "land_static": 4,
    "land_moving": 5,
}


def make_perception_labels(
    observation: dict[str, Any],
    privileged: dict[str, Any],
    *,
    camera_fov_rad: float = DEFAULT_CAMERA_FOV_RAD,
    distance_bucket_edges_m: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0),
) -> dict[str, Any]:
    """Build training labels for platform perception from simulator truth."""

    state = np.asarray(observation["state"], dtype=np.float32).reshape(-1)
    roll, pitch, yaw = float(state[3]), float(state[4]), float(state[5])
    relative_platform = np.asarray(privileged["relative_platform"], dtype=np.float32).reshape(3)
    distance_m = float(np.linalg.norm(relative_platform))
    depth = np.squeeze(np.asarray(observation["depth"], dtype=np.float32))
    height, width = depth.shape[:2]
    pixel = relative_vector_to_pixel(
        relative_platform,
        roll=roll,
        pitch=pitch,
        yaw=yaw,
        image_height=height,
        image_width=width,
        camera_fov_rad=camera_fov_rad,
    )
    visible = pixel is not None
    if visible:
        row, col = pixel
        pixel_norm = np.array(
            [
                (2.0 * col / max(width - 1, 1)) - 1.0,
                (2.0 * row / max(height - 1, 1)) - 1.0,
            ],
            dtype=np.float32,
        )
    else:
        row, col = -1, -1
        pixel_norm = np.zeros(2, dtype=np.float32)

    bucket = int(np.searchsorted(np.asarray(distance_bucket_edges_m, dtype=np.float32), distance_m, side="right"))
    return {
        "visible": int(visible),
        "pixel_row": int(row),
        "pixel_col": int(col),
        "pixel_norm": pixel_norm.astype(np.float32),
        "distance_m": np.float32(distance_m),
        "distance_bucket": int(bucket),
        "relative_platform": relative_platform.astype(np.float32),
    }


def _episode_arrays(step_records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    if not step_records:
        raise ValueError("Cannot serialize an empty episode")

    first_obs = step_records[0]["observation"]
    num_steps = len(step_records)
    depth_shape = np.asarray(first_obs["depth"], dtype=np.float32).shape
    state_dim = int(np.asarray(first_obs["state"], dtype=np.float32).shape[0])
    teacher_dim = int(
        np.asarray(step_records[0]["info"]["privileged"]["teacher_state"], dtype=np.float32).shape[0]
    )

    depth = np.zeros((num_steps,) + depth_shape, dtype=np.float32)
    state = np.zeros((num_steps, state_dim), dtype=np.float32)
    action = np.zeros((num_steps, 5), dtype=np.float32)
    reward = np.zeros((num_steps,), dtype=np.float32)
    terminated = np.zeros((num_steps,), dtype=np.int8)
    truncated = np.zeros((num_steps,), dtype=np.int8)
    teacher_state = np.zeros((num_steps, teacher_dim), dtype=np.float32)
    visible = np.zeros((num_steps,), dtype=np.int8)
    pixel_norm = np.zeros((num_steps, 2), dtype=np.float32)
    distance_m = np.zeros((num_steps,), dtype=np.float32)
    distance_bucket = np.zeros((num_steps,), dtype=np.int64)
    relative_platform = np.zeros((num_steps, 3), dtype=np.float32)
    mode_id = np.zeros((num_steps,), dtype=np.int64)
    challenge_type = np.zeros((num_steps,), dtype=np.int64)
    moving_platform = np.zeros((num_steps,), dtype=np.int8)

    for idx, record in enumerate(step_records):
        obs = record["observation"]
        info = record["info"]
        privileged = dict(info["privileged"])
        labels = make_perception_labels(obs, privileged)

        depth[idx] = np.asarray(obs["depth"], dtype=np.float32)
        state[idx] = np.asarray(obs["state"], dtype=np.float32)
        action[idx] = np.asarray(record["action"], dtype=np.float32)
        reward[idx] = np.float32(record["reward"])
        terminated[idx] = int(record["terminated"])
        truncated[idx] = int(record["truncated"])
        teacher_state[idx] = np.asarray(privileged["teacher_state"], dtype=np.float32)
        visible[idx] = labels["visible"]
        pixel_norm[idx] = labels["pixel_norm"]
        distance_m[idx] = labels["distance_m"]
        distance_bucket[idx] = labels["distance_bucket"]
        relative_platform[idx] = labels["relative_platform"]
        mode_name = str(record.get("metadata", {}).get("expert_mode", "idle"))
        mode_id[idx] = MODE_TO_ID.get(mode_name, MODE_TO_ID["idle"])
        challenge_type[idx] = int(privileged["challenge_type"])
        moving_platform[idx] = int(privileged["moving_platform"])

    return {
        "depth": depth,
        "state": state,
        "action": action,
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "teacher_state": teacher_state,
        "visible": visible,
        "pixel_norm": pixel_norm,
        "distance_m": distance_m,
        "distance_bucket": distance_bucket,
        "relative_platform": relative_platform,
        "mode_id": mode_id,
        "challenge_type": challenge_type,
        "moving_platform": moving_platform,
    }


def save_episode_dataset(
    output_dir: str | Path,
    episode_name: str,
    step_records: list[dict[str, Any]],
    summary: EpisodeSummary,
) -> Path:
    output_dir = ensure_dir(output_dir)
    arrays = _episode_arrays(step_records)
    npz_path = output_dir / f"{episode_name}.npz"
    meta_path = output_dir / f"{episode_name}.json"
    np.savez_compressed(npz_path, **arrays)
    save_json(meta_path, asdict(summary))
    return npz_path


def list_episode_files(dataset_root: str | Path) -> list[Path]:
    return sorted(Path(dataset_root).rglob("*.npz"))


def load_episode_dataset(npz_path: str | Path) -> dict[str, np.ndarray]:
    """Load a saved episode file into memory for smoke validation."""

    with np.load(Path(npz_path)) as episode:
        return {key: np.asarray(value) for key, value in episode.items()}


def validate_episode_dataset(npz_path: str | Path) -> dict[str, tuple[int, ...]]:
    """Validate the dataset schema and return the observed tensor shapes."""

    arrays = load_episode_dataset(npz_path)
    required_keys = {
        "depth",
        "state",
        "action",
        "reward",
        "terminated",
        "truncated",
        "teacher_state",
        "visible",
        "pixel_norm",
        "distance_m",
        "distance_bucket",
        "relative_platform",
        "mode_id",
        "challenge_type",
        "moving_platform",
    }
    missing = required_keys.difference(arrays.keys())
    if missing:
        raise ValueError(f"Episode dataset is missing keys: {sorted(missing)}")

    shapes = {key: tuple(value.shape) for key, value in arrays.items()}
    if arrays["depth"].ndim != 4:
        raise ValueError(f"depth must be 4D, got shape {arrays['depth'].shape}")
    if arrays["state"].ndim != 2:
        raise ValueError(f"state must be 2D, got shape {arrays['state'].shape}")
    if arrays["action"].ndim != 2 or arrays["action"].shape[-1] != 5:
        raise ValueError(f"action must be (T, 5), got shape {arrays['action'].shape}")
    if arrays["teacher_state"].ndim != 2:
        raise ValueError(f"teacher_state must be 2D, got shape {arrays['teacher_state'].shape}")
    return shapes
