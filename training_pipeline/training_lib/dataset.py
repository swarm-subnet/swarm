"""Dataset builders for expert rollouts and perception labels."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .common import EpisodeSummary, ensure_dir, load_json, save_json
from .experts import EXPERT_ACTION_DIM, expert_mode_id, map_category_label
from .geometry import DEFAULT_CAMERA_FOV_RAD, relative_vector_to_pixel

EXPECTED_STATE_DIM = 141
EXPECTED_TEACHER_STATE_DIM = 15

DATASET_WEIGHTING_POLICIES = (
    "uniform",
    "balanced_by_map_category",
    "balanced_by_teacher",
)


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
    action = np.zeros((num_steps, EXPERT_ACTION_DIM), dtype=np.float32)
    reward = np.zeros((num_steps,), dtype=np.float32)
    terminated = np.zeros((num_steps,), dtype=np.int8)
    truncated = np.zeros((num_steps,), dtype=np.int8)
    teacher_state = np.zeros((num_steps, teacher_dim), dtype=np.float32)
    visible = np.zeros((num_steps,), dtype=np.int8)
    pixel_row = np.zeros((num_steps,), dtype=np.int64)
    pixel_col = np.zeros((num_steps,), dtype=np.int64)
    pixel_norm = np.zeros((num_steps, 2), dtype=np.float32)
    distance_m = np.zeros((num_steps,), dtype=np.float32)
    distance_bucket = np.zeros((num_steps,), dtype=np.int64)
    relative_platform = np.zeros((num_steps, 3), dtype=np.float32)
    mode_id = np.zeros((num_steps,), dtype=np.int64)
    challenge_type = np.zeros((num_steps,), dtype=np.int64)
    moving_platform = np.zeros((num_steps,), dtype=np.int8)
    first_metadata = dict(step_records[0].get("metadata", {}))
    first_privileged = dict(step_records[0]["info"]["privileged"])
    teacher_id = np.asarray(str(first_metadata.get("teacher_id", "expert_unknown")))
    teacher_version = np.asarray(str(first_metadata.get("teacher_version", "v0")))
    map_category = np.asarray(
        str(
            first_metadata.get(
                "map_category",
                map_category_label(
                    int(first_privileged["challenge_type"]),
                    bool(first_privileged["moving_platform"]),
                ),
            )
        )
    )
    expert_mode_vocab_version = np.asarray(str(first_metadata.get("expert_mode_vocab_version", "unknown")))
    expert_action_dim = np.asarray(int(first_metadata.get("expert_action_dim", EXPERT_ACTION_DIM)), dtype=np.int64)

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
        pixel_row[idx] = labels["pixel_row"]
        pixel_col[idx] = labels["pixel_col"]
        pixel_norm[idx] = labels["pixel_norm"]
        distance_m[idx] = labels["distance_m"]
        distance_bucket[idx] = labels["distance_bucket"]
        relative_platform[idx] = labels["relative_platform"]
        mode_name = str(record.get("metadata", {}).get("expert_mode", "idle"))
        mode_id[idx] = expert_mode_id(mode_name)
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
        "pixel_row": pixel_row,
        "pixel_col": pixel_col,
        "pixel_norm": pixel_norm,
        "distance_m": distance_m,
        "distance_bucket": distance_bucket,
        "relative_platform": relative_platform,
        "mode_id": mode_id,
        "challenge_type": challenge_type,
        "moving_platform": moving_platform,
        "teacher_id": teacher_id,
        "teacher_version": teacher_version,
        "map_category": map_category,
        "expert_mode_vocab_version": expert_mode_vocab_version,
        "expert_action_dim": expert_action_dim,
    }


def _episode_metadata(step_records: list[dict[str, Any]], summary: EpisodeSummary) -> dict[str, Any]:
    metadata = dict(step_records[0].get("metadata", {})) if step_records else {}
    teacher_id = str(metadata.get("teacher_id", "expert_unknown"))
    teacher_version = str(metadata.get("teacher_version", "v0"))
    map_category = str(
        metadata.get(
            "map_category",
            map_category_label(summary.challenge_type, summary.moving_platform),
        )
    )
    return {
        **asdict(summary),
        "teacher_id": teacher_id,
        "teacher_version": teacher_version,
        "map_category": map_category,
        "expert_mode_vocab_version": str(metadata.get("expert_mode_vocab_version", "unknown")),
        "expert_action_dim": int(metadata.get("expert_action_dim", EXPERT_ACTION_DIM)),
    }


def save_episode_dataset(
    output_dir: str | Path,
    episode_name: str,
    step_records: list[dict[str, Any]],
    summary: EpisodeSummary,
    *,
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    output_dir = ensure_dir(output_dir)
    arrays = _episode_arrays(step_records)
    npz_path = output_dir / f"{episode_name}.npz"
    meta_path = output_dir / f"{episode_name}.json"
    np.savez_compressed(npz_path, **arrays)
    episode_metadata = _episode_metadata(step_records, summary)
    if extra_metadata:
        episode_metadata.update(extra_metadata)
    save_json(meta_path, episode_metadata)
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
        "pixel_row",
        "pixel_col",
        "pixel_norm",
        "distance_m",
        "distance_bucket",
        "relative_platform",
        "mode_id",
        "challenge_type",
        "moving_platform",
        "teacher_id",
        "teacher_version",
        "map_category",
        "expert_mode_vocab_version",
        "expert_action_dim",
    }
    missing = required_keys.difference(arrays.keys())
    if missing:
        raise ValueError(f"Episode dataset is missing keys: {sorted(missing)}")

    shapes = {key: tuple(value.shape) for key, value in arrays.items()}
    num_steps = int(arrays["action"].shape[0])
    if arrays["depth"].ndim != 4:
        raise ValueError(f"depth must be 4D, got shape {arrays['depth'].shape}")
    if arrays["depth"].shape[-1] != 1:
        raise ValueError(f"depth must have channel dimension 1, got shape {arrays['depth'].shape}")
    if arrays["state"].ndim != 2:
        raise ValueError(f"state must be 2D, got shape {arrays['state'].shape}")
    if arrays["state"].shape[-1] != EXPECTED_STATE_DIM:
        raise ValueError(
            f"state must have dim {EXPECTED_STATE_DIM}, got shape {arrays['state'].shape}"
        )
    if arrays["action"].ndim != 2 or arrays["action"].shape[-1] != EXPERT_ACTION_DIM:
        raise ValueError(f"action must be (T, {EXPERT_ACTION_DIM}), got shape {arrays['action'].shape}")
    if arrays["teacher_state"].ndim != 2:
        raise ValueError(f"teacher_state must be 2D, got shape {arrays['teacher_state'].shape}")
    if arrays["teacher_state"].shape[-1] != EXPECTED_TEACHER_STATE_DIM:
        raise ValueError(
            "teacher_state must have dim "
            f"{EXPECTED_TEACHER_STATE_DIM}, got shape {arrays['teacher_state'].shape}"
        )
    if arrays["pixel_norm"].shape != (num_steps, 2):
        raise ValueError(f"pixel_norm must be (T, 2), got shape {arrays['pixel_norm'].shape}")
    if arrays["relative_platform"].shape != (num_steps, 3):
        raise ValueError(f"relative_platform must be (T, 3), got shape {arrays['relative_platform'].shape}")
    if arrays["visible"].shape != (num_steps,):
        raise ValueError(f"visible must be (T,), got shape {arrays['visible'].shape}")
    if arrays["pixel_row"].shape != (num_steps,) or arrays["pixel_col"].shape != (num_steps,):
        raise ValueError(
            "pixel_row and pixel_col must be (T,), got shapes "
            f"{arrays['pixel_row'].shape} and {arrays['pixel_col'].shape}"
        )
    temporal_keys = (
        "depth",
        "state",
        "action",
        "reward",
        "terminated",
        "truncated",
        "teacher_state",
        "visible",
        "pixel_row",
        "pixel_col",
        "pixel_norm",
        "distance_m",
        "distance_bucket",
        "relative_platform",
        "mode_id",
        "challenge_type",
        "moving_platform",
    )
    for key in temporal_keys:
        if int(arrays[key].shape[0]) != num_steps:
            raise ValueError(
                f"Temporal key {key} has inconsistent leading dimension {arrays[key].shape[0]} "
                f"(expected {num_steps})"
            )
    if int(np.asarray(arrays["expert_action_dim"]).item()) != EXPERT_ACTION_DIM:
        raise ValueError(
            f"expert_action_dim must be {EXPERT_ACTION_DIM}, got {int(np.asarray(arrays['expert_action_dim']).item())}"
        )
    return shapes


def build_weighted_dataset_manifest(
    dataset_root: str | Path,
    *,
    weighting_policy: str = "balanced_by_map_category",
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    if weighting_policy not in DATASET_WEIGHTING_POLICIES:
        raise ValueError(
            f"Unsupported weighting_policy={weighting_policy!r}. "
            f"Expected one of {DATASET_WEIGHTING_POLICIES}."
        )

    episode_files = list_episode_files(dataset_root)
    entries: list[dict[str, Any]] = []
    for episode_path in episode_files:
        meta_path = episode_path.with_suffix(".json")
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing episode metadata file for {episode_path}")
        meta = load_json(meta_path)
        entries.append(
            {
                "episode_path": str(episode_path),
                "episode_meta_path": str(meta_path),
                "map_category": str(meta.get("map_category", "unknown")),
                "teacher_id": str(meta.get("teacher_id", "expert_unknown")),
                "teacher_version": str(meta.get("teacher_version", "v0")),
                "base_weight": float(meta.get("dataset_weight", 1.0)),
            }
        )

    if not entries:
        raise FileNotFoundError(f"No .npz episodes found in {dataset_root}")

    group_key = None
    if weighting_policy == "balanced_by_map_category":
        group_key = "map_category"
    elif weighting_policy == "balanced_by_teacher":
        group_key = "teacher_id"

    if group_key is None:
        for entry in entries:
            entry["sample_weight"] = float(entry["base_weight"])
    else:
        group_counts: dict[str, int] = {}
        for entry in entries:
            group = str(entry[group_key])
            group_counts[group] = group_counts.get(group, 0) + 1
        num_groups = max(1, len(group_counts))
        total_episodes = len(entries)
        for entry in entries:
            group = str(entry[group_key])
            reweight = float(total_episodes) / float(num_groups * group_counts[group])
            entry["sample_weight"] = float(entry["base_weight"] * reweight)

    manifest = {
        "format_version": 1,
        "dataset_root": str(Path(dataset_root)),
        "weighting_policy": weighting_policy,
        "episode_count": len(entries),
        "episodes": entries,
    }
    if output_path is not None:
        save_json(output_path, manifest)
    return manifest
