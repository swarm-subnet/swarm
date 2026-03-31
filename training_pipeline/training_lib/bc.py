"""Behavior cloning dataset and training helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover - optional during non-training usage
    raise ImportError(
        "training_lib.bc requires PyTorch to train the student policy."
    ) from exc

from .common import ensure_dir, load_json, save_json
from .dataset import list_episode_files
from .models import StudentModelConfig, StudentPolicy, load_model_config, save_checkpoint


@dataclass
class BehaviorCloningConfig:
    dataset_root: str
    output_dir: str
    model_config_path: str
    init_checkpoint: str | None = None
    seq_len: int = 16
    stride: int = 8
    batch_size: int = 16
    epochs: int = 5
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    action_loss_weight: float = 1.0
    visibility_loss_weight: float = 0.25
    pixel_loss_weight: float = 0.10
    distance_bucket_loss_weight: float = 0.10
    device: str = "cpu"
    num_workers: int = 0
    max_episodes: int | None = None


class EpisodeSequenceDataset(Dataset):
    """Windowed recurrent dataset over `.npz` episode files."""

    def __init__(self, episode_files: list[Path], *, seq_len: int, stride: int):
        self.episode_files = list(episode_files)
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.index: list[tuple[Path, int, int]] = []
        for file_path in self.episode_files:
            with np.load(file_path) as episode:
                length = int(episode["action"].shape[0])
            if length < seq_len:
                self.index.append((file_path, 0, length))
                continue
            for start in range(0, max(1, length - seq_len + 1), stride):
                stop = min(length, start + seq_len)
                self.index.append((file_path, start, stop))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        path, start, stop = self.index[idx]
        with np.load(path) as episode:
            depth = torch.as_tensor(episode["depth"][start:stop], dtype=torch.float32)
            state = torch.as_tensor(episode["state"][start:stop], dtype=torch.float32)
            action = torch.as_tensor(episode["action"][start:stop], dtype=torch.float32)
            visible = torch.as_tensor(episode["visible"][start:stop], dtype=torch.float32)
            pixel_norm = torch.as_tensor(episode["pixel_norm"][start:stop], dtype=torch.float32)
            distance_bucket = torch.as_tensor(episode["distance_bucket"][start:stop], dtype=torch.long)
        return {
            "depth": depth,
            "state": state,
            "action": action,
            "visible": visible,
            "pixel_norm": pixel_norm,
            "distance_bucket": distance_bucket,
        }


def _collate_sequences(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    max_len = max(sample["action"].shape[0] for sample in batch)

    def pad_tensor(tensor: torch.Tensor, pad_value: float = 0.0) -> torch.Tensor:
        if tensor.ndim == 1:
            padded = torch.full((max_len,), pad_value, dtype=tensor.dtype)
            padded[: tensor.shape[0]] = tensor
            return padded
        shape = (max_len,) + tuple(tensor.shape[1:])
        padded = torch.full(shape, pad_value, dtype=tensor.dtype)
        padded[: tensor.shape[0]] = tensor
        return padded

    output = {}
    for key in batch[0].keys():
        pad_value = 0 if key == "distance_bucket" else 0.0
        output[key] = torch.stack([pad_tensor(sample[key], pad_value) for sample in batch], dim=0)
    output["mask"] = torch.stack(
        [pad_tensor(torch.ones(sample["action"].shape[0], dtype=torch.float32)) for sample in batch],
        dim=0,
    )
    return output


def build_dataloader(config: BehaviorCloningConfig, *, episode_files: list[Path], shuffle: bool) -> DataLoader:
    dataset = EpisodeSequenceDataset(episode_files, seq_len=config.seq_len, stride=config.stride)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        collate_fn=_collate_sequences,
    )


def _masked_mean(loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    while mask.ndim < loss.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.to(dtype=loss.dtype)
    expanded_mask = torch.broadcast_to(mask, loss.shape)
    masked = loss * expanded_mask
    denom = torch.clamp(expanded_mask.sum(), min=1.0)
    return masked.sum() / denom


def run_behavior_cloning(config: BehaviorCloningConfig) -> dict[str, Any]:
    output_dir = ensure_dir(config.output_dir)
    model_config = load_model_config(config.model_config_path)
    model = StudentPolicy(model_config).to(config.device)

    if config.init_checkpoint:
        payload = torch.load(config.init_checkpoint, map_location=config.device)
        model.load_state_dict(payload["model_state_dict"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    ce = torch.nn.CrossEntropyLoss(reduction="none")

    episode_files = list_episode_files(config.dataset_root)
    if config.max_episodes is not None:
        episode_files = episode_files[: config.max_episodes]
    if not episode_files:
        raise FileNotFoundError(f"No .npz episodes found in {config.dataset_root}")

    split_idx = max(1, int(0.8 * len(episode_files)))
    train_files = episode_files[:split_idx]
    val_files = episode_files[split_idx:] or episode_files[:1]
    train_loader = build_dataloader(config, episode_files=train_files, shuffle=True)
    val_loader = build_dataloader(config, episode_files=val_files, shuffle=False)

    best_val = float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(config.epochs):
        model.train()
        train_loss_total = 0.0
        train_batches = 0
        for batch in train_loader:
            depth = batch["depth"].to(config.device)
            state = batch["state"].to(config.device)
            target_action = batch["action"].to(config.device)
            mask = batch["mask"].to(config.device)
            outputs = model(depth, state)

            action_loss = _masked_mean((outputs["action"] - target_action) ** 2, mask)
            loss = config.action_loss_weight * action_loss

            if outputs["visibility_logit"] is not None:
                target_visible = batch["visible"].to(config.device).unsqueeze(-1)
                target_pixel = batch["pixel_norm"].to(config.device)
                target_bucket = batch["distance_bucket"].to(config.device)
                visible_loss = _masked_mean(
                    bce(outputs["visibility_logit"], target_visible),
                    mask,
                )
                pixel_loss = _masked_mean(
                    (outputs["pixel_norm"] - target_pixel) ** 2,
                    mask,
                )
                bucket_loss = _masked_mean(
                    ce(
                        outputs["distance_bucket_logits"].reshape(-1, model_config.perception_num_buckets),
                        target_bucket.reshape(-1),
                    ).reshape(mask.shape),
                    mask,
                )
                loss = loss + (
                    config.visibility_loss_weight * visible_loss
                    + config.pixel_loss_weight * pixel_loss
                    + config.distance_bucket_loss_weight * bucket_loss
                )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_total += float(loss.item())
            train_batches += 1

        model.eval()
        val_loss_total = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                depth = batch["depth"].to(config.device)
                state = batch["state"].to(config.device)
                target_action = batch["action"].to(config.device)
                mask = batch["mask"].to(config.device)
                outputs = model(depth, state)
                loss = _masked_mean((outputs["action"] - target_action) ** 2, mask)
                val_loss_total += float(loss.item())
                val_batches += 1

        train_loss = train_loss_total / max(train_batches, 1)
        val_loss = val_loss_total / max(val_batches, 1)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                output_dir / "best_student.pt",
                model,
                optimizer=optimizer,
                extra={"epoch": epoch, "train_history": history},
            )

    save_json(output_dir / "bc_config.json", asdict(config))
    save_json(output_dir / "bc_history.json", history)
    return {
        "best_val_loss": best_val,
        "history_path": str(output_dir / "bc_history.json"),
        "checkpoint_path": str(output_dir / "best_student.pt"),
    }
