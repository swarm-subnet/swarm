"""Behavior cloning dataset and training helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Any

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
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
    dataset_manifest: str | None = None
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
    early_stopping_patience: int | None = 5
    early_stopping_min_delta: float = 1e-4
    early_stopping_min_epochs: int = 3
    log_every_batches: int = 0


class EpisodeSequenceDataset(Dataset):
    """Windowed recurrent dataset over `.npz` episode files."""

    def __init__(self, episode_entries: list[dict[str, Any]], *, seq_len: int, stride: int):
        self.episode_entries = [
            {
                "episode_path": Path(entry["episode_path"]),
                "sample_weight": float(entry.get("sample_weight", 1.0)),
            }
            for entry in episode_entries
        ]
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.index: list[tuple[Path, int, int]] = []
        self.sample_weights: list[float] = []
        for entry in self.episode_entries:
            file_path = Path(entry["episode_path"])
            sample_weight = float(entry["sample_weight"])
            with np.load(file_path) as episode:
                length = int(episode["action"].shape[0])
            if length < seq_len:
                self.index.append((file_path, 0, length))
                self.sample_weights.append(sample_weight)
                continue
            for start in range(0, max(1, length - seq_len + 1), stride):
                stop = min(length, start + seq_len)
                self.index.append((file_path, start, stop))
                self.sample_weights.append(sample_weight)

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


def build_dataloader(
    config: BehaviorCloningConfig,
    *,
    episode_entries: list[dict[str, Any]],
    shuffle: bool,
) -> DataLoader:
    dataset = EpisodeSequenceDataset(episode_entries, seq_len=config.seq_len, stride=config.stride)
    sampler = None
    if shuffle:
        weights = torch.as_tensor(dataset.sample_weights, dtype=torch.double)
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(dataset), replacement=True)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False if sampler is not None else shuffle,
        sampler=sampler,
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


def _compute_loss_terms(
    *,
    outputs: dict[str, torch.Tensor | None],
    batch: dict[str, torch.Tensor],
    mask: torch.Tensor,
    model_config: StudentModelConfig,
    bce,
    ce,
) -> dict[str, torch.Tensor]:
    target_action = batch["action"]
    action_loss = _masked_mean((outputs["action"] - target_action) ** 2, mask)
    zero = action_loss.new_zeros(())
    visibility_loss = zero
    pixel_loss = zero
    bucket_loss = zero

    if outputs["visibility_logit"] is not None:
        target_visible = batch["visible"].unsqueeze(-1)
        target_pixel = batch["pixel_norm"]
        target_bucket = batch["distance_bucket"]
        visibility_loss = _masked_mean(
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

    return {
        "action_loss": action_loss,
        "visibility_loss": visibility_loss,
        "pixel_loss": pixel_loss,
        "bucket_loss": bucket_loss,
    }


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

    if config.dataset_manifest:
        manifest_payload = load_json(config.dataset_manifest)
        episode_entries = list(manifest_payload.get("episodes", []))
        if not episode_entries:
            raise FileNotFoundError(f"No weighted dataset entries found in {config.dataset_manifest}")
    else:
        episode_files = list_episode_files(config.dataset_root)
        episode_entries = [{"episode_path": str(path), "sample_weight": 1.0} for path in episode_files]

    if config.max_episodes is not None:
        episode_entries = episode_entries[: config.max_episodes]
    if not episode_entries:
        raise FileNotFoundError(
            f"No dataset episodes found for root={config.dataset_root} manifest={config.dataset_manifest}"
        )

    split_idx = max(1, int(0.8 * len(episode_entries)))
    train_entries = episode_entries[:split_idx]
    val_entries = episode_entries[split_idx:] or episode_entries[:1]
    train_loader = build_dataloader(config, episode_entries=train_entries, shuffle=True)
    val_loader = build_dataloader(config, episode_entries=val_entries, shuffle=False)

    best_val = float("inf")
    best_val_action = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    stopped_early = False
    history: list[dict[str, float]] = []
    print(
        "[BC] starting "
        f"episodes={len(episode_entries)} train_episodes={len(train_entries)} val_episodes={len(val_entries)} "
        f"train_windows={len(train_loader.dataset)} val_windows={len(val_loader.dataset)} "
        f"epochs={config.epochs} batch_size={config.batch_size} seq_len={config.seq_len} stride={config.stride} "
        f"lr={config.learning_rate} device={config.device}",
        flush=True,
    )
    for epoch in range(config.epochs):
        epoch_start = time.perf_counter()
        model.train()
        train_loss_total = 0.0
        train_action_loss_total = 0.0
        train_batches = 0
        for batch_idx, batch in enumerate(train_loader, start=1):
            depth = batch["depth"].to(config.device)
            state = batch["state"].to(config.device)
            target_action = batch["action"].to(config.device)
            mask = batch["mask"].to(config.device)
            visible = batch["visible"].to(config.device)
            pixel_norm = batch["pixel_norm"].to(config.device)
            distance_bucket = batch["distance_bucket"].to(config.device)
            outputs = model(depth, state)

            loss_terms = _compute_loss_terms(
                outputs=outputs,
                batch={
                    "action": target_action,
                    "visible": visible,
                    "pixel_norm": pixel_norm,
                    "distance_bucket": distance_bucket,
                },
                mask=mask,
                model_config=model_config,
                bce=bce,
                ce=ce,
            )
            action_loss = loss_terms["action_loss"]
            loss = config.action_loss_weight * action_loss
            loss = loss + (
                config.visibility_loss_weight * loss_terms["visibility_loss"]
                + config.pixel_loss_weight * loss_terms["pixel_loss"]
                + config.distance_bucket_loss_weight * loss_terms["bucket_loss"]
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_total += float(loss.item())
            train_action_loss_total += float(action_loss.item())
            train_batches += 1
            if config.log_every_batches and (
                batch_idx == 1
                or batch_idx % config.log_every_batches == 0
                or batch_idx == len(train_loader)
            ):
                running_train_loss = train_loss_total / max(train_batches, 1)
                running_train_action_loss = train_action_loss_total / max(train_batches, 1)
                print(
                    f"[BC][epoch {epoch + 1}/{config.epochs}][batch {batch_idx}/{len(train_loader)}] "
                    f"train_total_loss={running_train_loss:.6f} "
                    f"train_action_loss={running_train_action_loss:.6f}",
                    flush=True,
                )

        model.eval()
        val_loss_total = 0.0
        val_action_loss_total = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                depth = batch["depth"].to(config.device)
                state = batch["state"].to(config.device)
                target_action = batch["action"].to(config.device)
                mask = batch["mask"].to(config.device)
                visible = batch["visible"].to(config.device)
                pixel_norm = batch["pixel_norm"].to(config.device)
                distance_bucket = batch["distance_bucket"].to(config.device)
                outputs = model(depth, state)
                loss_terms = _compute_loss_terms(
                    outputs=outputs,
                    batch={
                        "action": target_action,
                        "visible": visible,
                        "pixel_norm": pixel_norm,
                        "distance_bucket": distance_bucket,
                    },
                    mask=mask,
                    model_config=model_config,
                    bce=bce,
                    ce=ce,
                )
                loss = config.action_loss_weight * loss_terms["action_loss"]
                loss = loss + (
                    config.visibility_loss_weight * loss_terms["visibility_loss"]
                    + config.pixel_loss_weight * loss_terms["pixel_loss"]
                    + config.distance_bucket_loss_weight * loss_terms["bucket_loss"]
                )
                val_loss_total += float(loss.item())
                val_action_loss_total += float(loss_terms["action_loss"].item())
                val_batches += 1

        train_loss = train_loss_total / max(train_batches, 1)
        train_action_loss = train_action_loss_total / max(train_batches, 1)
        val_loss = val_loss_total / max(val_batches, 1)
        val_action_loss = val_action_loss_total / max(val_batches, 1)
        improved = val_loss < (best_val - float(config.early_stopping_min_delta))
        history.append(
            {
                "epoch": epoch,
                "train_total_loss": train_loss,
                "train_action_loss": train_action_loss,
                "val_total_loss": val_loss,
                "val_action_loss": val_action_loss,
                "best_val_total_loss_so_far": min(best_val, val_loss) if best_epoch >= 0 else val_loss,
                "improved": bool(improved),
                "epoch_wall_time_sec": time.perf_counter() - epoch_start,
            }
        )
        if improved:
            best_val = val_loss
            best_val_action = val_action_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                output_dir / "best_student.pt",
                model,
                optimizer=optimizer,
                extra={"epoch": epoch, "train_history": history},
            )
        else:
            epochs_without_improvement += 1

        print(
            f"[BC][epoch {epoch + 1}/{config.epochs}] "
            f"train_total_loss={train_loss:.6f} train_action_loss={train_action_loss:.6f} "
            f"val_total_loss={val_loss:.6f} val_action_loss={val_action_loss:.6f} "
            f"best_val_total={best_val:.6f} best_epoch={best_epoch + 1 if best_epoch >= 0 else 0} "
            f"improved={'yes' if improved else 'no'} no_improve={epochs_without_improvement} "
            f"epoch_sec={history[-1]['epoch_wall_time_sec']:.2f}",
            flush=True,
        )

        if (
            config.early_stopping_patience is not None
            and config.early_stopping_patience >= 0
            and (epoch + 1) >= int(config.early_stopping_min_epochs)
            and epochs_without_improvement >= int(config.early_stopping_patience)
        ):
            stopped_early = True
            print(
                f"[BC] early stopping at epoch {epoch + 1}: "
                f"no validation improvement for {epochs_without_improvement} epoch(s)",
                flush=True,
            )
            break

    save_json(output_dir / "bc_config.json", asdict(config))
    save_json(output_dir / "bc_history.json", history)
    return {
        "best_val_total_loss": best_val,
        "best_val_action_loss": best_val_action,
        "best_epoch": (best_epoch + 1) if best_epoch >= 0 else None,
        "best_epoch_index": best_epoch,
        "epochs_completed": len(history),
        "stopped_early": stopped_early,
        "history_path": str(output_dir / "bc_history.json"),
        "checkpoint_path": str(output_dir / "best_student.pt"),
    }
