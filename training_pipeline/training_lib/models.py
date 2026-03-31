"""Deployable student model and training-time critic skeletons."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover - optional during non-training usage
    raise ImportError(
        "training_lib.models requires PyTorch for model definition and training."
    ) from exc


@dataclass
class StudentModelConfig:
    depth_channels: int = 1
    state_dim: int = 141
    teacher_state_dim: int = 15
    action_dim: int = 5
    hidden_dim: int = 256
    gru_hidden_dim: int = 256
    perception_num_buckets: int = 6
    use_aux_heads: bool = True


def validate_student_contract(config: StudentModelConfig) -> None:
    """Enforce the deploy-time model contract used by the default pipeline."""

    if config.state_dim != 141:
        raise ValueError(f"Expected state_dim=141, got {config.state_dim}")
    if config.action_dim != 5:
        raise ValueError(f"Expected action_dim=5, got {config.action_dim}")
    if config.teacher_state_dim != 15:
        raise ValueError(f"Expected teacher_state_dim=15, got {config.teacher_state_dim}")
    if config.depth_channels != 1:
        raise ValueError(f"Expected depth_channels=1, got {config.depth_channels}")


class DepthEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        return self.proj(self.conv(depth))


class StateEncoder(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class StudentPolicy(nn.Module):
    """Student actor with optional auxiliary perception heads."""

    def __init__(self, config: StudentModelConfig):
        super().__init__()
        self.config = config
        self.depth_encoder = DepthEncoder(config.depth_channels, config.hidden_dim)
        self.state_encoder = StateEncoder(config.state_dim, config.hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(2 * config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRU(
            input_size=config.hidden_dim,
            hidden_size=config.gru_hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.action_head = nn.Sequential(
            nn.Linear(config.gru_hidden_dim, config.gru_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.gru_hidden_dim, config.action_dim),
        )
        if config.use_aux_heads:
            self.visibility_head = nn.Linear(config.gru_hidden_dim, 1)
            self.pixel_head = nn.Linear(config.gru_hidden_dim, 2)
            self.distance_bucket_head = nn.Linear(config.gru_hidden_dim, config.perception_num_buckets)
        else:
            self.visibility_head = None
            self.pixel_head = None
            self.distance_bucket_head = None

    def _prepare_inputs(self, depth: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        if depth.ndim == 4:
            depth = depth.unsqueeze(1)
        if state.ndim == 2:
            state = state.unsqueeze(1)
        batch_size, time_steps = depth.shape[0], depth.shape[1]
        depth = depth.permute(0, 1, 4, 2, 3).contiguous().view(batch_size * time_steps, self.config.depth_channels, depth.shape[2], depth.shape[3])
        state = state.contiguous().view(batch_size * time_steps, self.config.state_dim)
        return depth, state, batch_size, time_steps

    def _squash_action(self, action_raw: torch.Tensor) -> torch.Tensor:
        direction = torch.tanh(action_raw[..., 0:3])
        speed = torch.sigmoid(action_raw[..., 3:4])
        yaw = torch.tanh(action_raw[..., 4:5])
        return torch.cat([direction, speed, yaw], dim=-1)

    def forward(
        self,
        depth: torch.Tensor,
        state: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        depth, state, batch_size, time_steps = self._prepare_inputs(depth, state)
        depth_features = self.depth_encoder(depth)
        state_features = self.state_encoder(state)
        fused = self.fusion(torch.cat([depth_features, state_features], dim=-1))
        fused = fused.view(batch_size, time_steps, self.config.hidden_dim)
        gru_out, hidden_state = self.gru(fused, hidden_state)
        action_raw = self.action_head(gru_out)
        outputs: dict[str, torch.Tensor | None] = {
            "latent": gru_out,
            "action_raw": action_raw,
            "action": self._squash_action(action_raw),
            "hidden_state": hidden_state,
        }
        if self.visibility_head is not None:
            outputs["visibility_logit"] = self.visibility_head(gru_out)
            outputs["pixel_norm"] = torch.tanh(self.pixel_head(gru_out))
            outputs["distance_bucket_logits"] = self.distance_bucket_head(gru_out)
        else:
            outputs["visibility_logit"] = None
            outputs["pixel_norm"] = None
            outputs["distance_bucket_logits"] = None
        return outputs


class StudentRuntimePolicy(nn.Module):
    """Deployable TorchScript-friendly student that returns a single action."""

    def __init__(self, config: StudentModelConfig):
        super().__init__()
        validate_student_contract(config)
        self.depth_channels = int(config.depth_channels)
        self.state_dim = int(config.state_dim)
        self.hidden_dim = int(config.hidden_dim)
        self.gru_hidden_dim = int(config.gru_hidden_dim)
        self.action_dim = int(config.action_dim)
        self.depth_encoder = DepthEncoder(self.depth_channels, self.hidden_dim)
        self.state_encoder = StateEncoder(self.state_dim, self.hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.gru_hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.action_head = nn.Sequential(
            nn.Linear(self.gru_hidden_dim, self.gru_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.gru_hidden_dim, self.action_dim),
        )

    def _prepare_inputs(self, depth: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        if depth.ndim == 4:
            depth = depth.unsqueeze(1)
        if state.ndim == 2:
            state = state.unsqueeze(1)
        batch_size, time_steps = depth.shape[0], depth.shape[1]
        depth = depth.permute(0, 1, 4, 2, 3).contiguous().view(
            batch_size * time_steps,
            self.depth_channels,
            depth.shape[2],
            depth.shape[3],
        )
        state = state.contiguous().view(batch_size * time_steps, self.state_dim)
        return depth, state, batch_size, time_steps

    def _squash_action(self, action_raw: torch.Tensor) -> torch.Tensor:
        direction = torch.tanh(action_raw[..., 0:3])
        speed = torch.sigmoid(action_raw[..., 3:4])
        yaw = torch.tanh(action_raw[..., 4:5])
        return torch.cat([direction, speed, yaw], dim=-1)

    def forward(self, depth: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        depth, state, batch_size, time_steps = self._prepare_inputs(depth, state)
        depth_features = self.depth_encoder(depth)
        state_features = self.state_encoder(state)
        fused = self.fusion(torch.cat([depth_features, state_features], dim=-1))
        fused = fused.view(batch_size, time_steps, self.hidden_dim)
        gru_out, _ = self.gru(fused, None)
        action_raw = self.action_head(gru_out[:, -1, :])
        return self._squash_action(action_raw)


class AsymmetricCritic(nn.Module):
    """Critic that can ingest student latent state plus privileged teacher state."""

    def __init__(self, config: StudentModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.gru_hidden_dim + config.teacher_state_dim, config.gru_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.gru_hidden_dim, config.gru_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.gru_hidden_dim, 1),
        )

    def forward(self, latent: torch.Tensor, teacher_state: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([latent, teacher_state], dim=-1))


class StudentInferencePolicy:
    """Thin inference wrapper so stage scripts can roll out a torch student."""

    def __init__(self, model: StudentPolicy, *, device: str = "cpu"):
        self.model = model.to(device)
        self.device = torch.device(device)
        self.hidden_state: torch.Tensor | None = None
        self.model.eval()
        self.last_metadata: dict[str, Any] = {}

    def reset(self) -> None:
        self.hidden_state = None
        self.last_metadata = {}

    def get_last_metadata(self) -> dict[str, Any]:
        return dict(self.last_metadata)

    @torch.no_grad()
    def act(self, observation: dict[str, Any], info: dict[str, Any]) -> Any:
        depth = torch.as_tensor(observation["depth"], dtype=torch.float32, device=self.device).unsqueeze(0)
        state = torch.as_tensor(observation["state"], dtype=torch.float32, device=self.device).unsqueeze(0)
        outputs = self.model(depth, state, hidden_state=self.hidden_state)
        self.hidden_state = outputs["hidden_state"]
        action = outputs["action"][0, 0].detach().cpu().numpy()
        self.last_metadata = {
            "source": "student_model",
        }
        return action


def smoke_test_student_policy(
    model: StudentPolicy | StudentRuntimePolicy,
    *,
    state_dim: int = 141,
    depth_shape: tuple[int, int, int] = (128, 128, 1),
    batch_size: int = 2,
    time_steps: int = 2,
) -> dict[str, Any]:
    """Run a small shape smoke test against a student model."""

    model.eval()
    with torch.no_grad():
        depth = torch.zeros((batch_size, time_steps) + depth_shape, dtype=torch.float32)
        state = torch.zeros((batch_size, time_steps, state_dim), dtype=torch.float32)
        outputs = model(depth, state) if isinstance(model, StudentPolicy) else model(depth[:, 0], state[:, 0])
        if isinstance(model, StudentPolicy):
            action = outputs["action"]
            latent = outputs["latent"]
        else:
            action = outputs
            latent = None
    summary = {
        "input_depth_shape": list(depth.shape),
        "input_state_shape": list(state.shape),
        "action_shape": list(action.shape),
    }
    if latent is not None:
        summary["latent_shape"] = list(latent.shape)
    return summary


def build_student_runtime(model: StudentPolicy) -> StudentRuntimePolicy:
    """Create a deployable TorchScript-friendly runtime copy of a student."""

    runtime = StudentRuntimePolicy(model.config)
    runtime.load_state_dict(model.state_dict(), strict=False)
    runtime.eval()
    return runtime


def export_student_torchscript(
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    map_location: str = "cpu",
) -> dict[str, Any]:
    """Load a student checkpoint and export a TorchScript runtime artifact."""

    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    model, metadata = load_checkpoint(checkpoint_path, map_location=map_location)
    runtime = build_student_runtime(model)
    scripted = torch.jit.script(runtime)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(output_path))
    return {
        "checkpoint_path": str(checkpoint_path),
        "runtime_path": str(output_path),
        "state_dim": model.config.state_dim,
        "action_dim": model.config.action_dim,
        "metadata_keys": sorted(metadata.keys()),
    }


def save_model_config(path: str | Path, config: StudentModelConfig) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2, sort_keys=True)


def load_model_config(path: str | Path) -> StudentModelConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return StudentModelConfig(**payload)


def save_checkpoint(
    path: str | Path,
    model: StudentPolicy,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "model_config": asdict(model.config),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if extra is not None:
        payload["extra"] = extra
    torch.save(payload, Path(path))


def load_checkpoint(path: str | Path, *, map_location: str = "cpu") -> tuple[StudentPolicy, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=map_location)
    config = StudentModelConfig(**payload["model_config"])
    model = StudentPolicy(config)
    model.load_state_dict(payload["model_state_dict"])
    model.to(map_location)
    metadata = {
        key: value
        for key, value in payload.items()
        if key not in {"model_state_dict", "model_config"}
    }
    return model, metadata
