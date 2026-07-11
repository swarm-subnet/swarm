"""Export a trained SB3 policy as a model-graph artifact source."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import torch as th

from swarm.model_graph.constants import (
    EXECUTION_PROFILE_ID,
    FAMILY_GRAPH_CONTRACTS,
    MODEL_GRAPH_VERSION,
    ONNX_IR_VERSION,
    ONNX_OPSET_VERSION,
    RUNNER_ABI,
    family_has_swarm_axis,
)

POLICY_DEPTH_SIZE = 64


class _GraphPolicy(th.nn.Module):
    """The deterministic PPO actor over the family's raw observations."""

    def __init__(
        self, policy, depth_stride: int, low: np.ndarray, high: np.ndarray, swarm_axis: bool
    ):
        super().__init__()
        self.mlp_extractor = policy.mlp_extractor
        self.action_net = policy.action_net
        self.stride = depth_stride
        self.swarm_axis = swarm_axis
        self.act_dim = int(low.shape[-1])
        self.register_buffer("low", th.as_tensor(low, dtype=th.float32))
        self.register_buffer("high", th.as_tensor(high, dtype=th.float32))

    def forward(self, depth, state):
        small = depth[..., :: self.stride, :: self.stride, :]
        if self.swarm_axis:
            features = th.cat([th.flatten(small, start_dim=1), state], dim=1)
        else:
            features = th.cat([small.reshape(1, -1), state.reshape(1, -1)], dim=1)
        latent = self.mlp_extractor.forward_actor(features)
        action = self.action_net(latent)
        if not self.swarm_axis:
            action = action.reshape(self.act_dim)
        return th.min(th.max(action, self.low), self.high)


def _example_inputs(family_id: str) -> tuple[th.Tensor, th.Tensor]:
    contract = FAMILY_GRAPH_CONTRACTS[family_id]
    depth_shape = list(contract["observations"]["depth"])
    state_shape = list(contract["observations"]["state"])
    if family_has_swarm_axis(family_id):
        depth_shape[0] = 3
        state_shape[0] = 3
    return (
        th.rand(*depth_shape, dtype=th.float32),
        th.rand(*state_shape, dtype=th.float32),
    )


def export_policy(model, family_id: str, out_dir: Path) -> Path:
    """Write manifest.json + models/policy.onnx for the trained PPO model.

    Returns the source directory, ready for ``swarm model package``.
    """
    contract = FAMILY_GRAPH_CONTRACTS[family_id]
    depth_res = int(contract["observations"]["depth"][-3])
    graph = _GraphPolicy(
        model.policy,
        depth_stride=max(1, depth_res // POLICY_DEPTH_SIZE),
        low=np.asarray(contract["action_lower"], dtype=np.float32),
        high=np.asarray(contract["action_upper"], dtype=np.float32),
        swarm_axis=family_has_swarm_axis(family_id),
    ).eval()

    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = models_dir / "policy.onnx"

    dynamic_axes = None
    if family_has_swarm_axis(family_id):
        dynamic_axes = {"depth": {0: "N"}, "state": {0: "N"}, "action": {0: "N"}}
    th.onnx.export(
        graph,
        _example_inputs(family_id),
        str(onnx_path),
        input_names=["depth", "state"],
        output_names=["action"],
        opset_version=ONNX_OPSET_VERSION,
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )

    exported = onnx.load(str(onnx_path))
    exported.ir_version = ONNX_IR_VERSION
    onnx.save(exported, str(onnx_path))

    _check_parity(model, graph, family_id)

    manifest = {
        "model_graph_version": MODEL_GRAPH_VERSION,
        "family_id": family_id,
        "execution_profile": EXECUTION_PROFILE_ID,
        "runner_abi": RUNNER_ABI,
        "models": [
            {
                "id": "policy",
                "file": "models/policy.onnx",
                "sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
            }
        ],
        "nodes": [
            {
                "id": "policy",
                "model": "policy",
                "inputs": {"depth": "obs.depth", "state": "obs.state"},
            }
        ],
        "action": "policy.action",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out_dir


def _check_parity(model, graph: _GraphPolicy, family_id: str) -> None:
    """The exported graph must act exactly like model.predict(deterministic=True)."""
    depth, state = _example_inputs(family_id)
    with th.no_grad():
        exported_action = graph(depth, state).numpy()

    small = depth[..., :: graph.stride, :: graph.stride, :].numpy()
    sb3_action, _ = model.predict(
        {"depth": small, "state": state.numpy()}, deterministic=True
    )
    sb3_action = np.clip(
        sb3_action, graph.low.numpy(), graph.high.numpy()
    ).reshape(exported_action.shape)
    if not np.allclose(exported_action, sb3_action, atol=1e-5):
        raise RuntimeError("exported graph diverges from the trained policy")
