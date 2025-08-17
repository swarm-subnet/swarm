#!/usr/bin/env python3
"""
test_secure_RL.py

Offline validator for a single Stable‑Baselines3 PPO policy using a
**secure, weights‑only** loader. It reads a small, safe JSON
(`safe_policy_meta.json`) embedded in the checkpoint `.zip` to get the
activation function, `net_arch`, and SDE flag, then loads only the
policy weights (no pickles, no code execution).

Assumptions (kept simple on purpose):
- The zip always contains:
    - `policy.pth` (policy weights)
    - `safe_policy_meta.json` with fields:
        { "activation_fn": "<Name>", "net_arch": {... or ...}, "use_sde": <bool> }
- Observation space is not image-based (MlpPolicy is fine). If your env
  uses Dict observations, the script will auto-select MultiInputPolicy.

Usage
-----
$ python -m RL.test_secure_RL --model model/ppo_policy.zip [--seed 42] [--gui]
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch as th
import numpy as np
from stable_baselines3 import PPO

# Project imports
from swarm.constants import SIM_DT, HORIZON_SEC
from swarm.validator.task_gen import random_task
from swarm.validator.forward import _run_episode  # public helper
from swarm.validator.reward import flight_reward
from swarm.utils.env_factory import make_env
from swarm.protocol import ValidationResult

# Try gymnasium first, then gym (for policy selection based on obs space)
try:
    from gymnasium import spaces as _spaces  # type: ignore
except Exception:
    try:
        from gym import spaces as _spaces  # type: ignore
    except Exception:
        _spaces = None


# Import from centralized constants
from swarm.constants import SAFE_META_FILENAME


# ──────────────────────────────────────────────────────────────────────
# Minimal helpers
# ──────────────────────────────────────────────────────────────────────

_ACT_MAP = {
    "relu": th.nn.ReLU,
    "tanh": th.nn.Tanh,
    "elu": th.nn.ELU,
    "leakyrelu": th.nn.LeakyReLU,
    "silu": th.nn.SiLU,
    "gelu": th.nn.GELU,
    "mish": th.nn.Mish,
    "selu": th.nn.SELU,
    "celu": th.nn.CELU,
}


def _parse_activation(name: str) -> type[th.nn.Module]:
    # Accept "Tanh", "torch.nn.Tanh", etc.; we just use the last component lowercased
    key = name.strip().split(".")[-1].lower()
    return _ACT_MAP.get(key, th.nn.ReLU)


def _torch_supports_weights_only() -> bool:
    try:
        from inspect import signature

        return "weights_only" in signature(th.load).parameters
    except Exception:
        return False


def _choose_policy_class_from_env(env) -> str:
    if _spaces is None:
        return "MlpPolicy"
    try:
        return "MultiInputPolicy" if isinstance(env.observation_space, _spaces.Dict) else "MlpPolicy"
    except Exception:
        return "MlpPolicy"


def _read_text(zf: zipfile.ZipFile, name: str) -> str:
    with zf.open(name, "r") as fh:
        return fh.read().decode("utf-8")


def _read_bytes(zf: zipfile.ZipFile, name: str) -> bytes:
    with zf.open(name, "r") as fh:
        return fh.read()


def _extract_policy_state_dict(raw_obj: Any) -> Mapping[str, th.Tensor]:
    # SB3 usually saves a plain state_dict for policy.pth; older formats may nest under "policy"
    if isinstance(raw_obj, Mapping):
        if "mlp_extractor.policy_net.0.weight" in raw_obj or "action_net.weight" in raw_obj or "log_std" in raw_obj:
            return raw_obj  # direct state_dict
        if "policy" in raw_obj and isinstance(raw_obj["policy"], Mapping):
            return raw_obj["policy"]
    raise RuntimeError("Could not interpret the loaded object as a policy state_dict.")


# ──────────────────────────────────────────────────────────────────────
# Secure, JSON‑guided weights loader
# ──────────────────────────────────────────────────────────────────────

def secure_ppo_load_weights_only(checkpoint_zip: str | Path, *, env, device: str = "cpu") -> PPO:
    """
    Build a clean PPO and load ONLY the policy weights from an SB3 .zip, using
    the activation/net_arch/use_sde from `safe_policy_meta.json`.
    """
    if not _torch_supports_weights_only():
        raise RuntimeError(
            "This PyTorch build does not support `weights_only=True` on torch.load. "
            "Please upgrade to a newer PyTorch."
        )

    checkpoint_zip = str(checkpoint_zip)
    with zipfile.ZipFile(checkpoint_zip, "r") as zf:
        names = set(zf.namelist())
        if SAFE_META_FILENAME not in names:
            raise FileNotFoundError(f"Missing {SAFE_META_FILENAME} inside the checkpoint zip.")
        if "policy.pth" not in names:
            raise FileNotFoundError("Missing policy.pth inside the checkpoint zip.")

        # Read safe JSON metadata
        meta = json.loads(_read_text(zf, SAFE_META_FILENAME))
        # Expect exactly the keys shown in the user's example
        act_name: str = meta["activation_fn"]
        net_arch: Any = meta["net_arch"]
        use_sde: bool = bool(meta["use_sde"])

        # Load tensors safely (no pickle execution)
        raw = _read_bytes(zf, "policy.pth")

    obj = th.load(io.BytesIO(raw), map_location=device, weights_only=True)
    state_dict = _extract_policy_state_dict(obj)

    # Build fresh PPO with the desired policy config
    policy_class = _choose_policy_class_from_env(env)
    policy_kwargs = dict(activation_fn=_parse_activation(act_name), net_arch=net_arch)
    model = PPO(policy_class, env, device=device, policy_kwargs=policy_kwargs, use_sde=use_sde)

    # Strictly load weights
    incompat = model.policy.load_state_dict(state_dict, strict=True)
    if getattr(incompat, "missing_keys", []) or getattr(incompat, "unexpected_keys", []):
        raise RuntimeError(
            f"State dict mismatch.\n"
            f"Missing: {getattr(incompat, 'missing_keys', [])}\n"
            f"Unexpected: {getattr(incompat, 'unexpected_keys', [])}"
        )

    if getattr(model, "use_sde", False):
        model.policy.reset_noise()

    return model


def _run_episode_speed_limit(task, uid, model, *, gui=False):
    class _Pilot:
        def __init__(self, m): self.m = m
        def reset(self, task): pass
        def act(self, obs, t):
            act, _ = self.m.predict(obs, deterministic=True)
            return act.squeeze()

    pilot = _Pilot(model)
    env = make_env(task, gui=gui)
    
    try:
        obs, _ = env.reset(seed=task.map_seed)
    except:
        obs = env._computeObs()
    
    if isinstance(obs, dict):
        obs = obs[next(iter(obs))]

    pos0 = np.asarray(task.start, dtype=float)
    t_sim = 0.0
    energy = 0.0
    success = False
    speeds = []
    
    lo, hi = env.action_space.low.flatten(), env.action_space.high.flatten()
    last_pos = pos0
    overspeed_streak = 0

    while t_sim < task.horizon:
        act = np.clip(np.asarray(pilot.act(obs, t_sim), dtype=np.float32).reshape(-1), lo, hi)
        
        if (hasattr(env, 'ACT_TYPE') and hasattr(env, 'SPEED_LIMIT') and overspeed_streak >= 2):
            from gym_pybullet_drones.utils.enums import ActionType
            if env.ACT_TYPE == ActionType.VEL and env.SPEED_LIMIT:
                n = max(np.linalg.norm(act[:3]), 1e-6)
                scale = min(1.0, 0.9 / n)
                act[:3] *= scale
                act = np.clip(act, lo, hi)
        
        prev = last_pos
        obs, _r, terminated, truncated, info = env.step(act[None, :])
        last_pos = env._getDroneStateVector(0)[0:3]
        
        speed = np.linalg.norm(last_pos - prev) / SIM_DT
        speeds.append(speed)
        
        if hasattr(env, 'SPEED_LIMIT') and env.SPEED_LIMIT:
            ratio = float(speed) / env.SPEED_LIMIT
            overspeed_streak = (overspeed_streak + 1) if ratio > 1.2 else 0

        t_sim += SIM_DT
        energy += np.abs(act).sum() * SIM_DT

        if terminated or truncated:
            success = info.get("success", False)
            break

    if not gui:
        env.close()

    score = flight_reward(success=success, t=t_sim, e=energy, horizon=task.horizon)
    avg_speed = np.mean(speeds) if speeds else 0.0
    result = ValidationResult(uid, success, t_sim, energy, score)
    return result, avg_speed


# ──────────────────────────────────────────────────────────────────────
# CLI + evaluation
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Local Swarm policy validator (secure, JSON‑guided weights loader)")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("model/ppo_policy.zip"),
        help="Path to the Stable‑Baselines3 .zip file.",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed for MapTask generation")
    parser.add_argument("--gui", action="store_true", default=False,
                        help="After evaluation, replay the episode in a PyBullet GUI")
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Policy file not found: {args.model}")

    # Deterministic task
    task = random_task(sim_dt=SIM_DT, horizon=HORIZON_SEC, seed=args.seed)

    print(f"Evaluating policy at {args.model} …")
    # Build a small env just to instantiate the policy with correct spaces
    _init_env = make_env(task, gui=False)
    try:
        model = secure_ppo_load_weights_only(args.model, env=_init_env, device="cpu")
    finally:
        try:
            _init_env.close()  # type: ignore[attr-defined]
        except Exception:
            pass

    # Run the episode exactly like validator
    result, avg_speed = _run_episode_speed_limit(task=task, uid=0, model=model, gui=args.gui)

    print("----------------------------------------------------")
    print(f"Success: {result.success}")
    print(f"Time    : {result.time_sec:.2f} s")
    print(f"Energy  : {result.energy:.1f} J")
    print(f"Score   : {result.score:.3f}")
    print(f"Avg Speed: {avg_speed:.3f} m/s")
    print("----------------------------------------------------")


if __name__ == "__main__":
    main()
