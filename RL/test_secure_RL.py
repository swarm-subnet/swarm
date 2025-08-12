#!/usr/bin/env python3
"""
test_secure_RL.py

Offline validator for a single Stable‑Baselines3 PPO policy, but with a
**secure, weights‑only** loader that reconstructs the network architecture
from the checkpoint tensors (no unpickling of metadata, no code execution).

It mirrors your existing `test_RL.py`:
- same CLI
- same task generation
- same head‑less evaluation flow
- same printed metrics
- optional GUI replay via the validator helper

Key differences:
- We never call `PPO.load()`.
- We open the `.zip`, read only raw tensors (with torch.load(..., weights_only=True)),
  infer `net_arch` (shared/pi/vf) and whether SDE was used, then build a clean PPO
  and load the policy weights strictly.

Usage
-----
$ python test_secure_RL.py --model model/ppo_policy.zip [--seed 42] [--gui]
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch as th
from stable_baselines3 import PPO

# your project imports
from swarm.constants import SIM_DT, HORIZON_SEC
from swarm.validator.task_gen import random_task
from swarm.validator.forward import _run_episode  # public helper
from swarm.utils.gui_isolation import run_isolated  # noqa: F401 (kept for parity with test_RL.py)
from swarm.utils.env_factory import make_env

# Try gymnasium first, then gym (names differ across installs)
try:
    from gymnasium import spaces as _spaces  # type: ignore
except Exception:  # pragma: no cover
    try:
        from gym import spaces as _spaces  # type: ignore
    except Exception:  # pragma: no cover
        _spaces = None  # we won't rely on this if import fails


# ──────────────────────────────────────────────────────────────────────
# Utilities for safe, weights‑only loading
# ──────────────────────────────────────────────────────────────────────

_ACT_MAP = {
    # normalized keys (lowercased, stripped)
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


def _torch_supports_weights_only() -> bool:
    # Torch >= 2.0 has weights_only parameter on torch.load
    try:
        from inspect import signature

        return "weights_only" in signature(th.load).parameters
    except Exception:
        return False


def _zip_read_bytes(zf: zipfile.ZipFile, name: str) -> bytes:
    with zf.open(name, "r") as fh:
        return fh.read()


def _find_policy_tensor_file(zf: zipfile.ZipFile) -> str:
    """
    Locate the tensor file holding the policy weights inside an SB3 .zip.

    Preference order (common SB3 layouts):
      1) 'policy.pth'
      2) any '*.pth' that looks like policy (but not optimizer)
      3) 'parameters' / 'parameters.pth' (older bundles with a dict containing 'policy')
    """
    names = zf.namelist()

    if "policy.pth" in names:
        return "policy.pth"

    candidates = [
        n
        for n in names
        if n.lower().endswith((".pth", ".pt"))
        and ("policy" in n.lower())
        and ("optimizer" not in n.lower())
    ]
    if candidates:
        return sorted(candidates, key=len)[0]

    for fallback in ("parameters", "parameters.pth", "params.pth", "params"):
        if fallback in names:
            return fallback

    raise FileNotFoundError(
        f"Could not locate policy weights inside archive. Contents:\n{names}"
    )


def _extract_policy_state_dict(raw_obj: Any) -> Mapping[str, th.Tensor]:
    """
    Normalize whatever we loaded into a plain state_dict mapping for the policy.
    Handles:
      - direct state_dict (OrderedDict[str, Tensor])
      - dict with 'policy' key (older SB3 'parameters' bundle)
    """
    if isinstance(raw_obj, Mapping):
        # direct state_dict?
        if raw_obj and all(isinstance(k, str) for k in raw_obj.keys()):
            # Heuristic: policy dict typically has Linear weights/bias for mlp_extractor/action_net/value_net
            sample_keys = list(raw_obj.keys())[:6]
            if any(
                ("mlp_extractor." in k)
                or ("action_net." in k)
                or (k == "log_std")
                or (k.startswith("value_net."))
                for k in sample_keys
            ):
                return raw_obj  # type: ignore[return-value]
        # parameters bundle with nested 'policy'
        if "policy" in raw_obj and isinstance(raw_obj["policy"], Mapping):
            return raw_obj["policy"]  # type: ignore[return-value]

    raise RuntimeError(
        "Unable to interpret loaded object as a policy state_dict. "
        "This archive may not be an SB3 PPO policy or uses an unsupported layout."
    )


def _collect_linear_dims(sd: Mapping[str, th.Tensor], prefix: str) -> List[int]:
    """
    Collect the out_features of all Linear layers under a given sequential prefix.

    Example keys:
      'mlp_extractor.policy_net.0.weight' -> (idx=0, out=256, in=115)
      'mlp_extractor.policy_net.2.weight' -> (idx=2, out=256, in=256)
    """
    pat = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.weight$")
    layers: List[Tuple[int, int, int]] = []
    for k, v in sd.items():
        m = pat.match(k)
        if m and isinstance(v, th.Tensor) and v.ndim == 2:
            idx = int(m.group(1))
            out_f, in_f = int(v.shape[0]), int(v.shape[1])
            layers.append((idx, out_f, in_f))
    layers.sort(key=lambda t: t[0])
    return [out_f for _, out_f, _ in layers]


def _infer_net_arch_and_sde(sd: Mapping[str, th.Tensor]) -> Tuple[Any, bool]:
    """
    Infer SB3 `net_arch` and whether SDE was used, from a policy state_dict.

    Returns:
      net_arch: either dict(pi=[...], vf=[...]) or [shared..., dict(pi=[...], vf=[...])]
      use_sde: bool (True if log_std is a matrix of [features_dim, action_dim])
    """
    shared = _collect_linear_dims(sd, "mlp_extractor.shared_net")
    pi = _collect_linear_dims(sd, "mlp_extractor.policy_net")
    vf = _collect_linear_dims(sd, "mlp_extractor.value_net")

    # Construct net_arch
    if shared:
        net_arch: Any = shared + [dict(pi=pi, vf=vf)]
    else:
        net_arch = dict(pi=pi, vf=vf)

    # Detect SDE from shape of log_std
    use_sde = False
    log_std = sd.get("log_std", None)
    if isinstance(log_std, th.Tensor):
        # Non-SDE: shape [action_dim]; SDE: shape [features_dim, action_dim]
        use_sde = (log_std.ndim == 2)

    return net_arch, use_sde


def _maybe_read_policy_kwargs_from_json(zf: zipfile.ZipFile) -> Dict[str, Any]:
    """
    Best‑effort, SAFE attempt to read policy kwargs from any JSON files in the archive.
    We only accept simple, JSON‑serializable fields. Classes (like activation_fn)
    must be given as simple names (e.g., "ReLU") or "torch.nn.ReLU".
    """
    out: Dict[str, Any] = {}
    for name in zf.namelist():
        if not name.lower().endswith(".json"):
            continue
        try:
            data = json.loads(_zip_read_bytes(zf, name).decode("utf-8"))
        except Exception:
            continue

        # Policy kwargs sometimes appear nested
        pk = None
        if isinstance(data, dict) and "policy_kwargs" in data and isinstance(data["policy_kwargs"], dict):
            pk = data["policy_kwargs"]
        elif isinstance(data, dict) and "policy" in data and isinstance(data["policy"], dict) and "policy_kwargs" in data["policy"]:
            if isinstance(data["policy"]["policy_kwargs"], dict):
                pk = data["policy"]["policy_kwargs"]

        if pk is None:
            continue

        # sanitize known fields
        sanitized: Dict[str, Any] = {}
        # net_arch may be dict or list
        if "net_arch" in pk and isinstance(pk["net_arch"], (list, dict)):
            sanitized["net_arch"] = pk["net_arch"]

        # activation_fn as string like "ReLU", "torch.nn.ReLU", "nn.ReLU", etc.
        if "activation_fn" in pk and isinstance(pk["activation_fn"], (str,)):
            raw = pk["activation_fn"].strip()
            key = raw.split(".")[-1].lower()  # e.g., "relu"
            if key in _ACT_MAP:
                sanitized["activation_fn"] = _ACT_MAP[key]

        # ortho_init (bool) is safe to copy (it won't matter after loading weights)
        if "ortho_init" in pk and isinstance(pk["ortho_init"], bool):
            sanitized["ortho_init"] = pk["ortho_init"]

        # only keep if anything was found
        if sanitized:
            out.update(sanitized)

    return out


def _choose_policy_class_from_env(env) -> str:
    if _spaces is None:
        # Fallback: assume vector obs
        return "MlpPolicy"
    try:
        obs_space = env.observation_space
        if isinstance(obs_space, _spaces.Dict):
            return "MultiInputPolicy"
        return "MlpPolicy"
    except Exception:
        return "MlpPolicy"


def secure_ppo_load_weights_only(
    checkpoint_zip: str | Path,
    *,
    env,  # trusted env providing observation_space/action_space
    device: str = "cpu",
    # You can override these if you *know* them, otherwise we infer/guess.
    policy: Optional[str] = None,
    policy_kwargs: Optional[Dict[str, Any]] = None,
    **algo_kwargs,
) -> PPO:
    """
    Build a clean PPO and load ONLY the policy weights from an SB3 .zip.

    - Never unpickles metadata (we ignore/avoid any *.pkl in the archive).
    - Uses torch.load(..., weights_only=True) to avoid code execution.
    - Reconstructs net_arch (shared/pi/vf) and SDE usage from the tensors.
    - Does not restore optimizer state or env objects.

    Requirements:
    - PyTorch with `weights_only=True` support (>= 2.0). Newer is safer.
    - The checkpoint must be SB3‑style (policy state_dict present).
    """
    if not _torch_supports_weights_only():
        raise RuntimeError(
            "Your PyTorch build does not support safe loading with `weights_only=True`.\n"
            "Please upgrade PyTorch (recommended: latest stable)."
        )

    checkpoint_zip = str(checkpoint_zip)

    with zipfile.ZipFile(checkpoint_zip, "r") as zf:
        # We *do not* read any pickled metadata. Warn if present.
        if any(n.lower().endswith(".pkl") for n in zf.namelist()):
            print(
                "[secure-load] Archive contains pickled metadata (*.pkl). "
                "We will IGNORE it and load tensors only.",
                file=sys.stderr,
            )

        # Try to grab hints from any JSON present (safe)
        json_policy_kwargs = _maybe_read_policy_kwargs_from_json(zf)

        # Load tensors safely
        weight_member = _find_policy_tensor_file(zf)
        raw_bytes = _zip_read_bytes(zf, weight_member)

    # Safe tensor-only load
    obj = th.load(io.BytesIO(raw_bytes), map_location=device, weights_only=True)
    state_dict = _extract_policy_state_dict(obj)

    # Infer net_arch + SDE from tensors (authoritative — shapes don't lie)
    inferred_net_arch, inferred_use_sde = _infer_net_arch_and_sde(state_dict)

    # Merge policy_kwargs (JSON hints -> caller overrides -> inferred net_arch)
    pk: Dict[str, Any] = {}
    # Use JSON activation_fn if present; otherwise default to ReLU (common in PPO setups).
    if "activation_fn" in (policy_kwargs or {}):
        pk["activation_fn"] = (policy_kwargs or {})["activation_fn"]
    elif "activation_fn" in json_policy_kwargs:
        pk["activation_fn"] = json_policy_kwargs["activation_fn"]
    else:
        pk["activation_fn"] = th.nn.ReLU  # fallback; if you truly need Tanh, change here.

    # Prefer our inferred net_arch; if caller passed one explicitly, keep it.
    if "net_arch" in (policy_kwargs or {}):
        pk["net_arch"] = (policy_kwargs or {})["net_arch"]
    else:
        pk["net_arch"] = inferred_net_arch

    # Optionally carry ortho_init from JSON/caller (not necessary for correctness)
    if "ortho_init" in (policy_kwargs or {}):
        pk["ortho_init"] = (policy_kwargs or {})["ortho_init"]
    elif "ortho_init" in json_policy_kwargs:
        pk["ortho_init"] = json_policy_kwargs["ortho_init"]

    # Decide policy class if not provided
    policy_class = policy or _choose_policy_class_from_env(env)

    # Build a fresh, trusted PPO using *your* env/hparams
    # Important: set use_sde to what we inferred from tensors.
    model = PPO(policy_class, env, device=device, policy_kwargs=pk, use_sde=inferred_use_sde, **algo_kwargs)

    # Strictly load weights (fail clearly on mismatched architecture)
    incompat = model.policy.load_state_dict(state_dict, strict=True)
    if getattr(incompat, "missing_keys", []) or getattr(incompat, "unexpected_keys", []):
        raise RuntimeError(
            f"State dict mismatch.\n"
            f"Missing: {getattr(incompat, 'missing_keys', [])}\n"
            f"Unexpected: {getattr(incompat, 'unexpected_keys', [])}\n"
            "This usually means the checkpoint was trained with a different policy/net_arch.\n"
            "If you believe the dims match but activation differs, set `activation_fn` explicitly."
        )

    # Ensure SDE buffers (if any) are on the right device
    if getattr(model, "use_sde", False):
        model.policy.reset_noise()

    return model


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Local Swarm policy validator (secure weights‑only loader)")
parser.add_argument(
    "--model",
    type=Path,
    default=Path("model/ppo_policy.zip"),
    help="Path to the Stable‑Baselines3 .zip file (policy weights inside).",
)
parser.add_argument(
    "--seed",
    type=int,
    default=1,
    help="Random seed for MapTask generation",
)
parser.add_argument(
    "--gui",
    action="store_true",
    default=False,
    help="After evaluation, replay the episode in a PyBullet GUI",
)
args = parser.parse_args()

if not args.model.exists():
    raise FileNotFoundError(f"Policy file not found: {args.model}")


# ──────────────────────────────────────────────────────────────────────
# Deterministic MapTask
# ──────────────────────────────────────────────────────────────────────
task = random_task(sim_dt=SIM_DT, horizon=HORIZON_SEC, seed=args.seed)

# ──────────────────────────────────────────────────────────────────────
# Head‑less evaluation (bit‑for‑bit like the validator)
# We build a small, head‑less env ONLY to initialize policy shapes.
# This env is not reused for rollout; _run_episode() will make its own.
# ──────────────────────────────────────────────────────────────────────
print(f"Evaluating policy at {args.model} …")

_init_env = make_env(task, gui=False)
try:
    model = secure_ppo_load_weights_only(
        args.model,
        env=_init_env,
        device="cpu",        # CPU to match on‑chain validator
        # policy=None,       # auto‑choose MlpPolicy vs MultiInputPolicy from env
        # policy_kwargs=None # we infer net_arch; activation falls back to ReLU
    )
finally:
    try:
        _init_env.close()  # type: ignore[attr-defined]
    except Exception:
        pass

result = _run_episode(task=task, uid=0, model=model, gui=args.gui)

print("----------------------------------------------------")
print(f"Success: {result.success}")
print(f"Time    : {result.time_sec:.2f} s")
print(f"Energy  : {result.energy:.1f} J")
print(f"Score   : {result.score:.3f}")
print("----------------------------------------------------")
