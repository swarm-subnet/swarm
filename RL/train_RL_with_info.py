#!/usr/bin/env python3

import sys
import importlib

import argparse
import os
import json
import zipfile
from pathlib import Path
from typing import Any, Dict

import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from swarm.utils.env_factory import make_env
from swarm.validator.task_gen import random_task
from swarm.constants import SIM_DT, HORIZON_SEC, SAFE_META_FILENAME


def information_save(model: PPO, save_stem: str) -> None:
    """
    Append a small, SAFE JSON file to the SB3 checkpoint zip with just the
    non-executable metadata the secure loader needs (activation + net_arch + SDE).

    Args:
        model: Trained PPO model
        save_stem: Path used with `model.save(...)`. Can be with or without ".zip".
                   We will write the JSON into that zip archive.
    """
    # Resolve the actual .zip path that SB3 produced
    zip_path = Path(save_stem)
    if zip_path.suffix != ".zip":
        zip_path = zip_path.with_suffix(".zip")

    # --- Gather minimal metadata (no pickle, no code) ---
    # Activation function name
    act_attr = getattr(model.policy, "activation_fn", th.nn.ReLU)
    if isinstance(act_attr, type):
        act_name = act_attr.__name__          # e.g., "ReLU"
    else:
        act_name = act_attr.__class__.__name__  # instance -> "ReLU", "Tanh", ...

    # net_arch (read back if available; only for reference)
    def _infer_net_arch_from_policy() -> Any:
        me = getattr(model.policy, "mlp_extractor", None)
        if me is not None and hasattr(me, "net_arch"):
            return me.net_arch

        # Fallback: reconstruct sizes from Linear layers
        def _layers(seq) -> list[int]:
            out = []
            for m in getattr(seq, "_modules", {}).values():
                if isinstance(m, th.nn.Linear):
                    out.append(int(m.out_features))
            return out

        shared = _layers(getattr(me, "shared_net", th.nn.Sequential()))
        pi = _layers(getattr(me, "policy_net", th.nn.Sequential()))
        vf = _layers(getattr(me, "value_net", th.nn.Sequential()))
        return (shared + [dict(pi=pi, vf=vf)]) if shared else dict(pi=pi, vf=vf)

    net_arch = _infer_net_arch_from_policy()
    use_sde = bool(getattr(model, "use_sde", False))

    meta: Dict[str, Any] = {
        "format": "sb3-safe-meta@1",
        "algo": "PPO",
        "activation_fn": act_name,   # e.g., "ReLU", "Tanh"
        "net_arch": net_arch,        # informational; secure loader still infers from weights
        "use_sde": use_sde,
    }

    # --- Write/replace JSON inside the zip ---
    with zipfile.ZipFile(zip_path, mode="a", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(SAFE_META_FILENAME, json.dumps(meta, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1000)
    args = parser.parse_args()

    task = random_task(sim_dt=SIM_DT, horizon=HORIZON_SEC, seed=1)

    def _build_env():
        return make_env(task, gui=False)

    env = DummyVecEnv([_build_env])
    env = VecTransposeImage(env)
    model = PPO("MultiInputPolicy", env, verbose=1)

    model.learn(args.timesteps)

    os.makedirs("model", exist_ok=True)

    save_stem = "model/ppo_policy"
    model.save(save_stem)

    information_save(model, save_stem)

    env.close()


if __name__ == "__main__":
    main()
