# ---------------------------------------------------------------
#  Swarm validator – Policy API v2   (hardened, 50 MiB limits)
# ---------------------------------------------------------------
from __future__ import annotations

import asyncio
import gc
import traceback
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import bittensor as bt
import numpy as np
from stable_baselines3 import PPO                       # SB‑3 loader
from zipfile import ZipFile, BadZipFile
from swarm.core.drone       import track_drone


from swarm.protocol import MapTask, PolicySynapse, PolicyRef, ValidationResult
from swarm.utils.uids import get_random_uids
from swarm.utils.hash import sha256sum
from swarm.utils.env_factory import make_env


from .task_gen import random_task
from .reward   import flight_reward

from swarm.constants import (
    SIM_DT,
    HORIZON_SEC,
    SAMPLE_K,
    QUERY_TIMEOUT,
    FORWARD_SLEEP_SEC,
    GOAL_TOL,
)

# ----------------------------------------------------------------
# 0.  Helpers – run one episode with a Policy (.zip)
# ----------------------------------------------------------------
def _run_episode(
    task: "MapTask",
    uid: int,
    model: PPO,
    *,
    gui: bool = False,
) -> ValidationResult:
    """
    Executes one closed‑loop flight using *model* as the policy.

    Parameters
    ----------
    gui
        If *True* the episode is rendered in a PyBullet GUI.  Rendering runs
        in real‑time; a small delay is inserted after every step.
    """
    # ─── tiny adapter so we can keep the validator helper unchanged ────
    class _Pilot:
        def __init__(self, m): self.m = m
        def reset(self, task):  pass
        def act(self, obs, t):
            act, _ = self.m.predict(obs, deterministic=True)
            return act.squeeze()

    pilot = _Pilot(model)

    # Environment is *already reset* by make_env()
    env = make_env(task, gui=gui)

    # Get initial observation without another reset
    try:
        obs = env._computeObs()                # type: ignore[attr-defined]
    except AttributeError:
        # Fallback for envs that expose a public getter
        obs = env.get_observation()            # type: ignore[attr-defined]

    # For safety make sure obs is an array (unwrap dict‑of‑dicts if needed)
    if isinstance(obs, dict):
        obs = obs[next(iter(obs))]

    # Pos0 can be taken directly from the task description
    pos0 = np.asarray(task.start, dtype=float)

    # Distances for reward
    d_start  = float(np.linalg.norm(pos0 - task.goal))
    last_pos = pos0.copy()

    # Book‑keeping ------------------------------------------------------
    t_sim        = 0.0
    energy       = 0.0
    success      = False
    step_counter = 0
    frames_per_cam = max(1, int(round(1.0 / (SIM_DT * 60.0))))   # ≈60 Hz

    # Episode loop ------------------------------------------------------
    while t_sim < task.horizon:
        rpm  = pilot.act(obs, t_sim)
        obs, _reward, terminated, truncated, info = env.step(rpm[None, :])

        t_sim   += SIM_DT
        energy  += np.abs(rpm).sum() * SIM_DT
        last_pos = obs[:3] if obs.ndim == 1 else obs[0, :3]

        # Camera follow every ~60 GUI frames
        if gui and step_counter % frames_per_cam == 0:
            try:
                cli_id = getattr(env, "CLIENT", getattr(env, "_cli", 0))
                track_drone(cli=cli_id, drone_id=env.DRONE_IDS[0])
            except Exception:
                # Stay robust if the env internals change
                pass

        if gui:
            time.sleep(SIM_DT)                 # real‑time pacing

        if terminated or truncated:
            success = info.get("success", False)
            break

        step_counter += 1

    # Close Bullet session only when no GUI is open
    if not gui:
        env.close()

    # Reward / score ----------------------------------------------------
    d_final = float(np.linalg.norm(last_pos - task.goal))
    score = flight_reward(
        success   = success,
        t_alive   = t_sim,
        d_start   = d_start,
        d_final   = d_final,
        horizon   = task.horizon,
        goal_tol  = GOAL_TOL,
        t_to_goal = t_sim if success else None,
        e_used    = energy if success else None,
    )
    return ValidationResult(uid, success, t_sim, energy, score)


# ----------------------------------------------------------------
# 1.  Model handshake, caching & safety limits
# ----------------------------------------------------------------
MODEL_DIR        = Path("miner_models")      # all zips stored here
CHUNK_SIZE       = 2 << 20                   # 2 MiB – matches iter_chunks default
MAX_MODEL_BYTES  = 50 * 1024 * 1024          # 50 MiB hard cap (compressed + uncompressed)

def _zip_is_safe(path: Path, *, max_uncompressed: int) -> bool:
    """
    Cheap static inspection to reject dangerous ZIPs *without* extraction.

    • Total uncompressed size must not exceed `max_uncompressed`.
    • No absolute paths and no '..' path traversal sequences.
    """
    try:
        with ZipFile(path) as zf:
            total_uncompressed = 0
            for info in zf.infolist():
                # 1. forbid absolute paths or traversal
                name = info.filename
                if name.startswith("/") or name.startswith("\\") or ".." in Path(name).parts:
                    bt.logging.error(f"ZIP path traversal attempt: {name}")
                    return False
                # 2. track size
                total_uncompressed += info.file_size
                if total_uncompressed > max_uncompressed:
                    bt.logging.error(
                        f"ZIP too large when decompressed "
                        f"({total_uncompressed/1e6:.1f} MB > {max_uncompressed/1e6:.1f} MB)"
                    )
                    return False
            return True
    except BadZipFile:
        bt.logging.error("Corrupted ZIP archive.")
        return False
    except Exception as e:
        bt.logging.error(f"ZIP inspection error: {e}")
        return False


async def _ensure_models(self, uids: List[int]) -> Dict[int, Path]:
    """
    For every UID return the local Path to its latest .zip.
    Downloads if the cached SHA differs from the miner's PolicyRef.
    """
    MODEL_DIR.mkdir(exist_ok=True)
    paths: Dict[int, Path] = {}

    for uid in uids:
        axon = self.metagraph.axons[uid]

        # 1 – ask for current PolicyRef ------------------------------
        try:
            syn: PolicySynapse = await self.dendrite(
                axons=[axon],
                synapse=PolicySynapse.request_ref(),
                deserialize=True,
                timeout=QUERY_TIMEOUT,
            )
            if not syn.ref:
                bt.logging.warning(f"Miner {uid} returned no PolicyRef.")
                continue
            ref = PolicyRef(**syn.ref)
        except Exception as e:
            bt.logging.warning(f"Handshake with miner {uid} failed: {e}")
            continue

        # 2 – compare with cache ------------------------------------
        model_fp = MODEL_DIR / f"UID_{uid}.zip"
        up_to_date = model_fp.exists() and sha256sum(model_fp) == ref.sha256
        if up_to_date:
            # confirm cached file is still within limits (allows lowering MAX_MODEL_BYTES later)
            if model_fp.stat().st_size <= MAX_MODEL_BYTES and _zip_is_safe(model_fp, max_uncompressed=MAX_MODEL_BYTES):
                paths[uid] = model_fp
                continue
            else:
                bt.logging.warning(f"Cached model for {uid} violates new limits; redownloading.")
                model_fp.unlink(missing_ok=True)

        # 3 – request payload ---------------------------------------
        await _download_model(self, axon, ref, model_fp)
        if (
            model_fp.exists()
            and sha256sum(model_fp) == ref.sha256
            and model_fp.stat().st_size <= MAX_MODEL_BYTES
            and _zip_is_safe(model_fp, max_uncompressed=MAX_MODEL_BYTES)
        ):
            paths[uid] = model_fp
        else:
            bt.logging.warning("No valid FlightPlans returned by miners.")

        # -------- 4) (optional) persist FlightPlans ----------
        save_flightplans(task, results, plans)

        # -------- 5) weight update ---------------------------
        _apply_weight_update(self, results)

    except Exception as err:
        bt.logging.error(f"Validator forward error: {err}")

    # -------- 6) sleep --------------------------------------
    await asyncio.sleep(FORWARD_SLEEP_SEC)
