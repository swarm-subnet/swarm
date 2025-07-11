# ---------------------------------------------------------------
# Swarm validator – Policy API v2  (memory‑efficient version)
# ---------------------------------------------------------------
from __future__ import annotations

import asyncio
import gc
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import bittensor as bt
import numpy as np
from stable_baselines3 import PPO                        # SB‑3 loader

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
def _run_episode(task: MapTask, uid: int, model: PPO) -> ValidationResult:
    """
    Executes one closed‑loop flight using *model* as the policy.

    Fixes:
    • correctly unpack `env.reset()` which returns (obs, info)
    • handles both 1‑D (single‑drone) and 2‑D (N×state) observation shapes
    """
    # --- light wrapper so the rest of the code can stay the same --------
    class _Pilot:
        def __init__(self, m): self.m = m
        def reset(self, task):  pass
        def act(self, obs, t):
            act, _ = self.m.predict(obs, deterministic=True)
            return act.squeeze()

    pilot = _Pilot(model)
    env   = make_env(task, gui=False) 
    obs, _info = env.reset()                 # ← unpack the info dict

    # Make sure we can index the position regardless of shape
    if isinstance(obs, tuple):               # safety for unexpected returns
        obs = obs[0]
    pos0 = obs[:3] if obs.ndim == 1 else obs[0, :3]

    d_start  = float(np.linalg.norm(pos0 - task.goal))
    t_sim    = 0.0
    energy   = 0.0
    last_pos = pos0
    success  = False

    while t_sim < task.horizon:
        rpm  = pilot.act(obs, t_sim)
        obs, _reward, terminated, truncated, info = env.step(rpm[None, :])
        t_sim   += SIM_DT
        energy  += np.abs(rpm).sum() * SIM_DT
        last_pos = obs[:3] if obs.ndim == 1 else obs[0, :3]

        if terminated or truncated:
            success = info.get("success", False)
            break

    env.close()
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
# 1.  Model handshake & caching
# ----------------------------------------------------------------
MODEL_DIR = Path("miner_models")          # all zips stored here
CHUNK_SIZE = 2 << 20                      # 2 MiB – matches iter_chunks default

async def _ensure_models(self, uids: List[int]) -> Dict[int, Path]:
    """
    Make sure we have the latest .zip for every *uid*.
    Returns {uid: Path-to-zip}.  Does **not** load them into RAM.
    """
    MODEL_DIR.mkdir(exist_ok=True)
    paths: Dict[int, Path] = {}

    for uid in uids:
        axon = self.metagraph.axons[uid]

        # 1 – ask for PolicyRef ---------------------------------------
        try:
            syn: PolicySynapse = await self.dendrite(
                axons=[axon],
                synapse=PolicySynapse.query_update(),
                deserialize=True,
                timeout=QUERY_TIMEOUT,
            )
            if syn.no_update:
                # unchanged – keep current file name
                model_fp = MODEL_DIR / f"UID_{uid}.zip"
                if not model_fp.exists():
                    bt.logging.warning(f"Miner {uid} claims no_update but file missing.")
                    continue
                paths[uid] = model_fp
                continue

            if not syn.ref:
                bt.logging.warning(f"Miner {uid} returned neither ref nor no_update.")
                continue

            ref = PolicyRef(**syn.ref)

        except Exception as e:
            bt.logging.warning(f"Handshake failed with miner {uid}: {e}")
            continue

        # 2 – download if sha changed -------------------------------
        model_fp = MODEL_DIR / f"UID_{uid}.zip"
        if model_fp.exists() and sha256sum(model_fp) == ref.sha256:
            paths[uid] = model_fp
            continue   # already up‑to‑date

        # need fresh blob -------------------------------------------
        await _download_model(self, axon, ref, model_fp)
        if model_fp.exists() and sha256sum(model_fp) == ref.sha256:
            paths[uid] = model_fp
        else:
            bt.logging.warning(f"Model download for miner {uid} failed sha check.")

    return paths

async def _download_model(self, axon, ref: PolicyRef, dest: Path) -> None:
    """Stream a .zip from *axon* into *dest* atomically."""
    tmp = dest.with_suffix(".part")
    tmp.unlink(missing_ok=True)
    try:
        # tell miner we need the blob
        await self.dendrite(
            axons=[axon],
            synapse=PolicySynapse.request_blob(),
            timeout=QUERY_TIMEOUT,
        )

        async for msg in self.dendrite.stream(axon):
            if not msg or not msg.chunk:
                break
            with tmp.open("ab") as fh:
                fh.write(msg.chunk["data"])

        if sha256sum(tmp) != ref.sha256:
            bt.logging.error(f"SHA mismatch for miner blob {axon.hotkey}.")
            tmp.unlink(missing_ok=True)
            return

        tmp.replace(dest)
        bt.logging.info(f"Stored model for {axon.hotkey} at {dest}.")

    except Exception as e:
        bt.logging.warning(f"Streaming error ({axon.hotkey}): {e}")
        tmp.unlink(missing_ok=True)

# ----------------------------------------------------------------
# 2.  Weight update helper
# ----------------------------------------------------------------
def _apply_weight_update(self, results: List[ValidationResult]):
    if not results:
        bt.logging.warning("No results → no weight update.")
        return
    uids   = np.asarray([r.uid   for r in results], dtype=np.int64)
    scores = np.asarray([r.score for r in results], dtype=np.float32)
    self.update_scores(scores, uids)
    bt.logging.info(f"Pushed scores for {len(results)} miners.")

# ----------------------------------------------------------------
# 3.  Evaluate one miner (load → run → free)
# ----------------------------------------------------------------
def _evaluate_uid(task: MapTask, uid: int, model_fp: Path) -> ValidationResult:
    try:
        model = PPO.load(model_fp, device="cpu")
        res   = _run_episode(task, uid, model)
    except Exception as e:
        bt.logging.warning(f"Miner {uid} evaluation error: {e}")
        res = ValidationResult(uid, False, 0.0, 0.0, 0.0)
    finally:
        # free RAM
        try:
            del model
        except UnboundLocalError:
            pass
        gc.collect()
    return res

# ----------------------------------------------------------------
# 4.  Public coroutine – called by neurons/validator.py
# ----------------------------------------------------------------
async def forward(self) -> None:
    """
    One full validator iteration:
        1. sample miners
        2. ensure we have their policy zips on disk
        3. evaluate miners one‑by‑one (low RAM)
        4. push scores on‑chain
    """
    try:
        self.forward_count = getattr(self, "forward_count", 0) + 1
        bt.logging.info(f"[Forward #{self.forward_count}] start")

        # -------- task --------------------------------------------
        task = random_task(sim_dt=SIM_DT, horizon=HORIZON_SEC)

        # -------- pick miners -------------------------------------
        uids = get_random_uids(self, k=SAMPLE_K)
        bt.logging.info(f"Sampled miners: {uids}")

        # -------- make sure models are cached ---------------------
        model_paths = await _ensure_models(self, uids)
        bt.logging.info(f"Have models for {len(model_paths)} miners.")

        # -------- evaluate sequentially ---------------------------
        results: List[ValidationResult] = []
        for uid, fp in model_paths.items():
            res = _evaluate_uid(task, uid, fp)
            results.append(res)

        if results:
            best = max(r.score for r in results)
            avg  = np.mean([r.score for r in results])
            bt.logging.info(f"Scores: best={best:.3f}  avg={avg:.3f}")
        else:
            bt.logging.warning("No miners yielded a valid result.")

        # -------- push weights ------------------------------------
        _apply_weight_update(self, results)

    except Exception as err:
        bt.logging.error(f"Validator forward error: {err}")

    await asyncio.sleep(FORWARD_SLEEP_SEC)
