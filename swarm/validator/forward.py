# ---------------------------------------------------------------
#  Swarm validator – Policy API v2  (simplified, no subprocess)
# ---------------------------------------------------------------
from __future__ import annotations

import asyncio
import base64
import gc
import os
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import bittensor as bt
import numpy as np
from stable_baselines3 import PPO                       # SB‑3 loader
from zipfile import ZipFile, BadZipFile

from swarm.core.drone import track_drone
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
    BURN_EMISSIONS,
)

BURN_FRACTION  = 0.90            # 90 % burn (weight for UID 0)
KEEP_FRACTION  = 1.0 - BURN_FRACTION
UID_ZERO       = 0

# ──────────────────────────────────────────────────────────────────────────
# 0.  Global hardening parameters
# ──────────────────────────────────────────────────────────────────────────
MODEL_DIR         = Path("miner_models")           # all zips stored here
MAX_MODEL_BYTES   = 50 * 1024 * 1024               # 50 MiB compressed cap

# ──────────────────────────────────────────────────────────────────────────
# 1.  Helpers – secure ZIP inspection
# ──────────────────────────────────────────────────────────────────────────
def _zip_is_safe(path: Path, *, max_uncompressed: int) -> bool:
    """
    Reject dangerous ZIP files *without* extracting them.

    • Total uncompressed size must not exceed `max_uncompressed`.
    • No absolute paths or “..” traversal sequences.
    """
    try:
        with ZipFile(path) as zf:
            total_uncompressed = 0
            for info in zf.infolist():
                # (1) forbid absolute paths or traversal
                name = info.filename
                if name.startswith(("/", "\\")) or ".." in Path(name).parts:
                    bt.logging.error(f"ZIP path traversal attempt: {name}")
                    return False

                # (2) track size
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


# ──────────────────────────────────────────────────────────────────────────
# 2.  Episode roll‑out (unchanged)
# ──────────────────────────────────────────────────────────────────────────
def _run_episode(
    task: "MapTask",
    uid: int,
    model: PPO,
    *,
    gui: bool = False,
) -> ValidationResult:
    """
    Executes one closed‑loop flight using *model* as the policy.
    Returns a fully‑populated ValidationResult.
    """
    class _Pilot:
        def __init__(self, m): self.m = m
        def reset(self, task):  pass
        def act(self, obs, t):
            act, _ = self.m.predict(obs, deterministic=True)
            return act.squeeze()

    pilot = _Pilot(model)
    env   = make_env(task, gui=gui)

    # initial observation
    try:
        obs = env._computeObs()                # type: ignore[attr-defined]
    except AttributeError:
        obs = env.get_observation()            # type: ignore[attr-defined]

    if isinstance(obs, dict):
        obs = obs[next(iter(obs))]

    t_sim      = 0.0
    energy     = 0.0
    success    = False
    step_count = 0
    frames_per_cam = max(1, int(round(1.0 / (SIM_DT * 60.0))))   # ≈60 Hz

    while t_sim < task.horizon:
        rpm  = pilot.act(obs, t_sim)
        obs, _r, terminated, truncated, info = env.step(rpm[None, :])

        t_sim   += SIM_DT
        energy  += np.abs(rpm).sum() * SIM_DT

        if gui and step_count % frames_per_cam == 0:
            try:
                cli_id = getattr(env, "CLIENT", getattr(env, "_cli", 0))
                track_drone(cli=cli_id, drone_id=env.DRONE_IDS[0])
            except Exception:
                pass
        if gui:
            time.sleep(SIM_DT)

        if terminated or truncated:
            success = info.get("success", False)
            break

        step_count += 1

    if not gui:
        env.close()

    # ── reward ──────────────────────────────────────────────────────────
    score = flight_reward(
        success = success,
        t       = t_sim,
        e       = energy,
        horizon = task.horizon,
    )

    return ValidationResult(uid, success, t_sim, energy, score)


# ──────────────────────────────────────────────────────────────────────────
# 3.  Secure, cached model download (unchanged)
# ──────────────────────────────────────────────────────────────────────────
async def _download_model(self, axon, ref: PolicyRef, dest: Path) -> None:
    """
    Ask the miner for the full ZIP in one message (base‑64 encoded)
    and save it to *dest*.  All integrity and size checks still apply.
    """
    tmp = dest.with_suffix(".part")
    tmp.unlink(missing_ok=True)

    try:
        # 1 – request the blob
        responses = await self.dendrite(
            axons       = [axon],
            synapse     = PolicySynapse.request_blob(),
            deserialize = True,
            timeout     = QUERY_TIMEOUT,
        )

        if not responses:
            bt.logging.warning(f"Miner {axon.hotkey} sent no reply to blob request")
            return

        syn = responses[0]

        # 2 – make sure we actually got chunk data
        if not syn.chunk or "data" not in syn.chunk:
            bt.logging.warning(f"Miner {axon.hotkey} reply lacked chunk data")
            return

        # 3 – decode base‑64 → raw bytes
        try:
            raw_bytes = base64.b64decode(syn.chunk["data"])
        except Exception as e:
            bt.logging.warning(f"Base‑64 decode failed from miner {axon.hotkey}: {e}")
            return

        if len(raw_bytes) > MAX_MODEL_BYTES:
            bt.logging.error(
                f"Miner {axon.hotkey} sent oversized blob "
                f"({len(raw_bytes)/1e6:.1f} MB > 50 MB)"
            )
            return

        # 4 – write to temp file
        with tmp.open("wb") as fh:
            fh.write(raw_bytes)

        # 5 – SHA‑256 integrity
        if sha256sum(tmp) != ref.sha256:
            bt.logging.error(f"SHA mismatch for miner blob {axon.hotkey}.")
            tmp.unlink(missing_ok=True)
            return

        # 6 – ZIP sanity check
        if not _zip_is_safe(tmp, max_uncompressed=MAX_MODEL_BYTES):
            bt.logging.error(f"Unsafe ZIP from miner {axon.hotkey}.")
            tmp.unlink(missing_ok=True)
            return

        # 7 – promote
        tmp.replace(dest)
        bt.logging.info(f"Stored model for {axon.hotkey} at {dest}.")

    except Exception as e:
        bt.logging.warning(f"Download error ({axon.hotkey}): {e}")
        tmp.unlink(missing_ok=True)


async def _ensure_models(self, uids: List[int]) -> Dict[int, Path]:
    """
    For every UID return the local Path to its latest .zip.
    Downloads if the cached SHA differs from the miner's PolicyRef.
    """
    MODEL_DIR.mkdir(exist_ok=True)
    paths: Dict[int, Path] = {}

    for uid in uids:
        axon = self.metagraph.axons[uid]

        # 1 – ask for current PolicyRef
        try:
            responses: list[PolicySynapse] = await self.dendrite(
                axons=[axon],
                synapse=PolicySynapse.request_ref(),
                deserialize=True,
                timeout=QUERY_TIMEOUT,
            )

            if not responses:
                bt.logging.warning(f"Miner {uid} returned no response.")
                continue

            syn = responses[0]
            if not syn.ref:
                bt.logging.warning(f"Miner {uid} returned no PolicyRef.")
                continue

            ref = PolicyRef(**syn.ref)
        except Exception as e:
            bt.logging.warning(f"Handshake with miner {uid} failed: {e}")
            continue

        # 2 – compare with cache
        model_fp = MODEL_DIR / f"UID_{uid}.zip"
        up_to_date = model_fp.exists() and sha256sum(model_fp) == ref.sha256
        if up_to_date:
            if (
                model_fp.stat().st_size <= MAX_MODEL_BYTES
                and _zip_is_safe(model_fp, max_uncompressed=MAX_MODEL_BYTES)
            ):
                paths[uid] = model_fp
                continue
            else:
                bt.logging.warning(f"Cached model for {uid} violates limits; redownloading.")
                model_fp.unlink(missing_ok=True)

        # 3 – request payload
        await _download_model(self, axon, ref, model_fp)
        if (
            model_fp.exists()
            and sha256sum(model_fp) == ref.sha256
            and model_fp.stat().st_size <= MAX_MODEL_BYTES
            and _zip_is_safe(model_fp, max_uncompressed=MAX_MODEL_BYTES)
        ):
            paths[uid] = model_fp
        else:
            bt.logging.warning(f"Failed to obtain valid model for miner {uid}.")
            model_fp.unlink(missing_ok=True)

    return paths


# ──────────────────────────────────────────────────────────────────────────
# 4.  Single‑process evaluation helper
# ──────────────────────────────────────────────────────────────────────────
def _evaluate_uid(task: MapTask, uid: int, model_fp: Path) -> ValidationResult:
    """
    Load the miner's model and run one episode.
    Returns a ValidationResult; on any exception the score is 0.
    """
    try:
        bt.logging.debug(f"Loading model for UID {uid} from {model_fp}")
        custom_objects = {
            "lr_schedule": 2.5e-4,   # replace lambdas if present
            "clip_range": 0.2,
        }
        model = PPO.load(model_fp, device="cpu", custom_objects=custom_objects)
        res   = _run_episode(task, uid, model)
        return res
    except Exception as e:
        tb = traceback.format_exc()
        bt.logging.warning(f"[eval] Miner {uid} failed: {e}\n{tb}")
        return ValidationResult(uid, False, 0.0, 0.0, 0.0)
    finally:
        try:
            del model
        except Exception:
            pass
        gc.collect()


# ──────────────────────────────────────────────────────────────────────────
# 5.  Weight boosting (unchanged)
# ──────────────────────────────────────────────────────────────────────────
def _boost_scores(raw: np.ndarray, *, beta: float = 5.0) -> np.ndarray:
    """
    Exponential boost driven by absolute gap to the best score,
    scaled by batch standard deviation.
    """
    if raw.size == 0:
        return raw

    s_max = float(raw.max())
    sigma = float(raw.std())
    if sigma < 1e-9:                          # all miners identical
        weights = (raw == s_max).astype(np.float32)
    else:
        weights = np.exp(beta * (raw - s_max) / sigma)
        weights /= weights.max()              # normalise so best → 1

    return weights.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────
# 6.  Public coroutine – called by neurons/validator.py
# ──────────────────────────────────────────────────────────────────────────
async def forward(self) -> None:
    """Full validator tick with boosted weighting + optional burn."""
    try:
        self.forward_count = getattr(self, "forward_count", 0) + 1
        bt.logging.info(f"[Forward #{self.forward_count}] start")

        # ------------------------------------------------------------------
        # 1. build a secret task
        task = random_task(sim_dt=SIM_DT, horizon=HORIZON_SEC)

        # ------------------------------------------------------------------
        # 2. sample miners & secure their models
        uids = get_random_uids(self, k=SAMPLE_K)
        bt.logging.info(f"Sampled miners: {uids}")

        model_paths = await _ensure_models(self, uids)
        bt.logging.info(f"Verified models: {list(model_paths)}")

        # ------------------------------------------------------------------
        # 3. evaluation *in‑process*
        results = [_evaluate_uid(task, uid, fp) for uid, fp in model_paths.items()]
        if not results:
            bt.logging.warning("No valid results this round.")
            await asyncio.sleep(FORWARD_SLEEP_SEC)
            return

        raw_scores = np.asarray([r.score for r in results], dtype=np.float32)
        uids_np    = np.asarray([r.uid   for r in results], dtype=np.int64)

        # ------------------------------------------------------------------
        # 4. adaptive boost
        boosted = _boost_scores(raw_scores, beta=5.0)

        # ------------------------------------------------------------------
        # 5. optional burn logic
        if BURN_EMISSIONS:
            if UID_ZERO in uids_np:
                mask      = uids_np != UID_ZERO
                boosted   = boosted[mask]
                uids_np   = uids_np[mask]

            total_boost = boosted.sum()
            if total_boost > 0.0:
                boosted *= KEEP_FRACTION / total_boost
            else:
                boosted = np.zeros_like(boosted)

            uids_np   = np.concatenate(([UID_ZERO], uids_np))
            boosted   = np.concatenate(([BURN_FRACTION], boosted))

            bt.logging.info(
                f"Burn enabled → {BURN_FRACTION:.0%} to UID 0, "
                f"{KEEP_FRACTION:.0%} distributed over {len(boosted)-1} miners."
            )
        else:
            bt.logging.info("Burn disabled – using boosted weights as is.")

        # ------------------------------------------------------------------
        # 6. push weights on‑chain
        self.update_scores(boosted, uids_np)

    except Exception as e:
        bt.logging.error(f"Validator forward error: {e}")

    # ----------------------------------------------------------------------
    # 7. pace the main loop
    await asyncio.sleep(FORWARD_SLEEP_SEC)
