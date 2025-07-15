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
            bt.logging.warning(f"Failed to obtain valid model for miner {uid}.")
            model_fp.unlink(missing_ok=True)

    return paths


async def _download_model(self, axon, ref: PolicyRef, dest: Path) -> None:
    """
    Stream a .zip from *axon* into *dest* atomically.
    Enforces both compressed & uncompressed ≤ 50 MiB.
    """
    tmp = dest.with_suffix(".part")
    tmp.unlink(missing_ok=True)
    received = 0

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

            chunk = msg.chunk["data"]
            received += len(chunk)

            # --- hard cap on compressed size -----------------------
            if received > MAX_MODEL_BYTES:
                bt.logging.error(
                    f"Miner {axon.hotkey} sent oversized blob "
                    f"({received/1e6:.1f} MB > 50 MB). Aborting."
                )
                tmp.unlink(missing_ok=True)
                return

            with tmp.open("ab") as fh:
                fh.write(chunk)

        # --- verify compressed size --------------------------------
        if tmp.stat().st_size > MAX_MODEL_BYTES:
            bt.logging.error(
                f"Compressed ZIP from {axon.hotkey} exceeds 50 MB "
                f"({tmp.stat().st_size/1e6:.1f} MB)."
            )
            tmp.unlink(missing_ok=True)
            return

        # --- SHA‑256 integrity check -------------------------------
        if sha256sum(tmp) != ref.sha256:
            bt.logging.error(f"SHA mismatch for miner blob {axon.hotkey}.")
            tmp.unlink(missing_ok=True)
            return

        # --- dry inspection of ZIP (no extraction) -----------------
        if not _zip_is_safe(tmp, max_uncompressed=MAX_MODEL_BYTES):
            bt.logging.error(f"Unsafe ZIP from miner {axon.hotkey}.")
            tmp.unlink(missing_ok=True)
            return

        # passed all checks → promote
        tmp.replace(dest)
        bt.logging.info(f"Stored model for {axon.hotkey} at {dest}.")

    except Exception as e:
        bt.logging.warning(f"Streaming error ({axon.hotkey}): {e}")
        tmp.unlink(missing_ok=True)

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
def _boost_scores(raw: np.ndarray, *, beta: float = 5.0) -> np.ndarray:
    """
    Exponential boost driven by *absolute* gap to the best score, *scaled
    by the batch standard deviation*.

        w_i = exp( beta · (s_i − s_max) / σ )

    • `β` controls sharpness (β ≈ 5 works well).  
    • The best miner always gets weight 1.  
    • When scores are tightly clustered (σ small) even a 0.002 gap is
      magnified; when they are spread out the curve relaxes automatically.
    """
    if raw.size == 0:
        return raw

    s_max = float(raw.max())
    sigma = float(raw.std())
    if sigma < 1e-9:                          # all miners identical
        weights = (raw == s_max).astype(np.float32)  # leader gets 1, rest 0
    else:
        weights = np.exp(beta * (raw - s_max) / sigma)
        weights /= weights.max()              # normalise so best → 1

    return weights.astype(np.float32)


async def forward(self) -> None:
    """Full validator tick with boosted weighting."""
    try:
        self.forward_count = getattr(self, "forward_count", 0) + 1
        bt.logging.info(f"[Forward #{self.forward_count}] start")

        # ----- secret task -----------------------------------------------------------------
        task = random_task(sim_dt=SIM_DT, horizon=HORIZON_SEC)

        # ----- sample a subset of miners ---------------------------------------------------
        uids = get_random_uids(self, k=SAMPLE_K)
        bt.logging.info(f"Sampled miners: {uids}")

        # ----- handshake & secure download -------------------------------------------------
        model_paths = await _ensure_models(self, uids)
        bt.logging.info(f"Verified models: {list(model_paths)}")

        # ----- sandboxed evaluation --------------------------------------------------------
        results = [_evaluate_uid(task, uid, fp) for uid, fp in model_paths.items()]
        if not results:
            bt.logging.warning("No valid results this round.")
            await asyncio.sleep(FORWARD_SLEEP_SEC)
            return

        raw_scores = np.asarray([r.score for r in results], dtype=np.float32)
        uids_np    = np.asarray([r.uid   for r in results], dtype=np.int64)

        # ----- apply adaptive boost --------------------------------------------------------
        boosted = _boost_scores(raw_scores, beta=5.0)
        bt.logging.info(
            f"Boost: raw max {raw_scores.max():.3f}, "
            f"raw median {np.median(raw_scores):.3f} → "
            f"median weight {np.median(boosted):.3e}"
        )

        # ----- push weights on‑chain -------------------------------------------------------
        self.update_scores(boosted, uids_np)

        # ─────── Silent wandb weight logging ───────
        if hasattr(self, 'wandb_helper') and self.wandb_helper:
            try:
                self.wandb_helper.log_weight_update(
                    uids=uids_np.tolist(),
                    scores=boosted.tolist()
                )
                bt.logging.debug(f"Weight update logged to wandb for {len(uids_np)} miners")
                self.wandb_helper.finish()             # close current run
                time.sleep(10)  # brief pause to ensure run closure
                self.wandb_helper._init_wandb()        # open a new run
            except Exception as e:
                bt.logging.debug(f"Wandb weight logging failed: {e}")

    except Exception as e:
        bt.logging.error(f"Validator forward error: {e}")

    await asyncio.sleep(FORWARD_SLEEP_SEC)