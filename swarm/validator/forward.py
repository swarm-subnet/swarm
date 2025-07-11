# ---------------------------------------------------------------
#  Swarm validator – Policy API v2.4
#  Hardened against large files, ZIP bombs, malicious payloads,
#  and runaway evaluation (CPU/RAM/time capped).
# ---------------------------------------------------------------
from __future__ import annotations

import asyncio, gc, multiprocessing as mp, os, signal, zipfile
from pathlib import Path
from typing import Dict, List

import bittensor as bt
import numpy as np
from stable_baselines3 import PPO

from swarm.protocol   import MapTask, PolicySynapse, PolicyRef, ValidationResult
from swarm.utils.uids import get_random_uids
from swarm.utils.hash import sha256sum
from swarm.utils.env_factory import make_env

from .task_gen import random_task
from .reward   import flight_reward
from swarm.constants import (
    SIM_DT, HORIZON_SEC, SAMPLE_K, QUERY_TIMEOUT, FORWARD_SLEEP_SEC, GOAL_TOL
)

# ----------------------------------------------------------------------
# 0 · Security parameters (edit as needed)
# ----------------------------------------------------------------------
MAX_MODEL_BYTES       = 50 * 1024 * 1024          # 50 MiB on disk
MAX_ZIP_CONTENT_BYTES = 100 * 1024 * 1024         # 100 MiB uncompressed
EVAL_TIMEOUT_SEC      = HORIZON_SEC + 20          # safety margin
MEM_LIMIT_MB          = 1024                      # per‑process address‑space
MODEL_DIR             = Path("miner_models")
CHUNK_SIZE            = 2 << 20                   # 2 MiB

# ----------------------------------------------------------------------
# 1 · Episode runner  (pure, no side‑effects, can run in subprocess)
# ----------------------------------------------------------------------
def _run_episode(task: MapTask, uid: int, model_path: str) -> ValidationResult:
    """Sub‑process entry: loads model & runs one secret episode."""
    # --- resource limits (Linux only) ----------------------------------
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (MEM_LIMIT_MB << 20, MEM_LIMIT_MB << 20))
        resource.setrlimit(resource.RLIMIT_CPU, (EVAL_TIMEOUT_SEC, EVAL_TIMEOUT_SEC + 2))
    except Exception:
        pass  # best‑effort; ignored on non‑POSIX or if perms missing

    # --- load model ----------------------------------------------------
    try:
        model = PPO.load(model_path, device="cpu")
    except Exception as e:
        return ValidationResult(uid, False, 0.0, 0.0, 0.0)

    # --- lightweight pilot wrapper ------------------------------------
    class _Pilot:
        def __init__(self, m): self.m = m
        def reset(self, task):  pass
        def act(self, obs, t):
            act, _ = self.m.predict(obs, deterministic=True)
            return act.squeeze()

    pilot = _Pilot(model)
    env   = make_env(task, gui=False)
    obs, _ = env.reset()

    pos0 = obs[:3] if obs.ndim == 1 else obs[0, :3]
    d_start = float(np.linalg.norm(pos0 - task.goal))
    t_sim = energy = 0.0
    success = False
    last_pos = pos0

    while t_sim < task.horizon:
        rpm = pilot.act(obs, t_sim)
        obs, _, terminated, truncated, info = env.step(rpm[None, :])
        t_sim   += SIM_DT
        energy  += np.abs(rpm).sum() * SIM_DT
        last_pos = obs[:3] if obs.ndim == 1 else obs[0, :3]
        if terminated or truncated:
            success = info.get("success", False)
            break

    env.close()
    d_final = float(np.linalg.norm(last_pos - task.goal))

    score = flight_reward(
        success, t_sim, d_start, d_final, task.horizon, GOAL_TOL,
        t_to_goal=t_sim if success else None,
        e_used=energy if success else None,
    )
    return ValidationResult(uid, success, t_sim, energy, score)

# ----------------------------------------------------------------------
# 2 · ZIP sanity check (prevents ZIP bombs & corruption)
# ----------------------------------------------------------------------
def _is_zip_safe(fp: Path) -> bool:
    try:
        with zipfile.ZipFile(fp) as zf:
            total = sum(i.file_size for i in zf.infolist())
            return total <= MAX_ZIP_CONTENT_BYTES
    except Exception:
        return False

# ----------------------------------------------------------------------
# 3 · Handshake + secure download
# ----------------------------------------------------------------------
async def _ensure_models(self, uids: List[int]) -> Dict[int, Path]:
    MODEL_DIR.mkdir(exist_ok=True)
    paths: Dict[int, Path] = {}

    for uid in uids:
        axon = self.metagraph.axons[uid]

        # --- manifest ---------------------------------------------------
        try:
            syn: PolicySynapse = await self.dendrite(
                axons=[axon],
                synapse=PolicySynapse.request_ref(),
                deserialize=True,
                timeout=QUERY_TIMEOUT,
            )
            if not syn.ref:
                bt.logging.warning(f"Miner {uid} sent no PolicyRef.")
                continue
            ref = PolicyRef(**syn.ref)
        except Exception as e:
            bt.logging.warning(f"Handshake with miner {uid} failed: {e}")
            continue

        if ref.size_bytes > MAX_MODEL_BYTES:
            bt.logging.warning(f"Miner {uid} model {ref.size_bytes/1e6:.1f} MB > limit.")
            continue

        model_fp = MODEL_DIR / f"UID_{uid}.zip"
        if model_fp.exists() and sha256sum(model_fp) == ref.sha256:
            paths[uid] = model_fp
            continue

        # --- download ---------------------------------------------------
        if not await _download_blob(self, axon, ref, model_fp):
            continue

        paths[uid] = model_fp
    return paths

async def _download_blob(self, axon, ref: PolicyRef, dest: Path) -> bool:
    tmp = dest.with_suffix(".part")
    tmp.unlink(missing_ok=True)
    received = 0

    try:
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
            if received > MAX_MODEL_BYTES:
                bt.logging.warning(f"Download from {axon.hotkey} exceeds limit.")
                tmp.unlink(missing_ok=True)
                return False
            with tmp.open("ab") as fh:
                fh.write(chunk)
    except Exception as e:
        bt.logging.warning(f"Stream error {axon.hotkey}: {e}")
        tmp.unlink(missing_ok=True)
        return False

    if sha256sum(tmp) != ref.sha256 or not _is_zip_safe(tmp):
        bt.logging.warning(f"Model from {axon.hotkey} failed integrity/safety checks.")
        tmp.unlink(missing_ok=True)
        return False

    tmp.replace(dest)
    bt.logging.info(f"Cached {dest.name} ({received/1e6:.1f} MB).")
    return True

# ----------------------------------------------------------------------
# 4 · Evaluate in sandboxed subprocess
# ----------------------------------------------------------------------
def _evaluate_uid(task: MapTask, uid: int, model_fp: Path) -> ValidationResult:
    queue: mp.Queue = mp.Queue(maxsize=1)
    proc = mp.Process(target=lambda q: q.put(_run_episode(task, uid, str(model_fp))),
                      args=(queue,), daemon=True)
    proc.start()
    proc.join(EVAL_TIMEOUT_SEC)
    if proc.is_alive():
        proc.kill()
        bt.logging.warning(f"Timeout evaluating miner {uid}.")
        return ValidationResult(uid, False, 0.0, 0.0, 0.0)
    try:
        res: ValidationResult = queue.get_nowait()
    except Exception:
        res = ValidationResult(uid, False, 0.0, 0.0, 0.0)
    finally:
        queue.close(); queue.join_thread()
        gc.collect()
    return res

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

    except Exception as e:
        bt.logging.error(f"Validator forward error: {e}")

    await asyncio.sleep(FORWARD_SLEEP_SEC)