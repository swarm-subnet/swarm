# ---------------------------------------------------------------
# Swarm validator – Policy API v2
# ---------------------------------------------------------------
from __future__ import annotations

import asyncio
import importlib
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import bittensor as bt
import numpy as np

from swarm.protocol import (
    MapTask,
    PolicySynapse,
    PolicyRef,
    ValidationResult,
)
from swarm.utils.uids import get_random_uids
from swarm.utils.hash import sha256sum
from swarm.validator.loader import temp_venv
from swarm.utils.env_factory import make_env

from .task_gen import random_task
from .reward import flight_reward

from swarm.constants import (
    SIM_DT,
    HORIZON_SEC,
    SAMPLE_K,
    QUERY_TIMEOUT,
    FORWARD_SLEEP_SEC,
)


# ----------------------------------------------------------------
# 0.  Helpers – run one episode with a Pilot
# ----------------------------------------------------------------
def _run_episode(task: MapTask, uid: int, pilot: object) -> ValidationResult:
    env = make_env(task, gui=False, raw_rpm=True, randomise=True)
    obs = env.reset()
    pilot.reset(task)

    t_sim, energy, success = 0.0, 0.0, False
    while t_sim < task.horizon:
        rpm = pilot.act(obs, t_sim)
        obs, _, done, info = env.step(rpm[None, :])
        t_sim += SIM_DT
        energy += np.abs(rpm).sum() * SIM_DT
        if done:
            success = info.get("success", False)
            break

    env.close()
    score = flight_reward(success, t_sim, energy, task.horizon)
    return ValidationResult(uid, success, t_sim, energy, score)


# ----------------------------------------------------------------
# 1.  Handshake & caching logic
# ----------------------------------------------------------------
async def _fetch_pilots(self, uids: List[int]) -> Dict[int, object]:
    """
    End‑to‑end handshake for a **set of miners**:

        • ask for PolicyRef
        • download wheel if missing
        • import Pilot class in temp venv
        • return {uid: pilot‑instance}
    """
    # one‑time init of blob cache dir
    if not hasattr(self, "blob_cache"):
        self.blob_cache = Path(
            getattr(self.config, "blob_cache", ".swarm_cache")
        )
        self.blob_cache.mkdir(exist_ok=True)

    pilots: Dict[int, object] = {}

    for uid in uids:
        axon = self.metagraph.axons[uid]

        # -------- 1. ask for PolicyRef -------------------------
        try:
            reply: PolicySynapse = await self.dendrite(
                axons=[axon],
                synapse=PolicySynapse.query_update(),
                deserialize=True,
                timeout=QUERY_TIMEOUT,
            )
            ref_dict = reply.ref
            if not ref_dict and reply.no_update:
                # We already cached that miner’s latest wheel – find it
                # via metagraph.stored_sha[uid] or fallback to “sha256”
                sha = getattr(self.metagraph, "stored_sha", {}).get(uid)
                wheel_fp = self.blob_cache / sha if sha else None
                if not wheel_fp or not wheel_fp.exists():
                    bt.logging.warning(
                        f"No cached wheel for miner {uid} although they "
                        "sent no_update=True – skipping."
                    )
                    continue
                ref = PolicyRef(
                    sha256=sha,
                    entrypoint="",  # unused when we already have wheel
                    framework="",
                    size_bytes=wheel_fp.stat().st_size,
                )
            elif ref_dict:
                ref = PolicyRef(**ref_dict)
            else:
                bt.logging.warning(f"Miner {uid} gave neither ref nor no_update flag.")
                continue

        except Exception as e:
            bt.logging.warning(f"Could not obtain PolicyRef from miner {uid}: {e}")
            continue

        wheel_fp: Path = self.blob_cache / ref.sha256

        # -------- 2. download if needed ------------------------
        if not wheel_fp.exists():
            await _download_wheel(self, axon, ref, wheel_fp)

        if not wheel_fp.exists():
            continue  # download failed

        # -------- 3. import Pilot class ------------------------
        try:
            with temp_venv(wheel_fp):
                mod_name, cls_name = ref.entrypoint.split(":")
                cls = getattr(importlib.import_module(mod_name), cls_name)
                pilots[uid] = cls()
        except Exception as e:
            bt.logging.warning(f"Import error for miner {uid}: {type(e).__name__} – {e}")

    return pilots


async def _download_wheel(self, axon, ref: PolicyRef, wheel_fp: Path):
    """Request blob streaming and write it to *wheel_fp* atomically."""
    tmp_fp = wheel_fp.with_suffix(".part")
    try:
        # tell miner we need it
        await self.dendrite(
            axons=[axon],
            synapse=PolicySynapse.request_blob(),
            timeout=QUERY_TIMEOUT,
        )

        async for msg in self.dendrite.stream(axon):
            if msg and msg.chunk:
                with tmp_fp.open("ab") as out:
                    out.write(msg.chunk["data"])
            else:
                break

        if sha256sum(tmp_fp) != ref.sha256:
            tmp_fp.unlink(missing_ok=True)
            bt.logging.error("SHA‑256 mismatch – discarded corrupt wheel.")
            return

        tmp_fp.rename(wheel_fp)
        bt.logging.info(
            f"Cached wheel {wheel_fp.name} ({wheel_fp.stat().st_size/1e6:.1f} MB)."
        )
    except Exception as e:
        tmp_fp.unlink(missing_ok=True)
        bt.logging.warning(f"Streaming wheel from miner failed: {e}")


# ----------------------------------------------------------------
# 2.  Weight update helper
# ----------------------------------------------------------------
def _apply_weight_update(self, results: List[ValidationResult]):
    if not results:
        bt.logging.warning("No validation results – skipping weight update.")
        return

    uids_np = np.asarray([r.uid for r in results], dtype=np.int64)
    scores_np = np.asarray([r.score for r in results], dtype=np.float32)
    self.update_scores(scores_np, uids_np)
    bt.logging.info(f"Pushed scores for {len(results)} miners.")


# ----------------------------------------------------------------
# 3.  Public coroutine (called from neurons/validator.py)
# ----------------------------------------------------------------
async def forward(self) -> None:
    """
    One full validator iteration:
        1. sample miners
        2. fetch/instantiate their pilots
        3. run closed‑loop simulations
        4. update scores on‑chain
    """
    try:
        self.forward_count = getattr(self, "forward_count", 0) + 1
        bt.logging.info(f"[Forward #{self.forward_count}] start")

        # ---------- build task ------------------------------
        task = random_task(sim_dt=SIM_DT, horizon=HORIZON_SEC)

        # ---------- choose miners ---------------------------
        uids = get_random_uids(self, k=SAMPLE_K)
        bt.logging.info(f"Sampled miners: {uids}")

        # ---------- get pilots ------------------------------
        pilots = await _fetch_pilots(self, uids)
        bt.logging.info(f"Loaded {len(pilots)} pilots.")

        # ---------- simulate / score ------------------------
        results: List[ValidationResult] = []
        for uid, pilot in pilots.items():
            try:
                results.append(_run_episode(task, uid, pilot))
            except Exception as e:  # pilot crashed mid‑flight
                bt.logging.warning(f"Episode failed for miner {uid}: {e}")
                traceback.print_exc()

        if results:
            best = max(r.score for r in results)
            avg = np.mean([r.score for r in results])
            bt.logging.info(f"Scores: best={best:.3f}  avg={avg:.3f}")
        else:
            bt.logging.warning("No successful episodes this round.")

        # ---------- push weights ----------------------------
        _apply_weight_update(self, results)

    except Exception as err:
        bt.logging.error(f"Validator forward error: {err}")

    await asyncio.sleep(FORWARD_SLEEP_SEC)
