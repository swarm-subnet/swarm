# ---------------------------------------------------------------
# Swarm – Validator neuron (Phase‑3 rewrite)
# ---------------------------------------------------------------
from __future__ import annotations

import asyncio
import importlib
import sys
import traceback
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import bittensor as bt
import numpy as np

from swarm.protocol import (
    MapTask,
    PolicyRef,
    PolicySynapse,
    ValidationResult,
)
from swarm.utils.uids import get_random_uids
from swarm.utils.hash import sha256sum
from swarm.utils.chunking import iter_chunks
from swarm.validator.loader import temp_venv
from swarm.utils.env_factory import make_env

# Local helpers (same modules you already have)
from .task_gen import random_task
from .reward    import flight_reward

from swarm.constants import (
    SIM_DT,
    HORIZON_SEC,
    SAMPLE_K,
    QUERY_TIMEOUT,
    FORWARD_SLEEP_SEC,
)


class Validator(bt.neurons.core.validator.ValidatorNeuron):
    """
    Swarm validator neuron implementing the v2 PolicyRef/Chunk handshake.

    The class is self‑contained: spin it up with `bt.validator`.
    """
    version: str = "1"

    # -----------------------------------------------------------
    # Construction / helpers
    # -----------------------------------------------------------
    def __init__(self, config: bt.Config | None = None):
        super().__init__(config)

        # Where we cache wheels streamed from miners
        self.blob_cache: Path = Path(self.config.get("blob_cache", ".swarm_cache"))
        self.blob_cache.mkdir(exist_ok=True)

        bt.logging.info(f"Blob cache directory: {self.blob_cache.resolve()}")

    # -----------------------------------------------------------
    # Miner‑handshake utilities
    # -----------------------------------------------------------
    async def _get_pilots(
        self, uids: List[int], task: MapTask
    ) -> Dict[int, object]:
        """
        For every UID in *uids*:

            1. Ask for a PolicyRef.
            2. If we do not have the wheel, request streaming chunks.
            3. Load the Pilot class inside a tiny venv and instantiate it.

        Returns
        -------
        Dict[uid, Pilot‑instance]
        """
        pilots: Dict[int, object] = {}

        for uid in uids:
            axon = self.metagraph.axons[uid]

            # --- 1) Request PolicyRef -----------------------------------
            try:
                ref_syn = await self.dendrite(
                    axons=[axon],
                    synapse=PolicySynapse(task=asdict(task)),
                    deserialize=True,
                    timeout=QUERY_TIMEOUT,
                )
                # dendrite returns a list when axons=list; unwrap
                ref_syn = ref_syn[0] if isinstance(ref_syn, (list, tuple)) else ref_syn
                if not ref_syn.ref:
                    bt.logging.warning(f"Miner {uid} returned no PolicyRef.")
                    continue
                ref = PolicyRef(**ref_syn.ref)
            except Exception as e:
                bt.logging.warning(f"Handshake with miner {uid} failed: {e}")
                continue

            wheel_fp: Path = self.blob_cache / ref.sha256

            # --- 2) Download wheel if missing --------------------------
            if not wheel_fp.exists():
                bt.logging.info(f"Fetching wheel from miner {uid} ({ref.sha256[:8]}…).")

                # Tell miner we need the blob
                await self.dendrite(
                    axons=[axon],
                    synapse=PolicySynapse(need_blob=True),
                    timeout=QUERY_TIMEOUT,
                )

                # Stream chunks until miner signals end (empty chunk/ref/result msg)
                async for msg in self.dendrite.stream(axon):
                    if msg and msg.chunk:
                        with wheel_fp.open("ab") as out:
                            out.write(msg.chunk["data"])
                    else:  # end of stream
                        break

                # Integrity check
                if sha256sum(wheel_fp) != ref.sha256:
                    wheel_fp.unlink(missing_ok=True)
                    bt.logging.error(f"SHA‑256 mismatch for blob from miner {uid}; skipped.")
                    continue
                bt.logging.info(f"Wheel {wheel_fp.name} cached ({wheel_fp.stat().st_size/1e6:.1f} MB).")

            # --- 3) Import & instantiate Pilot --------------------------
            try:
                with temp_venv(wheel_fp):
                    mod_name, cls_name = ref.entrypoint.split(":")
                    cls = getattr(importlib.import_module(mod_name), cls_name)
                    pilots[uid] = cls()
            except Exception as e:
                bt.logging.warning(f"Failed to load Pilot from miner {uid}: {type(e).__name__} — {e}")

        return pilots

    # -----------------------------------------------------------
    # Scoring helpers
    # -----------------------------------------------------------
    @staticmethod
    def _run_episode(task: MapTask, uid: int, pilot: object) -> ValidationResult:
        """
        Executes one closed‑loop simulation with the miner's Pilot implementation.
        """
        env = make_env(task, gui=False, raw_rpm=True, randomise=True)
        obs = env.reset()                  # numpy array
        pilot.reset(task)

        t_sim    = 0.0
        energy   = 0.0
        success  = False

        while t_sim < task.horizon:
            rpm = pilot.act(obs, t_sim)    # (4,) RPM command
            obs, _, done, info = env.step(rpm[None, :])  # env expects batch
            t_sim += SIM_DT
            energy += np.abs(rpm).sum() * SIM_DT
            if done:
                success = info.get("success", False)
                break

        score = flight_reward(success, t_sim, energy, task.horizon)
        return ValidationResult(uid, success, t_sim, energy, score)

    def _apply_weight_update(self, results: List[ValidationResult]) -> None:
        """
        Writes miner scores into the metagraph’s score cache and
        triggers a weight‑push on‑chain.
        """
        if not results:
            bt.logging.warning("No validation results – skipping weight update.")
            return

        uids_np   = np.asarray([r.uid   for r in results], dtype=np.int64)
        scores_np = np.asarray([r.score for r in results], dtype=np.float32)

        self.update_scores(scores_np, uids_np)   # provided by base class
        bt.logging.info(f"Pushed scores for {len(results)} miners.")

    # -----------------------------------------------------------
    # Public API (called by bittensor framework)
    # -----------------------------------------------------------
    async def forward(self) -> None:
        """
        One full validator iteration (≈ 1–3 s):
          1. Build a random MapTask.
          2. Fetch/instantiate pilots for K miners.
          3. Run closed‑loop episodes, score, and update weights.
        """
        try:
            # -------- bookkeeping -------------------------------
            self.forward_count = getattr(self, "forward_count", 0) + 1
            bt.logging.info(f"[Forward #{self.forward_count}] start")

            # -------- 1) build task ------------------------------
            task: MapTask = random_task(sim_dt=SIM_DT, horizon=HORIZON_SEC)

            # -------- 2) fetch pilots ----------------------------
            uids: List[int] = get_random_uids(self, k=SAMPLE_K)
            bt.logging.info(f"Sampled miners: {uids}")
            pilots = await self._get_pilots(uids, task)

            # -------- 3) simulate & score ------------------------
            results: List[ValidationResult] = []
            for uid, pilot in pilots.items():
                try:
                    results.append(self._run_episode(task, uid, pilot))
                except Exception as e:
                    bt.logging.warning(
                        f"Episode failed for miner {uid}: {type(e).__name__} — {e}"
                    )
                    traceback.print_exc()

            # quick telemetry
            if results:
                best = max(r.score for r in results)
                avg  = sum(r.score for r in results) / len(results)
                bt.logging.info(
                    f"Scored {len(results)} miners | best={best:.3f} avg={avg:.3f}"
                )
            else:
                bt.logging.warning("No valid pilots returned by miners.")

            # -------- 4) weight update ---------------------------
            self._apply_weight_update(results)

        except Exception as err:
            bt.logging.error(f"Validator forward error: {err}")

        # -------- 5) relax --------------------------------------
        await asyncio.sleep(FORWARD_SLEEP_SEC)
