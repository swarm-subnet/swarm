# ---------------------------------------------------------------
#  Swarm validator – forward loop (Policy API v2)
# ---------------------------------------------------------------
from __future__ import annotations

import asyncio
import traceback

import bittensor as bt
import numpy as np

from typing import Dict, List

from swarm.constants import (
    EPOCH_FREEZE_SECONDS,
    FORWARD_SLEEP_SEC,
    MODEL_DIR,
)
from swarm.utils.hash import sha256sum


def _merge_per_type_medians(
    a: Dict[str, List[float]], b: Dict[str, List[float]]
) -> Dict[str, float]:
    merged: Dict[str, float] = {}
    all_keys = set(a) | set(b)
    for key in all_keys:
        combined = a.get(key, []) + b.get(key, [])
        merged[key] = float(np.median(combined)) if combined else 0.0
    return merged

from .backend_api import BackendApiClient
from .docker.docker_evaluator import DockerSecureEvaluator
from .runtime_telemetry import tracker_call
from .seed_manager import BenchmarkSeedManager
from .utils import (
    NORMAL_MODEL_QUEUE_PROCESS_LIMIT,
    _apply_backend_weights_to_scores,
    _detect_new_models,
    _ensure_models_from_backend,
    _get_processable_queue_keys,
    _get_validator_stake,
    _process_normal_queue_item,
    _refresh_normal_model_queue,
    _run_full_benchmark,
    _run_screening,
    _submit_score_with_ack,
    clear_benchmark_cache,
    clear_normal_model_queue,
    load_normal_model_queue,
    save_normal_model_queue,
    set_cached_score,
)


async def forward(self) -> None:
    """
    Benchmark-style validator forward.

    Flow:
    1. Sync with backend → get weights + reeval queue
    2. Apply weights from backend (set on-chain)
    3. Process re-eval queue (if any)
    4. Detect new/changed models
    5. For each new model: screening → full benchmark → submit
    """
    try:
        self.forward_count = getattr(self, "forward_count", 0) + 1
        tracker_call(self, "mark_forward_started", forward_count=self.forward_count)
        bt.logging.info(f"[Forward #{self.forward_count}] start")

        # ──────────────────────────────────────────────────────────────
        # STEP 0: Initialize components if needed
        # ──────────────────────────────────────────────────────────────
        if not hasattr(self, 'seed_manager'):
            self.seed_manager = BenchmarkSeedManager()
            _invalidate_local_state_for_regenerated_seeds(self)

        if not hasattr(self, 'backend_api'):
            try:
                self.backend_api = BackendApiClient(wallet=self.wallet)
            except ValueError as e:
                tracker_call(self, "mark_forward_failed", error=str(e))
                bt.logging.error(f"Backend API initialization failed: {e}")
                bt.logging.error("Set SWARM_BACKEND_API_URL environment variable")
                await asyncio.sleep(FORWARD_SLEEP_SEC)
                return

        if not hasattr(self, 'docker_evaluator') or not DockerSecureEvaluator._base_ready:
            tracker_call(self, "mark_forward_failed", error="Docker evaluator not ready")
            bt.logging.error("Docker evaluator not ready")
            await asyncio.sleep(FORWARD_SLEEP_SEC)
            return
        if hasattr(self, "docker_evaluator"):
            setattr(self.docker_evaluator, "runtime_tracker", getattr(self, "runtime_tracker", None))

        validator_hotkey = self.wallet.hotkey.ss58_address
        validator_stake = _get_validator_stake(self)

        # ──────────────────────────────────────────────────────────────
        # STEP 0.25: Publish any unpublished old epoch seeds
        # ──────────────────────────────────────────────────────────────
        for pub in self.seed_manager.get_pending_publications():
            ep = pub.get("epoch_number")
            try:
                await self.backend_api.publish_epoch_seeds(
                    epoch_number=ep,
                    seeds=pub.get("seeds", []),
                    started_at=pub.get("started_at", ""),
                    ended_at=pub.get("ended_at", ""),
                    benchmark_version=pub.get("benchmark_version"),
                )
                self.seed_manager.mark_epoch_published(ep)
                bt.logging.info(f"Published epoch {ep} seeds to backend")
            except Exception as e:
                bt.logging.warning(f"Failed to publish epoch {ep} seeds: {e}")

        # ──────────────────────────────────────────────────────────────
        # STEP 0.5: Epoch transition detection
        # ──────────────────────────────────────────────────────────────
        if self.seed_manager.check_epoch_transition():
            old_epoch = self.seed_manager.advance_to_new_epoch()
            tracker_call(
                self,
                "mark_epoch_transition",
                old_epoch=old_epoch,
                new_epoch=self.seed_manager.epoch_number,
            )
            bt.logging.info(
                f"Epoch transition: {old_epoch} -> {self.seed_manager.epoch_number}"
            )

            for pub in self.seed_manager.get_pending_publications():
                ep = pub.get("epoch_number")
                try:
                    await self.backend_api.publish_epoch_seeds(
                        epoch_number=ep,
                        seeds=pub.get("seeds", []),
                        started_at=pub.get("started_at", ""),
                        ended_at=pub.get("ended_at", ""),
                        benchmark_version=pub.get("benchmark_version"),
                    )
                    self.seed_manager.mark_epoch_published(ep)
                    bt.logging.info(f"Published epoch {ep} seeds to backend")
                except Exception as e:
                    bt.logging.warning(f"Failed to publish epoch {ep} seeds: {e}")
            cleanup_started = asyncio.get_running_loop().time()
            self.docker_evaluator.cleanup()
            tracker_call(
                self,
                "mark_docker_cleanup",
                duration_sec=asyncio.get_running_loop().time() - cleanup_started,
                reason="epoch_transition",
            )
            clear_normal_model_queue()
            clear_benchmark_cache()
            self._epoch_just_transitioned = True
            bt.logging.info("Epoch transition: killed Docker, cleared queue and cache")

        # ──────────────────────────────────────────────────────────────
        # STEP 1: Sync with backend
        # ──────────────────────────────────────────────────────────────
        bt.logging.info("📡 Syncing with backend...")
        tracker_call(self, "mark_backend_sync_started")
        sync_data = await self.backend_api.sync()

        if sync_data.get("fallback"):
            bt.logging.warning("Backend unavailable — burning 100% emissions")

        self._current_top = sync_data.get("current_top", {})
        reeval_queue = sync_data.get("reeval_queue", [])
        backend_epoch = int(
            sync_data.get("benchmark_epoch")
            or sync_data.get("current_epoch")
            or 0
        )
        tracker_call(
            self,
            "mark_backend_sync_completed",
            fallback=bool(sync_data.get("fallback", False)),
            pending_models_count=len(sync_data.get("pending_models", [])),
            reeval_queue_count=len(reeval_queue),
            leaderboard_version=sync_data.get("leaderboard_version"),
            error=str(sync_data.get("error", "")),
        )

        if backend_epoch > 0 and backend_epoch != self.seed_manager.epoch_number:
            old_epoch = self.seed_manager.align_to_epoch(backend_epoch)
            if old_epoch is not None:
                self.docker_evaluator.cleanup()
                clear_normal_model_queue()
                clear_benchmark_cache()
                self._epoch_just_transitioned = True
                bt.logging.info(
                    f"Aligned validator seed epoch to backend benchmark epoch: "
                    f"{old_epoch} -> {self.seed_manager.epoch_number}"
                )

        epoch_just_transitioned = getattr(self, '_epoch_just_transitioned', False)
        if epoch_just_transitioned and self._current_top.get("uid") is not None:
            champion_uid = self._current_top["uid"]
            reeval_queue.insert(0, {"uid": champion_uid, "reason": "epoch_transition"})
            bt.logging.info(f"👑 Champion UID {champion_uid} queued for epoch re-evaluation")

        # ──────────────────────────────────────────────────────────────
        # STEP 2: Apply weights from backend
        # ──────────────────────────────────────────────────────────────
        backend_weights = {} if sync_data.get("fallback") else sync_data.get("weights", {})
        _apply_backend_weights_to_scores(self, backend_weights)
        nonzero_uids = int(np.count_nonzero(self.scores))
        bt.logging.info(
            f"⚖️ Applied backend weights to local scores: {nonzero_uids} non-zero UID(s)"
        )

        # ──────────────────────────────────────────────────────────────
        # STEP 3: Process re-eval queue
        # ──────────────────────────────────────────────────────────────
        if sync_data.get("fallback") and reeval_queue:
            bt.logging.warning(
                f"Backend unavailable: skipping {len(reeval_queue)} cached re-eval item(s)"
            )
            if epoch_just_transitioned:
                bt.logging.info("Keeping epoch transition flag — will retry champion re-eval next cycle")
            else:
                self._epoch_just_transitioned = False
        else:
            epoch_reeval_succeeded = not epoch_just_transitioned
            for reeval_item in reeval_queue:
                uid = reeval_item.get("uid")
                reason = reeval_item.get("reason", "unknown")
                tracker_call(self, "mark_reeval_started", uid=int(uid), reason=str(reason))
                bt.logging.info(f"🔄 Re-evaluation requested for UID {uid}: {reason}")

                model_path = MODEL_DIR / f"UID_{uid}.zip"
                if not model_path.exists():
                    tracker_call(self, "mark_reeval_missing_model", uid=int(uid), reason=str(reason))
                    bt.logging.warning(f"Model not found for re-eval UID {uid}")
                    continue

                model_hash = sha256sum(model_path)

                if reason == "epoch_transition":
                    all_seeds = self.seed_manager.get_all_seeds()
                    bt.logging.info(
                        f"👑 Champion UID {uid}: {len(all_seeds)} seeds directly (no screening)"
                    )
                    full_score, per_type_scores, full_scores, _ = await _run_full_benchmark(
                        self, uid, model_path, seeds=all_seeds
                    )
                    screening_score = full_score
                    screening_scores = []
                    combined_scores = full_scores
                else:
                    screening_score, screening_scores, scr_per_type = await _run_screening(
                        self, uid, model_path
                    )
                    full_score, _, full_scores, bench_per_type = await _run_full_benchmark(
                        self, uid, model_path
                    )
                    combined_scores = screening_scores + full_scores
                    per_type_scores = _merge_per_type_medians(scr_per_type, bench_per_type)
                all_seeds_count = len(combined_scores)
                total_score = (
                    float(np.median(combined_scores)) if combined_scores else 0.0
                )

                recorded, terminal, ack_reason = await _submit_score_with_ack(
                    self,
                    uid=uid,
                    validator_hotkey=validator_hotkey,
                    validator_stake=validator_stake,
                    model_hash=model_hash,
                    total_score=total_score,
                    per_type_scores=per_type_scores,
                    seeds_evaluated=all_seeds_count,
                    epoch_number=self.seed_manager.epoch_number,
                )

                if not recorded:
                    tracker_call(
                        self,
                        "mark_reeval_completed",
                        uid=int(uid),
                        reason=str(reason),
                        success=False,
                        total_score=total_score,
                        error=str(ack_reason),
                    )
                    if terminal:
                        bt.logging.error(
                            f"Re-eval score rejected permanently for UID {uid}: "
                            f"{ack_reason}"
                        )
                    else:
                        bt.logging.warning(
                            f"Re-eval score submit failed for UID {uid}, "
                            f"will retry next sync: {ack_reason}"
                        )
                    continue

                if reason == "epoch_transition":
                    epoch_reeval_succeeded = True

                tracker_call(
                    self,
                    "mark_reeval_completed",
                    uid=int(uid),
                    reason=str(reason),
                    success=True,
                    total_score=total_score,
                )
                set_cached_score(model_hash, self.seed_manager.epoch_number, {
                    "uid": uid,
                    "total_score": total_score,
                    "screening_score": screening_score,
                    "full_score": full_score,
                    "per_type_scores": per_type_scores,
                    "seeds_evaluated": all_seeds_count
                })

                bt.logging.info(
                    f"Re-eval complete for UID {uid}: score={total_score:.4f}"
                )
            if epoch_reeval_succeeded:
                self._epoch_just_transitioned = False
            elif epoch_just_transitioned:
                bt.logging.warning("Champion epoch re-eval not submitted, will retry next cycle")

        # ──────────────────────────────────────────────────────────────
        # STEP 4: Discovery from backend (no axon polling)
        # ──────────────────────────────────────────────────────────────
        remaining = self.seed_manager.seconds_until_epoch_end()
        in_freeze = remaining < EPOCH_FREEZE_SECONDS
        tracker_call(
            self,
            "mark_epoch_state",
            epoch_number=self.seed_manager.epoch_number,
            seconds_until_end=remaining,
            freeze_active=in_freeze,
        )

        if in_freeze:
            bt.logging.info(
                f"⏸️ Epoch freeze active — {remaining / 60:.0f} min remaining. "
                "Skipping new model discovery and evaluation."
            )
            pending = []
        else:
            pending = sync_data.get("pending_models", [])
            if pending:
                bt.logging.info(f"Backend reports {len(pending)} pending model(s)")

        model_paths = await _ensure_models_from_backend(self, pending)
        if not model_paths:
            bt.logging.info("No models to process this cycle")
            new_models = {}
        else:
            new_models = _detect_new_models(self, model_paths)
            if new_models:
                bt.logging.info(
                    f"🆕 Found {len(new_models)} new/changed models for queue"
                )
            else:
                bt.logging.info("No new/changed models detected")

        queue = (
            _refresh_normal_model_queue(new_models)
            if new_models
            else load_normal_model_queue()
        )
        tracker_call(self, "update_queue_state", queue)

        # ──────────────────────────────────────────────────────────────
        # STEP 5: Queue worker (normal-model pipeline consumer)
        # ──────────────────────────────────────────────────────────────
        if in_freeze:
            processable_keys = []
        else:
            processable_keys = _get_processable_queue_keys(
                queue, NORMAL_MODEL_QUEUE_PROCESS_LIMIT
            )
        if not processable_keys:
            bt.logging.info("No normal-model queue items ready this cycle")
        else:
            bt.logging.info(f"📦 Processing {len(processable_keys)} queued model(s)")
            for queue_key in processable_keys:
                await _process_normal_queue_item(
                    self,
                    queue=queue,
                    key=queue_key,
                    validator_hotkey=validator_hotkey,
                    validator_stake=validator_stake,
                )

        items = queue.get("items", {})
        completed_keys = [
            key for key, item in items.items()
            if item.get("status") in ("completed", "terminal_rejected")
        ]
        for key in completed_keys:
            items.pop(key, None)

        queue["items"] = items
        save_normal_model_queue(queue)
        tracker_call(self, "update_queue_state", queue)

        cleanup_started = asyncio.get_running_loop().time()
        self.docker_evaluator.cleanup()
        tracker_call(
            self,
            "mark_docker_cleanup",
            duration_sec=asyncio.get_running_loop().time() - cleanup_started,
            reason="forward_end",
        )
        bt.logging.info(f"[Forward #{self.forward_count}] complete")
        tracker_call(self, "mark_forward_completed", forward_count=self.forward_count)
        tracker_call(self, "flush")

    except Exception as e:
        tracker_call(self, "mark_forward_failed", error=str(e))
        bt.logging.error(f"Validator forward error: {e}")
        bt.logging.error(traceback.format_exc())


def _invalidate_local_state_for_regenerated_seeds(self) -> None:
    """Drop persisted local evaluation state if the current epoch seeds were rebuilt."""
    seed_manager = getattr(self, "seed_manager", None)
    if seed_manager is None:
        return
    if not getattr(seed_manager, "current_epoch_requires_state_invalidation", False):
        return
    clear_normal_model_queue()
    clear_benchmark_cache()
    seed_manager.current_epoch_requires_state_invalidation = False
    bt.logging.warning(
        "Current epoch seeds were regenerated locally; cleared queued progress and benchmark cache"
    )
