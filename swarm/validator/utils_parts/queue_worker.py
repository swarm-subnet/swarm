import math

from ._shared import *
from .heartbeat import HeartbeatManager
from swarm.validator.runtime_telemetry import tracker_call

from swarm.constants import N_DOCKER_WORKERS, BENCHMARK_TOTAL_SEED_COUNT, BENCHMARK_SCREENING_SEED_COUNT


def _utils_facade():
    from swarm.validator import utils as validator_utils

    return validator_utils


async def _process_normal_queue_item(
    self,
    queue: dict,
    key: str,
    validator_hotkey: str,
    validator_stake: float,
) -> None:
    items = queue.get("items", {})
    item = items.get(key)
    if not item:
        return

    try:
        uid = int(item.get("uid", -1))
        model_hash = str(item.get("model_hash", ""))
        model_path = Path(str(item.get("model_path", "")))
        github_url = str(item.get("github_url", ""))

        if uid < 0 or not model_hash:
            item["status"] = "terminal_rejected"
            item["last_error"] = "invalid queue item"
            item["updated_at"] = time.time()
            return

        item["status"] = "processing"
        item["updated_at"] = time.time()
        tracker_call(self, "mark_queue_item_stage", queue=queue, key=key, item=item, stage="processing")

        if not model_path.exists():
            _utils_facade()._schedule_queue_retry(item, "model file missing")
            tracker_call(
                self,
                "mark_queue_item_stage",
                queue=queue,
                key=key,
                item=item,
                stage="retry",
                severity="warning",
                note="model file missing",
            )
            return

        current_hash = _utils_facade().sha256sum(model_path)
        if current_hash != model_hash:
            item["status"] = "terminal_rejected"
            item["last_error"] = "model hash changed before processing"
            item["updated_at"] = time.time()
            tracker_call(
                self,
                "mark_queue_item_stage",
                queue=queue,
                key=key,
                item=item,
                stage="terminal_rejected",
                severity="warning",
                note="model hash changed before processing",
            )
            return

        miner_hotkey = ""
        try:
            miner_hotkey = self.metagraph.hotkeys[uid]
        except Exception:
            pass

        if not item.get("registered", False):
            if item.get("from_backend", False):
                item["registered"] = True
                item["status"] = "registered"
                item["updated_at"] = time.time()
                tracker_call(
                    self,
                    "mark_queue_item_stage",
                    queue=queue,
                    key=key,
                    item=item,
                    stage="registered",
                )
            else:
                accepted, terminal, reason = await _utils_facade()._register_new_model_with_ack(
                    self,
                    uid=uid,
                    model_hash=model_hash,
                    validator_hotkey=validator_hotkey,
                    github_url=github_url,
                    miner_hotkey=miner_hotkey,
                )
                if not accepted:
                    if terminal:
                        item["status"] = "terminal_rejected"
                        item["last_error"] = reason
                        item["updated_at"] = time.time()
                        _utils_facade().mark_model_hash_processed(uid, model_hash)
                        tracker_call(
                            self,
                            "mark_queue_item_stage",
                            queue=queue,
                            key=key,
                            item=item,
                            stage="terminal_rejected",
                            severity="warning",
                            note=str(reason),
                        )
                    else:
                        _utils_facade()._schedule_queue_retry(item, f"register failed: {reason}")
                        tracker_call(
                            self,
                            "mark_queue_item_stage",
                            queue=queue,
                            key=key,
                            item=item,
                            stage="retry",
                            severity="warning",
                            note=f"register failed: {reason}",
                        )
                    return

                item["registered"] = True
                item["status"] = "registered"
                item["retry_attempts"] = 0
                item["next_retry_at"] = 0
                item["last_error"] = ""
                item["updated_at"] = time.time()
                tracker_call(
                    self,
                    "mark_queue_item_stage",
                    queue=queue,
                    key=key,
                    item=item,
                    stage="registered",
                )

        epoch = self.seed_manager.epoch_number
        cached = (
            _utils_facade().get_cached_score(model_hash, epoch)
            if _utils_facade().has_cached_score(model_hash, epoch)
            else None
        )

        if item.get("screening_score") is None:
            if cached:
                item["screening_score"] = float(cached.get("screening_score", 0.0))
            else:
                tracker_call(self, "mark_queue_item_stage", queue=queue, key=key, item=item, stage="screening")
                screening_score, screening_scores, screening_per_type = (
                    await _utils_facade()._run_screening(self, uid, model_path)
                )
                item["screening_score"] = float(screening_score)
                item["screening_scores"] = screening_scores
                item["screening_per_type"] = {
                    k: v for k, v in screening_per_type.items() if v
                }
            item["updated_at"] = time.time()

        if item.get("screening_passed") is None:
            item["screening_passed"] = _utils_facade()._passes_screening(
                self, float(item.get("screening_score", 0.0))
            )

        screening_passed = bool(item.get("screening_passed", False))
        bt.logging.info(
            f"{'PASSED' if screening_passed else 'FAILED'} Screening UID {uid} | "
            f"score={item.get('screening_score', 0):.4f}"
        )

        if not item.get("screening_recorded", False):
            tracker_call(
                self,
                "mark_queue_item_stage",
                queue=queue,
                key=key,
                item=item,
                stage="screening_submit",
            )
            tracker_call(self, "mark_submission_started", stage="screening", uid=uid, model_hash=model_hash)
            recorded, terminal, reason = await _utils_facade()._submit_screening_with_ack(
                self,
                uid=uid,
                validator_hotkey=validator_hotkey,
                validator_stake=validator_stake,
                screening_score=float(item.get("screening_score", 0.0)),
                passed=screening_passed,
            )
            if not recorded:
                bt.logging.warning(
                    f"Screening submit failed for UID {uid} | "
                    f"terminal={terminal} | {reason}"
                )
                tracker_call(
                    self,
                    "mark_submission_result",
                    stage="screening",
                    uid=uid,
                    success=False,
                    terminal=terminal,
                    reason=str(reason),
                    model_hash=model_hash,
                )
                if terminal:
                    item["status"] = "terminal_rejected"
                    item["last_error"] = reason
                    item["updated_at"] = time.time()
                    _utils_facade().mark_model_hash_processed(uid, model_hash)
                    tracker_call(
                        self,
                        "mark_queue_item_stage",
                        queue=queue,
                        key=key,
                        item=item,
                        stage="terminal_rejected",
                        severity="warning",
                        note=str(reason),
                    )
                else:
                    _utils_facade()._schedule_queue_retry(item, f"screening submit failed: {reason}")
                    tracker_call(
                        self,
                        "mark_queue_item_stage",
                        queue=queue,
                        key=key,
                        item=item,
                        stage="retry",
                        severity="warning",
                        note=f"screening submit failed: {reason}",
                    )
                return

            tracker_call(
                self,
                "mark_submission_result",
                stage="screening",
                uid=uid,
                success=True,
                terminal=False,
                model_hash=model_hash,
            )
            item["screening_recorded"] = True
            item["status"] = "screening_recorded"
            item["retry_attempts"] = 0
            item["next_retry_at"] = 0
            item["last_error"] = ""
            item["updated_at"] = time.time()
            bt.logging.info(f"Screening score submitted to backend for UID {uid}")
            tracker_call(
                self,
                "mark_queue_item_stage",
                queue=queue,
                key=key,
                item=item,
                stage="screening_recorded",
            )

        if not screening_passed:
            bt.logging.info(
                f"UID {uid} screening failed — skipping benchmark"
            )
            item["status"] = "completed"
            item["updated_at"] = time.time()
            item.pop("screening_scores", None)
            _utils_facade().mark_model_hash_processed(uid, model_hash)
            tracker_call(
                self,
                "mark_queue_item_stage",
                queue=queue,
                key=key,
                item=item,
                stage="completed",
            )
            return

        missing_score_payload = (
            item.get("total_score") is None
            or not isinstance(item.get("per_type_scores"), dict)
            or item.get("seeds_evaluated") is None
        )

        if missing_score_payload:
            if cached:
                per_type_scores = cached.get("per_type_scores", {})
                if not isinstance(per_type_scores, dict):
                    per_type_scores = {}
                item["full_score"] = float(
                    cached.get("full_score", cached.get("total_score", 0.0))
                )
                item["total_score"] = float(cached.get("total_score", 0.0))
                item["per_type_scores"] = per_type_scores
                item["seeds_evaluated"] = int(cached.get("seeds_evaluated", BENCHMARK_TOTAL_SEED_COUNT))
            else:
                all_benchmark_seeds = self.seed_manager.get_benchmark_seeds()
                partial_scores = item.get("benchmark_partial_scores", [])
                partial_per_type = item.get("benchmark_partial_per_type", {})
                done = len(partial_scores)
                remaining_seeds = all_benchmark_seeds[done:]

                if remaining_seeds:
                    tracker_call(
                        self,
                        "mark_queue_item_stage",
                        queue=queue,
                        key=key,
                        item=item,
                        stage="benchmark",
                        progress_done=done,
                        progress_total=len(all_benchmark_seeds),
                    )
                    total_benchmark_seeds = len(all_benchmark_seeds)
                    hb = HeartbeatManager(self.backend_api, asyncio.get_running_loop())
                    hb.start("evaluating_benchmark", uid, total_benchmark_seeds)
                    if done > 0:
                        with hb._lock:
                            hb._progress = done
                            hb._last_sent = done
                    round_size = max(1, math.ceil(total_benchmark_seeds / N_DOCKER_WORKERS))
                    try:
                        for i in range(0, len(remaining_seeds), round_size):
                            chunk = remaining_seeds[i:i + round_size]
                            prior_avg = float(np.mean(partial_scores)) if partial_scores else 0.0
                            chunk_scores, chunk_per_type, chunk_details = await _utils_facade()._evaluate_seeds(
                                self, uid, model_path, chunk,
                                f"benchmark [{done + 1}..{done + len(chunk)}]",
                                on_seed_complete=hb.on_seed_complete,
                                prior_seeds_done=done,
                                prior_total_seeds=total_benchmark_seeds,
                                prior_avg=prior_avg,
                            )
                            try:
                                seed_batch = [
                                    {"seed_index": BENCHMARK_SCREENING_SEED_COUNT + done + j, "score": d["score"], "map_type": d["map_type"]}
                                    for j, d in enumerate(chunk_details) if d.get("map_type") != "unknown"
                                ]
                                if seed_batch:
                                    await self.backend_api.post_seed_scores_batch(
                                        model_uid=uid, epoch_number=epoch, scores=seed_batch,
                                    )
                            except Exception as seed_err:
                                bt.logging.warning(f"Seed score upload failed for UID {uid}: {seed_err}")
                            partial_scores.extend(chunk_scores)
                            for tname, tscores in chunk_per_type.items():
                                partial_per_type.setdefault(tname, []).extend(tscores)
                            done = len(partial_scores)
                            item["benchmark_partial_scores"] = partial_scores
                            item["benchmark_partial_per_type"] = partial_per_type
                            item["updated_at"] = time.time()
                            tracker_call(
                                self,
                                "mark_queue_item_stage",
                                queue=queue,
                                key=key,
                                item=item,
                                stage="benchmark",
                                progress_done=done,
                                progress_total=len(all_benchmark_seeds),
                                note=f"chunk {done}/{len(all_benchmark_seeds)}",
                            )
                            _utils_facade().save_normal_model_queue(queue)
                    finally:
                        hb.finish()

                screening_scores = item.get("screening_scores", [])
                if not isinstance(screening_scores, list):
                    screening_scores = []
                scr_per_type = item.get("screening_per_type", {})

                combined_scores = screening_scores + partial_scores
                total_score = float(np.mean(combined_scores)) if combined_scores else 0.0
                full_score = float(np.mean(partial_scores)) if partial_scores else 0.0

                merged_per_type = {}
                all_type_keys = set(scr_per_type) | set(partial_per_type)
                for type_key in all_type_keys:
                    combined = scr_per_type.get(type_key, []) + partial_per_type.get(type_key, [])
                    merged_per_type[type_key] = float(np.mean(combined)) if combined else 0.0

                item["full_score"] = full_score
                item["total_score"] = total_score
                item["per_type_scores"] = merged_per_type
                item["seeds_evaluated"] = len(combined_scores)
                item.pop("benchmark_partial_scores", None)
                item.pop("benchmark_partial_per_type", None)
            item["updated_at"] = time.time()

        per_type_str = " | ".join(
            f"{k}={v:.2f}" for k, v in sorted(item.get("per_type_scores", {}).items())
        )
        bt.logging.info(
            f"Benchmark UID {uid} | "
            f"total={item.get('total_score', 0):.4f} | "
            f"{item.get('seeds_evaluated', 0)} seeds | {per_type_str}"
        )

        if not item.get("score_recorded", False):
            tracker_call(
                self,
                "mark_queue_item_stage",
                queue=queue,
                key=key,
                item=item,
                stage="score_submit",
            )
            tracker_call(self, "mark_submission_started", stage="score", uid=uid, model_hash=model_hash)
            recorded, terminal, reason = await _utils_facade()._submit_score_with_ack(
                self,
                uid=uid,
                validator_hotkey=validator_hotkey,
                validator_stake=validator_stake,
                model_hash=model_hash,
                total_score=float(item.get("total_score", 0.0)),
                per_type_scores=dict(item.get("per_type_scores", {})),
                seeds_evaluated=int(item.get("seeds_evaluated", 0) or 0),
                epoch_number=self.seed_manager.epoch_number,
            )
            if not recorded:
                bt.logging.warning(
                    f"Score submit failed for UID {uid} | "
                    f"terminal={terminal} | {reason}"
                )
                tracker_call(
                    self,
                    "mark_submission_result",
                    stage="score",
                    uid=uid,
                    success=False,
                    terminal=terminal,
                    reason=str(reason),
                    model_hash=model_hash,
                )
                if terminal:
                    item["status"] = "terminal_rejected"
                    item["last_error"] = reason
                    item["updated_at"] = time.time()
                    _utils_facade().mark_model_hash_processed(uid, model_hash)
                    tracker_call(
                        self,
                        "mark_queue_item_stage",
                        queue=queue,
                        key=key,
                        item=item,
                        stage="terminal_rejected",
                        severity="warning",
                        note=str(reason),
                    )
                else:
                    _utils_facade()._schedule_queue_retry(item, f"score submit failed: {reason}")
                    tracker_call(
                        self,
                        "mark_queue_item_stage",
                        queue=queue,
                        key=key,
                        item=item,
                        stage="retry",
                        severity="warning",
                        note=f"score submit failed: {reason}",
                    )
                return

            tracker_call(
                self,
                "mark_submission_result",
                stage="score",
                uid=uid,
                success=True,
                terminal=False,
                model_hash=model_hash,
            )
            item["score_recorded"] = True
            item["status"] = "completed"
            item["retry_attempts"] = 0
            item["next_retry_at"] = 0
            item["last_error"] = ""
            item["updated_at"] = time.time()
            bt.logging.info(f"Score submitted to backend for UID {uid}")
            tracker_call(
                self,
                "mark_queue_item_stage",
                queue=queue,
                key=key,
                item=item,
                stage="completed",
            )

        bt.logging.info(
            f"UID {uid} evaluation complete | "
            f"screening={item.get('screening_score', 0):.4f} "
            f"total={item.get('total_score', 0):.4f} | "
            f"{item.get('seeds_evaluated', 0)} seeds"
        )

        _utils_facade().set_cached_score(model_hash, epoch, {
            "uid": uid,
            "total_score": float(item.get("total_score", 0.0)),
            "screening_score": float(item.get("screening_score", 0.0)),
            "full_score": float(item.get("full_score", item.get("total_score", 0.0))),
            "per_type_scores": dict(item.get("per_type_scores", {})),
            "seeds_evaluated": int(item.get("seeds_evaluated", 0) or 0),
        })

        item.pop("screening_scores", None)
        item.pop("screening_per_type", None)
        _utils_facade().mark_model_hash_processed(uid, model_hash)
        tracker_call(self, "update_queue_state", queue)
        tracker_call(self, "increment_counter", "models_processed_total")

    except Exception as e:
        _utils_facade()._schedule_queue_retry(item, f"queue worker exception: {e}")
        tracker_call(
            self,
            "mark_queue_item_stage",
            queue=queue,
            key=key,
            item=item,
            stage="retry",
            severity="error",
            note=f"queue worker exception: {e}",
        )


# ──────────────────────────────────────────────────────────────────────────
# Weight application
# ──────────────────────────────────────────────────────────────────────────
