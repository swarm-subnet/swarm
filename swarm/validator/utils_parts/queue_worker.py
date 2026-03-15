from ._shared import *


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

        if not model_path.exists():
            _utils_facade()._schedule_queue_retry(item, "model file missing")
            return

        current_hash = _utils_facade().sha256sum(model_path)
        if current_hash != model_hash:
            item["status"] = "terminal_rejected"
            item["last_error"] = "model hash changed before processing"
            item["updated_at"] = time.time()
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
                    else:
                        _utils_facade()._schedule_queue_retry(item, f"register failed: {reason}")
                    return

                item["registered"] = True
                item["status"] = "registered"
                item["retry_attempts"] = 0
                item["next_retry_at"] = 0
                item["last_error"] = ""
                item["updated_at"] = time.time()

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
                screening_score, screening_scores = await _utils_facade()._run_screening(
                    self, uid, model_path
                )
                item["screening_score"] = float(screening_score)
                item["screening_scores"] = screening_scores
            item["updated_at"] = time.time()

        if item.get("screening_passed") is None:
            item["screening_passed"] = _utils_facade()._passes_screening(
                self, float(item.get("screening_score", 0.0))
            )

        screening_passed = bool(item.get("screening_passed", False))

        if not item.get("screening_recorded", False):
            recorded, terminal, reason = await _utils_facade()._submit_screening_with_ack(
                self,
                uid=uid,
                validator_hotkey=validator_hotkey,
                validator_stake=validator_stake,
                screening_score=float(item.get("screening_score", 0.0)),
                passed=screening_passed,
            )
            if not recorded:
                if terminal:
                    item["status"] = "terminal_rejected"
                    item["last_error"] = reason
                    item["updated_at"] = time.time()
                    _utils_facade().mark_model_hash_processed(uid, model_hash)
                else:
                    _utils_facade()._schedule_queue_retry(item, f"screening submit failed: {reason}")
                return

            item["screening_recorded"] = True
            item["status"] = "screening_recorded"
            item["retry_attempts"] = 0
            item["next_retry_at"] = 0
            item["last_error"] = ""
            item["updated_at"] = time.time()

        if not screening_passed:
            item["status"] = "completed"
            item["updated_at"] = time.time()
            item.pop("screening_scores", None)
            _utils_facade().mark_model_hash_processed(uid, model_hash)
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
                item["seeds_evaluated"] = int(cached.get("seeds_evaluated", 1200))
            else:
                full_score, per_type_scores, full_scores = await _utils_facade()._run_full_benchmark(
                    self, uid, model_path
                )
                screening_scores = item.get("screening_scores", [])
                if not isinstance(screening_scores, list):
                    screening_scores = []
                combined_scores = screening_scores + full_scores
                total_score = float(np.median(combined_scores)) if combined_scores else 0.0
                item["full_score"] = float(full_score)
                item["total_score"] = total_score
                item["per_type_scores"] = per_type_scores
                item["seeds_evaluated"] = len(combined_scores)
            item["updated_at"] = time.time()

        if not item.get("score_recorded", False):
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
                if terminal:
                    item["status"] = "terminal_rejected"
                    item["last_error"] = reason
                    item["updated_at"] = time.time()
                    _utils_facade().mark_model_hash_processed(uid, model_hash)
                else:
                    _utils_facade()._schedule_queue_retry(item, f"score submit failed: {reason}")
                return

            item["score_recorded"] = True
            item["status"] = "completed"
            item["retry_attempts"] = 0
            item["next_retry_at"] = 0
            item["last_error"] = ""
            item["updated_at"] = time.time()

        _utils_facade().set_cached_score(model_hash, epoch, {
            "uid": uid,
            "total_score": float(item.get("total_score", 0.0)),
            "screening_score": float(item.get("screening_score", 0.0)),
            "full_score": float(item.get("full_score", item.get("total_score", 0.0))),
            "per_type_scores": dict(item.get("per_type_scores", {})),
            "seeds_evaluated": int(item.get("seeds_evaluated", 0) or 0),
        })

        item.pop("screening_scores", None)
        _utils_facade().mark_model_hash_processed(uid, model_hash)

    except Exception as e:
        _utils_facade()._schedule_queue_retry(item, f"queue worker exception: {e}")


# ──────────────────────────────────────────────────────────────────────────
# Weight application
# ──────────────────────────────────────────────────────────────────────────
