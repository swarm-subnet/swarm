from ._shared import *
from .detection import _get_miner_coldkey


async def _register_new_model_with_ack(
    self,
    uid: int,
    model_hash: str,
    validator_hotkey: str,
    github_url: str = "",
    miner_hotkey: str = "",
) -> Tuple[bool, bool, str]:
    coldkey = _get_miner_coldkey(self, uid)
    response = await self.backend_api.post_new_model(
        uid=uid,
        model_hash=model_hash,
        coldkey=coldkey,
        validator_hotkey=validator_hotkey,
        github_url=github_url,
        miner_hotkey=miner_hotkey,
    )

    if response.get("accepted", False):
        model_path = MODEL_DIR / f"UID_{uid}.zip"
        if model_path.is_file():
            try:
                upload_resp = await self.backend_api.upload_model_file(uid, model_path)
                bt.logging.info(f"Model upload for UID {uid}: {upload_resp}")
            except Exception as e:
                bt.logging.warning(f"Model upload failed for UID {uid} (non-fatal): {e}")
        return True, False, ""

    terminal, reason = classify_backend_failure(response, "new_model")
    return False, terminal, reason


async def _submit_screening_with_ack(
    self,
    uid: int,
    validator_hotkey: str,
    validator_stake: float,
    screening_score: float,
) -> Tuple[bool, bool, str]:
    response = await self.backend_api.post_screening(
        uid=uid,
        validator_hotkey=validator_hotkey,
        validator_stake=validator_stake,
        screening_score=screening_score,
    )

    if response.get("recorded", False):
        return True, False, ""

    terminal, reason = classify_backend_failure(response, "screening")
    return False, terminal, reason


async def _publish_pending_epoch_seeds(self) -> None:
    """Publish unpublished past-epoch seed records to the backend.

    Marks an epoch as published only when the backend confirms acceptance,
    so rejected or failed calls remain pending and are retried on the next
    forward cycle.
    """
    for pub in self.seed_manager.get_pending_publications():
        ep = pub.get("epoch_number")
        try:
            response = await self.backend_api.publish_epoch_seeds(
                epoch_number=ep,
                seeds=pub.get("seeds", []),
                started_at=pub.get("started_at", ""),
                ended_at=pub.get("ended_at", ""),
                benchmark_version=pub.get("benchmark_version"),
            )
        except Exception as e:
            bt.logging.warning(f"Failed to publish epoch {ep} seeds: {e}")
            continue

        accepted = isinstance(response, dict) and (
            response.get("published") or response.get("accepted")
        )
        if accepted:
            self.seed_manager.mark_epoch_published(ep)
            bt.logging.info(f"Published epoch {ep} seeds to backend")
        else:
            bt.logging.warning(
                f"Backend rejected epoch {ep} publish; will retry next cycle: {response}"
            )


async def _submit_score_with_ack(
    self,
    uid: int,
    validator_hotkey: str,
    validator_stake: float,
    model_hash: str,
    total_score: float,
    per_type_scores: Dict[str, float],
    seeds_evaluated: int,
    epoch_number: Optional[int] = None,
) -> Tuple[bool, bool, str]:
    response = await self.backend_api.post_score(
        uid=uid,
        validator_hotkey=validator_hotkey,
        validator_stake=validator_stake,
        model_hash=model_hash,
        total_score=total_score,
        per_type_scores=per_type_scores,
        seeds_evaluated=seeds_evaluated,
        epoch_number=epoch_number,
    )

    if response.get("recorded", False):
        return True, False, ""

    terminal, reason = classify_backend_failure(response, "score")
    return False, terminal, reason


# ──────────────────────────────────────────────────────────────────────────
# Queue worker – processes a single normal-model pipeline item
# ──────────────────────────────────────────────────────────────────────────

