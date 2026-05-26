from ._shared import *
from .detection import _get_miner_coldkey
from swarm.challenge_families import DEFAULT_RUNTIME_FAMILY_ID


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
    family_id: str = DEFAULT_RUNTIME_FAMILY_ID,
) -> Tuple[bool, bool, str]:
    response = await self.backend_api.post_screening(
        uid=uid,
        validator_hotkey=validator_hotkey,
        validator_stake=validator_stake,
        screening_score=screening_score,
        family_id=family_id,
    )

    if response.get("recorded", False):
        return True, False, ""

    terminal, reason = classify_backend_failure(response, "screening")
    return False, terminal, reason


PUBLISH_BACKOFF_CAP_CYCLES = 12


def _seed_manager_method(seed_manager, method_name: str, *args, **kwargs):
    method = getattr(seed_manager, method_name)
    try:
        return method(*args, **kwargs)
    except TypeError:
        return method(*args)


def _publish_backoff_cycles(failures: int) -> int:
    """Return how many forward cycles to skip before retrying a failed publish.

    Sequence: 1 → 1, 2 → 2, 3 → 4, 4 → 8, 5+ → 12 (capped). Doubles per
    failure so a permanently broken backend stops spamming logs while
    transient outages still recover quickly.
    """
    if failures <= 0:
        return 0
    return min(PUBLISH_BACKOFF_CAP_CYCLES, 2 ** (failures - 1))


async def _publish_pending_epoch_seeds(self) -> None:
    """Publish unpublished past-epoch seed records to the backend.

    Marks an epoch as published only when the backend confirms acceptance,
    so rejected or failed calls remain pending. Repeatedly failing epochs
    back off exponentially (up to ``PUBLISH_BACKOFF_CAP_CYCLES``) so
    permanent rejections do not flood logs every cycle.
    """
    skip_until: Dict[tuple[int, str], int] = getattr(self, "_epoch_publish_skip_until", {})
    failures: Dict[tuple[int, str], int] = getattr(self, "_epoch_publish_failures", {})
    current_cycle = int(getattr(self, "forward_count", 0))

    for pub in _seed_manager_method(self.seed_manager, "get_pending_publications"):
        ep = pub.get("epoch_number")
        family_id = str(pub.get("family_id") or "cf_search_and_rescue")
        if ep is None:
            continue
        scope = (int(ep), family_id)

        if current_cycle < skip_until.get(scope, 0):
            continue

        try:
            response = await self.backend_api.publish_epoch_seeds(
                epoch_number=ep,
                family_id=family_id,
                seeds=pub.get("seeds", []),
                started_at=pub.get("started_at", ""),
                ended_at=pub.get("ended_at", ""),
                benchmark_version=pub.get("benchmark_version"),
            )
        except Exception as e:
            bt.logging.warning(f"Failed to publish epoch {ep} ({family_id}) seeds: {e}")
            failures[scope] = failures.get(scope, 0) + 1
            skip_until[scope] = current_cycle + _publish_backoff_cycles(failures[scope])
            continue

        accepted = isinstance(response, dict) and (
            response.get("published") or response.get("accepted")
        )
        if accepted:
            _seed_manager_method(
                self.seed_manager,
                "mark_epoch_published",
                ep,
                family_id=family_id,
            )
            failures.pop(scope, None)
            skip_until.pop(scope, None)
            bt.logging.info(f"Published epoch {ep} ({family_id}) seeds to backend")
        else:
            failures[scope] = failures.get(scope, 0) + 1
            backoff = _publish_backoff_cycles(failures[scope])
            skip_until[scope] = current_cycle + backoff
            bt.logging.warning(
                f"Backend rejected epoch {ep} ({family_id}) publish "
                f"(attempt {failures[scope]}); "
                f"retry in {backoff} cycle(s): {response}"
            )

    self._epoch_publish_skip_until = skip_until
    self._epoch_publish_failures = failures


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
    family_id: str = DEFAULT_RUNTIME_FAMILY_ID,
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
        family_id=family_id,
    )

    if response.get("recorded", False):
        return True, False, ""

    terminal, reason = classify_backend_failure(response, "score")
    return False, terminal, reason


# ──────────────────────────────────────────────────────────────────────────
# Queue worker – processes a single normal-model pipeline item
# ──────────────────────────────────────────────────────────────────────────
