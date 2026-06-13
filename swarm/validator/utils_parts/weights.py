from ._shared import *

from swarm.constants import UID_ZERO
from swarm.validator import koth as _koth


_ADVISORY_DIVERGENCE_EPS: float = 1e-3
_advisory_warn_state: Dict[str, float] = {"last_log_ts": 0.0}
_ADVISORY_WARN_INTERVAL_SEC: float = 60.0


def _parse_king_entries(rows: Any) -> list:
    entries = []
    for raw in rows or []:
        if not isinstance(raw, dict):
            continue
        try:
            entries.append(_koth.KingEntry.from_sync_dict(raw))
        except _koth.MalformedKingEntry as exc:
            bt.logging.warning(f"KotH: dropping malformed king entry: {exc}")
    return entries


def _make_uid_validator(metagraph: Any):
    """is_valid_uid for the combine: reject UID0 / out-of-range / re-registered
    hotkeys so their share burns. Falls back to UID0-only when no metagraph.

    Bounds-checks against len(hotkeys), and treats any lookup error (e.g. a
    list/np-array shape mismatch during a metagraph refresh) as invalid -> burn,
    so a transient metagraph state can never raise and abort weight setting.
    """
    if metagraph is None:
        return _koth.default_uid_validator
    hotkeys = list(getattr(metagraph, "hotkeys", []) or [])
    n = len(hotkeys)

    def _valid(entry: "_koth.KingEntry") -> bool:
        uid = int(entry.uid)
        if uid == UID_ZERO or uid < 0 or uid >= n:
            return False
        try:
            return hotkeys[uid] == entry.hotkey
        except (IndexError, TypeError):
            return False

    return _valid


def compute_koth_weights_from_sync(
    sync_data: Dict[str, Any], *, metagraph: Any = None
) -> Dict[int, float]:
    """Local {uid: weight} from the sync payload; advisory backend weights ignored.

    Per-family when the payload carries ``kings_by_family`` + ``family_shares``
    (weight = Σ family_share · koth_share); otherwise falls back to the flat
    single-family ``kings`` list.
    """
    family_shares = sync_data.get("family_shares") or {}
    kings_by_family_raw = sync_data.get("kings_by_family") or {}

    # A modern payload always carries kings_by_family; use the per-family path
    # even when family_shares is empty (every family score-gated to zero -> full
    # burn), so we converge with the backend instead of falling back to the flat
    # window. Only a legacy payload (no kings_by_family) takes the flat path.
    if kings_by_family_raw:
        kings_by_family = {
            str(fid): _parse_king_entries(rows)
            for fid, rows in kings_by_family_raw.items()
        }
        shares = {str(k): float(v) for k, v in family_shares.items()}
        combined = _koth.combine_family_weights(
            shares, kings_by_family, is_valid_uid=_make_uid_validator(metagraph)
        )
        local_weights = combined.miner_raw
    else:
        local_weights = _koth.compute_weights(_parse_king_entries(sync_data.get("kings")))

    if not sync_data.get("fallback"):
        advisory = sync_data.get("weights") or {}
        if advisory:
            # Soft, rate-limited diagnostic. Some divergence is EXPECTED and
            # correct: the validator applies live-metagraph chain identity
            # (re-registered-hotkey detection) that the backend cannot, so the
            # validator may legitimately burn a king's share the backend kept.
            # Eligibility ownership: backend owns repo-eligibility (already
            # filtered in kings_by_family); validator owns chain identity.
            _maybe_warn_advisory_divergence(advisory, local_weights)

    return local_weights


def stamp_local_weights_on_kings(
    kings_payload: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Replace each entry's `weight` with validator-computed per-row share."""
    if not kings_payload:
        return []
    entries: List[_koth.KingEntry] = []
    valid_index: List[int] = []
    for i, raw in enumerate(kings_payload):
        if not isinstance(raw, dict):
            continue
        try:
            entries.append(_koth.KingEntry.from_sync_dict(raw))
            valid_index.append(i)
        except _koth.MalformedKingEntry:
            continue

    if entries:
        row_weights = _koth.compute_row_weights(entries)
        by_orig: Dict[int, float] = dict(zip(valid_index, row_weights))
    else:
        by_orig = {}

    stamped: List[Dict[str, Any]] = []
    for i, raw in enumerate(kings_payload):
        if not isinstance(raw, dict):
            stamped.append({})
            continue
        entry = dict(raw)
        if i in by_orig:
            entry["weight"] = float(by_orig[i])
        else:
            entry["weight"] = 0.0
            entry["local_weight_error"] = "malformed"
        stamped.append(entry)
    return stamped


def _maybe_warn_advisory_divergence(
    advisory: Dict[Any, Any], local: Dict[int, float]
) -> None:
    """Rate-limited warning when backend-advisory weights diverge from local."""
    try:
        adv_norm: Dict[int, float] = {}
        for k, v in advisory.items():
            try:
                adv_norm[int(k)] = float(v)
            except (ValueError, TypeError):
                continue
    except AttributeError:
        return

    diverged = False
    all_uids = set(adv_norm) | set(local)
    for uid in all_uids:
        if abs(adv_norm.get(uid, 0.0) - local.get(uid, 0.0)) > _ADVISORY_DIVERGENCE_EPS:
            diverged = True
            break
    if not diverged:
        return

    import time as _time
    now = _time.time()
    if now - _advisory_warn_state["last_log_ts"] < _ADVISORY_WARN_INTERVAL_SEC:
        return
    _advisory_warn_state["last_log_ts"] = now
    bt.logging.warning(
        "KotH advisory divergence: backend weights differ from "
        f"local-computed weights beyond {_ADVISORY_DIVERGENCE_EPS}. "
        f"backend={adv_norm} local={local}"
    )


def _apply_backend_weights_to_scores(self, backend_weights: Dict[Any, Any]) -> None:
    """Apply backend weights to validator scores with deterministic reset."""
    scores_lock = getattr(self, "_scores_lock", None)
    if scores_lock is not None:
        with scores_lock:
            _apply_backend_weights_to_scores_unlocked(self, backend_weights)
    else:
        _apply_backend_weights_to_scores_unlocked(self, backend_weights)

    mark_ready = getattr(self, "_mark_weights_ready_for_setting", None)
    if mark_ready is not None:
        mark_ready()


def _apply_backend_weights_to_scores_unlocked(self, backend_weights: Dict[Any, Any]) -> None:
    """Apply the per-family KotH miner map with explicit burn.

    Each weight is already an ABSOLUTE fraction of total emissions
    (family_allocation × koth_share), so miner[uid] = raw[uid] directly and
    UID0 = 1 - Σ miner. No keep-fraction scaling and no proportional rescale:
    an unallocated family slice, a warming-up family's throttled remainder, an
    empty family (absent from backend_weights), and any invalid row (burn UID /
    out-of-range / non-finite) all genuinely burn rather than being
    redistributed onto the surviving miners.
    """
    self.scores = np.zeros(self.metagraph.n, dtype=np.float32)

    distributed = 0.0
    for uid_str, weight in (backend_weights or {}).items():
        try:
            uid = int(uid_str)
            raw = float(weight)
        except (ValueError, TypeError):
            continue
        if uid == UID_ZERO or uid < 0 or uid >= self.metagraph.n:
            continue
        if not np.isfinite(raw) or raw <= 0.0:
            continue
        self.scores[uid] = raw
        distributed += raw

    if distributed > 1.0:
        self.scores /= distributed
        distributed = 1.0

    if BURN_EMISSIONS and 0 <= UID_ZERO < self.metagraph.n:
        self.scores[UID_ZERO] = max(0.0, 1.0 - float(distributed))
