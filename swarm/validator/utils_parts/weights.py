from ._shared import *

from swarm.validator import koth as _koth


_ADVISORY_DIVERGENCE_EPS: float = 1e-3
_advisory_warn_state: Dict[str, float] = {"last_log_ts": 0.0}
_ADVISORY_WARN_INTERVAL_SEC: float = 60.0


def compute_koth_weights_from_sync(sync_data: Dict[str, Any]) -> Dict[int, float]:
    """Local {uid: weight} from sync kings; advisory backend weights ignored."""
    raw_kings = sync_data.get("kings") or []
    entries = []
    for raw in raw_kings:
        if not isinstance(raw, dict):
            continue
        try:
            entries.append(_koth.KingEntry.from_sync_dict(raw))
        except _koth.MalformedKingEntry as exc:
            bt.logging.warning(f"KotH: dropping malformed king entry: {exc}")
            continue

    local_weights = _koth.compute_weights(entries)

    if not sync_data.get("fallback"):
        advisory = sync_data.get("weights") or {}
        if advisory:
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
    self.scores = np.zeros(self.metagraph.n, dtype=np.float32)

    if not backend_weights:
        if BURN_EMISSIONS and 0 <= UID_ZERO < self.metagraph.n:
            self.scores[UID_ZERO] = 1.0
        return

    uids_list = []
    weights_list = []

    for uid_str, weight in backend_weights.items():
        try:
            uid = int(uid_str)
            parsed_weight = float(weight)
            if uid < 0 or uid >= self.metagraph.n:
                continue
            uids_list.append(uid)
            weights_list.append(parsed_weight)
        except (ValueError, TypeError):
            continue

    if not uids_list:
        if BURN_EMISSIONS and 0 <= UID_ZERO < self.metagraph.n:
            self.scores[UID_ZERO] = 1.0
        return

    uids_np = np.array(uids_list, dtype=np.int64)
    weights_np = np.array(weights_list, dtype=np.float32)

    if BURN_EMISSIONS and UID_ZERO not in uids_np and 0 <= UID_ZERO < self.metagraph.n:
        total_weight = weights_np.sum()
        if total_weight > 0:
            weights_np *= KEEP_FRACTION / total_weight
        uids_np = np.concatenate(([UID_ZERO], uids_np))
        weights_np = np.concatenate(([BURN_FRACTION], weights_np))

    self.scores[uids_np] = weights_np
