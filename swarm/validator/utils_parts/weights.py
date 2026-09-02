# The MIT License (MIT)
# Copyright © 2026 Swarm

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

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


def accept_sync_version(validator: Any, sync_data: Dict[str, Any]) -> bool:
    """False when this sync payload is older than one already applied.

    Backend weights are absolute, so a response that lost the race would
    overwrite a newer allocation. A refused payload leaves the validator on the
    last weights it accepted, which is also what the offline fallback holds.
    """
    try:
        version = int(sync_data.get("leaderboard_version") or 0)
    except (TypeError, ValueError):
        version = 0
    applied = int(getattr(validator, "_applied_leaderboard_version", 0) or 0)
    if version < applied:
        bt.logging.info(
            f"Ignoring stale backend weights: leaderboard v{version} < applied v{applied}"
        )
        return False
    validator._applied_leaderboard_version = version
    return True


def compute_koth_weights_from_sync(
    sync_data: Dict[str, Any], *, metagraph: Any = None
) -> Optional[Dict[int, float]]:
    """Local {uid: weight} from the sync payload; advisory backend weights ignored.

    Per-family when the payload carries ``kings_by_family`` + ``family_shares``
    (weight = Σ family_slice · koth_share; the slices are absolute, so whatever
    a family does not pay out burns). An empty map is a valid full-burn state. A
    legacy payload without ``kings_by_family`` is refused (returns None) so the
    caller holds its last good weights instead of burning on a bad payload.
    """
    family_shares = sync_data.get("family_shares") or {}
    kings_by_family_raw = sync_data.get("kings_by_family") or {}

    # A modern payload always carries kings_by_family; use the per-family path
    # even when family_shares is empty (no family payable -> full burn), so we
    # converge with the backend. A legacy payload (no kings_by_family) is refused.
    if kings_by_family_raw or "kings_by_family" in sync_data:
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
        if sync_data.get("kings"):
            bt.logging.warning(
                "KotH: legacy flat payload without kings_by_family; refusing "
                "(V5 requires the per-family payload)."
            )
        return None

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
    """Apply the per-family KotH miner map to the score vector.

    Each weight is an ABSOLUTE fraction of total emissions
    (family_slice × koth_share). Every slice the backend did not assign to a
    live miner — kingless families, archived families, dropped shares — goes to
    the burn UID, so ``set_weights``' normalization is a no-op and no unpaid
    slice ever leaks onto the paid miners. An empty map burns everything.
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

    if 0 <= UID_ZERO < self.metagraph.n:
        self.scores[UID_ZERO] = max(0.0, 1.0 - distributed)
