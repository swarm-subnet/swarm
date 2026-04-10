from ._shared import *


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
