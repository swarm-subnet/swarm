import random
import json
import bittensor as bt
import numpy as np
from typing import List
from pathlib import Path


def is_low_performer(uid: int) -> bool:
    """Check if UID is a low performer based on recent evaluation history.

    Args:
        uid: UID to check

    Returns:
        True if UID is a low performer and should be filtered, False otherwise
    """
    from swarm.constants import (
        LOW_PERFORMER_FILTER_ENABLED,
        MIN_AVG_SCORE_THRESHOLD,
        MIN_EVALUATION_RUNS
    )

    if not LOW_PERFORMER_FILTER_ENABLED:
        return False

    history_file = Path("/tmp/victory_history.json")
    if not history_file.exists():
        return False

    try:
        with open(history_file, 'r') as f:
            history = json.load(f)

        uid_str = str(uid)
        if uid_str not in history:
            return False

        runs = history[uid_str].get("runs", [])
        if len(runs) < MIN_EVALUATION_RUNS:
            return False

        recent_runs = runs[-MIN_EVALUATION_RUNS:]
        scores = [run.get("score", 0.0) for run in recent_runs]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        return avg_score < MIN_AVG_SCORE_THRESHOLD

    except Exception as e:
        bt.logging.debug(f"Error checking low performer status for UID {uid}: {e}")
        return False


def check_uid_availability(
    metagraph: "bt.metagraph.Metagraph", uid: int, vpermit_tao_limit: int
) -> bool:
    """Check if uid is available. The UID should be available if it is serving and has less than vpermit_tao_limit stake
    Args:
        metagraph (:obj: bt.metagraph.Metagraph): Metagraph object
        uid (int): uid to be checked
        vpermit_tao_limit (int): Validator permit tao limit
    Returns:
        bool: True if uid is available, False otherwise
    """
    # Filter non serving axons.
    if not metagraph.axons[uid].is_serving:
        return False
    # Filter validator permit > 1024 stake.
    if metagraph.validator_permit[uid]:
        if metagraph.S[uid] > vpermit_tao_limit:
            return False
    # Available otherwise.
    return True


def get_random_uids(self, k: int, exclude: List[int] = None) -> np.ndarray:
    """Returns k available random uids from the metagraph.
    Args:
        k (int): Number of uids to return.
        exclude (List[int]): List of uids to exclude from the random sampling.
    Returns:
        uids (np.ndarray): Randomly sampled available uids.
    Notes:
        If `k` is larger than the number of available `uids`, set `k` to the number of available `uids`.
    """
    candidate_uids = []
    avail_uids = []
    filtered_low_performers = []

    for uid in range(self.metagraph.n.item()):
        uid_is_available = check_uid_availability(
            self.metagraph, uid, self.config.neuron.vpermit_tao_limit
        )
        uid_is_not_excluded = exclude is None or uid not in exclude
        uid_is_low_performer = is_low_performer(uid)

        if uid_is_available:
            avail_uids.append(uid)
            if uid_is_not_excluded and not uid_is_low_performer:
                candidate_uids.append(uid)
            elif uid_is_low_performer:
                filtered_low_performers.append(uid)

    if filtered_low_performers:
        bt.logging.info(f"Filtered {len(filtered_low_performers)} low-performer UIDs: {filtered_low_performers[:10]}{'...' if len(filtered_low_performers) > 10 else ''}")

    k = min(k, len(avail_uids))
    available_uids = candidate_uids
    if len(candidate_uids) < k:
        available_uids += random.sample(
            [uid for uid in avail_uids if uid not in candidate_uids],
            k - len(candidate_uids),
        )
    uids = np.array(random.sample(available_uids, k))
    return uids
