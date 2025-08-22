"""
Swarm Validator Rewards System

A hardcoded reward distribution system for subnet validators that provides
fair and predictable reward allocation across different performance tiers.

Key Features:
- Fixed percentage allocation for top 5 performers
- Balanced distribution for ranks 6-100 with minimum guarantees
- Proportional redistribution of zero-score rewards
- Deterministic UID-based tiebreaking for identical scores
- Support for up to 100 rewarded miners (ranks 101+ receive 0%)

"""

import numpy as np
from typing import Tuple, Dict, List, Any
import logging

# Configure logging
logger = logging.getLogger(__name__)

# System constants
MAX_REWARDED_MINERS = 100
MIN_SCORE_THRESHOLD = 0.0
TIEBREAK_EPSILON = 0.9999


class RewardDistributionConfig:
    """Configuration class for reward distribution parameters."""
    
    # Hardcoded percentages for top performers
    TOP_TIER_PERCENTAGES = {
        1: 0.20,    # 20.00%
        2: 0.15,    # 15.00%
        3: 0.12,    # 12.00%
        4: 0.08,    #  8.00%
        5: 0.0425   #  4.25%
    }
    
    # Distribution parameters for ranks 6-100
    LINEAR_WEIGHT = 0.3         # Weight of linear decay component
    EXPONENTIAL_WEIGHT = 0.7    # Weight of exponential decay component
    EXPONENTIAL_DECAY = 0.015   # Exponential decay rate
    MINIMUM_RANK_100_WEIGHT = 0.0022  # Minimum weight for rank 100
    
    @classmethod
    def get_top_tier_total(cls) -> float:
        """Calculate total percentage allocated to top 5 ranks."""
        return sum(cls.TOP_TIER_PERCENTAGES.values())
    
    @classmethod
    def get_remaining_allocation(cls) -> float:
        """Calculate remaining percentage for ranks 6-100."""
        return 1.0 - cls.get_top_tier_total()


def _validate_inputs(uids: np.ndarray, raw_scores: np.ndarray) -> Dict[str, Any]:
    """
    Validate input arrays and return error information if invalid.
    
    Args:
        uids: Array of miner unique identifiers
        raw_scores: Array of performance scores
        
    Returns:
        Dictionary with validation results and error information
    """
    if len(uids) == 0 or len(raw_scores) == 0:
        return {"valid": False, "error": "Empty input arrays"}
    
    if len(uids) != len(raw_scores):
        return {"valid": False, "error": f"Input length mismatch: {len(uids)} UIDs vs {len(raw_scores)} scores"}
    
    # Check for valid numeric data
    valid_mask = np.isfinite(raw_scores) & np.isfinite(uids.astype(float))
    if not np.any(valid_mask):
        return {"valid": False, "error": "No valid numeric data found"}
    
    return {"valid": True, "valid_mask": valid_mask}


def _sort_miners_by_performance(uids: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sort miners by performance with deterministic tiebreaking.
    
    Primary sort: Score (descending - higher is better)
    Secondary sort: UID (ascending - lower UID wins ties)
    
    Args:
        uids: Array of miner UIDs
        scores: Array of performance scores
        
    Returns:
        Tuple of (sorted_uids, sorted_scores)
    """
    # Sort by score (descending) then by UID (ascending) for deterministic tiebreaking
    sort_indices = np.lexsort((uids, -scores))
    return uids[sort_indices], scores[sort_indices]


def _compute_decay_weights_for_ranks_6_to_100(num_ranks: int, config: RewardDistributionConfig) -> np.ndarray:
    """
    Compute balanced decay weights for ranks 6-100 using hybrid linear-exponential approach.
    
    This function creates a more equitable distribution by combining:
    - Linear decay: Ensures lower ranks receive meaningful rewards
    - Exponential decay: Maintains performance-based differentiation
    
    Args:
        num_ranks: Number of ranks to compute (n_rewarded - 5)
        config: Configuration object with distribution parameters
        
    Returns:
        Array of unnormalized weights for ranks 6 through n_rewarded
    """
    if num_ranks <= 0:
        return np.array([])
    
    decay_weights = []
    
    for rank_offset in range(num_ranks):
        # Linear component: decreases linearly from 1.0 to ~0.01
        linear_component = (num_ranks - rank_offset) / num_ranks
        
        # Exponential component: gentle exponential decay
        exp_component = np.exp(-config.EXPONENTIAL_DECAY * rank_offset)
        
        # Hybrid weight: weighted combination of both approaches
        weight = (config.LINEAR_WEIGHT * linear_component + 
                 config.EXPONENTIAL_WEIGHT * exp_component)
        decay_weights.append(weight)
    
    decay_weights = np.array(decay_weights, dtype=np.float32)
    
    # Ensure rank 100 gets minimum guaranteed amount (approximately 4-5 TAO)
    if num_ranks >= 95:  # Only apply if we have rank 100
        decay_weights[-1] = max(decay_weights[-1], config.MINIMUM_RANK_100_WEIGHT)
    
    return decay_weights


def _identify_zero_score_miners(scores: np.ndarray, weights: np.ndarray) -> Tuple[List[int], List[int], float]:
    """
    Identify miners with zero scores and calculate total rewards to redistribute.
    
    Args:
        scores: Array of sorted performance scores
        weights: Array of current weight allocations
        
    Returns:
        Tuple of (zero_score_indices, non_zero_indices, total_zero_rewards)
    """
    zero_score_indices = []
    non_zero_score_indices = []
    total_zero_rewards = 0.0
    
    for i in range(len(scores)):
        if scores[i] <= MIN_SCORE_THRESHOLD:
            zero_score_indices.append(i)
            total_zero_rewards += weights[i]
        else:
            non_zero_score_indices.append(i)
    
    return zero_score_indices, non_zero_score_indices, total_zero_rewards


def _redistribute_zero_rewards_proportionally(weights: np.ndarray, 
                                            zero_indices: List[int],
                                            non_zero_indices: List[int], 
                                            total_zero_rewards: float) -> None:
    """
    Redistribute rewards from zero-score miners proportionally among performing miners.
    
    This maintains the relative reward structure while ensuring zero performers get nothing.
    Each non-zero performer receives additional rewards proportional to their existing allocation.
    
    Args:
        weights: Array of weight allocations (modified in-place)
        zero_indices: Indices of miners with zero scores
        non_zero_indices: Indices of miners with non-zero scores
        total_zero_rewards: Total amount to redistribute
    """
    if total_zero_rewards <= 0.0 or len(non_zero_indices) == 0:
        return
    
    # Zero out the weights for zero-score miners
    for idx in zero_indices:
        weights[idx] = 0.0
    
    # Calculate current total of non-zero weights for proportional distribution
    non_zero_weights = weights[non_zero_indices]
    non_zero_total = non_zero_weights.sum()
    
    if non_zero_total > 0:
        # Redistribute proportionally based on existing weights
        for idx in non_zero_indices:
            proportion = weights[idx] / non_zero_total
            weights[idx] += total_zero_rewards * proportion
    
    logger.info(f"Redistributed {total_zero_rewards:.6f} from {len(zero_indices)} zero-score miners "
                f"to {len(non_zero_indices)} performing miners")


def _enforce_strict_ranking_order(weights: np.ndarray) -> None:
    """
    Ensure strictly decreasing weights to maintain ranking integrity.
    
    Applies minimal adjustments to prevent identical weights while preserving
    the overall distribution structure.
    
    Args:
        weights: Array of weight allocations (modified in-place)
    """
    if len(weights) <= 1:
        return
    
    for i in range(1, len(weights)):
        if weights[i] >= weights[i-1]:
            # Apply minimal decrease to maintain strict ordering
            weights[i] = weights[i-1] * TIEBREAK_EPSILON


def compute_hardcoded_weights(uids: np.ndarray, raw_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Compute reward weights using the hardcoded distribution system.
    
    This function implements a fair and predictable reward distribution:
    1. Fixed percentages for top 5 performers (20%, 15%, 12%, 8%, 4.25%)
    2. Balanced decay for ranks 6-100 ensuring reasonable minimum rewards
    3. Proportional redistribution of zero-score rewards
    4. Strict ranking enforcement with UID tiebreaking
    
    Args:
        uids: Array of miner unique identifiers
        raw_scores: Array of performance scores (higher = better)
        
    Returns:
        Tuple containing:
        - sorted_uids: UIDs ordered by performance (best to worst)
        - weights: Normalized reward weights (sum = 1.0)
        - debug_info: Dictionary with allocation statistics and metadata
        
    Raises:
        No exceptions raised - returns empty arrays and error info for invalid inputs
    """
    config = RewardDistributionConfig()
    
    # Step 1: Validate inputs
    validation = _validate_inputs(uids, raw_scores)
    if not validation["valid"]:
        logger.error(f"Input validation failed: {validation['error']}")
        return np.array([]), np.array([]), {"error": validation["error"]}
    
    # Step 2: Clean and sort data
    valid_mask = validation["valid_mask"]
    clean_uids = uids[valid_mask]
    clean_scores = raw_scores[valid_mask]
    
    sorted_uids, sorted_scores = _sort_miners_by_performance(clean_uids, clean_scores)
    
    # Step 3: Apply reward limit (top 100 only)
    n_total = len(sorted_uids)
    n_rewarded = min(MAX_REWARDED_MINERS, n_total)
    
    if n_rewarded < n_total:
        sorted_uids = sorted_uids[:n_rewarded]
        sorted_scores = sorted_scores[:n_rewarded]
        logger.info(f"Applied reward limit: {n_rewarded}/{n_total} miners will receive rewards")
    
    # Step 4: Initialize weight allocation
    weights = np.zeros(n_rewarded, dtype=np.float32)
    
    # Step 5: Assign hardcoded percentages for top 5
    for rank, percentage in config.TOP_TIER_PERCENTAGES.items():
        if rank <= n_rewarded:
            weights[rank - 1] = percentage
    
    # Step 6: Compute decay weights for ranks 6-100
    if n_rewarded > 5:
        num_decay_ranks = n_rewarded - 5
        decay_weights = _compute_decay_weights_for_ranks_6_to_100(num_decay_ranks, config)
        
        # Normalize decay weights to use remaining allocation
        remaining_allocation = config.get_remaining_allocation()
        if decay_weights.sum() > 0:
            decay_weights = decay_weights / decay_weights.sum() * remaining_allocation
        
        # Assign decay weights to positions 6 through n_rewarded
        weights[5:5+len(decay_weights)] = decay_weights
    
    # Step 7: Handle zero-score redistribution
    zero_indices, non_zero_indices, total_zero_rewards = _identify_zero_score_miners(
        sorted_scores, weights)
    
    if total_zero_rewards > 0:
        _redistribute_zero_rewards_proportionally(
            weights, zero_indices, non_zero_indices, total_zero_rewards)
    
    # Step 8: Enforce strict ranking order
    _enforce_strict_ranking_order(weights)
    
    # Step 9: Final normalization
    total_weight = weights.sum()
    if total_weight > 0:
        weights = weights / total_weight
    else:
        logger.warning("Total weight is zero after processing")
    
    # Step 10: Compile debug information
    debug_info = {
        "n_total": n_total,
        "n_rewarded": n_rewarded,
        "n_excluded": max(0, n_total - n_rewarded),
        "zero_score_miners": len(zero_indices),
        "non_zero_miners": len(non_zero_indices),
        "zero_redistribution_amount": total_zero_rewards,
        "top_tier_allocation": config.get_top_tier_total(),
        "decay_tier_allocation": config.get_remaining_allocation(),
        "final_weight_sum": weights.sum(),
        "winner_percentage": weights[0] * 100 if len(weights) > 0 else 0.0,
        "rank_100_percentage": weights[-1] * 100 if len(weights) == 100 else None
    }
    
    logger.info(f"Reward computation completed: {n_rewarded} miners, "
                f"{len(zero_indices)} zero scores, weight sum: {weights.sum():.8f}")
    
    return sorted_uids, weights, debug_info


def compute_tiered_weights(uids: np.ndarray, raw_scores: np.ndarray, cfg=None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Backwards compatibility wrapper for the hardcoded rewards system.
    
    This function maintains API compatibility with the previous tiered system
    while using the new hardcoded distribution approach.
    
    Args:
        uids: Array of miner unique identifiers
        raw_scores: Array of performance scores
        cfg: Configuration object (ignored, maintained for compatibility)
        
    Returns:
        Same format as compute_hardcoded_weights()
    """
    return compute_hardcoded_weights(uids, raw_scores)


# Export the main computation functions
__all__ = ['compute_hardcoded_weights', 'compute_tiered_weights', 'RewardDistributionConfig']