#!/usr/bin/env python
# analyse_realistic_batch.py
#
# Compares two boosting rules on a “more‑realistic” batch:
#
#   • 10 miners score 0.0
#   • 10 miners score 0.1
#   • the remaining miners (to reach 256 total) are **linearly spaced
#     between 0.1 and 0.9**           ← slight duplication of 0.1 is OK
#
# Boosting rules:
#   1. _boost_scores          (β = 5, σ on FULL batch)
#   2. _boost_scores_trimmed  (β = 2, σ on TOP‑20 % of batch)

import math
import numpy as np
import pandas as pd


# ────────────────────────────────────────────────────────────────
# 1.  Boosting functions
# ────────────────────────────────────────────────────────────────
def _boost_scores(raw: np.ndarray, *, beta: float = 5.0) -> np.ndarray:
    if raw.size == 0:
        return raw
    s_max = float(raw.max())
    sigma = float(raw.std())
    if sigma < 1e-9:
        weights = (raw == s_max).astype(np.float32)
    else:
        weights = np.exp(beta * (raw - s_max) / sigma)
        weights /= weights.max()
    return weights.astype(np.float32)


def _boost_scores_trimmed(
    raw: np.ndarray,
    *,
    beta: float = 2.0,
    top_frac: float = 0.20,
) -> np.ndarray:
    if raw.size == 0:
        return raw
    if not (0.0 < top_frac <= 1.0):
        raise ValueError("top_frac must be in (0, 1]")
    k = max(1, int(math.ceil(top_frac * raw.size)))
    top_scores = np.partition(raw, -k)[-k:]
    sigma = float(top_scores.std())
    s_max = float(raw.max())
    if sigma < 1e-9:
        weights = (raw == s_max).astype(np.float32)
    else:
        weights = np.exp(beta * (raw - s_max) / sigma)
        weights /= weights.max()
    return weights.astype(np.float32)


# ────────────────────────────────────────────────────────────────
# 2.  Build batch
# ────────────────────────────────────────────────────────────────
N_TOTAL            = 256
cluster_zero       = np.full(10, 0.0, dtype=np.float32)        # 10 miners at 0
cluster_point1     = np.full(10, 0.1, dtype=np.float32)        # 10 miners at 0.1
remaining          = N_TOTAL - 20
linear_rest        = np.linspace(0.1, 0.9, remaining,
                                 dtype=np.float32)             # 236 miners
raw_batch          = np.concatenate([cluster_zero,
                                     cluster_point1,
                                     linear_rest])
assert raw_batch.shape == (N_TOTAL,)

# ────────────────────────────────────────────────────────────────
# 3.  Helper – dump diagnostics for a booster
# ────────────────────────────────────────────────────────────────
def dump_stats(name: str,
               weights: np.ndarray,
               raw: np.ndarray,
               beta: float,
               sigma: float) -> None:

    total_w       = float(weights.sum())
    top_share_pct = 100.0 * weights.max() / total_w

    print(f"\n=== {name} (β = {beta}, σ = {sigma:.6e}) =============================")
    print(f"Batch size             : {raw.size}")
    print(f"Sum of weights         : {total_w:.6f}")
    print(f"Top miner share        : {top_share_pct:.3f}%\n")

    # ► Table by raw score (group identical scores)
    df = (
        pd.DataFrame({"score": raw, "weight": weights})
          .groupby("score")
          .agg(
              miners       = ("score",  "count"),
              weight_each  = ("weight", "first"),
              weight_total = ("weight", "sum"),
          )
          .sort_index()
    )
    df["% each"]  = 100.0 * df["weight_each"]  / total_w
    df["% group"] = 100.0 * df["weight_total"] / total_w
    print("Reward distribution by *raw score*:")
    print(df.to_string(float_format=lambda x: f"{x:10.5f}"))
    print()

    # ► Hypothetical curve 0.00 → 0.90
    grid_scores  = np.round(np.arange(0.00, 0.91, 0.01), 2)
    grid_weights = np.exp(beta * (grid_scores - raw.max()) / sigma)
    grid_weights /= grid_weights.max()
    df_grid = pd.DataFrame(
        {"score": grid_scores,
         "weight": grid_weights,
         "% of pool if solo": 100.0 * grid_weights / total_w}
    )
    print("Hypothetical weight curve (step 0.01):")
    print(df_grid.to_string(index=False,
                            float_format=lambda x: f"{x:10.5f}"))


# ────────────────────────────────────────────────────────────────
# 4.  Run both boosters
# ────────────────────────────────────────────────────────────────
w_full   = _boost_scores(raw_batch, beta=5.0)
sigma_full = float(raw_batch.std())

k_top20  = max(1, int(math.ceil(0.20 * raw_batch.size)))
sigma_trim = float(np.partition(raw_batch, -k_top20)[-k_top20:].std())
w_trim   = _boost_scores_trimmed(raw_batch, beta=2.0, top_frac=0.20)

dump_stats("FULL‑σ baseline", w_full,  raw_batch, beta=5.0, sigma=sigma_full)
dump_stats("TRIMMED‑σ (top‑20 %)", w_trim, raw_batch, beta=2.0, sigma=sigma_trim)
