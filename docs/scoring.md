<a id="scoring-top"></a>

# Scoring

How a model's score is computed by each validator and combined by the backend.

---

<details>
  <summary><b>Table of Contents</b></summary>
  <ol>
    <li><a href="#what-a-score-is">What a score is</a></li>
    <li><a href="#how-each-validator-scores-locally">How each validator scores locally</a></li>
    <li><a href="#how-the-backend-combines-validator-scores">How the backend combines validator scores</a></li>
    <li><a href="#the-screening-pass-rule">The screening pass rule</a></li>
    <li><a href="#glossary">Glossary</a></li>
  </ol>
</details>

---

## What a score is

A score is a single number in `[0, 1]` representing a model's average performance across a fixed set of evaluation seeds. Each validator runs every seed in an isolated Docker container, evaluates the per-seed result, and averages over the seeds it ran.

The same model evaluated on different validators may produce slightly different numbers because of small differences in docker timing, OS scheduling, and other host-side noise. The backend resolves these into one network-level score.

<p align="right">(<a href="#scoring-top">back to top</a>)</p>

---

## How each validator scores locally

For a given model and a given phase (screening or benchmark), each validator:

1. Receives a task with the seed range to evaluate.
2. Runs every seed and records a per-seed score.
3. Averages the per-seed scores into a phase score.
4. Reports the phase score to the backend.

There is one consensus set of validators whose reports are counted toward the network's decision. Reports from outside that set are stored for audit but ignored for aggregation.

<p align="right">(<a href="#scoring-top">back to top</a>)</p>

---

## How the backend combines validator scores

For a model with reports from validators `i ∈ submitters`:

```
weighted_score = Σ (stake_i × score_i)  /  Σ stake_i        (sums over submitters)
```

The denominator is the sum of stakes of the validators that actually reported, not the total consensus stake. A validator that did not submit does not appear in either side of the ratio.

The aggregate is the **stake-weighted average score across the validators that rated this model**. A validator with 10× more stake counts 10× more in the average.

<p align="right">(<a href="#scoring-top">back to top</a>)</p>

---

## The screening pass rule

A challenger model must pass two independent gates to advance from screening:

```
Gate 1 — QUORUM (validators must have reported enough)
    Σ stake_i  over submitters  >=  0.51 × Σ stake_j  over the consensus set

Gate 2 — IMPROVEMENT (the weighted score must clear the threshold)
    weighted_score  >=  current_champion.benchmark_score  +  floor(champion_score)
```

If no champion exists yet, Gate 2 falls back to `weighted_score >= 0.01`.

The **improvement floor** is not a constant; it relaxes as the champion's score climbs:

```
champion_score <= 0.50                   floor = 0.015   (flat anti-spam plateau)
champion_score  > 0.50                   floor = 0.005 + 0.010 × (1 − t²)
                                                   where t = (champion_score − 0.5) / 0.5
```

```
champion_score    floor
─────────────     ──────
0.00 — 0.50       0.0150     (plateau)
0.60              0.0146
0.70              0.0134
0.80              0.0114
0.85              0.0101
0.90              0.0086
0.95              0.0069
1.00              0.0050
```

Why the curve: in the bottom half (champion below 0.50) we keep the bar at the full 0.015 to discourage farming on a weak champion. Once the champion crosses 0.50, less headroom is left to fight for, so the bar smoothly relaxes — gently at first, then steeper near 1.0 where every fractional point is hard-won.

The two denominators are different on purpose: the **score** is averaged over the submitters only (a non-submitter contributes nothing), while the **quorum** is measured against the full consensus set (a non-submitter counts as missing).

<p align="right">(<a href="#scoring-top">back to top</a>)</p>

---

## Glossary

| Term | Meaning |
|---|---|
| **Consensus set** | The fixed set of validator hotkeys whose reports are counted toward network decisions. |
| **Submitter** | A consensus-set validator that has reported a phase score for a given model and epoch. |
| **Stake** | The on-chain Bittensor stake of a validator, as read from the network metagraph. |
| **Weighted score** | Submitter-only stake-weighted mean of validator-reported scores. |
| **Quorum** | The condition that the submitters' combined stake is at least 51 % of the consensus-set stake. |
| **Improvement floor** | The minimum gap a challenger's weighted score must show above the current champion's benchmark score. Flat at `0.015` while the champion is below `0.50`, then smoothly decays to `0.005` at champion score `1.00`. |

<p align="right">(<a href="#scoring-top">back to top</a>)</p>
