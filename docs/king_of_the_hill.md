<a id="koth-top"></a>

# King of the Hill

How emissions are distributed on Swarm Subnet 124.

This document describes how the King of the Hill (KotH) emissions mechanism works.

---

<details>
  <summary><b>Table of Contents</b></summary>
  <ol>
    <li><a href="#what-koth-is">What KotH is</a></li>
    <li><a href="#why-it-exists">Why it exists</a></li>
    <li><a href="#the-5-king-window">The 5-king window</a></li>
    <li><a href="#how-each-kings-share-is-calculated">How each king's share is calculated</a></li>
    <li><a href="#edge-cases">Edge cases</a></li>
    <li><a href="#faq">FAQ</a></li>
    <li><a href="#glossary">Glossary</a></li>
  </ol>
</details>

---

## What KotH is

When a benchmark cycle finishes and a new model is crowned champion, that model does not receive 100% of subnet emissions. Instead, **the last 5 champions share emissions**, with each one's slice proportional to how much they improved the network's best score when they took the throne.

- The **current champion** is always at the top of the lineage.
- The **four most recent past champions** continue earning until they age out of the window after subsequent dethronings.
- Each king's share is computed once at crowning time and never recomputed.

<p align="right">(<a href="#koth-top">back to top</a>)</p>

---

## Why it exists

Winner-take-all has two failure modes that KotH addresses:

1. **Copycat models.** Under winner-take-all, a miner can clone the current champion, add `0.015` of noise to pass the crowning floor, and take 100% of emissions without contributing real innovation. Under KotH, that miner's tiny jump translates to a tiny share — most of the emissions stay with the past kings whose jumps were larger.

2. **Innovation goes unpaid.** Under winner-take-all, the miner who pushed the network from 0.85 to 0.92 is forgotten the moment someone nudges it to 0.93. Under KotH, that 0.07 jump keeps paying for the next four dethronings — proportional to the real contribution.

KotH rewards **the act of moving the frontier**, not just the act of sitting on it.

<p align="right">(<a href="#koth-top">back to top</a>)</p>

---

## The 5-king window

The window holds **exactly 5 entries**: the current king plus the four most recent past kings.

```
Rank        Slot                          Earning
─────────────────────────────────────────────────────
 0          Reigning (current)            yes
−1          1 dethroning ago              yes
−2          2 dethronings ago             yes
−3          3 dethronings ago             yes
−4          4 dethronings ago             yes — rotates out next
                                          dethroning
```

After the next crowning, the king at slot `−4` leaves the window and stops earning. The new king takes slot `0`, every other king shifts one slot down.

A past king's hotkey **does not need to do anything** to keep earning — they simply stay in the lineage. Past kings are never re-evaluated; their model file may even be deleted from GitHub without affecting their share.

<p align="right">(<a href="#koth-top">back to top</a>)</p>

---

## How each king's share is calculated

Each king's share depends on **two things measured at the moment they were crowned**:

```
1.  How much they improved the score
2.  How hard that improvement was to make
```

The "how hard" part recognises that improving from `0.20 → 0.25` is easier than improving from `0.80 → 0.85`. There is less remaining headroom near the top, so each percentage point closer to perfect is rarer.

### The formula

For each king `i` in the 5-king window, with their score `score_i` and the previous king's score `prev_i`:

```
delta_i    = max(0, score_i − prev_i)                     # raw improvement, clamped
adjusted_i = delta_i / max(1 − prev_i, 0.05)              # headroom-adjusted, capped
weight_i   = adjusted_i / sum(adjusted_j in window)        # normalised to share of 100 %
```

The cap (`0.05`) prevents the formula from blowing up as scores approach 1.0. The maximum effective multiplier is 1 / 0.05 = 20×.

### Plain-English version

- **Compute how much each king improved the score** (their `delta`).
- **Divide that improvement by the remaining headroom** when they were crowned. A small jump near 1.0 turns into a big number; a small jump from a low baseline turns into a small number.
- **Add up everyone's adjusted numbers and divide each one by the total** to get their share.

<p align="right">(<a href="#koth-top">back to top</a>)</p>

---

## Edge cases

### The first king ever

When the subnet has zero past champions, the first king's `prev_score` is treated as `0`. Their `delta` equals their full score, and they take 100% of emissions until someone dethrones them.

### A king reaches the perfect score (1.0)

If a king's score is at or extremely close to 1.0, subsequent jumps look like dividing by zero. The cap of `0.05` in the formula bounds the headroom-adjusted multiplier at 20×, so no single late jump can take more than ~95% of the window.

### Backend unreachable

When the validator cannot reach the backend to fetch the active king window, the validator's existing emission-burn safety mechanism takes over — share is routed to UID 0 and effectively burned for that cycle. No bad weights are set. As soon as the backend recovers, normal KotH weights resume.

<p align="right">(<a href="#koth-top">back to top</a>)</p>

---

## FAQ

### How often is my share recalculated?

Never, once you have been crowned. Your `delta` and `adjusted` values are computed at the moment of crowning and locked. The only thing that changes about your share over time is the denominator (the sum of all 5 kings' `adjusted` values) — that changes only when a new king joins the window or you age out of it.

### What if I get dethroned?

You stay in the window at rank `−1`. You keep earning a share for the next four dethronings, after which you age out at rank `−4`.

### Can I become king twice?

No. The subnet enforces **one model per hotkey, lifetime**. A hotkey that has been crowned cannot submit a second model. If you want to compete again, register a new hotkey.

### What happens to a king who deletes their GitHub repo?

Their share continues to flow to their hotkey. Past kings are never re-evaluated; the share is locked in at crowning. The on-chain emission goes to the original hotkey regardless of whether the public repo is still available.

### Why is there a minimum jump (0.015) to take the throne?

The crowning floor (`champion + 0.015`) is an anti-noise threshold. Without it, the network would re-elect a "new" champion every time a benchmark produced a 0.0001 score variance. The floor is unchanged by KotH.

### What if the subnet emission rate changes?

KotH only determines **how** subnet emissions are split among the 5 kings. The total subnet emission rate is set by Bittensor consensus and is independent of KotH. If total subnet emissions go up, every king's share-of-100% stays the same but their dollar value rises proportionally.

### Can a coordinated team take multiple king slots?

Yes — this is a known limit of the V5.0.0 design. A team running multiple hotkeys could in principle release staged improvements across them to occupy several king slots. The "one model per hotkey, lifetime" rule limits but does not eliminate this. Detection is hard without invasive on-chain checks; the design choice for V5.0.0 is to accept this risk and revisit in a future release if it is observed.

<p align="right">(<a href="#koth-top">back to top</a>)</p>

---

## Glossary

| Term | Meaning |
|---|---|
| **King** | A model that took the throne by passing the screening + benchmark and beating the previous champion by ≥ 0.015. |
| **Lineage** | The permanent ordered list of every king ever, stored by the backend. |
| **Active window** | The current 5 kings whose shares are summed and used for emissions. |
| **Headroom** | The distance from the previous king's score to the perfect score of 1.0. The "room left to grow". |
| **Jump** | The absolute score improvement when a king was crowned (`score − prev_score`). |
| **Headroom-adjusted jump** | The jump divided by the remaining headroom, capped to prevent singularity. |
| **Share** | The percentage of subnet emissions a king receives. Sums to 100% across all 5 active kings. |
| **Aging out** | When a king reaches rank `−5` (i.e., five dethronings have happened since they took the throne) and leaves the window. |
| **Crowning floor** | The fixed minimum improvement (0.015) required to dethrone the current champion. |

<p align="right">(<a href="#koth-top">back to top</a>)</p>
