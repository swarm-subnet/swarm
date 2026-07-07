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
    <li><a href="#rank-weighting">Rank weighting</a></li>
    <li><a href="#taking-the-throne--the-dynamic-floor">Taking the throne — the dynamic floor</a></li>
    <li><a href="#per-family-emissions">Per-family emissions</a></li>
    <li><a href="#edge-cases">Edge cases</a></li>
    <li><a href="#faq">FAQ</a></li>
    <li><a href="#glossary">Glossary</a></li>
  </ol>
</details>

---

## What KotH is

Swarm runs **one King of the Hill per challenge family** (e.g. Autopilot, Search-and-Rescue). Each family keeps its own lineage of champions, and **the last 5 champions of that family share that family's slice of emissions**, with each one's slice proportional to how much they improved the family's best score when they took the throne.

- The **current champion** of a family is always at the top of that family's lineage.
- The **four most recent past champions** of the family keep earning until they age out of the window.
- Each king's gain is locked at crowning; the rank weight shifts as newer kings arrive, moving share toward the freshest champions.

How the family slices add up is covered in [Per-family emissions](#per-family-emissions). The within-a-family split below is identical to the original single-competition design.

<p align="right">(<a href="#koth-top">back to top</a>)</p>

---

## Why it exists

Winner-take-all has two failure modes that KotH addresses:

1. **Copycat models.** Under winner-take-all, a miner can clone the current champion, add `0.015` of noise to pass the crowning floor, and take 100% of emissions without contributing real innovation. Under KotH, that miner's tiny jump translates to a tiny share — most of the emissions stay with the past kings whose jumps were larger.

2. **Innovation goes unpaid.** Under winner-take-all, the miner who pushed the network from 0.85 to 0.92 is forgotten the moment someone nudges it to 0.93. Under KotH, that 0.07 jump keeps paying — proportional to the real contribution — for up to four more dethronings.

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

A king keeps earning while two things hold: they are still in the window and their submission repo is still reachable. Past kings are never re-evaluated — the gain is locked at crowning — but rank taper or a dead repo can reduce or switch off the share before they age out.

Taper ranks are assigned among the **payable** kings only. When a seat stops being payable (dead repo or an admin drop), the kings behind it move up one taper step and its slice renormalizes onto the survivors; the rank badge on the ladder still shows window position, so a badge and its paid share can briefly diverge while a seat is unreachable.

<p align="right">(<a href="#koth-top">back to top</a>)</p>

---

## How each king's share is calculated

Each king's share depends on the gain they locked at crowning and their current rank in the family window:

```
1.  How much remaining headroom they closed
2.  How fresh their crown is
```

The gain recognises that improving from `0.20 → 0.25` is easier than improving from `0.80 → 0.85`. There is less remaining headroom near the top, so closing the same fraction of the remaining distance matters more.

### The formula

For each king `i` in the 5-king window, with their score `score_i` and the previous king's score `prev_i`:

```
gain_i   = log( max(1 − prev_i, 0.01) / max(1 − score_i, 0.01) )   # ≥ 0
weight_i = (5 − rank_i) / 5                                        # rank 0 = reigning
share_i  = gain_i × weight_i / sum(gain_j × weight_j in window)
```

The `0.01` floor caps the headroom so improvements above `0.99` do not blow up.

### Plain-English version

- **Measure improvement in log-headroom** — how much of the remaining distance-to-perfect the king closed.
- **Taper by rank** — champion 100%, then 80%, 60%, 40%, 20%.
- **Normalise the tapered gains** so the family window sums to 100%.

<p align="right">(<a href="#koth-top">back to top</a>)</p>

---

## Rank weighting

A king's gain is tapered by where it sits in the family lineage. The reigning champion (rank 0) keeps its full gain; every step further back loses **20%**:

```
weight = (5 − rank) / 5
   rank 0 (champion) → 1.0     rank 2 → 0.6     rank 4 (oldest) → 0.2
```

So the freshest champions earn the most, and a king fades **as new champions are crowned and push it down the window** — not by any clock, and with no hard age cutoff. Every king in the window keeps a share (down to 20% weight at the bottom); a king only stops earning once a sixth crowning pushes it out of the window entirely.

Splitting one improvement across several crownings earns *less* — each earlier piece sits at a lower rank and is weighted down — so there is no advantage to gaming the split.

Rank weighting is separate from the crowning floor below: the floor decides who *takes* the throne, while rank weighting shapes how the throne's *earnings* are split.

<p align="right">(<a href="#koth-top">back to top</a>)</p>

---

## Taking the throne — the dynamic floor

To be crowned, a challenger must clear the current champion by an **improvement floor** that *shrinks* as the champion climbs:

```
champion ≤ 0.5      floor = 0.015     (flat — anti-noise while scores are low)
champion → 1.00     floor → 0.005     (smaller, since every point near the top is hard-won)
```

So a frozen top of the board becomes easier to dethrone, and champions cycle through the window faster. The gate is applied at final crowning (and at screening, when that phase is enabled), and a family's registry policy can override the numbers.

<p align="right">(<a href="#koth-top">back to top</a>)</p>

---

## Per-family emissions

Each family runs its own 5-king window. Three levels decide a UID's final weight:

```
1.  family_share(f)   — how big a slice family f gets of the emission pool
2.  best_score(f)     — how much of that slice is actually earned (0…1)
3.  koth_share(uid,f) — the UID's share WITHIN family f (the formula above)

weight(uid) = sum over families f of  family_share(f) × best_score(f) × koth_share(uid, f)
```

A hotkey that is a king in two families earns from both — the contributions add.

### How a family's slice is sized

`family_share` is **absolute**: each family has an `emission_allocation` — a direct fraction of TOTAL emissions, set by governance (not by miners) — multiplied by a status weight from its **emissions state**. Families are **not** normalised against each other; whatever is left unallocated is **burned**.

```
emissions state    status weight
───────────────    ─────────────
active             1.0
saturated          0.5
incubating         0.25
regression         0.1
archived           0.0   (does not participate)

family_share(f) = emission_allocation(f) × status_weight(f)
```

### Pay for accuracy: the score gate

A family only pays out as much of its slice as its **best model actually earns**. The family's slice is multiplied by `best_score(f)` — the current champion's benchmark score in `0…1` — and the rest **burns**. A family half-solved pays half its slice.

```
paid(f)   = family_share(f) × best_score(f)
burned(f) = family_share(f) × (1 − best_score(f))
```

`best_score` is the champion's **live** score (it refreshes each epoch on re-eval), and it is applied by the backend so every validator gates the same way. The 5-king split is unchanged — `best_score` only decides how much of the slice is paid versus burned, not how the paid part is divided among the kings.

Example — Autopilot allocation `0.10` `active` (1.0) with a champion at score `0.80`, SAR allocation `0.10` `incubating` (0.25) with a champion at `0.50`:

```
family_share(Autopilot) = 0.10 × 1.00          = 0.100
paid(Autopilot)         = 0.100 × 0.80          = 0.080   (8%)
family_share(SAR)       = 0.10 × 0.25           = 0.025
paid(SAR)               = 0.025 × 0.50          = 0.0125  (1.25%)
burned                  = 1 − 0.080 − 0.0125    = 0.9075  (90.75%)
```

Raising, throttling, or archiving one family does **not** change another's share — each family's slice is independent.

### Empty family → its slice burns

If a participating family has **no payable king yet** (e.g. a freshly activated family nobody has won), its slice is **not** redistributed to the other families — it is burned (routed to UID 0) until that family crowns its first king. This keeps each family's reserved emissions reserved.

### How weights reach the chain

The backend serves the **raw kings** (score + previous score) per family plus the family shares. Validators **recompute** the weights locally from those raw numbers and the unchanged formula, then apply the subnet emission burn on top. Because every validator uses the same kings and the same formula, they converge on the same weights without a shared secret.

<p align="right">(<a href="#koth-top">back to top</a>)</p>

---

## Edge cases

### The first king ever

When a family has zero past champions, its first king's `prev_score` is treated as `0`. Their `delta` equals their full score, and they take 100% of **that family's slice** until someone dethrones them.

### A king reaches the perfect score (1.0)

If a king's score is at or extremely close to 1.0, subsequent jumps can otherwise make headroom look like zero. The `0.01` floor bounds the gain at `log(1 / 0.01) ≈ 4.6`, and a jump entirely inside the top `0.01` (for example `0.995 → 1.0`) earns nothing.

### Backend unreachable

When the validator cannot reach the backend, it falls back to the **last per-family snapshot it cached** (kings + family shares) and recomputes weights from that. If it has no cached snapshot, share is routed to UID 0 and effectively burned for that cycle. No bad weights are set. As soon as the backend recovers, normal KotH weights resume.

<p align="right">(<a href="#koth-top">back to top</a>)</p>

---

## FAQ

### How often is my share recalculated?

Your gain is computed at the moment of crowning and locked. Your share moves when the window changes: a new crowning shifts every rank, changes the rank taper, and eventually ages older kings out. Your share also drops to **zero** if your repo becomes unreachable.

### Do I keep earning forever if nobody beats me?

Yes. You fade only as new champions push you down the ranks, and you leave after five crownings. There is no clock.

### What if I get dethroned?

You slide to rank `−1` at 80% taper and keep earning until you age out at rank `−4` after four more dethronings.

### Can I become king twice?

No. The subnet enforces **one model per hotkey, lifetime**. A hotkey that has been crowned cannot submit a second model. If you want to compete again, register a new hotkey.

### What happens to a king who deletes their GitHub repo?

Their slice stops and burns. A king's repo is health-checked; if it is deleted or becomes unreachable the seat is treated like a dropped seat — it keeps its window slot but pays nobody, so its share burns rather than flowing to a model that can no longer be verified.

### Why is there a minimum jump to take the throne?

The crowning floor is an anti-noise threshold: without it, the network would re-elect a "new" champion every time a benchmark produced a 0.0001 score variance. It is **dynamic** — flat at `0.015` while the champion is at or below `0.5`, then shrinking toward `0.005` as the score approaches 1.0 (see [Taking the throne](#taking-the-throne--the-dynamic-floor)).

### What if the subnet emission rate changes?

KotH only determines **how** subnet emissions are split among the 5 kings. The total subnet emission rate is set by Bittensor consensus and is independent of KotH. If total subnet emissions go up, every king's share-of-100% stays the same but their dollar value rises proportionally.

### Can a coordinated team take multiple king slots?

Yes — this is a known limit of the V5.0.0 design. A team running multiple hotkeys could in principle release staged improvements across them to occupy several king slots. The "one model per hotkey, lifetime" rule limits but does not eliminate this. Detection is hard without invasive on-chain checks; the design choice for V5.0.0 is to accept this risk and revisit in a future release if it is observed.

### Can I earn from more than one family?

Yes. Families are independent — a hotkey that is a king in two families collects from both, and the two slices simply add. There is one model per hotkey per family, so competing in several families means winning each one on its own merits.

### Does my family's share drop when another family launches?

No. Family slices are **absolute and independent** — a new family takes its slice from the **burn**, not from yours. Your `family_share` changes only if governance changes *your* family's `emission_allocation` or emissions state.

<p align="right">(<a href="#koth-top">back to top</a>)</p>

---

## Glossary

| Term | Meaning |
|---|---|
| **King** | A model that took the throne by passing the full benchmark and clearing the dynamic crowning floor. |
| **Challenge family** | An independent competition (e.g. Autopilot, Search-and-Rescue), each with its own lineage, window, and emission slice. |
| **Lineage** | The permanent ordered list of every king ever in a family, stored by the backend. |
| **Active window** | A family's current 5 kings whose shares are summed and used for that family's slice. |
| **Family share** | A family's absolute fraction of total emissions: `emission_allocation × status weight`. Independent of other families; unallocated emissions burn. |
| **Burn** | Emissions routed to UID 0 (not paid to any miner) — used for a participating family that has no payable king yet. |
| **Headroom** | The distance from the previous king's score to the perfect score of 1.0. The "room left to grow". |
| **Jump** | The absolute score improvement when a king was crowned (`score − prev_score`). |
| **Log-headroom gain** | `log((1 − prev) / (1 − score))`, with headroom floored at `0.01` to prevent singularity. |
| **Rank weighting** | `(5 − rank) / 5`; the champion keeps full gain, and each older king loses 20%. |
| **Share** | The fraction of emissions a king receives (`family_share × koth_share`). A family's 5 active kings sum to that family's slice, not to 100%. |
| **Aging out** | When a king reaches rank `−5` (i.e., five dethronings have happened since they took the throne) and leaves the window. |
| **Crowning floor** | The minimum improvement required to dethrone the champion: 0.015 while the score is ≤ 0.5, decaying to 0.005 near 1.0. |

<p align="right">(<a href="#koth-top">back to top</a>)</p>
