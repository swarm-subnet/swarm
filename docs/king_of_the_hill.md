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
- The **four most recent past champions** of the family keep earning until they age out of the window — but each seat is paid for at most **7 days from when it was crowned** (see [Emission decay](#emission-decay--the-7-day-timer)).
- Each king's share is computed once at crowning time and never recomputed.

How the family slices add up is covered in [Per-family emissions](#per-family-emissions). The within-a-family split below is identical to the original single-competition design.

<p align="right">(<a href="#koth-top">back to top</a>)</p>

---

## Why it exists

Winner-take-all has two failure modes that KotH addresses:

1. **Copycat models.** Under winner-take-all, a miner can clone the current champion, add `0.015` of noise to pass the crowning floor, and take 100% of emissions without contributing real innovation. Under KotH, that miner's tiny jump translates to a tiny share — most of the emissions stay with the past kings whose jumps were larger.

2. **Innovation goes unpaid.** Under winner-take-all, the miner who pushed the network from 0.85 to 0.92 is forgotten the moment someone nudges it to 0.93. Under KotH, that 0.07 jump keeps paying — proportional to the real contribution — for up to four more dethronings, capped at 7 days per seat.

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

A king keeps earning while three things hold: they are still in the window, their **7-day emission timer** has not run out (see below), and their submission repo is still reachable. Past kings are never re-evaluated — the share is locked at crowning — but the timer or a dead repo can switch the share off before they age out.

<p align="right">(<a href="#koth-top">back to top</a>)</p>

---

## Emission decay — the 7-day timer

A king is not paid forever for a model nobody beats. Each seat carries a **7-day timer that starts the moment it is crowned**:

- While the timer is running, the seat earns its share as normal.
- Once **7 days** pass since it was crowned — whatever its rank — the seat's slice **stops and burns**, routed to the burn UID exactly like a dropped seat. The other kings do **not** absorb it, so the subnet simply emits less. The clock is per seat: being dethroned does not pause or extend it.
- The seat keeps its place in the window (shown at 0%) and leaves only when later crownings push it past the 5-king window.

The only way to get a fresh 7 days is a **genuine improvement** that earns you a new crowning — a brand-new seat with its own clock. Surviving a weekly re-evaluation, or re-submitting the same model, does **not** reset it; the mechanism pays for progress, not for tenure.

If every seat in a family has timed out, that family's entire slice burns until someone crowns a new king.

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

If a king's score is at or extremely close to 1.0, subsequent jumps look like dividing by zero. The cap of `0.05` in the formula bounds the headroom-adjusted multiplier at 20×, so no single late jump can take more than ~95% of the window.

### Backend unreachable

When the validator cannot reach the backend, it falls back to the **last per-family snapshot it cached** (kings + family shares) and recomputes weights from that. If it has no cached snapshot, share is routed to UID 0 and effectively burned for that cycle. No bad weights are set. As soon as the backend recovers, normal KotH weights resume.

<p align="right">(<a href="#koth-top">back to top</a>)</p>

---

## FAQ

### How often is my share recalculated?

Your `delta` and `adjusted` values are computed at the moment of crowning and locked. The denominator (the sum of all 5 kings' `adjusted` values) changes when a new king joins the window or you age out of it. Your share also drops to **zero** when your 7-day emission timer runs out (see [Emission decay](#emission-decay--the-7-day-timer)) or your repo becomes unreachable.

### Do I keep earning forever if nobody beats me?

No. Each seat has a **7-day timer** from crowning. If nobody beats you and you do not ship a stronger model within 7 days, your slice stops and burns until the frontier moves again. The only way to refresh it is a new crowning — see [Emission decay](#emission-decay--the-7-day-timer).

### What if I get dethroned?

You stay in the window at rank `−1` and keep earning — but only until your **7-day emission timer** runs out or you age out at rank `−4` after four more dethronings, whichever comes first.

### Can I become king twice?

No. The subnet enforces **one model per hotkey, lifetime**. A hotkey that has been crowned cannot submit a second model. If you want to compete again, register a new hotkey.

### What happens to a king who deletes their GitHub repo?

Their slice stops and burns. A king's repo is health-checked; if it is deleted or becomes unreachable the seat is treated like a dropped seat — it keeps its window slot but pays nobody, so its share burns rather than flowing to a model that can no longer be verified.

### Why is there a minimum jump (0.015) to take the throne?

The crowning floor (`champion + 0.015`) is an anti-noise threshold. Without it, the network would re-elect a "new" champion every time a benchmark produced a 0.0001 score variance. The floor is unchanged by KotH.

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
| **King** | A model that took the throne by passing the screening + benchmark and beating the previous champion by ≥ 0.015. |
| **Challenge family** | An independent competition (e.g. Autopilot, Search-and-Rescue), each with its own lineage, window, and emission slice. |
| **Lineage** | The permanent ordered list of every king ever in a family, stored by the backend. |
| **Active window** | A family's current 5 kings whose shares are summed and used for that family's slice. |
| **Family share** | A family's absolute fraction of total emissions: `emission_allocation × status weight`. Independent of other families; unallocated emissions burn. |
| **Burn** | Emissions routed to UID 0 (not paid to any miner) — used for a participating family that has no payable king yet. |
| **Headroom** | The distance from the previous king's score to the perfect score of 1.0. The "room left to grow". |
| **Jump** | The absolute score improvement when a king was crowned (`score − prev_score`). |
| **Headroom-adjusted jump** | The jump divided by the remaining headroom, capped to prevent singularity. |
| **Share** | The fraction of emissions a king receives (`family_share × koth_share`). A family's 5 active kings sum to that family's slice, not to 100%. |
| **Aging out** | When a king reaches rank `−5` (i.e., five dethronings have happened since they took the throne) and leaves the window. |
| **Crowning floor** | The fixed minimum improvement (0.015) required to dethrone the current champion. |

<p align="right">(<a href="#koth-top">back to top</a>)</p>
