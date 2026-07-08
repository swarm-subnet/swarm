<a id="koth-top"></a>

# King of the Hill

How emissions are distributed on Swarm Subnet 124.

---

<details>
  <summary><b>Table of Contents</b></summary>
  <ol>
    <li><a href="#what-koth-is">What KotH is</a></li>
    <li><a href="#why-it-exists">Why it exists</a></li>
    <li><a href="#the-5-king-window">The 5-king window</a></li>
    <li><a href="#how-each-kings-share-is-calculated">How each king's share is calculated</a></li>
    <li><a href="#rank-weighting">Rank weighting</a></li>
    <li><a href="#taking-the-throne--the-dynamic-floor">Taking the throne: the dynamic floor</a></li>
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

1. **Copycat models.** Under winner-take-all, a miner can clone the current champion, add just enough noise to clear the crowning floor, and take 100% of emissions without contributing real innovation. Under KotH, that miner's tiny jump translates to a tiny share: most of the emissions stay with the past kings whose jumps were larger.

2. **Innovation goes unpaid.** Under winner-take-all, the miner who pushed the network from 0.85 to 0.92 is forgotten the moment someone nudges it to 0.93. Under KotH, that 0.07 jump keeps paying (proportional to the real contribution) for up to four more dethronings.

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
−4          4 dethronings ago             yes (leaves on next crowning)
```

After the next crowning, the king at slot `−4` leaves the window and stops earning. The new king takes slot `0`, every other king shifts one slot down.

Being dethroned does **not** remove a king from the window: the seat stays payable, tapered by rank, until five newer crownings push it out. The gain is locked at crowning: a window seat's score and prev-score are written once and never touched again, even when the reigning champion is re-scored at the weekly epoch re-eval.

A seat stops paying before it ages out if it fails the [payout eligibility check](#who-a-seat-can-pay): a dead or unreachable repo, or a missing artifact. An ineligible seat keeps its window slot (no sixth king is backfilled) but it is skipped at payout, so its slice renormalizes onto the family's surviving kings. Taper ranks are assigned among the **payable** kings only, ordered by crowning recency, so the kings behind a skipped seat move up one taper step; the rank badge on the ladder still shows window position, so a badge and its paid share can briefly diverge while a seat is unreachable.

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

For each king `i` in the 5-king window, with their score `score_i` and the previous king's score `prev_i` (both clamped to `[0, 1]`; non-finite values become `0`):

```
gain_i   = log( max(1 − prev_i, 0.01) / max(1 − score_i, 0.01) )   # 0 when score_i ≤ prev_i
weight_i = max(0, (5 − rank_i) / 5)                                # rank 0 = newest crowning
share_i  = gain_i × weight_i / sum(gain_j × weight_j in window)
```

The `0.01` floor caps the headroom so improvements above `0.99` do not blow up. Rank is derived from crowning recency (epoch, then lineage order), not from list position. If every `gain × weight` in a family is zero, the family pays nobody, and its slice redistributes to the other families (see below).

### Plain-English version

- **Measure improvement in log-headroom**: how much of the remaining distance-to-perfect the king closed.
- **Taper by rank**: champion 100%, then 80%, 60%, 40%, 20%.
- **Normalise the tapered gains** so the family window sums to 100%.

<p align="right">(<a href="#koth-top">back to top</a>)</p>

---

## Rank weighting

A king's gain is tapered by where it sits in the family lineage. The reigning champion (rank 0) keeps its full gain; every step further back loses **20%**:

```
weight = (5 − rank) / 5
   rank 0 (champion) → 1.0     rank 2 → 0.6     rank 4 (oldest) → 0.2
```

So the freshest champions earn the most, and a king fades **as new champions are crowned and push it down the window**, not by any clock, and with no hard age cutoff. Every king in the window keeps a share (down to 20% weight at the bottom); a king only stops earning once a sixth crowning pushes it out of the window entirely.

Splitting one improvement across several crownings earns *less* (each earlier piece sits at a lower rank and is weighted down), so there is no advantage to gaming the split.

Rank weighting is separate from the crowning floor below: the floor decides who *takes* the throne, while rank weighting shapes how the throne's *earnings* are split.

<p align="right">(<a href="#koth-top">back to top</a>)</p>

---

<a id="taking-the-throne--the-dynamic-floor"></a>
## Taking the throne: the dynamic floor

Every submission runs the full 1100-seed benchmark. To be crowned, a challenger must clear the current champion by an **improvement floor** that *shrinks* as the champion climbs. With champion score `s`:

```
s ≤ 0.5      floor = floor_max                                (flat, anti-noise while scores are low)
s > 0.5      floor = floor_min + (floor_max − floor_min) × (1 − t²),   t = (s − 0.5) / 0.5
```

The decay is convex: the floor stays near `floor_max` just past `0.5` and falls off toward `floor_min` as the champion approaches `1.0`, since every point near the top is hard-won. So a frozen top of the board becomes easier to dethrone, and champions cycle through the window faster.

The global defaults are `floor_max = 0.015` and `floor_min = 0.005`; the registry overrides them per family:

| Family | floor_max (champion ≤ 0.5) | floor_min (champion → 1.0) |
|---|---|---|
| Autopilot | 0.015 | 0.005 |
| Search-and-Rescue | 0.02 | 0.007 |
| Swarm Autopilot | 0.015 | 0.005 |
| Swarm SAR | 0.02 | 0.007 |
| Interceptor | 0.015 | 0.005 |

When a family has no champion at all, the first evaluated model takes the throne with no floor to clear: see [The first king ever](#the-first-king-ever).

A screening pre-phase exists in the code behind a build-time constant, but it is off: submissions go straight to the full benchmark, and the floor is applied once, at crowning.

<p align="right">(<a href="#koth-top">back to top</a>)</p>

---

## Per-family emissions

Each family runs its own 5-king window. Two levels decide a UID's final weight:

```
1.  family_share(f)   = how big a slice family f gets of the emission pool
2.  koth_share(uid,f) = the UID's share WITHIN family f (the formula above)

weight(uid) = sum over families f of  family_share(f) × koth_share(uid, f)
```

Every hotkey competes in a single family, so a UID's weight comes from that family's slice.

### How a family's slice is sized

Each family has an `emission_allocation` set by governance (not by miners). The current allocations:

| Family | emission_allocation |
|---|---|
| Autopilot | 0.10 |
| Search-and-Rescue | 0.30 |
| Swarm Autopilot | 0.15 |
| Swarm SAR | 0.30 |
| Interceptor | 0.15 |

Allocations are **normalised** so they sum to `1.0` across every family with a positive allocation (the table above already does). A family's **emissions state** then decides whether it participates at all:

```
emissions state                              weight
─────────────────────────────────────────    ──────
active / saturated / incubating / regression   1.0
archived                                       0.0   (out of payout)
```

All five families are currently `active`. Only `archived` changes anything today: the other states are labels on the lifecycle, not payout multipliers.

### Inactive slices redistribute

A family is **payable** when it is not archived and its window has at least one eligible king with a positive improvement gain. The normalised slices of every non-payable family (kingless, solved-and-archived, or archived for any other reason) are split **equally** across the payable families:

```
share(f) = base(f) + sum(base of non-payable families) / count(payable families)
```

Example: four families payable, Interceptor (base `0.15`) has no king yet:

```
share(Autopilot)          = 0.10 + 0.15/4 = 0.1375
share(Search-and-Rescue)  = 0.30 + 0.15/4 = 0.3375
share(Swarm Autopilot)    = 0.15 + 0.15/4 = 0.1875
share(Swarm SAR)          = 0.30 + 0.15/4 = 0.3375
```

The moment Interceptor crowns its first king, its slice snaps back to `0.15` and the top-ups disappear. If **no** family is payable, validators simply hold their last weights.

<a id="who-a-seat-can-pay"></a>
### Who a seat can pay

Before shipping a window to validators, the backend checks every seat:

| Submission type | Eligible when |
|---|---|
| Public (GitHub) | `repo_intact` is true, and the repo is accessible |
| Private | the sealed artifact is still on file |

Champion status is **not** required: past kings in the window are ordinary evaluated models. An ineligible seat is skipped at payout and its slice renormalizes onto the family's surviving kings. UID 0 is reserved and can never hold a seat.

### Solved families

Each family carries a secret `solve_threshold`. When a champion clears it, the family is **solved**: no new champions are crowned, and the current window keeps earning as-is. Seven days later the family is archived, and its slice redistributes equally to the remaining payable families.

### How weights reach the chain

The backend serves the **raw kings** (score + previous score) per family plus the family shares. Validators ignore the backend's advisory weight map and **recompute** the weights locally from those raw numbers and the unchanged formula. A payload without the per-family windows is refused outright. Each validator also checks every king's UID against the live metagraph: if the hotkey no longer matches (a re-registered UID), that share is dropped locally. Because every validator uses the same kings and the same formula, they converge on the same weights without a shared secret.

The final score vector is L1-normalised on chain, so any share not assigned to a live miner spreads pro-rata across the paid miners: no emissions are parked on UID 0.

<p align="right">(<a href="#koth-top">back to top</a>)</p>

---

## Edge cases

<a id="the-first-king-ever"></a>
### The first king ever

When a family has zero past champions, the first evaluated model is crowned unconditionally and its `prev_score` is recorded as `0`. Their gain covers their full score, and they take 100% of **that family's slice** until someone dethrones them.

### The weekly re-eval does not touch the window

Every current champion is re-evaluated on the new epoch's seeds each week. The champion keeps the crown with its fresh score **even if the fresh score is lower**: rival scores come from different epoch seeds, so a cross-epoch comparison is invalid. No new lineage row is written and the window seat keeps its original crowning numbers; a champion only falls to a challenger that clears the floor on the same epoch's seeds.

<p align="right">(<a href="#koth-top">back to top</a>)</p>

---

## FAQ

### How often is my share recalculated?

Your gain is computed at the moment of crowning and locked: the weekly re-eval never rewrites it. Your share moves when the window changes: a new crowning shifts every rank, changes the rank taper, and eventually ages older kings out. Your seat also stops paying if it fails the eligibility check (dead repo, missing artifact); that share renormalizes onto the family's other kings until the seat is healthy again.

### What if I get dethroned?

You slide to rank `−1` (80% taper) and keep earning, dropping one rank with each new crowning down to rank `−4` (20% taper). The fifth crowning after yours pushes you out of the 5-king window, and you stop earning.

### Can I become king twice?

Not on the same hotkey. Once your model is evaluated, the hotkey's slot is locked, so you cannot swap in a stronger model on it. To take a crown again, register a new hotkey and submit your improved model from it. Every hotkey is one family, one model, one final submission.

### What happens to a king who deletes their GitHub repo?

Their seat stops paying. A seat is only payable while its repo is intact and accessible (or, for private submissions, while the sealed artifact is on file). The seat keeps its window slot but is skipped at payout, and its slice renormalizes onto the family's surviving kings. It comes back if the repo does.

### Why is there a minimum jump to take the throne?

The crowning floor is an anti-noise threshold: without it, the network would re-elect a "new" champion every time a benchmark produced a 0.0001 score variance. It is **dynamic**: flat while the champion is at or below `0.5`, then shrinking as the score approaches 1.0, and per-family: `0.015 → 0.005` for most families, `0.02 → 0.007` for the two SAR families (see [Taking the throne](#taking-the-throne--the-dynamic-floor)).

### Can I earn from more than one family?

Yes, with separate hotkeys. Every hotkey competes in exactly one family, so entering several families means registering one hotkey per family and winning each crown on its own merits. The seats pay independently.

### Does my family's share drop when another family launches?

Yes, it can. Every family is allocated a percentage of the subnet's total emissions, and all the family slices add up to 100%. When a new family is added, it takes its own percentage out of that same total, so the existing families each end up with a little less.

For now there are only five families and their slices are already fixed, so nothing is changing today.

<p align="right">(<a href="#koth-top">back to top</a>)</p>

---

## Glossary

| Term | Meaning |
|---|---|
| **King** | A model that took the throne by passing the full benchmark and clearing the dynamic crowning floor. |
| **Challenge family** | An independent competition (e.g. Autopilot, Search-and-Rescue), each with its own lineage, window, and emission slice. |
| **Lineage** | The permanent ordered list of every king ever in a family, stored by the backend. |
| **Active window** | A family's current 5 kings whose shares are summed and used for that family's slice. |
| **Family share** | A family's normalised `emission_allocation`, plus an equal split of every non-payable family's slice. |
| **Payable seat** | A window seat that passes the eligibility check: intact + accessible repo for public submissions, sealed artifact on file for private ones. |
| **Headroom** | The distance from the previous king's score to the perfect score of 1.0. The "room left to grow". |
| **Jump** | The absolute score improvement when a king was crowned (`score − prev_score`). |
| **Log-headroom gain** | `log((1 − prev) / (1 − score))`, with headroom floored at `0.01` to prevent singularity; zero when the score does not exceed the previous king's. |
| **Rank weighting** | `(5 − rank) / 5`; the champion keeps full gain, and each older king loses 20%. Ranks run over the payable seats by crowning recency. |
| **Share** | The fraction of emissions a king receives (`family_share × koth_share`). A family's 5 active kings sum to that family's slice, not to 100%. |
| **Aging out** | When a king reaches rank `−5` (i.e., five dethronings have happened since they took the throne) and leaves the window. |
| **Crowning floor** | The minimum improvement required to dethrone the champion: flat up to a champion score of 0.5, then decaying, `0.015 → 0.005` by default, `0.02 → 0.007` for the SAR families. |
| **Solved family** | A family whose champion cleared the secret solve threshold; it crowns no new kings and is archived after 7 days, its slice redistributing to the payable families. |

<p align="right">(<a href="#koth-top">back to top</a>)</p>
