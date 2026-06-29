# King of the Hill

How emissions are split on Swarm Subnet 124.

Instead of paying only the single best model, Swarm pays the **last 10 champions** — each by how
much it moved the frontier when it was crowned, weighted toward the more recent champions. This
rewards the *act of improving*, not just sitting on top, and spreads emissions across many
independent miners.

---

## The 10-king window

The window holds the **10 most recent champions** of the current benchmark version. The newest
is the reigning champion; the rest are past champions that still earn while they remain in the
window.

```
rank   00   reigning champion        earns
rank  −1    1 dethroning ago         earns
 …
rank  −9    9 dethronings ago        earns — rotates out on the next crowning
```

When a new champion is crowned it enters at rank 00, everyone shifts down one, and the oldest
king leaves the window and stops earning.

A code release (new benchmark version) starts a **fresh** window — the maps and scoring change,
so old scores are not comparable. The weekly epoch re-evaluation (same version) keeps the window
intact: champions persist week to week, only their scores refresh.

---

## How each king's share is decided

Each king's slice depends on **how much it improved the score** and **how recent it is**. The
improvement is measured as log-headroom — worth more near the top, where there is little headroom
left — and is then tapered by the king's **rank** in the lineage so newer kings earn more.

For a king crowned at `score`, having beaten a previous best of `prev`, sitting at `rank` in the
window (`rank = 0` is the reigning champion):

```
gain   = log( max(1 − prev, 0.01) / max(1 − score, 0.01) )   # ≥ 0
weight = (10 − rank) / 10                                     # champion 1.0, each step down −0.1
share  = gain × weight / (sum of gain × weight over the window)
```

In words:

- Measure improvement in **log-headroom** — how much of the remaining distance-to-perfect the
  king closed.
- Taper it by **rank**: the champion keeps full weight, the king below it 90%, the next 80%, and so
  on — see *Rank weighting* below.
- Normalise across the window so the shares sum to 100%.

The `0.01` floor caps the headroom so improvements above `0.99` do not blow up.

The first king of a version starts from `prev = 0` and takes the full window while it stands alone.
As newer kings arrive, it slides down the ranks and its weight drops, shifting emissions toward the
more recent champions.

---

## Rank weighting

A king's gain is tapered by where it sits in the lineage. The reigning champion (rank 0) keeps its
full gain; every step further back loses **10%**:

```
weight = (10 − rank) / 10
   rank 0 (champion) → 1.0      rank 5 → 0.5      rank 9 (oldest) → 0.1
```

So the freshest champions earn the most, and a king fades **as new champions are crowned and push it
down the window** — not by any clock, and with no hard age cutoff. Every king in the top-10 always
keeps a share (down to 10% weight at the bottom); a king only stops earning once an 11th crowning
pushes it out of the window entirely.

Splitting one improvement across several crownings now earns *less* — each earlier piece sits at a
lower rank and is weighted down — so there is no advantage to gaming the split.

Rank weighting is separate from the crowning floor below: the crowning floor decides who *takes* the
throne, while rank weighting shapes how the throne's *earnings* are split.

---

## Taking the throne — the dynamic floor

To be crowned, a challenger's consensus score must clear the current champion by an **improvement
floor** that *shrinks* as the champion climbs:

```
champion ≤ 0.35     floor = 0.015     (flat — anti-noise while scores are low)
champion → 1.00     floor → 0.005     (smaller, since every point near the top is hard-won)
```

So a frozen top of the board becomes easier to dethrone, and champions cycle through the window
faster. The same gate is applied at screening (300 seeds) and at final crowning (full benchmark).
Pass/fail is always a stake-weighted consensus decision across validators, never one validator's
verdict.

---

## Edge cases

- **No king yet, but a champion exists** (e.g. just after a release): the sitting champion is
  re-evaluated under the new rules and seeded as the first king from its **real** new score — never
  from zero. It earns 100% until challengers arrive.
- **No king and no champion:** weight is routed to the burn UID until the first king is crowned.
- **A king's repo goes dead** (deleted or unreachable): it leaves the window and its slice is
  **redistributed to the other kings**, so emissions stay fully paid out.
- **One model per hotkey, lifetime:** a crowned hotkey cannot submit again — to compete again,
  register a new hotkey. This bounds how many seats one operator can take.

---

## How weights reach the chain

The backend serves the king window (each king's `score`, the `prev` it beat, and its UID). The
share map is recomputed from those numbers and sent to validators, which apply it on-chain each
forward cycle. The live window and a diagnostics snapshot (version, champion score, the crowning
floor, last crowning) are public at `/kings/active` and `/kings/diagnostics`.

---

## Glossary

| Term | Meaning |
|---|---|
| **King** | A model that took the throne by passing screening + benchmark and clearing the dynamic floor. |
| **Window** | The 10 most recent kings of the current version whose shares are summed. |
| **Headroom** | The distance from a score to a perfect 1.0 — the room left to grow. |
| **Log-headroom gain** | `log((1 − prev) / (1 − score))`, floored at 0.01 — how much of the remaining distance-to-perfect a king closed when crowned. |
| **Rank weighting** | `(10 − rank) / 10`. The champion (rank 0) keeps full gain; each older king loses 10%, down to 10% at the bottom of the window. |
| **Dynamic (crowning) floor** | The minimum improvement to dethrone the champion: 0.015 while ≤ 0.35, decaying to 0.005 at 1.0. |
| **Burn** | Weight routed to UID 0 (paid to no miner) when there is no payable king. |
