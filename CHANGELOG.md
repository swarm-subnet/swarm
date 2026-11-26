# Changelog

Two numbers describe a release. `swarm/__init__.py` holds the release version: changing it makes validators update. Each challenge family's `benchmark_version` in the registry is that family's own benchmark label: raising it re-evaluates that family's champions, and nothing else does.

## 6.2.0

- Swarm Test Field v2 (`cf_test_field_v2`) launches as a task of its own, at `1.0.0`, with 10% of emissions. Each seed draws its own field, cut from real ground, 1,378 to 4,133 m², open or reaching into trees; two seeds in five are nights. Up to 10 people stand, crouch, kneel, sit or walk on it, alone or together, some under trees.
- The model reports every person on the field and every goat, dog and grown boar that lives on it, as a person or as an animal, with a box and the spot within 1 m. A seed scores `0.50 people + 0.20 animals + 0.20 coverage + 0.10 flight`. Missed people, false alarms and edge trips turn 5 of the model's best seeds to 0; animals cost only their own seed.
- People and animals outside the field, birds, piglets, rats and mice are there to be left alone. A seed holds at most six animals.
- The first Swarm Test Field (`cf_test_field`) is completed: it takes no submissions and pays nothing, and its code, page and champion stay. Swarm Sentinel carries 90% of emissions.
- The real areas are fetched beside `swarm-worlds`, one archive an area, each checked against its SHA-256: `python -m swarm_worlds.areas`. The validator image and the miner setup run it.
- The thermal camera reads the ground of a real area against the field's own brightness, so tile edges and the grain of the ground photograph no longer show as heat. Swarm Sentinel's frames are unchanged.
- The benchmark's result table lists each seed's animals to report and its valid and failed animal reports.
- `swarm-bullet3` 2.0.0.7 and `swarm-worlds` v1.2.0.

## 6.1.0

- Every family's benchmark version restarts at `1.0.0`: `cf_solar_patrol` from `6.0.0`, `cf_test_field` from `6.0.1`, the other six from `5.1.6`. It is a rename only: nobody is re-evaluated and no score changes. Work from earlier epochs keeps its old label, and the backend lists each reset on `/version`.
- From here a family's version moves on its own and never follows the release number. MAJOR is a new task spec, MINOR a scoring change that re-evaluates that family's champions, PATCH stays `0`. A label change ships in a three-part release, never in a hotfix.
- Swarm Sentinel and Swarm Test Field re-check their seeds once after the reset, because each seed-check cache is filed per label. The verdicts are the same.
- Swarm Sentinel thieves act less by rule. A crew is up to five men, reaches the fence at any time in the first 200 s, and keeps together, spreads out or comes in two groups. Each man walks at his own pace. Wind, night and the wind's direction set how far a thief hears the drone, half of the thieves who run flee before a camera has seen them, and a thief who freezes or drops flat stays on his path.
- A report box is judged against the thief's body, not the tool he holds, and small stray pieces of a body seen through a gap are left out of its box. A thief seen through a far fence or under a tree crown is named in the object map, so a correct report on him counts. Swarm Test Field reads its boxes the same way.
- Swarm Sentinel evaluation is faster and takes less memory, with the same frames: a patrol is booked at 3,100 MiB on engine 2.0.0.6, down from 3,700.
- Every drone and the dock load from `swarm-description`, a package of its own pinned by tag like `swarm-worlds`.
- A validator whose worker stops turning restarts itself after 30 minutes and logs every thread's stack first. `SWARM_VALIDATOR_STALL_SEC` sets the limit; 0 turns it off.
- `swarm-bullet3` 2.0.0.6, `swarm-description` v1.0.0 and `swarm-worlds` v1.1.1.

## 6.0.0

- Swarm Sentinel (`cf_solar_patrol`) launches: the solar park patrol challenge, paying its last 10 kings through a per-family king window.
- Each challenge family carries its own benchmark version. A release re-evaluates only the families whose version it raises: Sentinel starts at `6.0.0`, the other six stay at `5.1.6`.
- Sentinel scoring heights are measured above the ground, not the dock, and the wind slows near the ground so strong-wind take-offs clear the dock.
- A model that abandons a seed is scored on it instead of the seed being treated as a machine fault, and every intruder on that seed counts as missed.
- Reports are capped at 20 per patrol, and a model's writable folder and submission zip are capped.
- Validators refuse Sentinel on an incomplete install and repair a stale `swarm-worlds`.
- `swarm-bullet3` 2.0.0.5 and `swarm-worlds` v1.1.0.

## 5.1.6.5

- The forest difficulty draws from its own seed stream, and the forest hills mesh cache is published only once it is fully written.
- The ground below a goal pad no longer counts against clearance on the final descent.
- Lying search-and-rescue victims sit on their tilted underside, so slopes no longer bury them.
- Placing the next drone's village pads keeps the earlier drones' pads.

## 5.1.6.4

- A king crowned inside the headroom floor keeps its ladder seat.

## 5.1.6.3

- Village landing pads are placed within the village distance range instead of the mountain one.

## 5.1.6.2

- The autopilot search clue noise draws from its own seed stream, so other draws no longer move the clues.
- The solar park map builder and its forest land, the groundwork for Swarm Sentinel.
- The export checker's material and texture rules match the engine.

## 5.1.6.1

- The validator ships as a published Docker image (`ghcr.io/swarm-subnet/swarm-validator`). Hosts already running the updater are moved onto it on this update; nothing is compiled or installed on the host any more.
- The seed clock now holds the thinking the per-step budget allows (autopilot 840 s to 2,640 s on a 1.0x host), and a seed that stops reporting progress is cut after 222 s. Slow but legal models finish on the first attempt; frozen ones free their worker early.
- A seed that hits the clock is handed back to the pool instead of being flown a second time on the same host, and a cancelled task stops the seeds already in flight.
- A backend timeout, error or rate limit is no longer read as an answer: a refused batch is dropped, an outage is waited out, no score is lost to an error reply.
- Docker cleanup runs off the event loop, every call bounded, and touches only this validator's own containers, images and buffers, so validators sharing a daemon leave each other alone.
- Every phase of a seed is timed and recorded; each run ends with one summary line in the log and `swarm monitor` gains a Seed Timing section.
