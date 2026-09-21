# Changelog

## 5.1.6.1

- The validator ships as a published Docker image (`ghcr.io/swarm-subnet/swarm-validator`). Hosts already running the updater are moved onto it on this update; nothing is compiled or installed on the host any more.
- The seed clock now holds the thinking the per-step budget allows (autopilot 840 s to 2,640 s on a 1.0x host), and a seed that stops reporting progress is cut after 222 s. Slow but legal models finish on the first attempt; frozen ones free their worker early.
- A seed that hits the clock is handed back to the pool instead of being flown a second time on the same host, and a cancelled task stops the seeds already in flight.
- A backend timeout, error or rate limit is no longer read as an answer: a refused batch is dropped, an outage is waited out, no score is lost to an error reply.
- Docker cleanup runs off the event loop, every call bounded, and touches only this validator's own containers, images and buffers, so validators sharing a daemon leave each other alone.
- Every phase of a seed is timed and recorded; each run ends with one summary line in the log and `swarm monitor` gains a Seed Timing section.
