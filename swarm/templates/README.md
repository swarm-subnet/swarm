# Swarm Subnet 124 — Model Submission

This repository contains a trained drone navigation model submitted to
[Swarm (Bittensor Subnet 124)](https://github.com/swarm-subnet/swarm).

## What is Swarm?

Swarm is a decentralized AI competition running on the
[Bittensor](https://bittensor.com) network. Miners compete to build
the best autonomous drone navigation agents. Models are evaluated
across diverse 3D environments and scored on speed, reliability and
path efficiency — with TAO rewards for top performers every epoch.

## Repository Structure

```
submission.zip   # Packaged model ready for evaluation
README.md        # This file (required by the Swarm validator)
```

## Running This Model

Install the Swarm package and run a local benchmark:

```bash
pip install -e .              # from the swarm repo root
swarm benchmark --full        # full evaluation across all seed maps
```

Or test a single seed:

```bash
swarm model test submission.zip
```

## Links

- **Subnet Repo** — https://github.com/swarm-subnet/swarm
- **Bittensor** — https://bittensor.com
- **Subnet ID** — 124

---

*This README is required by the Swarm validator. Do not modify it.*
