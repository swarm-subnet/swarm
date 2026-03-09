<div align="center">

<img src="https://raw.githubusercontent.com/swarm-subnet/swarm/main/swarm/assets/Swarm_2.png" alt="Swarm — Decentralized Drone Intelligence" width="100%">

# Swarm Subnet 124 — Drone Navigation Model

**Autonomous drone intelligence, competing on the Bittensor network.**

[![Subnet](https://img.shields.io/badge/Bittensor-Subnet_124-black?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0Ij48Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSIxMCIgZmlsbD0iI2ZmZDcwMCIvPjwvc3ZnPg==)](https://bittensor.com)
[![GitHub](https://img.shields.io/badge/Swarm-Repository-yellow?style=for-the-badge&logo=github)](https://github.com/swarm-subnet/swarm)
[![Discord](https://img.shields.io/badge/Join-Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/swarm-subnet)

</div>

---

This repository contains a trained **drone navigation model** submitted to [Swarm (Bittensor Subnet 124)](https://github.com/swarm-subnet/swarm) — a decentralized AI competition where miners build autonomous agents that navigate complex 3D environments.

## What is Swarm?

Swarm is the **first decentralized drone intelligence competition** running on the [Bittensor](https://bittensor.com) network. Miners train reinforcement learning agents to pilot drones through procedurally generated environments. Models are evaluated on **speed**, **reliability**, and **obstacle avoidance** — with TAO rewards distributed to top performers every epoch.

The drone receives only a **depth camera feed** and its own state vector — no GPS, no map, no cheating. Pure learned navigation.

## Challenge Environments

Models are tested across **diverse environment types**, each requiring different navigation strategies:

<table>
<tr>
<td align="center" width="50%">
<img src="https://raw.githubusercontent.com/swarm-subnet/swarm/main/swarm/assets/map_images/Type1_sub2.png" alt="City Navigation" width="100%">
<br><b>City Navigation</b><br>
<em>Navigate between buildings, streets and urban obstacles</em>
</td>
<td align="center" width="50%">
<img src="https://raw.githubusercontent.com/swarm-subnet/swarm/main/swarm/assets/map_images/Type3.png" alt="Mountain Terrain" width="100%">
<br><b>Mountain Terrain</b><br>
<em>Fly through peaks and valleys with elevation changes</em>
</td>
</tr>
<tr>
<td align="center" width="50%">
<img src="https://raw.githubusercontent.com/swarm-subnet/swarm/main/swarm/assets/map_images/Type4_2.png" alt="Warehouse Interior" width="100%">
<br><b>Warehouse Interior</b><br>
<em>Indoor navigation among shelves, forklifts and conveyors</em>
</td>
<td align="center" width="50%">
<img src="https://raw.githubusercontent.com/swarm-subnet/swarm/main/swarm/assets/map_images/landing_pad.png" alt="Drone on Landing Pad" width="100%">
<br><b>Precision Landing</b><br>
<em>Reach the target landing pad to complete the mission</em>
</td>
</tr>
</table>

## How It Works

```
Depth Camera (128x128) ──┐
                         ├──▶ Neural Network ──▶ Velocity Commands ──▶ Drone
Drone State Vector ──────┘
```

1. The drone observes the world through a **128x128 depth camera** and its own state (position, velocity, orientation)
2. A neural network processes these inputs and outputs **velocity commands**
3. The drone navigates toward the goal while avoiding obstacles
4. Performance is scored on **completion time**, **success rate**, and **flight safety**

## Running This Model

Clone the [Swarm repository](https://github.com/swarm-subnet/swarm) and install dependencies:

```bash
git clone https://github.com/swarm-subnet/swarm.git
cd swarm
pip install -e .
```

Run a full benchmark evaluation across all environment types:

```bash
python debugging/bench_full_eval.py --model submission.zip --workers 2
```

Or test on a specific seed:

```bash
python debugging/bench_full_eval.py --model submission.zip --seeds-per-group 1
```

## Repository Structure

```
submission.zip   # Trained model (zipped agent code + weights)
README.md        # This file (required by the Swarm validator)
```

## Scoring

| Component | Weight | Description |
|-----------|--------|-------------|
| Success   | 45%    | Did the drone reach the goal? |
| Speed     | 45%    | How fast relative to the time limit? |
| Safety    | 10%    | Minimum clearance from obstacles |

Final score: `0.45 * success + 0.45 * time_bonus + 0.10 * safety_bonus`

## Links

- **Swarm Repository** — [github.com/swarm-subnet/swarm](https://github.com/swarm-subnet/swarm)
- **Bittensor Network** — [bittensor.com](https://bittensor.com)
- **Subnet ID** — 124

---

<div align="center">

<img src="https://raw.githubusercontent.com/swarm-subnet/swarm/main/swarm/assets/Swarm.png" alt="Swarm Logo" width="80">

**Built for [Swarm Subnet 124](https://github.com/swarm-subnet/swarm) on the [Bittensor](https://bittensor.com) network.**

*This README is required by the Swarm validator. Do not modify it.*

</div>
