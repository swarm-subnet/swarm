<a id="readme-top"></a>

<p align="center">
  <img src="https://raw.githubusercontent.com/swarm-subnet/swarm/main/swarm/assets/Swarm_2.png" alt="Swarm" width="50%" />
</p>

<h1 align="center">Swarm Benchmark Entry</h1>

<p align="center">
  <b>A trained drone pilot, packaged as a verified entry to the Swarm benchmark.</b><br/>
  <i>Depth camera in, velocity commands out. Scored on 1,100 worlds it has never seen.</i>
</p>

<p align="center">
  <a href="https://github.com/swarm-subnet/swarm"><img alt="Benchmark" src="https://img.shields.io/badge/benchmark-Subnet%20124-F5D400?style=flat-square&labelColor=111111" /></a>
  <a href="https://discord.gg/8dPqPDw7GC"><img alt="Discord" src="https://img.shields.io/badge/Discord-Join-5865F2?style=flat-square&logo=discord&logoColor=white" /></a>
  <a href="https://x.com/SwarmSubnet"><img alt="X" src="https://img.shields.io/badge/X-Follow-111111?style=flat-square&logo=x&logoColor=white" /></a>
  <a href="https://swarm124.com"><img alt="Website" src="https://img.shields.io/badge/swarm124.com-visit-F5D400?style=flat-square&labelColor=111111" /></a>
</p>

<p align="center">
  <a href="#run-this-model"><img alt="Run This Model" src="https://img.shields.io/badge/Run%20This%20Model-Benchmark-F5D400?style=for-the-badge" /></a>
  &nbsp;
  <a href="https://swarm124.com/benchmark"><img alt="Leaderboard" src="https://img.shields.io/badge/Leaderboard-swarm124.com-111111?style=for-the-badge" /></a>
</p>

---

> [!NOTE]
> This repository is a verified entry to the [Swarm](https://swarm124.com) benchmark. This README is the benchmark's hash-locked template, checked byte-for-byte, so it is identical across every entry and must not be edited.

---

https://github.com/user-attachments/assets/ee579a55-5eb2-4f6c-83db-4b1a223b9bb2

<p align="center">
  <sub><b>Search and Rescue.</b> One of five missions on the Swarm benchmark: teaching a drone to find people.</sub>
</p>

---

## This Repository

This repository holds one trained flight model entered into [Swarm](https://swarm124.com), the open benchmark for autonomous drone AI. The artifact under `artifacts/` is a complete pilot: a policy that reads a depth camera and its own flight state, and flies. Models are evaluated across six procedurally generated world types, 1,100 fresh seeds per family every weekly epoch, with no privileged information and no pre-built maps.

```
README.md                              # This file, the byte-exact benchmark template
submission_manifest.json               # Declares which family this repo competes in
artifacts/<family_id>/submission.zip   # The trained agent for that family
```

The `.zip` carries the agent's `DroneFlightController` class in `drone_agent.py` plus its trained weights; the manifest pins its SHA-256. One artifact, one family: every entry competes in exactly one challenge.

## Run This Model

The model in this repo runs on your machine with the public benchmark engine: the same worlds and the same scoring the leaderboard uses.

**1. Clone and install**

```bash
git clone https://github.com/swarm-subnet/swarm.git
cd swarm
uv venv
source .venv/bin/activate
uv pip install -e .
```

**2. Run a benchmark**

```bash
python -m swarm.benchmark.engine \
  --model /path/to/this-repo/artifacts/cf_autopilot/submission.zip \
  --family-id cf_autopilot \
  --seeds-per-group 3 --workers 4
```

Point `--model` and `--family-id` at the artifact this repo ships. The artifact lives here, not in the swarm checkout, so pass its full path.

> [!NOTE]
> `--seeds-per-group 1` runs a quick pass; validators score the full 1,100 seeds.

## See It Fly

https://github.com/user-attachments/assets/a16e9453-663c-4483-a3b8-160c412fd3e7

<p align="center">
  <sub><b>Interceptor:</b> air-to-air pursuit, from a real benchmark run.</sub>
</p>

<table>
<tr>
<td align="center" width="50%">
<img src="https://raw.githubusercontent.com/swarm-subnet/swarm/main/swarm/assets/Drone_flying.gif" alt="Drone navigating a procedural city" width="100%">
<br><sub>Third-person view</sub>
</td>
<td align="center" width="50%">
<img src="https://raw.githubusercontent.com/swarm-subnet/swarm/main/swarm/assets/Drone_flying_FPV.gif" alt="Drone FPV depth view" width="100%">
<br><sub>FPV: what the drone actually sees</sub>
</td>
</tr>
</table>

The policy reads a depth image and its own flight state 50 times a second and answers with velocity commands: a direction, a speed, a turn. No waypoints, no obstacle coordinates, no map. It learns to read the world through the camera and react.

## Challenge Families

The benchmark runs **five challenge families**. Each is its own competition with its own champion, its own leaderboard lineage, and its own slice of subnet emissions. One repo enters exactly one of them: every hotkey competes in a single family.

| Family | ID | Mission | Emissions |
|--------|----|---------|:--------:|
| [Search and Rescue](https://github.com/swarm-subnet/swarm/blob/main/docs/families/search_and_rescue.md) | `cf_search_and_rescue` | Find a person, hold a steady 2–4 m hover overhead, never touch them | 10% |
| [Swarm SAR](https://github.com/swarm-subnet/swarm/blob/main/docs/families/swarm_sar.md) | `cf_swarm_sar` | 2–8 drones sweep together; one confirmed hover wins for the team | 10% |
| [Swarm Autopilot](https://github.com/swarm-subnet/swarm/blob/main/docs/families/swarm_autopilot.md) | `cf_swarm_autopilot` | One policy lands 2–8 drones on a shared pool of pads | 10% |
| [Interceptor](https://github.com/swarm-subnet/swarm/blob/main/docs/families/interceptor.md) | `cf_interceptor` | Hunt a fleeing, jinking drone and catch it before the 60 s clock | 10% |
| [Autopilot](https://github.com/swarm-subnet/swarm/blob/main/docs/families/autopilot.md) | `cf_autopilot` | Cross the world, find the pad inside a noisy search area, touch down clean | 10% |

Each guide spells out the full contract: observation layout, action bounds, maps, episode rules, and scoring.

## Scoring

| Component | Weight | What It Measures |
|-----------|--------|-----------------|
| **Success** | 45% | Did the drone complete the family's mission? |
| **Speed** | 45% | How fast relative to the time limit? |
| **Safety** | 10% | Minimum clearance from obstacles |

```
seed score = 0.45 × success + 0.45 × time + 0.10 × safety
```

Interceptor is the one exception: 50% success, 50% speed, no safety term. It is a pursuit.

Every submission runs the **full 1,100-seed benchmark** and is ranked by its **mean score across all 1,100 seeds**. Seeds rotate weekly (Monday 16:00 UTC) and champions are re-evaluated on the fresh set. To take a crown, a challenger must beat the champion by the family's improvement floor: **+0.015** (Autopilot, Swarm Autopilot, Interceptor) or **+0.02** (the two SAR families).

Each family's emission slice is split among its **last five champions**, weighted by how much each one raised the family's best score. Full spec in [King of the Hill](https://github.com/swarm-subnet/swarm/blob/main/docs/king_of_the_hill.md).

## Build Your Own Entry

This entry was trained by one miner. The benchmark is open, every family has a champion, and every champion can be beaten.

1. **Read the docs**: the [Miner guide](https://github.com/swarm-subnet/swarm/blob/main/miner/docs/miner.md) covers repo setup, packaging, submission, and training tips.
2. **Pick your family**: each family guide above defines the exact interface your model needs.
3. **Study the baseline**: the [training starters](https://github.com/swarm-subnet/swarm/tree/main/miner/src/RL) train a policy and package it into a ready artifact.
4. **Train and iterate**: benchmark locally, push your score higher.
5. **Submit and compete**: publish your repo and climb the [leaderboard](https://swarm124.com/benchmark).

Models trained on this benchmark fly on real hardware: see [Langostino](https://github.com/swarm-subnet/Langostino), Swarm's open-source ROS2 drone.

<p align="center">
  <a href="https://github.com/swarm-subnet/swarm/blob/main/miner/docs/miner.md"><img alt="Start Training" src="https://img.shields.io/badge/Start%20Training-Miner%20Guide-F5D400?style=for-the-badge" /></a>
</p>

## Community

<p align="center">
  <a href="https://discord.gg/8dPqPDw7GC"><img alt="Discord" src="https://img.shields.io/badge/Join%20Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" /></a>
  &nbsp;
  <a href="https://x.com/SwarmSubnet"><img alt="X" src="https://img.shields.io/badge/Follow-111111?style=for-the-badge&logo=x&logoColor=white" /></a>
  &nbsp;
  <a href="https://github.com/swarm-subnet"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" /></a>
</p>

---

<p align="center">
  <img src="https://raw.githubusercontent.com/swarm-subnet/swarm/main/swarm/assets/Swarm.png" alt="Swarm" width="60">
</p>

<p align="center">
  <b><a href="https://swarm124.com">Swarm</a>: where AI learns to fly.</b><br/>
  <sub>Subnet 124 on <a href="https://bittensor.com">Bittensor</a></sub>
</p>
