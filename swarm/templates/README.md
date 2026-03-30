<a id="readme-top"></a>

<p align="center">
  <img src="https://raw.githubusercontent.com/swarm-subnet/swarm/main/swarm/assets/Swarm_2.png" alt="Swarm" width="60%" />
</p>

<h1 align="center">Autonomous Drone Navigation</h1>

<p align="center">
  <i>No hand-crafted rules. No pre-built maps. No shortcuts.<br/>
  A neural network that learned to fly — from scratch.</i>
</p>

<p align="center">
  <a href="https://github.com/swarm-subnet/swarm/releases"><img alt="Version" src="https://img.shields.io/badge/version-v4.0.0-green?style=flat-square" /></a>
  <a href="https://discord.gg/8dPqPDw7GC"><img alt="Discord" src="https://img.shields.io/badge/Discord-Join-5865F2?style=flat-square&logo=discord&logoColor=white" /></a>
  <a href="https://x.com/SwarmSubnet"><img alt="X" src="https://img.shields.io/badge/X-Follow-000000?style=flat-square&logo=x&logoColor=white" /></a>
  <a href="https://swarm124.com"><img alt="Website" src="https://img.shields.io/badge/swarm124.com-visit-orange?style=flat-square&logo=googlechrome&logoColor=white" /></a>
</p>

---

This model was trained on the [Swarm](https://swarm124.com) open benchmark — a competition where AI agents learn to pilot drones through procedurally generated 3D worlds using nothing but a depth camera and their own flight state.

<!-- TABLE OF CONTENTS -->
<details>
  <summary><b>Table of Contents</b></summary>
  <ol>
    <li><a href="#about-swarm">About Swarm</a></li>
    <li><a href="#from-simulation-to-reality">From Simulation to Reality</a></li>
    <li><a href="#see-it-fly">See It Fly</a></li>
    <li><a href="#environments">Environments</a></li>
    <li><a href="#how-it-works">How It Works</a></li>
    <li><a href="#run-this-model">Run This Model</a></li>
    <li><a href="#scoring">Scoring</a></li>
    <li><a href="#train-your-own">Train Your Own</a></li>
    <li><a href="#community">Community</a></li>
  </ol>
</details>

<!-- ABOUT SWARM -->
## About Swarm

[Swarm](https://swarm124.com) is the **open benchmark for autonomous drone navigation**. AI agents learn to fly through complex 3D worlds from raw sensor input — no privileged information, no handholding.

* **Hard problem** — depth camera + state vector in, velocity commands out
* **No memorization** — 1,000 procedurally generated environments, every seed is unique
* **Fully reproducible** — deterministic seeds, containerized evaluation, public leaderboard
* **Open to everyone** — train, submit, compete — the best model wins

Autonomous navigation shouldn't live in closed labs. Swarm makes drone AI **measurable**, **comparable**, and **open**. Powered by the [Bittensor](https://bittensor.com) network (Subnet 124).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- FROM SIMULATION TO REALITY -->
## From Simulation to Reality

Models trained on the Swarm benchmark don't stay in simulation — **they fly on real hardware.**

<p align="center">
  <a href="https://github.com/swarm-subnet/Langostino">
    <img src="https://raw.githubusercontent.com/swarm-subnet/Langostino/main/assets/banner_section_1.png" alt="Langostino — The Swarm Drone" width="80%" />
  </a>
</p>

<p align="center">
  <b><a href="https://github.com/swarm-subnet/Langostino">Langostino</a></b> is our open-source autonomous drone — built from scratch with ROS2, Raspberry Pi, and INAV.<br/>
  Full assembly guide, bill of materials, and 3D-printable parts included. Build it yourself.
</p>

<p align="center">
  <a href="https://www.youtube.com/shorts/gf9mxroeurU" target="_blank" rel="noopener noreferrer">
    <img src="https://img.youtube.com/vi/gf9mxroeurU/maxresdefault.jpg" alt="Langostino autonomous flight" width="80%" loading="lazy" />
  </a>
</p>

<p align="center">
  <a href="https://www.youtube.com/shorts/gf9mxroeurU" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/Watch-Autonomous%20Flight-FF0000?style=for-the-badge&logo=youtube&logoColor=white" alt="Watch on YouTube" />
  </a>
  &nbsp;
  <a href="https://github.com/swarm-subnet/Langostino">
    <img src="https://img.shields.io/badge/Build%20Your%20Own-Langostino-111111?style=for-the-badge&logo=github" alt="Build your own" />
  </a>
</p>

<p align="center">
  <b>Train in Simulation</b> &nbsp;→&nbsp; <b>Compete on the Leaderboard</b> &nbsp;→&nbsp; <b>Deploy on Real Hardware</b><br/>
  <sub><a href="https://github.com/swarm-subnet/swarm">Swarm</a> · <a href="https://swarm124.com/benchmark">swarm124.com</a> · <a href="https://github.com/swarm-subnet/Langostino">Langostino</a></sub>
</p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- SEE IT FLY -->
## See It Fly

<table>
<tr>
<td align="center" width="50%">
<img src="https://raw.githubusercontent.com/swarm-subnet/swarm/main/swarm/assets/Drone_flying.gif" alt="Drone navigating a procedural city" width="100%">
<br><sub>Third-person view</sub>
</td>
<td align="center" width="50%">
<img src="https://raw.githubusercontent.com/swarm-subnet/swarm/main/swarm/assets/Drone_flying_FPV.gif" alt="Drone FPV view" width="100%">
<br><sub>FPV — what the drone sees</sub>
</td>
</tr>
</table>

<p align="center">
  <sub>No GPS. No pre-built map. Just a depth camera and learned instincts.</sub>
</p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ENVIRONMENTS -->
## Environments

<table>
<tr>
<td align="center" width="50%">
<img src="https://raw.githubusercontent.com/swarm-subnet/swarm/main/swarm/assets/map_images/Type1_sub2.png" alt="City" width="100%">
<br><b>City</b>
</td>
<td align="center" width="50%">
<img src="https://raw.githubusercontent.com/swarm-subnet/swarm/main/swarm/assets/map_images/Type3_sub2.png" alt="Ski Village" width="100%">
<br><b>Ski Village</b>
</td>
</tr>
<tr>
<td align="center" width="50%">
<img src="https://raw.githubusercontent.com/swarm-subnet/swarm/main/swarm/assets/map_images/Type3.png" alt="Mountains" width="100%">
<br><b>Mountains</b>
</td>
<td align="center" width="50%">
<img src="https://raw.githubusercontent.com/swarm-subnet/swarm/main/swarm/assets/map_images/Type4_2.png" alt="Warehouse" width="100%">
<br><b>Warehouse</b>
</td>
</tr>
<tr>
<td align="center" width="50%">
<img src="https://raw.githubusercontent.com/swarm-subnet/swarm/main/swarm/assets/map_images/Type6_sub1.png" alt="Forest Normal" width="100%">
<br><b>Forest — Normal</b>
</td>
<td align="center" width="50%">
<img src="https://raw.githubusercontent.com/swarm-subnet/swarm/main/swarm/assets/map_images/Type6_sub2.png" alt="Forest Autumn" width="100%">
<br><b>Forest — Autumn</b>
</td>
</tr>
<tr>
<td align="center" width="50%">
<img src="https://raw.githubusercontent.com/swarm-subnet/swarm/main/swarm/assets/map_images/Type6_sub3.png" alt="Forest Snow" width="100%">
<br><b>Forest — Snow</b>
</td>
<td align="center" width="50%">
<img src="https://raw.githubusercontent.com/swarm-subnet/swarm/main/swarm/assets/map_images/Type6_sub4.png" alt="Forest Dead" width="100%">
<br><b>Forest — Dead</b>
</td>
</tr>
</table>

<p align="center">
  <sub>6 environment types, each <b>procedurally generated</b> across <b>1,000 unique seeds</b> per epoch.</sub>
</p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- HOW IT WORKS -->
## How It Works

<table align="center">
<tr>
<td align="center"><b>Depth Camera</b><br><sub>128×128 image</sub></td>
<td align="center" rowspan="2"><b>&nbsp;→&nbsp; Neural Network &nbsp;→&nbsp;</b></td>
<td align="center" rowspan="2"><b>Velocity Commands</b><br><sub>3D movement</sub></td>
<td align="center" rowspan="2"><b>&nbsp;→&nbsp; Drone</b></td>
</tr>
<tr>
<td align="center"><b>State Vector</b><br><sub>position · velocity · orientation</sub></td>
</tr>
</table>

<br/>

The drone sees the world through a **128×128 depth image** and knows its own position, velocity, and orientation. A neural network maps these raw inputs directly to **3D velocity commands** — steering toward the goal while avoiding every obstacle in its path.

**No pre-built maps. No obstacle positions. Everything is learned from experience.**

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- RUN THIS MODEL -->
## Run This Model

**1. Clone and install**

```bash
git clone https://github.com/swarm-subnet/swarm.git
cd swarm
pip install -e .
```

**2. Run a benchmark** (3 seeds per environment type by default)

```bash
swarm benchmark --model submission.zip --workers 4
```

**3. Quick test** (one seed per environment type)

```bash
swarm benchmark --model submission.zip --seeds-per-group 1
```

<details>
  <summary><b>What's in this repo?</b></summary>

```
submission.zip   # Trained agent — neural network weights + inference code
README.md        # This file
```

The `.zip` contains the agent's `DroneFlightController` class and trained weights. The Swarm benchmark runner loads the agent, places it in a procedurally generated environment, and measures its performance.

</details>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- SCORING -->
## Scoring

| Component | Weight | What It Measures |
|-----------|--------|-----------------|
| **Success** | 45% | Did the drone reach the goal? |
| **Speed** | 45% | How fast relative to the time limit? |
| **Safety** | 10% | Minimum clearance from obstacles |

```
Final score = 0.45 × success + 0.45 × time_bonus + 0.10 × safety_bonus
```

Models are ranked by **average score across 1,000 seeds** — consistency matters more than lucky runs. To claim the top spot, a new model must beat the current champion by at least **+0.015**.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- TRAIN YOUR OWN -->
## Train Your Own

Think you can build a better pilot? The benchmark is open — **prove it.**

1. **Read the docs** — [Miner guide](https://github.com/swarm-subnet/swarm/blob/main/docs/miner.md) covers observations, actions, training tips, and submission format
2. **Study the baseline** — [Agent template](https://github.com/swarm-subnet/swarm/blob/main/swarm/submission_template/drone_agent.py) shows the exact interface your model needs
3. **Train and iterate** — Test locally against all environment types, push your score higher
4. **Submit and compete** — Push to a public GitHub repo and climb the [leaderboard](https://swarm124.com/benchmark)

<p align="center">
  <a href="https://github.com/swarm-subnet/swarm/blob/main/docs/miner.md">
    <img alt="Start training" src="https://img.shields.io/badge/Start%20Training-Miner%20Guide-111111?style=for-the-badge" />
  </a>
</p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- COMMUNITY -->
## Community

Swarm is built by researchers, engineers, and drone builders. **Join us.**

<p align="center">
  <a href="https://discord.gg/8dPqPDw7GC"><img alt="Discord" src="https://img.shields.io/badge/Join%20Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" /></a>
  &nbsp;
  <a href="https://x.com/SwarmSubnet"><img alt="X" src="https://img.shields.io/badge/Follow-000000?style=for-the-badge&logo=x&logoColor=white" /></a>
  &nbsp;
  <a href="https://github.com/swarm-subnet"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" /></a>
  &nbsp;
  <a href="https://swarm124.com"><img alt="Website" src="https://img.shields.io/badge/swarm124.com-visit-orange?style=for-the-badge&logo=googlechrome&logoColor=white" /></a>
</p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

<p align="center">
  <img src="https://raw.githubusercontent.com/swarm-subnet/swarm/main/swarm/assets/Swarm.png" alt="Swarm" width="60">
</p>

<p align="center">
  <b><a href="https://swarm124.com">Swarm</a> — where AI learns to fly.</b><br/>
  <sub>Subnet 124 on <a href="https://bittensor.com">Bittensor</a></sub>
</p>
