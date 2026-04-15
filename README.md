<a id="readme-top"></a>

<p align="center">
  <img src="https://raw.githubusercontent.com/swarm-subnet/swarm/main/swarm/assets/Swarm_2.png" alt="Swarm" width="60%" />
</p>

<h1 align="center">Swarm — Autonomous Drone Navigation</h1>

<p align="center">
  <b>The open benchmark where AI learns to fly.</b><br/>
  <i>Train a neural network to navigate drones through 3D worlds it has never seen —<br/>
  using nothing but a depth camera and raw flight state. No maps. No rules. No shortcuts.</i>
</p>

<p align="center">
  <a href="https://github.com/swarm-subnet/swarm/releases"><img alt="Version" src="https://img.shields.io/badge/version-v4.0.0-green?style=flat-square" /></a>
  <a href="https://discord.gg/8dPqPDw7GC"><img alt="Discord" src="https://img.shields.io/badge/Discord-Join-5865F2?style=flat-square&logo=discord&logoColor=white" /></a>
  <a href="https://x.com/SwarmSubnet"><img alt="X" src="https://img.shields.io/badge/X-Follow-000000?style=flat-square&logo=x&logoColor=white" /></a>
  <a href="https://swarm124.com"><img alt="Website" src="https://img.shields.io/badge/swarm124.com-visit-orange?style=flat-square&logo=googlechrome&logoColor=white" /></a>
</p>

<p align="center">
  <a href="docs/miner.md">
    <img alt="Start Training" src="https://img.shields.io/badge/Start%20Training-Miner%20Guide-111111?style=for-the-badge" />
  </a>
  &nbsp;
  <a href="#cli">
    <img alt="Run Benchmark" src="https://img.shields.io/badge/Run%20Benchmark-CLI-111111?style=for-the-badge" />
  </a>
  &nbsp;
  <a href="docs/validator.md">
    <img alt="Run Validator" src="https://img.shields.io/badge/Run%20Validator-Guide-111111?style=for-the-badge" />
  </a>
</p>

---

<details>
  <summary><b>Table of Contents</b></summary>
  <ol>
    <li><a href="#about-swarm">About Swarm</a></li>
    <li><a href="#see-it-fly">See It Fly</a></li>
    <li><a href="#environments">Environments</a></li>
    <li><a href="#cli">CLI</a></li>
    <li><a href="#how-it-works">How It Works</a></li>
    <li><a href="#scoring">Scoring</a></li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#community">Community</a></li>
    <li><a href="#from-simulation-to-reality">From Simulation to Reality</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>

---

<!-- ABOUT SWARM -->
## About Swarm

Delivery, inspection, search and rescue — autonomous drones are being deployed everywhere, but the AI behind them is still developed behind closed doors. There's no standard way to measure if one flight policy is better than another.

**Swarm changes that.** It's an open benchmark that puts every model on equal footing: 1,000 procedurally generated worlds, containerized evaluation, and a public [leaderboard](https://swarm124.com/benchmark). No data leaks, no memorization — just raw skill.

The rules are simple:
- Your model gets a **128×128 depth image** and a **state vector**
- It outputs **velocity commands** to fly the drone
- It has **60 seconds** to navigate to a landing platform
- It must do this across **cities, mountains, warehouses, forests, open terrain, and more** — environments it has never seen before

The best model wins. That's it. Powered by the [Bittensor](https://bittensor.com) network (Subnet 124).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

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

---

<!-- ENVIRONMENTS -->
## Environments

Every benchmark run generates unique worlds. Six environment types test completely different navigation skills — tight urban corridors, open-air precision, mountain terrain, village streets, indoor obstacle courses, and dense forests.

<table>
<tr>
<td align="center" width="50%">
<img src="https://raw.githubusercontent.com/swarm-subnet/swarm/main/swarm/assets/map_images/Type1_sub2.png" alt="City" width="100%">
<br><b>City</b> — dense streets, buildings, intersections
</td>
<td align="center" width="50%">
<img src="https://raw.githubusercontent.com/swarm-subnet/swarm/main/swarm/assets/map_images/Type3_sub2.png" alt="Ski Village" width="100%">
<br><b>Ski Village</b> — snow-roofed buildings, mountain backdrop
</td>
</tr>
<tr>
<td align="center" width="50%">
<img src="https://raw.githubusercontent.com/swarm-subnet/swarm/main/swarm/assets/map_images/Type3.png" alt="Mountains" width="100%">
<br><b>Mountains</b> — procedural terrain, peaks and valleys
</td>
<td align="center" width="50%">
<img src="https://raw.githubusercontent.com/swarm-subnet/swarm/main/swarm/assets/map_images/Type4_2.png" alt="Warehouse" width="100%">
<br><b>Warehouse</b> — indoor, racks, cranes, 12m ceiling
</td>
</tr>
</table>

<br>

<h3 align="center">Forest — 4 Seasonal Modes</h3>

<table>
<tr>
<td align="center" width="50%">
<img src="https://raw.githubusercontent.com/swarm-subnet/swarm/main/swarm/assets/map_images/Type6_sub1.png" alt="Forest Normal" width="100%">
<br><b>Normal</b> — green canopy, full foliage
</td>
<td align="center" width="50%">
<img src="https://raw.githubusercontent.com/swarm-subnet/swarm/main/swarm/assets/map_images/Type6_sub2.png" alt="Forest Autumn" width="100%">
<br><b>Autumn</b> — orange and brown tones
</td>
</tr>
<tr>
<td align="center" width="50%">
<img src="https://raw.githubusercontent.com/swarm-subnet/swarm/main/swarm/assets/map_images/Type6_sub3.png" alt="Forest Snow" width="100%">
<br><b>Snow</b> — white terrain, bare branches
</td>
<td align="center" width="50%">
<img src="https://raw.githubusercontent.com/swarm-subnet/swarm/main/swarm/assets/map_images/Type6_sub4.png" alt="Forest Dead" width="100%">
<br><b>Dead</b> — no leaves, dark ground
</td>
</tr>
</table>

<p align="center">
  <sub>1,000 unique seeds per epoch — 6 environment types, each procedurally generated with unique layouts every run.</sub>
</p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

<!-- CLI -->
## CLI

Develop, test, and benchmark your model without ever leaving the terminal.

```bash
pip install -e .
```

Once published on PyPI:

```bash
pip install swarm-sotapilot
```

```bash
swarm doctor                                          # Check environment readiness
swarm champion                                        # Download the current champion model
swarm model test --source my_agent/                   # Validate source folder
swarm model package --source my_agent/                # Bundle into Submission/submission.zip
swarm model verify --model Submission/submission.zip  # Verify structure and compliance
swarm benchmark --model Submission/submission.zip --workers 4  # Run benchmark
swarm benchmark --model Submission/submission.zip --seeds-per-group 1  # Quick test
swarm report                                          # View results
```

<p align="center">
  <sub>Full docs: <a href="docs/CLI_readme.md">CLI reference</a>.</sub>
</p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

<!-- HOW IT WORKS -->
## How It Works

<table align="center">
<tr>
<td align="center"><b>Depth Camera</b><br><sub>128×128 image</sub></td>
<td align="center" rowspan="2"><b>&nbsp;→&nbsp; Your Model &nbsp;→&nbsp;</b></td>
<td align="center" rowspan="2"><b>Flight Commands</b><br><sub>[dir_x, dir_y, dir_z, speed, yaw]</sub></td>
<td align="center" rowspan="2"><b>&nbsp;→&nbsp; Drone</b></td>
</tr>
<tr>
<td align="center"><b>State Vector</b><br><sub>position · velocity · orientation</sub></td>
</tr>
</table>

<br/>

Your model receives a depth image and flight state at 50 Hz. It outputs 5D velocity commands. The drone has 60 seconds to navigate to a landing platform and touch down safely.

There are no waypoints, no GPS, no obstacle coordinates. The model must learn to read the depth image and react — just like a real pilot would.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

<!-- SCORING -->
## Scoring

| Component | Weight | What It Measures |
|-----------|--------|-----------------|
| **Success** | 45% | Did the drone land on the platform? |
| **Speed** | 45% | How fast, relative to the time limit? |
| **Safety** | 10% | Minimum clearance from obstacles during flight |

```
score = 0.45 × success + 0.45 × time + 0.10 × safety
```

Ranking is by **average score across 1,000 seeds**. No lucky runs — you need consistency. New models must pass a screening gate (champion score + 0.015) before running the full benchmark.

Seeds rotate every **7 days** (Monday 16:00 UTC). Each validator generates its own 1,000 seeds per epoch. All seeds are published on [swarm124.com](https://swarm124.com) for transparency.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

<!-- GETTING STARTED -->
## Getting Started

<table>
<tr>
<td align="center" width="50%">
<h3>Train a Model</h3>
<p>Build a drone pilot from zero. The <a href="docs/miner.md">Miner Guide</a> covers the agent interface, CLI workflow, submission format, and how to push to the leaderboard.</p>
<a href="docs/miner.md">
  <img alt="Miner Guide" src="https://img.shields.io/badge/Miner%20Guide-Start%20Training-111111?style=for-the-badge" />
</a>
</td>
<td align="center" width="50%">
<h3>Run a Validator</h3>
<p>Evaluate models on your hardware. The <a href="docs/validator.md">Validator Guide</a> covers Docker setup, PM2 launch, and auto-updates.</p>
<a href="docs/validator.md">
  <img alt="Validator Guide" src="https://img.shields.io/badge/Validator%20Guide-Get%20Started-111111?style=for-the-badge" />
</a>
</td>
</tr>
</table>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

<!-- COMMUNITY -->
## Community

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

<!-- FROM SIMULATION TO REALITY -->
## From Simulation to Reality

The models trained here don't stay in simulation.

<p align="center">
  <a href="https://github.com/swarm-subnet/Langostino">
    <img src="https://raw.githubusercontent.com/swarm-subnet/Langostino/main/assets/banner_section_1.png" alt="Langostino — The Swarm Drone" width="80%" />
  </a>
</p>

**[Langostino](https://github.com/swarm-subnet/Langostino)** is the open-source drone we built to prove it — ROS2, Raspberry Pi, INAV, 3D-printed parts. Full assembly guide and bill of materials included. Anyone can build one.

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
  <b>Train in Simulation</b> &nbsp;→&nbsp; <b>Compete on the Leaderboard</b> &nbsp;→&nbsp; <b>Deploy on Real Hardware</b>
</p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <img src="https://raw.githubusercontent.com/swarm-subnet/swarm/main/swarm/assets/Swarm.png" alt="Swarm" width="60">
</p>

<p align="center">
  <b><a href="https://swarm124.com">Swarm</a> — where AI learns to fly.</b><br/>
  <sub>Subnet 124 on <a href="https://bittensor.com">Bittensor</a></sub>
</p>
