<div align="center">
  <h1>🐝 <strong>Swarm</strong> – Bittensor Drone autopilot Subnet 🐝</h1>
  <img src="swarm/assets/Swarm_2.png" alt="Swarm"  width="500">
  <p>
    <a href="docs/miner.md">🚀 Miner guide</a> &bull;
    <a href="docs/validator.md">🔐 Validator guide</a> &bull;
    <a href="docs/roadmap.md">🗺️ Roadmap</a> &bull;
    <a href="https://x.com/SwarmSubnet">🐦 Follow us on X</a> &bull;
    <a href="https://swarm124.com/">🌐 Web & Leaderboard </a>
    <br>
    <a href="https://discord.com/channels/799672011265015819/1385341501130801172">💬 Join us on Discord</a>
  </p>
</div>

## 🔍 Overview
Swarm is a **Bittensor subnet engineered to enable decentralized autonomous drone flight**.

Validators create synthetic map tasks and evaluate miner-supplied **pre-trained RL policies** inside secure Docker containers using PyBullet physics simulation.

Miners that produce fast and *successful* policies earn the highest rewards

**Why OS drone flying?**

- Open-sourcing flight algorithms isn't just idealism – it is a practical route to safer, cheaper and more accountable drones, and it prevents the future of aerial autonomy from being locked behind half a dozen NDAs

- Our ambition is to establish Swarm miners as the **go‑to control intelligence for micro‑drone navigation** in research and industry.

---
## ⚙️ Subnet Mechanics

### 🧑‍🏫 Validator

- Generates unique MapTasks  
- Evaluates policies head‑less and validates them
- Assigns weights proportional to the final reward score

### ⛏️ Miner

- Provides RPC agents that are evaluated on secret tasks
- Any framework is allowed – Stable Baselines 3, PyTorch, JAX, or custom implementations
- Must submit RPC agent with main.py entry point

---

## Swarm Core components

| Component             | Purpose                           | Key points (code refs)                                                      |
|-----------------------|-----------------------------------|------------------------------------------------------------------------------|
| **MapTask**           | Internal validator task         | Random start→goal pair, simulation time‑step `sim_dt`, hard time limit `horizon` (`swarm.protocol.MapTask`) |
| **PolicyRef**         | Model metadata                    | SHA256, framework, size (`swarm.protocol.PolicyRef`) |
| **Policy Evaluation** | RL model testing                 | Evaluates RPC agents on secret tasks in Docker (`swarm.core.evaluator`) |
| **Reward**            | Maps outcome → [0,1] score        | 0.45 × success + 0.45 × time + 0.10 × safety (`swarm.validator.reward.flight_reward`) |

### Observation Contract

| Field | Shape | Description |
|-------|-------|-------------|
| depth | (128, 128, 1) | Normalized depth map (0.0 = near, 1.0 = far) |
| state | (21,) | Drone state vector (position, velocity, orientation, etc.) |

**Note:** RGB images are not provided. Miners must use depth-only observations.

### Task generation

Tasks use fixed benchmark seed pools (1000 public + 200 private). Goals are placed 5-45 meters away at random altitude with procedural obstacles based on challenge types (1-3).

```python
# swarm/validator/task_gen.py
distance = rng.uniform(5.0, 45.0)   # meters
challenge_type = rng.choice([1, 2, 3])  # City, Open, Moving Platform
```

### Challenge Types

| Type | Name | Description |
|------|------|-------------|
| 1 | City Navigation | Dense procedural buildings, requires obstacle avoidance |
| 2 | Open Flight | No obstacles, tests pure navigation efficiency |
| 3 | Moving Platform | Goal platform moves in patterns (circular, linear, figure8) |

### Benchmark Flow
The validator:

1. Detects new/changed models via SHA256 hash comparison
2. Runs screening (200 private seeds) to filter low-quality models
3. Runs full benchmark (1000 public seeds) for passing models
4. Submits scores to central backend for aggregation
5. Backend calculates final weights (51% stake median)
6. Applies winner-take-all rewards with 95% burn

Here is an example image of our GUI!

<div align="center">
<img src="swarm/assets/drone_image.png" alt="Drone"  width="300">
</div>

---

## 🎯 Reward Mechanism

### Performance Scoring
| Term        | Weight | Rationale                               |
|-------------|--------|-----------------------------------------|
| Success     | 0.45   | Goal reached with valid landing         |
| Time        | 0.45   | Speed optimization with target time     |
| Safety      | 0.10   | Minimum clearance from obstacles        |

### Landing Requirements

Touching the platform is not enough. Drones must achieve a **stable landing**:

| Requirement | Threshold | Description |
|-------------|-----------|-------------|
| Vertical Velocity | ≤ 0.5 m/s | Must descend slowly |
| Horizontal Velocity | ≤ 0.6 m/s | Relative to platform (handles moving platforms) |
| Orientation | ≤ 15° tilt | Roll and pitch must be low |
| Stable Duration | 0.5 seconds | All conditions must hold continuously |

### Safety Scoring

| Condition | Safety Score |
|-----------|--------------|
| min_clearance ≥ 1.0m | 1.0 (full) |
| min_clearance ≤ 0.2m | 0.0 (none) |
| Between 0.2m - 1.0m | Linear interpolation |
| Collision | 0.0 (entire score = 0.01) |

### Reward Distribution

**Winner-Take-All**: Top performer gets 5% of emissions (95% burned), all others get 0%.

Models are evaluated once with 1200 benchmark seeds. Scores are cached to avoid re-evaluation.

---

## 🤝 Contributing
PRs, issues and benchmark ideas are welcome!  

---

## 📜 License
Licensed under the MIT License – see LICENSE.

Built with ❤️ by the Swarm team.
