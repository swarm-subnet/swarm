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
| **Reward**            | Maps outcome → [0,1] score        | 0.50 × success + 0.50 × time (`swarm.validator.reward.flight_reward`) |

### Task generation

Tasks are generated with synchronized seeds across all validators using 10-minute time windows. Goals are placed 5-15 meters away at random altitude with procedural obstacles based on challenge types (1-4).

```python
# swarm/validator/task_gen.py
distance = rng.uniform(5.0, 15.0)   # meters
challenge_type = rng.choice([1, 2, 3, 4])
```

### Validation loop  
The validator:

1. Generates synchronized seed based on time window
2. Samples miners and requests their models
3. Downloads new models and performs multi-layer fake detection
4. Evaluates each policy in isolated Docker container
5. Computes normalized scores across challenge types
6. Applies winner-take-all rewards and writes weights to chain

Here is an example image of our GUI!

<div align="center">
<img src="swarm/assets/drone_image.png" alt="Drone"  width="300">
</div>

---

## 🎯 Reward Mechanism

### Performance Scoring
| Term        | Weight | Rationale                               |
|-------------|--------|-----------------------------------------|
| Success     | 0.50   | Goal reached                            |
| Time        | 0.50   | Speed optimization with target time     |

### Reward Distribution

**Winner-Take-All**: Top performer gets 25% of the emissions, all others get 0%.

Miners must complete minimum 25 runs before eligibility. Scores are normalized across all challenge types for fair comparison.

---

## 🤝 Contributing
PRs, issues and benchmark ideas are welcome!  

---

## 📜 License
Licensed under the MIT License – see LICENSE.

Built with ❤️ by the Swarm team.
