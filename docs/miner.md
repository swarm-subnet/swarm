# ⛏️ Swarm Miner Guide

The Swarm subnet tasks your miner with developing pre-trained flight-control policies for a simulated drone. Your model is benchmarked across procedurally generated 3D environments — cities, mountains, warehouses, forests, and open terrain.

This guide shows how to install, configure, and run a Swarm miner.

## 🔒 RPC Submission

All submissions must be RPC agents. Miner code runs in isolated Docker containers while evaluation executes on the validator host. See [Creating Your Agent](#️-creating-your-agent) for full details on structure, templates, and examples.

## 💻 System Requirements

Mining (serving your model) requires minimal resources. Training is up to you — use whatever hardware fits your approach.

| Component | Mining (Minimal) | Training | Notes |
|-----------|-----------------|----------|-------|
| CPU | 2 cores | 4+ cores | Mining is lightweight |
| RAM | 4 GB | 16 GB+ | Training depends on your setup |
| Disk | 20 GB | 100 GB+ | Repository + venv + models |
| GPU | None | Optional | Training only, depends on your approach |
| Python | 3.8+ | 3.10+ | SB3 and PyTorch compatibility |
| OS | Linux / macOS / WSL2 | Ubuntu 22.04+ | Scripts optimized for Ubuntu |

## 🚀 Installation

```bash
# 1. Clone the repo
git clone https://github.com/swarm-subnet/swarm
cd swarm

# 2. Install dependencies
chmod +x scripts/miner/install_dependencies.sh
./scripts/miner/install_dependencies.sh

# 3. Miner setup
chmod +x scripts/miner/setup.sh
./scripts/miner/setup.sh

# 4. Activate virtual env
source miner_env/bin/activate
```

## 🔧 Configuration

All runtime parameters are passed via CLI flags.

| Flag | Description | Example |
|------|-------------|---------|
| `--netuid` | Subnet netuid | `--netuid 124` |
| `--wallet.name` | Your coldkey name | `--wallet.name my_cold` |
| `--wallet.hotkey` | Hotkey used for mining | `--wallet.hotkey my_hot` |
| `--subtensor.network` | Network (finney, test) | `--subtensor.network finney` |
| `--axon.port` | TCP port your miner listens on | `--axon.port 8091` |

Create keys if you haven't:

```bash
btcli wallet new_coldkey --wallet.name my_cold
btcli wallet new_hotkey  --wallet.name my_cold --wallet.hotkey my_hot
```

## 🏃‍♂️ Running the Miner

### PM2 Launch

```bash
source miner_env/bin/activate

pm2 start neurons/miner.py --name swarm_miner -- \
     --netuid 124 \
     --subtensor.network finney \
     --wallet.name my_cold \
     --wallet.hotkey my_hot \
     --axon.port 8091
```

### Logs

```bash
pm2 logs swarm_miner
```

### Stop / Restart

```bash
pm2 restart swarm_miner
pm2 stop    swarm_miner
```

## 🛠️ Creating Your Agent

### Using the Submission Template

```bash
cp swarm/submission_template/drone_agent.py your_agent/
cd your_agent
# Customize drone_agent.py with your controller
```

**What you need:**
- `drone_agent.py` — Customize this file with your controller (REQUIRED)
- `requirements.txt` — Optional: additional dependencies
- Your model files — SB3, PyTorch, etc.

**Template files (auto-provided, no need to include):**
- `main.py` — Automatically injected during evaluation
- `agent.capnp` — Automatically injected during evaluation
- `agent_server.py` — Automatically injected during evaluation

### Basic Agent Structure

```python
class DroneFlightController:
    def __init__(self):
        # Load your model here (SB3, PyTorch, JAX, etc.)
        from stable_baselines3 import PPO
        self.model = PPO.load("./my_model.zip")

    def act(self, observation):
        # observation: dict with "depth" (128,128,1) and "state" (N,)
        # Return action array [vx, vy, vz, speed, yaw]
        action, _ = self.model.predict(observation, deterministic=True)
        return action

    def reset(self):
        # Reset state between missions
        pass
```

### Custom PyTorch Example

```python
import torch

class DroneFlightController:
    def __init__(self):
        self.model = torch.load("my_model.pt")
        self.model.eval()

    def act(self, observation):
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation)
            action = self.model(obs_tensor).numpy()
        return action
```

### Deploy Your Agent

```bash
# Create ZIP with only drone_agent.py + model files
zip -r agent_submission.zip drone_agent.py [your_model_files]

# Optional: include requirements.txt
zip -r agent_submission.zip requirements.txt

# Place in Submission directory
mkdir -p Submission
cp agent_submission.zip Submission/submission.zip
```

## 🔍 Observations

| Field | Shape | Description |
|-------|-------|-------------|
| `depth` | (128, 128, 1) | Normalized depth map (0.5m – 20m range) |
| `state` | (N,) | Position, velocity, orientation, action history, altitude, search area direction |

The search area gives a direction toward the goal with up to ±10m noise — sometimes close, sometimes off. The drone must use its depth sensor to find the actual landing platform.

### Action Space

| Index | Name | Range | Description |
|-------|------|-------|-------------|
| 0 | vx | [-1, 1] | Velocity X direction |
| 1 | vy | [-1, 1] | Velocity Y direction |
| 2 | vz | [-1, 1] | Velocity Z direction |
| 3 | speed | [0, 1] | Thrust multiplier |
| 4 | yaw | [-1, 1] | Target yaw angle (maps to [-π, π]) |

**Constraints:**
- Max velocity: 3.0 m/s
- Max yaw rate: 3.141 rad/s (180°/s)
- Simulation rate: 50 Hz

## 🏆 Scoring

```
score = 0.45 × success + 0.45 × time + 0.10 × safety
```

| Term | Weight | Description |
|------|--------|-------------|
| **Success** | 0.45 | 1.0 if valid landing, 0.0 otherwise |
| **Time** | 0.45 | 1.0 if within target time, decays linearly to 0.0 at horizon |
| **Safety** | 0.10 | Based on minimum clearance from obstacles during flight |

Collision with any obstacle sets the score to **0.01**.

### Landing

For **static platforms**, touching is not enough — the drone must hold a **stable landing** for 0.5 seconds:

| Condition | Threshold |
|-----------|-----------|
| Vertical velocity | ≤ 0.5 m/s |
| Horizontal velocity | ≤ 0.6 m/s (relative to platform) |
| Tilt (roll / pitch) | ≤ 15° |

For **moving platforms**, contact with the platform is enough to count as success.

## ✈️ How the Miner Works

| Step | Direction | What happens |
|------|-----------|--------------|
| 1 | Validator → Miner | Empty `PolicySynapse` — "Send me your manifest." |
| 2 | Miner → Validator | `PolicyRef` with SHA-256 hash, file size, framework tag (`rpc-agent`). |
| 3 | Validator | Compares hash to cache. If identical → done. If different → proceed. |
| 4 | Validator → Miner | `need_blob=True` — "Stream me the new zip." |
| 5 | Miner → Validator | Series of `PolicyChunk` messages until EOF. |
| 6 | Validator | Stores agent, runs in Docker, evaluates on benchmark seeds. |

Miners never see the evaluation maps — only the RPC agent is tested.

### Required Folder Layout

```
swarm/
└── Submission/
    └── submission.zip
        ├── drone_agent.py       ← Your controller (REQUIRED)
        ├── requirements.txt     ← Optional
        └── [your model files]   ← Optional
```

**`drone_agent.py` is mandatory** — missing it results in automatic rejection. Submissions must be ≤ **50 MiB** compressed.

## 🔄 Updating Your Agent

```bash
# Create ZIP with your files
zip -r agent_submission.zip drone_agent.py [your_model_files]

# Place in Submission directory
mkdir -p Submission
cp agent_submission.zip Submission/submission.zip

# Restart miner to serve new hash
pm2 restart swarm_miner
```

The miner computes SHA-256 at startup. Validators fetch new agents automatically at the next handshake.

## 🧪 Local Testing

```bash
# Test your agent locally
python tests/test_rpc.py swarm/submission_template/ --seed 42

# Test and create submission.zip automatically
python tests/test_rpc.py swarm/submission_template/ --zip
```

## 🔧 Troubleshooting

**"Missing drone_agent.py"** — Ensure your ZIP contains `drone_agent.py`. Template files are auto-injected.

**"Dangerous executable files detected"** — Remove `.exe`, `.so`, `.dll` files. Only Python code and model files allowed.

**"Agent too large"** — Submissions must be ≤ 50 MiB compressed.

**"RPC connection failed"** — Ensure your agent starts correctly and responds to ping requests.

## 🆘 Support

- Discord — ping @Miguelikk or @AliSaaf
- GitHub issues — open a ticket with logs & error trace

Happy mining, and may your drones fly far 🚀
