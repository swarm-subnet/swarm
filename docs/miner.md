# ⛏️ Swarm Miner Guide
*(Swarm subnet)*

The Swarm subnet tasks your miner with developing pre‑trained flight‑control policies which dynamically generate safe flight paths for a simulated drone across a procedurally generated world. 
This guide shows how to install, configure and run a Swarm miner

## 🔒 RPC Submission Requirements

**CRITICAL**: All submissions must be RPC agents. Validators evaluate agents in secure Docker containers.

### Required RPC Agent Structure
Your submission ZIP **must contain**:
```
agent_submission.zip
├── main.py              ← REQUIRED entry point
├── drone_agent.py       ← Your flight controller
├── agent_server.py      ← Cap'n Proto RPC server
├── agent.capnp          ← RPC schema
└── [your model files]   ← Optional: SB3, PyTorch, etc.
```

**Missing main.py = automatic rejection**

### Using SB3 Models in RPC Agent
You can use Stable Baselines 3 models inside your RPC agent:

```python
# drone_agent.py
from stable_baselines3 import PPO

class DroneFlightController:
    def __init__(self):
        self.model = PPO.load("./my_sb3_model.zip")
    
    def act(self, observation):
        action, _ = self.model.predict(observation, deterministic=True)
        return action
    
    def reset(self):
        pass
```

## 💻 System Requirements

| Component | Minimal | Recommended | Notes                                         |
|-----------|---------|-------------|-----------------------------------------------|
| CPU       | 3 cores  | 6 cores      | Model training and inference                   |
| RAM       | 8 GB     | 16 GB        | Larger for training, 8GB sufficient for mining |
| Disk      | 20 GB     | 100 GB       | Repository + virtual‑env + model storage      |
| GPU       | none     | Optional     | Depends on your training setup                |
| PyTorch   | 1.9.0+   | Latest       | **REQUIRED**: Must support `weights_only=True` |
| Python    | 3.8+     | 3.10+        | SB3 and PyTorch compatibility                 |
| OS        | Linux / macOS / WSL2 | Ubuntu 22.04+ | Scripts optimized for Ubuntu                |

## 🚀 Installation

```bash
# 1) clone the repo
git clone https://github.com/swarm-subnet/swarm
cd swarm

# 2) install dependencies
chmod +x scripts/miner/install_dependencies.sh
./scripts/miner/install_dependencies.sh

# 3) Miner setup
chmod +x scripts/miner/setup.sh
./scripts/miner/setup.sh

# 4) Activate virtual env
source miner_env/bin/activate
```

## 🔧 Configuration

All runtime parameters are passed via CLI flags; nothing needs editing inside the repo.

| Flag                   | Description                     | Example                   |
|------------------------|---------------------------------|---------------------------|
| `--netuid`             | Subnet netuid on-chain          | `--netuid 124`            |
| `--wallet.name`        | Your coldkey name               | `--wallet.name my_cold`   |
| `--wallet.hotkey`      | Hotkey used for mining          | `--wallet.hotkey my_hot`  |
| `--subtensor.network`  | Network (finney, test)          | `--subtensor.network finney` |
| `--axon.port`          | TCP port your miner listens on  | `--axon.port 8091`        |

Create the keys first if you have not:

```bash
btcli wallet new_coldkey --wallet.name my_cold
btcli wallet new_hotkey  --wallet.name my_cold --wallet.hotkey my_hot
```

## 🏃‍♂️ Running the miner (PM2 example)

```bash
source miner_env/bin/activate

pm2 start neurons/miner.py --name swarm_miner -- \
     --netuid 124 \
     --subtensor.network finney \
     --wallet.name my_cold \
     --wallet.hotkey my_hot \
     --axon.port 8091
```

Check logs:

```bash
pm2 logs swarm_miner
```

Stop / restart:

```bash
pm2 restart swarm_miner
pm2 stop     swarm_miner
```


## 🛠️ Creating RPC Agents

### Using Submission Template
Swarm provides a submission template in `swarm/submission_template/`:

```bash
# Copy template files
cp -r swarm/submission_template/* your_agent/
cd your_agent
```

### Basic RPC Agent Structure

**main.py** (entry point):
```python
from drone_agent import DroneFlightController
from agent_server import start_server

if __name__ == "__main__":
    agent = DroneFlightController()
    start_server(agent, port=8000)
```

**drone_agent.py** (your controller):
```python
class DroneFlightController:
    def __init__(self):
        # Load your model here (SB3, PyTorch, JAX, etc.)
        from stable_baselines3 import PPO
        self.model = PPO.load("./my_model.zip")
    
    def act(self, observation):
        # Return action array [vx, vy, vz, speed, yaw]
        action, _ = self.model.predict(observation, deterministic=True)
        return action
    
    def reset(self):
        # Reset state between missions
        pass
```

### Using SB3 Models
You can use any SB3 model inside your RPC agent. Just load it in `__init__` and use it in `act()`.

### Using Custom PyTorch Models
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
# Create ZIP with all required files
zip -r agent_submission.zip main.py drone_agent.py agent_server.py agent.capnp [your_model_files]

# Place in model directory
cp agent_submission.zip model/ppo_policy.zip
```

## ✈️ How the Miner Works

1. **Validator sends an empty `PolicySynapse`** to request your agent manifest.
2. **Your miner responds with a `PolicyRef`** containing the SHA256 hash, file size, and framework tag (`rpc-agent`) of your RPC agent.
3. **Validator compares the SHA‑256 to its cache.**
   - If identical → **done** (uses cached agent).
   - If different → **proceed** to download.
4. **Validator requests the agent** by sending `need_blob=True`.
5. **Your miner streams the agent** as a series of `PolicyChunk` messages until EOF.
6. **Validator stores the agent** as `miner_models_v2/UID_<uid>.zip`, extracts it, runs `main.py`, connects via RPC, and evaluates on secret tasks. Score ∈ [0, 1] is written on‑chain.


| Step | Direction | Payload | What happens |
|------|-----------|---------|--------------|
| 1 | **Validator ➜ Miner** | empty `PolicySynapse` | “Send me your manifest.” |
| 2 | **Miner ➜ Validator** | `ref` (`PolicyRef`) | Contains **sha256**, file size & framework tag (`rpc-agent`). |
| 3 | **Validator** compares the SHA‑256 to its cache. | — | If identical → **done**. If different → **proceed**. |
| 4 | **Validator ➜ Miner** | `need_blob=True` | “Stream me the new zip.” |
| 5 | **Miner ➜ Validator** | series of `chunk` messages (`PolicyChunk`) | Raw bytes until EOF. |
| 6 | **Validator** stores `miner_models_v2/UID_<uid>.zip`, loads it with SB3 and evaluates it on secret tasks. | — | Score ∈ [0 … 1] is written on‑chain. |

There is **no MapTask in the handshake**.  
Miners never see the evaluation maps; only their RPC agent is tested.

### Required Folder Layout

```
swarm/
└── model/
    └── ppo_policy.zip     ← your RPC agent submission
        ├── main.py              ← Entry point (REQUIRED)
        ├── drone_agent.py       ← Your controller
        ├── agent_server.py      ← RPC server
        ├── agent.capnp          ← RPC schema
        └── [your model files]   ← Optional: SB3, PyTorch, etc.
```

**main.py is mandatory** - missing it results in automatic rejection.

Update the path or filename in `neurons/miner.py` if you organize files differently.

## 🏆 Reward formula

| Term            | Weight | Description                                      |
|-----------------|--------|--------------------------------------------------|
| Mission success | 0.50   | 1.0 if goal reached, else 0      |
| Time factor     | 0.50   | 1.0 if time ≤ target, linear decay otherwise    |

Target time is computed as `(distance / 3.0 m/s) × 1.06` to allow a 6% buffer for optimal flight.

*Full logic: `swarm/validator/reward.py`.*


## 🔄 Updating your agent  

**ALWAYS test your agent locally before deployment:**
```bash
# Test your RPC agent locally
cd your_agent_directory
python main.py
# In another terminal, test RPC connection

# If test passes, create ZIP and deploy
zip -r agent_submission.zip main.py drone_agent.py agent_server.py agent.capnp [your_model_files]
cp agent_submission.zip model/ppo_policy.zip

# Restart miner to serve new hash
pm2 restart swarm_miner
```

The miner computes SHA‑256 at startup. Validators fetch new agents automatically at the next handshake.

## 🔧 Troubleshooting

### Agent Rejection Issues

**❌ "Missing main.py"**
```
Error: Missing main.py - RPC agent submission required
```
**Solution:** Ensure your ZIP contains `main.py` as entry point

**❌ "Dangerous executable files detected"**
```
Error: Dangerous executable files detected: [.exe, .so, .dll]
```
**Solution:** Remove executable files from your submission. Only Python code and model files allowed.

**❌ "Agent too large"**
```
Error: Agent exceeds size limit
```
**Solution:** Submissions must be ≤ **50 MiB** compressed. Reduce model size or remove unnecessary files.

**❌ "RPC connection failed"**
```
Error: RPC ping failed
```
**Solution:** Ensure your `main.py` starts RPC server on port 8000 and responds to ping requests

## 🆘 Need help?

- Discord – ping @Miguelikk or @AliSaaf
- GitHub issues – open a ticket with logs & error trace

Happy mining, and may your drones fly far 🚀!
