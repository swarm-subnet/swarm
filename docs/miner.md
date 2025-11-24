# ⛏️ Swarm Miner Guide
*(Swarm subnet)*

The Swarm subnet tasks your miner with developing pre‑trained flight‑control policies which dynamically generate safe flight paths for a simulated drone across a procedurally generated world. 
This guide shows how to install, configure and run a Swarm miner

## 🔒 RPC Submission Requirements

**CRITICAL**: All submissions must be RPC agents. Miner code runs in isolated Docker containers while evaluation executes on the validator host.

### Required RPC Agent Structure
Your submission ZIP **must contain**:
```
agent_submission.zip
├── drone_agent.py       ← Your flight controller (REQUIRED)
├── requirements.txt     ← Optional: additional dependencies
└── [your model files]   ← SB3, PyTorch, etc.
```

**Template files (`main.py`, `agent.capnp`, `agent_server.py`) are automatically provided by validators** - you don't need to include them in your submission.

### Template Files (Auto-Injected)

The following files are automatically injected from official templates during evaluation:
- `main.py` - Entry point (provided automatically)
- `agent.capnp` - RPC schema (provided automatically)
- `agent_server.py` - Cap'n Proto RPC server (provided automatically)

**You only need to submit `drone_agent.py`** - this is where you implement your controller and load your model.

All customization belongs in `drone_agent.py` where you have complete freedom to:
- Load any ML framework (SB3, PyTorch, TensorFlow, JAX)
- Implement custom preprocessing/postprocessing
- Use any model architecture

### Optional: requirements.txt

If your agent needs additional Python packages, include a `requirements.txt` file:
```
numpy>=1.20.0
torch>=1.9.0
stable-baselines3>=2.0.0
```

Dependencies will be installed automatically before evaluation.



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
# Copy only drone_agent.py (template files are auto-injected)
cp swarm/submission_template/drone_agent.py your_agent/
cd your_agent

# Customize drone_agent.py with your controller
```

**What you need:**
- `drone_agent.py` - Customize this file with your controller (REQUIRED)
- `requirements.txt` - Optional: additional dependencies
- Your model files - SB3, PyTorch, etc.

**Template files (auto-provided, no need to include):**
- `agent.capnp` - Automatically injected during evaluation
- `agent_server.py` - Automatically injected during evaluation
- `main.py` - Automatically injected during evaluation

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
# Create ZIP with only drone_agent.py + model files
zip -r agent_submission.zip drone_agent.py [your_model_files]

# Optional: include requirements.txt if needed
zip -r agent_submission.zip requirements.txt

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
        ├── drone_agent.py       ← Your controller (REQUIRED)
        ├── requirements.txt     ← Optional: additional dependencies
        └── [your model files]   ← Optional: SB3, PyTorch, etc.
```

**drone_agent.py is mandatory** - missing it results in automatic rejection.

Template files (`main.py`, `agent.capnp`, `agent_server.py`) are automatically injected during evaluation.

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
# Test your RPC agent locally (copy template files for local testing)
cp swarm/submission_template/* your_agent_directory/
cd your_agent_directory
python main.py
# In another terminal, test RPC connection

# If test passes, create ZIP with only your files
zip -r agent_submission.zip drone_agent.py [your_model_files]
# Optional: include requirements.txt
zip -r agent_submission.zip requirements.txt

cp agent_submission.zip model/ppo_policy.zip

# Restart miner to serve new hash
pm2 restart swarm_miner
```

The miner computes SHA‑256 at startup. Validators fetch new agents automatically at the next handshake.

## 🧪 Test Before Submitting

```bash
python tests/test_rpc.py swarm/submission_template/ --seed 42
```

Test your agent locally using the same evaluation logic as validators.

---

## 🔧 Troubleshooting

### Agent Rejection Issues

**❌ "Missing drone_agent.py"**
```
Error: Missing drone_agent.py - RPC agent submission required
```
**Solution:** Ensure your ZIP contains `drone_agent.py`. Template files (`main.py`, `agent.capnp`, `agent_server.py`) are automatically provided - you don't need to include them.

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
