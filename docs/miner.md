# ⛏️ Swarm Miner Guide
*(Swarm subnet)*

The Swarm subnet tasks your miner with developing pre‑trained flight‑control policies which dynamically generate safe, energy‑efficient flight paths for a simulated drone across a procedurally generated world. 
This guide shows how to install, configure and run a Swarm miner

## 🔒 Model Security Requirements

**CRITICAL**: Validators use Docker-based secure evaluation with **strict model requirements**. Models missing required security metadata are **automatically rejected** and scored 0.0.

### Why Security Matters
- **Untrusted code execution**: Your models run inside validator Docker containers for security isolation
- **Malicious model protection**: Validators scan for fake models and maintain shared blacklists
- **Secure loading**: Models must support PyTorch `weights_only=True` loading (no pickle execution)
- **Automatic rejection**: Non-compliant models are rejected before evaluation

### Required Model Structure
Your model ZIP **must contain exactly these files**:
```
ppo_policy.zip
├── policy.pth              ← PyTorch weights (no pickle objects)
└── safe_policy_meta.json   ← REQUIRED security metadata
```

**Missing either file = automatic rejection**

### safe_policy_meta.json Requirements
This JSON file enables secure weights-only loading and **must contain**:
```json
{
  "activation_fn": "relu",
  "net_arch": {"pi": [64, 64], "vf": [64, 64]},
  "use_sde": false
}
```

**Required fields:**
- `activation_fn`: Activation function name (relu, tanh, elu, leakyrelu, silu, gelu, mish, selu, celu)
- `net_arch`: Network architecture - `{"pi": [...], "vf": [...]}` for policy/value networks
- `use_sde`: State-dependent exploration boolean (usually `false`)

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
# 1) clone the repo (no sub‑modules required)
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
```bash
sudo apt update && sudo apt install -y \
     build-essential git pkg-config libgl1-mesa-glx mesa-utils
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
source miner_env/bin/activate      # if not already active

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


## 🛠️ Creating Compliant Models

### Basic Training Scripts (Starting Points)
Swarm provides **basic example scripts** in `RL/` that miners should **improve upon**:

**Basic Training Example:**
```bash
# Basic PPO training example - IMPROVE THIS for better performance
python swarm/RL/train_RL_with_info.py
```

**Model Compliance Testing:**
```bash
# Test ANY model for security compliance (ALWAYS use this)
python -m RL.test_secure_RL --model model/ppo_policy.zip [--seed 42] [--gui]
```

⚠️ **The provided scripts are minimal examples** - successful miners develop their own advanced training approaches.

### Making ANY Model Compliant

**Step 1: Export Compatible Format**
Whatever training framework you use, export your model as:
- `policy.pth` - PyTorch state dict (weights only)
- `safe_policy_meta.json` - Metadata for secure loading

**Step 2: Create Metadata File**
Generate `safe_policy_meta.json` matching your model architecture:
```json
{
  "activation_fn": "relu",
  "net_arch": {"pi": [64, 64], "vf": [64, 64]},
  "use_sde": false
}
```

**Step 3: Test Compliance**
```bash
python -m RL.test_secure_RL --model your_model.zip
```

**Step 4: Deploy**
Place compliant model in `model/ppo_policy.zip`

### Advanced Training Approaches
Successful miners typically:
- Use advanced RL algorithms beyond basic PPO
- Implement curriculum learning for complex navigation
- Optimize for speed, energy efficiency, and success rate
- Use custom reward shaping and exploration strategies
- Train on diverse procedurally generated environments

The provided scripts are **starting points** - build better ones to compete effectively.

## ✈️ How the Miner Works

1. **Validator sends an empty `PolicySynapse`** to request your model manifest.
2. **Your miner responds with a `PolicyRef`** containing the SHA256 hash, file size, and framework tag (`sb3‑ppo`) of your trained model.
3. **Validator compares the SHA‑256 to its cache.**
   - If identical → **done** (uses cached model).
   - If different → **proceed** to download.
4. **Validator requests the model** by sending `need_blob=True`.
5. **Your miner streams the model** as a series of `PolicyChunk` messages until EOF.
6. **Validator stores the model** as `miner_models_v2/UID_<uid>.zip`, loads it with SB3, and evaluates it on secret tasks. Score ∈ [0, 1] is written on‑chain.


| Step | Direction | Payload | What happens |
|------|-----------|---------|--------------|
| 1 | **Validator ➜ Miner** | empty `PolicySynapse` | “Send me your manifest.” |
| 2 | **Miner ➜ Validator** | `ref` (`PolicyRef`) | Contains **sha256**, file size & framework tag (`sb3‑ppo`). |
| 3 | **Validator** compares the SHA‑256 to its cache. | — | If identical → **done**. If different → **proceed**. |
| 4 | **Validator ➜ Miner** | `need_blob=True` | “Stream me the new zip.” |
| 5 | **Miner ➜ Validator** | series of `chunk` messages (`PolicyChunk`) | Raw bytes until EOF. |
| 6 | **Validator** stores `miner_models_v2/UID_<uid>.zip`, loads it with SB3 and evaluates it on secret tasks. | — | Score ∈ [0 … 1] is written on‑chain. |

There is **no MapTask in the handshake**.  
Miners never see the evaluation maps; only their exported policy is tested.

### Required Folder Layout

```
swarm/
└── model/
    └── ppo_policy.zip     ← your trained SB3 PPO policy
        ├── policy.pth              ← PyTorch weights (REQUIRED)
        └── safe_policy_meta.json   ← Security metadata (REQUIRED)
```

**Both files inside the ZIP are mandatory** - missing either file results in automatic model rejection.

Update the path or filename in `neurons/miner.py` if you organize files differently.

## 🏆 Reward formula

| Term            | Weight | Description                                      |
|-----------------|--------|--------------------------------------------------|
| Mission success | 0.70   | 1.0 if goal reached, else 0                      |
| Time factor     | 0.15   | 1 − t_goal / horizon, clamped to [0,1]           |
| Energy factor   | 0.15   | 1 − E_used / E_budget, clamped to [0,1]          |

*Full logic: `swarm/validator/reward.py`.*


## 🔄 Updating your model  

**ALWAYS test compliance locally before deployment:**
```bash
# Test your new model BEFORE deploying
python -m RL.test_secure_RL --model your_new_model.zip

# If test passes, deploy to miner
cp your_new_model.zip model/ppo_policy.zip

# Restart miner to serve new hash
pm2 restart swarm_miner
```

The miner computes SHA‑256 at startup. Validators fetch new models automatically at the next handshake.

## 🔧 Troubleshooting

### Model Rejection Issues

**❌ "Missing safe_policy_meta.json"**
```
Error: Model missing secure metadata
```
**Solution:** Create `safe_policy_meta.json` with required fields inside your model ZIP

**❌ "Invalid JSON structure"**  
```
Error: Invalid JSON in safe_policy_meta.json
```
**Solution:** Check JSON syntax and ensure all required fields: `activation_fn`, `net_arch`, `use_sde`

**❌ "Model too large"**
```
Error: Model exceeds size limit
```
**Solution:** Models must be ≤ **10 MiB** compressed. Reduce network size or remove unnecessary files.

**❌ "PyTorch weights_only not supported"**
```
Error: PyTorch version doesn't support weights_only=True
```
**Solution:** Upgrade PyTorch: `pip install torch>=1.9.0`

### Model Compliance Check
```bash
# Always test before deployment
python -m RL.test_secure_RL --model model/ppo_policy.zip
```

## 🆘 Need help?


- Discord – ping @Miguelikk or @AliSaaf
- GitHub issues – open a ticket with logs & error trace

Happy mining, and may your drones fly far 🚀!
