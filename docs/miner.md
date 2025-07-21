# ⛏️ Swarm Miner Guide
*(Swarm subnet)*

The Swarm subnet tasks your miner with developing pre‑trained flight‑control policies which dynamically generate safe, energy‑efficient flight paths for a simulated drone across a procedurally generated world. 
This guide shows how to install, configure and run a Swarm miner

## 💻 System requirements to run the default miner code

| Component | Minimal | Recommended | Notes                                         |
|-----------|---------|-------------|-----------------------------------------------|
| CPU       | 3 cores  | 6 cores      | Path‑planning is light‑weight                 |
| RAM       | 8 GB     | 8 GB         |                                               |
| Disk      | 20 GB     | 100 GB         | Repository + virtual‑env                      |
| CPU       | 3 cores  | 6 cores      | Path‑planning is light‑weight                 |
| RAM       | 8 GB     | 8 GB         |                                               |
| Disk      | 20 GB     | 100 GB         | Repository + virtual‑env                      |
| GPU       | none     | Optional     | Depends on your model             |
| OS        | Linux / macOS / WSL2 | —           | Scripts are written for Ubuntu 22.04          |

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


## ✈️ How does the miner work now?

1. **Validator sends an empty `PolicySynapse`** to request your model manifest.
2. **Your miner responds with a `PolicyRef`** containing the SHA256 hash, file size, and framework tag (`sb3‑ppo`) of your trained model.
3. **Validator compares the SHA‑256 to its cache.**
   - If identical → **done** (uses cached model).
   - If different → **proceed** to download.
4. **Validator requests the model** by sending `need_blob=True`.
5. **Your miner streams the model** as a series of `PolicyChunk` messages until EOF.
6. **Validator stores the model** as `miner_models/UID_<uid>.zip`, loads it with SB3, and evaluates it on secret tasks. Score ∈ [0, 1] is written on‑chain.


| Step | Direction | Payload | What happens |
|------|-----------|---------|--------------|
| 1 | **Validator ➜ Miner** | empty `PolicySynapse` | “Send me your manifest.” |
| 2 | **Miner ➜ Validator** | `ref` (`PolicyRef`) | Contains **sha256**, file size & framework tag (`sb3‑ppo`). |
| 3 | **Validator** compares the SHA‑256 to its cache. | — | If identical → **done**. If different → **proceed**. |
| 4 | **Validator ➜ Miner** | `need_blob=True` | “Stream me the new zip.” |
| 5 | **Miner ➜ Validator** | series of `chunk` messages (`PolicyChunk`) | Raw bytes until EOF. |
| 6 | **Validator** stores `miner_models/UID_<uid>.zip`, loads it with SB3 and evaluates it on secret tasks. | — | Score ∈ [0 … 1] is written on‑chain. |

There is **no MapTask in the handshake**.  
Miners never see the evaluation maps; only their exported policy is tested.

### Folder layout expected by the reference miner

swarm/
└── model/
    └── ppo_policy.zip     ← your trained SB3 PPO policy
   
Update the path or filename in neurons/miner.py if you organise files differently.

## 🏆 Reward formula

| Term            | Weight | Description                                      |
|-----------------|--------|--------------------------------------------------|
| Mission success | 0.70   | 1.0 if goal reached, else 0                      |
| Time factor     | 0.15   | 1 − t_goal / horizon, clamped to [0,1]           |
| Energy factor   | 0.15   | 1 − E_used / E_budget, clamped to [0,1]          |

*Full logic: `swarm/validator/reward.py`.*


## 🔄 Updating your model  

Simply overwrite `model/ppo_policy.zip` with a new file; the miner computes
its SHA‑256 at start‑up. Restart the process (or run `pm2 reload`) to serve
the new hash. Validators will fetch it automatically at the next handshake.

## 🆘 Need help?


- Discord – ping @Miguelikk or @AliSaaf
- GitHub issues – open a ticket with logs & error trace

Happy mining, and may your drones fly far 🚀!
