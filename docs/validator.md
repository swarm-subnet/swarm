# 🚀 Swarm Validator Guide
*(Swarm subnet – netuid 124)*

This document shows how to install and operate the Swarm validator that evaluates models received from miners on dynamically generated maps

## 🖥️ System requirements

| Resource | Minimal | Notes                                |
|----------|---------|--------------------------------------|
| CPU      | 3 cores  | Miners are evaluated 1 by 1, no no need for much spec |
| RAM      | 8 GB     |                     |
| Disk     | 20 GB     | Environment                   |
| GPU      | none     |  |

**Supported & tested Linux distros:**

- Ubuntu 22.04 LTS (Jammy)
- Ubuntu 24.04 LTS (Noble)

Other distros should work – install equivalent packages manually.

## 📦 1 · Clone & install

```bash
git clone https://github.com/swarm-subnet/swarm
cd swarm

# Install general system dependencies
chmod +x scripts/validator/main/install_dependencies.sh
./scripts/validator/main/install_dependencies.sh

# Setup Python environment and packages
chmod +x scripts/validator/main/setup.sh
./scripts/validator/main/setup.sh

sudo apt update && sudo apt install -y build-essential git pkg-config libgl1-mesa-glx mesa-utils
```


## 🔑 2 · Create wallet keys

```bash
btcli wallet new_coldkey --wallet.name my_cold
btcli wallet new_hotkey  --wallet.name my_cold --wallet.hotkey my_validator
```
And register in the subnet

## ⚙️ 3 · Run the validator

### PM2 launch example

```bash
source validator_env/bin/activate   

pm2 start neurons/validator.py --name swarm_validator -- \
  --netuid 124 \
  --subtensor.network finney \
  --wallet.name coldkey \
  --wallet.hotkey hotkey \
  --logging.debug
```

### Logs:

```bash
pm2 logs swarm_validator
```

### Stop / restart:

```bash
pm2 restart swarm_validator
pm2 stop     swarm_validator
```

## 🔄 4 · Automatic update & deploy

**scripts/auto_update_deploy.sh**

**What it does**

- Every _n_ minutes checks `origin/main` for a higher `swarm/__init__.py::__version__`.
- Pulls, resets to the new commit and calls `scripts/update_deploy.sh` to rebuild & restart the PM2 validator process – zero downtime.

**How to use**

```bash
chmod +x ./scripts/validator/update/auto_update_deploy.sh 
chmod +x ./scripts/validator/update/update_deploy.sh

# edit the variables at the top of auto_update_deploy.sh
nano ./scripts/validator/update/auto_update_deploy.sh
#   PROCESS_NAME="swarm_validator"
#   WALLET_NAME="my_cold"
#   WALLET_HOTKEY="my_validator"
#   SUBTENSOR_PARAM="--subtensor.network finney"

# then run it under pm2 / tmux / systemd
pm2 start --name auto_update_validator \
          --interpreter /bin/bash \
          scripts/validator/update/auto_update_deploy.sh
```

## 🧩 5 · What the validator actually does 
1. **Build a secret task**  
   A random MapTask (world limits, obstacles, physics Δt, horizon) is produced  
   by `swarm/validator/task_gen.py`.  
   *The task is **never** sent to miners.*

2. **Discover miners’ models**  
   *File cache:* `miner_models/UID_<uid>.zip`  
   For each sampled UID the validator  
   1. sends an empty **`PolicySynapse`** → miner replies with a **`PolicyRef`**;  
   2. compares the `sha256` to the cached file. If it differs, it sends  
      `need_blob=True` and streams the new zip through `PolicyChunk` messages  
      (`_download_model`).  
   All handshake and caching logic lives in `_ensure_models()` inside  
   `swarm/validator/forward.py`.

3. **Evaluate miners one‑by‑one (low‑RAM loop)**  
   Each model is loaded, exercised on the secret task, and immediately freed.  
   The episode runner `_run_episode` measures:  
   * success flag  
   * time alive  
   * distance closed (progress)  
   * energy used  
   These metrics are converted into a **score ∈ [0, 1]** by  
   `swarm/validator/reward.py`.

4. **Update on‑chain weights**  
   Scores are fed into `BaseValidatorNeuron.update_scores()` and pushed to
   subtensor, rewarding miners proportionally.

5. **Sleep & repeat**  
   The loop pauses for `FORWARD_SLEEP_SEC`, then returns to step 1.

Everything is orchestrated by the coroutine
`swarm/validator/forward.py::forward`.



## 🆘 Support

- Discord: #swarm-dev – ping @Miguelikk or @AliSaaf

Happy validating! 🚀