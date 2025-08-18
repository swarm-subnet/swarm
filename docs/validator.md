# 🚀 Swarm Validator Guide
*(Swarm subnet – netuid 124)*

This document shows how to install and operate the Swarm validator that securely evaluates models received from miners on dynamically generated maps using Docker-based isolation.

## 🔒 Security Overview

**Why Docker is Required:** Miners submit potentially untrusted RL policy code that runs during evaluation. The Swarm validator uses Docker containers to:
- **Isolate untrusted code** in secure sandboxed environments
- **Prevent malicious models** from accessing your system
- **Detect fake models** using 3-layer verification system
- **Blacklist malicious models** automatically

Without Docker, **your validator cannot safely evaluate miner models**.

## 🖥️ System Requirements

| Resource | Minimal | Notes                                |
|----------|---------|--------------------------------------|
| CPU      | 3 cores  | Miners are evaluated 1 by 1, no no need for much spec |
| RAM      | 8 GB     |                     |
| Disk     | 50 GB     | Environment                   |
| GPU      | none     |  |

**Supported & tested Linux distros:**

- Ubuntu 22.04 LTS (Jammy)
- Ubuntu 24.04 LTS (Noble)

Other distros should work – install equivalent packages manually.

## 🐳 Docker Installation (REQUIRED)

**Docker is mandatory** for validator operation. The validator cannot start without Docker.

### Ubuntu 22.04 / 24.04 Installation

```bash
# 1. Update system packages
sudo apt update && sudo apt upgrade -y

# 2. Install Docker dependencies
sudo apt install -y apt-transport-https ca-certificates curl gnupg lsb-release

# 3. Add Docker official GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# 4. Add Docker repository
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# 5. Install Docker Engine
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# 6. Add your user to docker group (avoid sudo)
sudo usermod -aG docker $USER

# 7. Start and enable Docker service
sudo systemctl start docker
sudo systemctl enable docker

# 8. Log out and back in (or reboot) to apply group membership
```

### Docker Installation Verification

```bash
# Test Docker installation
docker --version
docker run hello-world

# Verify Docker service is running
sudo systemctl status docker

# Test Docker without sudo
docker ps
```

**Expected Output:**
```
Docker version 24.0.0+
Hello from Docker! (success message)
● docker.service - Docker Application Container Engine
   Active: active (running)
```

### Manual Docker Installation (Alternative)

If automatic installation fails:

```bash
# Download Docker installation script
curl -fsSL https://get.docker.com -o get-docker.sh

# Review the script (recommended)
less get-docker.sh

# Run installation script
sudo sh get-docker.sh

# Post-installation setup
sudo usermod -aG docker $USER
sudo systemctl start docker
sudo systemctl enable docker
```

## 📦 Installation & Setup

### 1. Clone Repository

```bash
git clone https://github.com/swarm-subnet/swarm
cd swarm
```

### 2. Install System Dependencies

```bash
# Install general system dependencies
chmod +x scripts/validator/main/install_dependencies.sh
./scripts/validator/main/install_dependencies.sh

# Install additional required packages
sudo apt update && sudo apt install -y build-essential git pkg-config libgl1-mesa-glx mesa-utils
```

### 3. Setup Python Environment

```bash
# Setup Python environment and packages
chmod +x scripts/validator/main/setup.sh
./scripts/validator/main/setup.sh

# Activate the validator environment
source validator_env/bin/activate
```

### 4. Verify Docker Integration

```bash
# Test Docker access from Python environment
python -c "import subprocess; print('Docker status:', subprocess.run(['docker', 'ps'], capture_output=True).returncode == 0)"
```

**Expected Output:** `Docker status: True`


## 🔑 Wallet & Registration Setup

### Create Wallet Keys

```bash
btcli wallet new_coldkey --wallet.name my_cold
btcli wallet new_hotkey  --wallet.name my_cold --wallet.hotkey my_validator
```

### Register on Subnet 124

```bash
# Register your validator on Swarm subnet
btcli subnet register --wallet.name my_cold --wallet.hotkey my_validator --netuid 124 --subtensor.network finney

# Check registration status
btcli wallet overview --wallet.name my_cold --subtensor.network finney
```

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


## 🧩 5 · What the validator actually does (v2.0.3)

1. **Build a secret task**  
   A random MapTask (world limits, obstacles, physics Δt, horizon) is produced  
   by `swarm/validator/task_gen.py`.  
   *The task is **never** sent to miners.*

2. **Discover miners’ models**  
   *File cache:* `miner_models_v2/UID_<uid>.zip`  
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

## 🔧 Troubleshooting

### Docker Issues

**Docker not installed:**
```bash
Error: docker: command not found
```
**Solution:** Follow the Docker installation section above.

**Docker permission denied:**
```bash
Permission denied while trying to connect to Docker daemon
```
**Solution:** 
```bash
sudo usermod -aG docker $USER
# Log out and back in, or reboot
```

**Docker service not running:**
```bash
Cannot connect to the Docker daemon
```
**Solution:**
```bash
sudo systemctl start docker
sudo systemctl enable docker
```

### Validator Startup Issues

**PyBullet/OpenGL errors:**
```bash
# Install missing graphics libraries
sudo apt update && sudo apt install -y libgl1-mesa-glx mesa-utils
```

**Model cache permissions:**
```bash
# Fix model cache directory permissions
mkdir -p miner_models_v2
chmod 755 miner_models_v2
```

**Docker container creation fails:**
```bash
# Check Docker system status
docker system df
docker system prune  # Clean up if needed
```

### Performance Optimization

**High memory usage:**
- Increase system RAM (minimum 12GB recommended)
- Monitor with `docker stats` during evaluation
- Clean model cache periodically: `rm -rf miner_models_v2/*.zip`

**Slow model evaluation:**
- Increase CPU cores (6+ cores recommended)
- Check Docker resource limits
- Monitor system load with `htop`

### Security Warnings

**Blacklisted model detected:**
```
🚫 FAKE MODEL DETECTED during verification
```
**Action:** Model automatically blacklisted, no manual action needed.

**Docker container timeout:**
```
⏰ Verification timeout for model
```
**Action:** Model evaluation skipped for safety, container cleaned up automatically.

## 🆘 Support

- Discord: #swarm-dev – ping @Miguelikk or @AliSaaf

Happy validating! 🚀