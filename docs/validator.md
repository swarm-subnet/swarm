# 🚀 Swarm Validator Guide
*(Swarm subnet – netuid 124)*

This document shows how to install and operate the Swarm validator that securely evaluates models received from miners on dynamically generated maps using Docker-based isolation.

## 🖥️ System Requirements

| Resource | Minimal | Notes                                |
|----------|---------|--------------------------------------|
| CPU      | 3 cores  | Miners are evaluated 1 by 1 |
| RAM      | 8 GB     |                     |
| Disk     | 50 GB     | Environment                   |
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

### 5. Configure Environment Variables

Create `.env` file in repository root:

```bash
# REQUIRED for synchronized seeds (Contact the team)
VALIDATOR_SECRET_KEY=your_secret_key_here 

# Optional: WandB logging
WANDB_API_KEY=your_wandb_key_here
VALIDATOR_NAME=my_validator_name
```

**Note:** `VALIDATOR_SECRET_KEY` is used for synchronized seed generation across validators. Contact the team to obtain it.

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

## ⚙️ Run the validator

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

## 🔄 Automatic update & deploy

**scripts/validator/update/auto_update_deploy.sh**

**What it does**

- Every *n* minutes checks `origin/main` for a higher `swarm/__init__.py::__version__`.
- Pulls, resets to the new commit and calls `scripts/validator/update/update_deploy.sh` to rebuild & restart the PM2 validator process.

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
#   CHECK_INTERVAL_MINUTES=30

# then run it under pm2
pm2 start --name auto_update_validator \
          --interpreter /bin/bash \
          scripts/validator/update/auto_update_deploy.sh
```


## 🧩 What the validator actually does

1. **Generate synchronized seed**  
   All validators generate identical seeds based on 10-minute time windows using `VALIDATOR_SECRET_KEY`.  
   This ensures all validators evaluate miners on identical maps.

2. **Sample miners**  
   Sample up to 256 miners from metagraph, filtering out low performers (avg score < 0.2) and validators.

3. **Discover miners' models**  
   For each sampled UID:
   - Send empty `PolicySynapse` → miner replies with `PolicyRef`
   - Compare SHA256 to cache at `miner_models_v2/UID_<uid>.zip`
   - If hash differs, request `need_blob=True` and stream via `PolicyChunk` messages

4. **Multi-layer fake detection**  
   Models are validated for security and authenticity before evaluation.

5. **Docker evaluation**  
   Each model evaluated in isolated container on secret tasks.

6. **Score normalization**  
   Compute normalized score as weighted average across all challenge types.  
   Minimum 25 runs required before miner is eligible for rewards.

7. **Winner-take-all rewards**  
   Winner take all

8. **Update weights and repeat**  
   Write weights to chain, sleep until next seed window, cleanup Docker resources.

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

**Missing VALIDATOR_SECRET_KEY:**
```
Error: VALIDATOR_SECRET_KEY not set
```
**Solution:** Create `.env` file with `VALIDATOR_SECRET_KEY=your_secret_here`

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
docker system prune -f  # Clean up if needed
```

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

- Discord – ping @Miguelikk or @AliSaaf

Happy validating! 🚀