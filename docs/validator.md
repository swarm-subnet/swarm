# 🔐 Swarm Validator Guide

This document shows how to install and operate the Swarm validator. The validator securely evaluates miner models on procedurally generated maps — cities, mountains, warehouses, open terrain, and forests. Miner code runs in isolated Docker containers while evaluation and scoring execute on the validator host.

## 🖥️ System Requirements

| Resource | Minimal | Notes |
|----------|---------|-------|
| CPU | 3 cores | |
| RAM | 8 GB | |
| Disk | 50 GB | Environment + model cache |
| GPU | None | |

**Supported Linux distros:**

- Ubuntu 22.04 LTS (Jammy)
- Ubuntu 24.04 LTS (Noble)

Other distros should work — install equivalent packages manually.

## 🐳 Docker Installation (Required)

**Docker is mandatory** for validator operation. The validator cannot start without Docker.

### Ubuntu 22.04 / 24.04

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

### Verify Docker

```bash
docker --version
docker run hello-world
docker ps
sudo systemctl status docker
```

## 📦 Installation

### 1. Clone Repository

```bash
git clone https://github.com/swarm-subnet/swarm
cd swarm
```

### 2. Install System Dependencies

```bash
chmod +x scripts/validator/main/install_dependencies.sh
./scripts/validator/main/install_dependencies.sh

sudo apt update && sudo apt install -y build-essential git pkg-config libgl1-mesa-glx mesa-utils
```

### 3. Setup Python Environment

```bash
chmod +x scripts/validator/main/setup.sh
./scripts/validator/main/setup.sh

source validator_env/bin/activate
```

### 4. Configure Environment Variables

Create `.env` file in repository root:

```bash
# REQUIRED — Backend API endpoint
SWARM_BACKEND_API_URL=https://api.example.com

# REQUIRED — Private benchmark secret (contact the team)
SWARM_PRIVATE_BENCHMARK_SECRET=your_private_secret_here

# Optional: WandB logging
WANDB_API_KEY=your_wandb_key_here
VALIDATOR_NAME=my_validator_name
```

Both `SWARM_BACKEND_API_URL` and `SWARM_PRIVATE_BENCHMARK_SECRET` are required. Contact the team to obtain values.

## 🔑 Wallet & Registration

### Create Wallet Keys

```bash
btcli wallet new_coldkey --wallet.name my_cold
btcli wallet new_hotkey  --wallet.name my_cold --wallet.hotkey my_validator
```

### Register on Subnet 124

```bash
btcli subnet register --wallet.name my_cold --wallet.hotkey my_validator --netuid 124 --subtensor.network finney

btcli wallet overview --wallet.name my_cold --subtensor.network finney
```

## ⚙️ Run the Validator

### PM2 Launch

```bash
source validator_env/bin/activate

pm2 start neurons/validator.py --name swarm_validator -- \
  --netuid 124 \
  --subtensor.network finney \
  --wallet.name coldkey \
  --wallet.hotkey hotkey \
  --logging.debug
```

### Logs

```bash
pm2 logs swarm_validator
```

### Stop / Restart

```bash
pm2 restart swarm_validator
pm2 stop    swarm_validator
```

## 🔄 Auto-Update

**`scripts/validator/update/auto_update_deploy.sh`** checks `origin/main` for version bumps every *n* minutes. When a new version is found, it pulls, resets, and restarts the PM2 process.

```bash
chmod +x ./scripts/validator/update/auto_update_deploy.sh
chmod +x ./scripts/validator/update/update_deploy.sh

# Edit variables at the top of auto_update_deploy.sh
nano ./scripts/validator/update/auto_update_deploy.sh

# Run under PM2
pm2 start --name auto_update_validator \
          --interpreter /bin/bash \
          scripts/validator/update/auto_update_deploy.sh
```

## 🧩 What the Validator Does

1. **Detect new models**
   For each miner UID, compare SHA-256 hash to cache. If hash differs, download the new model.

2. **Screening (200 seeds)**
   New models are first evaluated on 200 private seeds (derived via HMAC-SHA256). Must score within 80% of the top model to proceed.

3. **Full benchmark (800 seeds)**
   Models that pass screening are evaluated on the remaining 800 benchmark seeds across all five map types. Evaluation runs in parallel Docker containers.

4. **Submit scores to backend**
   Final score (median of all 1,000 seeds) is submitted to the backend.

5. **Backend aggregation**
   Backend aggregates scores from all validators (51% stake consensus).

6. **Apply weights**
   Validators fetch final weights from the backend and apply them on-chain.

7. **Caching**
   Results are cached by model hash + benchmark version. Same model is never re-evaluated within the same epoch.

## 🔧 Troubleshooting

### Docker Issues

**Docker not installed:**
```
docker: command not found
```
Follow the Docker installation section above.

**Docker permission denied:**
```
Permission denied while trying to connect to Docker daemon
```
```bash
sudo usermod -aG docker $USER
# Log out and back in
```

**Docker service not running:**
```
Cannot connect to the Docker daemon
```
```bash
sudo systemctl start docker
sudo systemctl enable docker
```

### Validator Startup Issues

**PyBullet/OpenGL errors:**
```bash
sudo apt update && sudo apt install -y libgl1-mesa-glx mesa-utils
```

**Model cache permissions:**
```bash
mkdir -p miner_models_v2
chmod 755 miner_models_v2
```

**Docker container issues:**
```bash
docker system df
docker system prune -f
```

## 🆘 Support

- Discord — ping @Miguelikk or @AliSaaf

Happy validating! 🚀
