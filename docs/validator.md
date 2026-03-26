# 🔐 Swarm Validator Guide

This document shows how to install and operate the Swarm validator. The validator securely evaluates miner models on procedurally generated maps — cities, open terrain, mountains, villages, warehouses, and forests. Miner code runs in isolated Docker containers while evaluation and scoring execute on the validator host.

Run `swarm doctor` after installation to verify your environment is ready.

## 🖥️ System Requirements

| Resource | Minimal | Notes |
|----------|---------|-------|
| CPU | 12 cores | |
| RAM | 48 GB | |
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
SWARM_BACKEND_API_URL=<contact the team>

# REQUIRED — WandB logging
WANDB_API_KEY=<contact the team>
VALIDATOR_NAME=my_validator_name
```

Contact the team on [Discord](https://discord.gg/8dPqPDw7GC) to obtain `SWARM_BACKEND_API_URL` and `WANDB_API_KEY`.

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

## 📡 Telemetry and Monitor

The validator now writes a local runtime snapshot and an append-only event stream while it runs.

### Files

By default, telemetry is written here:

```bash
swarm/state/validator_runtime.json
swarm/state/validator_events.jsonl
```

- `validator_runtime.json`
  - Current point-in-time health snapshot.
  - Best source for dashboards and alert state.
- `validator_events.jsonl`
  - Structured event log.
  - Useful for reconstructing stage transitions and debugging stalls.

### Monitor Command

Run the live terminal monitor:

```bash
source validator_env/bin/activate
swarm monitor
```

Useful variants:

```bash
swarm monitor --once --no-clear
swarm monitor --refresh-sec 2.0
swarm monitor --max-events 20
```

### What the Monitor Shows

- **Forward**
  - Last forward start/completion, duration, and whether a cycle is still running.
- **Backend**
  - Sync state, fallback mode, pending model count, re-eval queue size, leaderboard version.
- **Epoch**
  - Current epoch, seconds until epoch end, and whether freeze mode is active.
- **Queue**
  - Local queue counts by status, oldest item age, retry pressure, and the active queue items.
- **Evaluation**
  - Current screening and benchmark progress.
- **Docker**
  - Requested/effective workers, adaptive backoff, worker failures/restarts, cleanup timing.
- **Chain / Weights**
  - Post-forward chain sync state and the latest weight-set attempt/result.
- **Alerts**
  - Automatic warning/critical classification for common stalls and dead zones.

### What to Expect in Healthy Operation

- `pending_models_count` rises and falls, but does not grow forever.
- Queue items move through:
  - `processing`
  - `registered`
  - `screening`
  - `screening_submit`
  - `benchmark`
  - `score_submit`
  - `completed`
- `last_completed_forward_count` keeps increasing.
- `backend.fallback` stays `false` most of the time.
- Docker `active_worker_cap` usually matches the requested worker count.
- `oldest_age_sec` stays bounded instead of drifting upward for hours.

### Common Warning Signs

- **Backend fallback stays active**
  - New `pending_models` discovery is effectively stalled.
- **Queue oldest age keeps increasing**
  - Work is arriving faster than it is draining, or a stage is stuck.
- **Retries dominate the queue**
  - Backend submission failures or internal exceptions are recycling items.
- **Freeze active with processable queue items**
  - The validator is intentionally pausing queue work near epoch end.
- **Docker backoff active for long periods**
  - The host is overloaded or worker execution is unhealthy.
- **Repeated re-eval warnings**
  - Champion or queued re-evals are being recomputed repeatedly instead of finishing cleanly.
- **No forward completion**
  - The validator thread, backend sync, Docker cleanup, or post-forward chain sync may be stalled.

### Recommended First Checks

If the monitor looks unhealthy:

1. Confirm Docker is healthy:
   ```bash
   docker ps
   docker stats --no-stream
   ```
2. Check whether backend sync is falling back:
   - look for `backend.fallback=true`
   - look at `last_sync_success_at`
3. Check whether queue items are stuck in one stage:
   - especially `screening`, `benchmark`, or `score_submit`
4. Check whether adaptive backoff reduced worker capacity:
   - compare `requested_workers` vs `active_worker_cap`
5. Check whether epoch freeze is active:
   - the queue may be waiting by design near epoch rollover

### Notes

- The monitor is local-only and reads the validator's own telemetry files.
- If the validator is not running yet, the telemetry files may not exist.
- The event log is append-only; if it grows too much, rotate or truncate it during maintenance windows.

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

1. **Sync with backend**
   Fetch pending models list via backend API (`GET /validators/sync`). Discover new models submitted by miners.

2. **Download from GitHub**
   Download `submission.zip` from the miner's public GitHub repository. Verify SHA-256 hash matches the backend's stored hash. README.md hash is verified by the backend during submission.

3. **Screening (200 seeds)**
   New models are first evaluated on 200 seeds. Must score >= **101%** of the current champion to proceed (`SCREENING_TOP_MODEL_FACTOR = 1.01`), or >= 0.1 if no champion exists.

4. **Full benchmark (800 seeds)**
   Models that pass screening are evaluated on the remaining 800 benchmark seeds across all environment types. Evaluation runs in parallel Docker containers.

5. **Submit scores to backend**
   Final score (average of all 1,000 seeds) is submitted to the backend.

6. **Backend aggregation**
   Backend aggregates scores from all validators (51% stake consensus).

7. **Apply weights**
   Validators fetch final weights from the backend and apply them on-chain.

8. **Caching**
   Results are cached by model hash + benchmark version + epoch. Same model is never re-evaluated within the same epoch.

### Per-Validator Seeds

Each validator independently generates its own 1,000 random seeds per epoch using `random.SystemRandom()`. With 1,000 seeds, the statistical variance across validators is negligible.

Seeds rotate every **7 days** (Monday 16:00 UTC). At the end of each epoch, per-validator seeds are published on [swarm124.com](https://swarm124.com) for full transparency.

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

- **Discord** — [discord.gg/8dPqPDw7GC](https://discord.gg/8dPqPDw7GC) (ping @Miguelikk or @AliSaaf)
- **GitHub Issues** — open a ticket with logs & error trace
- **Website** — [swarm124.com](https://swarm124.com)

Happy validating!
