# 🔐 Swarm Validator Guide

This document shows how to install and operate the Swarm validator. The validator securely evaluates miner models across six challenge families on procedurally generated maps: cities, open terrain, mountains, villages, warehouses, forests, and offices. Miner models run in isolated Docker containers under the subnet-owned runner, while evaluation and scoring execute on the validator host.

Run `swarm doctor` after installation to verify your environment is ready.

## 🎯 What You Evaluate

Swarm runs **six challenge families**: five active and one completed. Evaluation is family-scoped: every task the backend hands you names one family, and the validator builds that family's environment, maps, and seeds from the task metadata; you never pick a family yourself.

| Family | ID | Mission | Emissions |
|--------|-----|---------|-----------|
| [Autopilot](../../docs/families/autopilot.md) | `cf_autopilot` | One drone crosses a generated world and lands on a pad inside a noisy search area | 15% |
| [Search and Rescue](../../docs/families/search_and_rescue.md) | `cf_search_and_rescue` | One drone finds a downed victim by depth camera and holds a confirmation hover overhead | 15% |
| [Swarm Autopilot](../../docs/families/swarm_autopilot.md) | `cf_swarm_autopilot` | One policy lands 2–8 drones on a shared pool of pads | 20% |
| [Swarm Search and Rescue](../../docs/families/swarm_sar.md) | `cf_swarm_sar` | One policy sweeps the map with 2–8 drones until any drone confirms the victim | 20% |
| [Interceptor](../../docs/families/interceptor.md) | `cf_interceptor` | Completed open-terrain pursuit; winning solution preserved as open source | 0% (historical 30%) |
| [Office Interceptor](../../docs/families/office_interceptor.md) | `cf_interceptor_office` | One drone hunts down a validator-flown target inside a fixed office | 30% |

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

Other distros should work; install equivalent packages manually.

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

# 6. Start and enable Docker service
sudo systemctl start docker
sudo systemctl enable docker
```

### Recommended hardening

Run the validator under its own user instead of your login account:

```bash
sudo useradd -m -s /bin/bash swarm-validator
sudo usermod -aG docker swarm-validator
```

Keep evaluation containers isolated from each other by setting `"icc": false`
in `/etc/docker/daemon.json`:

```bash
echo '{ "icc": false }' | sudo tee /etc/docker/daemon.json
sudo systemctl restart docker
```

For an extra layer you can run [rootless Docker](https://docs.docker.com/engine/security/rootless/).

### Verify Docker

```bash
docker --version
docker run hello-world
docker ps
sudo systemctl status docker
```

## 📦 Installation

The validator ships as a container image. Nothing is compiled on your machine and no
Python environment is created, so a validator behaves the same on every host.

### 1. Clone the repository

The checkout provides the compose file and the update scripts. The validator itself
comes from the image.

```bash
git clone https://github.com/swarm-subnet/swarm
cd swarm
```

### 2. Configure

```bash
cp .env.example .env
nano .env
```

Fill in at least these:

| Variable | What it is |
|---|---|
| `SWARM_BACKEND_API_URL` | Backend endpoint, from the team |
| `WANDB_API_KEY` | Logging key, from the team |
| `SWARM_WALLET_NAME` | Your coldkey name |
| `SWARM_WALLET_HOTKEY` | Your validator hotkey |

`SWARM_STATE_DIR` defaults to `/opt/swarm-validator-state` and is mounted at the same
path inside the container. Leave it absolute: the validator hands that path to the host's
Docker daemon when it starts evaluation containers, so it has to mean the same thing on
both sides.

Contact the team on [Discord](https://discord.gg/8dPqPDw7GC) for the backend URL and the
WandB key.

### 3. Start it

```bash
bash validator/scripts/update/update_deploy.sh
```

That pulls the published image, prepares the state directory, stops any old host-process
validator, and starts the container. Run the same command any time you want to force an
update by hand.

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

### Start, stop, restart

```bash
cd swarm
docker compose -f .docker/docker-compose.yml --profile validator up -d validator
docker compose -f .docker/docker-compose.yml --profile validator stop validator
docker compose -f .docker/docker-compose.yml --profile validator restart validator
```

### Logs

```bash
docker compose -f .docker/docker-compose.yml logs -f validator
```

### Which version is running

```bash
docker inspect --format '{{index .Config.Labels "swarm.__version__"}}' \
  "$(docker compose -f .docker/docker-compose.yml --profile validator ps -q validator)"
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

  The `screening` and `screening_submit` stages run only when the backend enables the screening phase, which is off by default, so new models go straight to `benchmark`.
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

**`validator/scripts/update/auto_update_deploy.sh`** watches the registry. Every *n*
minutes it pulls `:latest`, compares that image's `swarm.__version__` label against the
one the running container was started from, and redeploys when the published version is
newer. No git pull, no reinstall, nothing built on your machine.

### If you already run the updater

**Nothing to do.** The watcher you have running checks a version and runs
`update_deploy.sh` from disk. That script now deploys the container, so your host moves
onto the container flow on its next update without you restarting anything.

### Fresh install, under systemd

```bash
sudo cp validator/scripts/update/swarm-validator-updater.service /etc/systemd/system/
# adjust WorkingDirectory and User if your checkout is not /root/swarm
sudo systemctl daemon-reload
sudo systemctl enable --now swarm-validator-updater
journalctl -u swarm-validator-updater -f
```

### Fresh install, under PM2

```bash
pm2 start --name auto_update_validator \
          --interpreter /bin/bash \
          validator/scripts/update/auto_update_deploy.sh
pm2 save
```

The legacy path `scripts/validator/update/auto_update_deploy.sh` still forwards to the
one above, so an updater registered before the scripts moved keeps working.

### Pinning a version, and rolling back

`:latest` is what the updater follows. To hold a specific version, set the tag in `.env`
and the updater stops moving you:

```bash
SWARM_VALIDATOR_TAG=5.1.5.6
```

```bash
bash validator/scripts/update/update_deploy.sh
```

Every published version keeps its own tag, so rolling back is setting the tag to the
previous version and running that command. Set the tag back to `latest` to resume
automatic updates.

### Coming from a host install

The first update does it for you: it stops the PM2 host process and starts the container
in its place. If you would rather do it by hand:

```bash
pm2 stop swarm_validator
bash validator/scripts/update/update_deploy.sh
```

Your wallet and `.env` are unchanged. State moves to `SWARM_STATE_DIR`; the validator
rebuilds anything it finds missing there on its next cycle.

## 🧩 What the Validator Does

1. **Sync with the backend**
   `GET /validators/sync` returns the current epoch, the per-family King of the Hill windows and family shares, the champions, and the latest weight map. Runs once per forward cycle. Evaluation work arrives separately via the `GET /validators/next-task` long-poll, which assigns the model under evaluation; individual seeds are then leased on demand through `POST /validators/tasks/{id}/claim-seeds` as workers free up.

2. **Fetch the model**
   Fetch the archive from the backend vault (every family is on the private track) and verify its SHA-256 against the backend record. The bytes are written owner-only, never kept for forensics, and deleted once the task is done. A public-track family, if one is ever reopened, is downloaded from the miner's GitHub repo instead.

3. **Full benchmark (1,000 seeds)**
   Every new model runs its family's full 1,000-seed benchmark in parallel Docker containers. As workers free up, the validator claims up to that many pending seeds from the backend's shared pool, so validators of different speeds share one model without long idle tails. The task metadata carries the family and phase, so no local configuration is needed. A screening pre-phase (the first 300 seeds, with a pass bar tied to the champion's score) exists behind a backend constant but is off by default: submissions go straight to the full benchmark.

4. **Report scores**
   Per-seed and aggregate scores are submitted to the backend as they are computed.

5. **Apply weights**
   Validators recompute the weight map locally from the per-family windows on every forward cycle and set it on-chain on the chain's epoch-length cadence. The weights come from per-family King of the Hill windows: each family's last five champions share that family's emission slice; see [king_of_the_hill.md](../../docs/king_of_the_hill.md).

6. **Caching**
   Results are cached by model hash + benchmark version + epoch. The same model is not re-evaluated within the same epoch unless a re-eval is explicitly queued (for example, a benchmark version bump).

### Shared Epoch Seeds

Every validator flies the same 1,000 seeds per family per epoch. The backend holds one secret key per epoch and serves it to trusted validators over `/validators/sync`; each seed is `HMAC-SHA256(key, "v1|<family>|<epoch>|<index>")` truncated to 32 bits, so the whole network derives an identical list without any of it being predictable in advance. Seed index N is therefore the same mission everywhere, and two models in one epoch are compared on the same maps rather than on two different draws.

The key is never needed by hand: it arrives with the regular sync, and a task assigned for a future epoch carries that epoch's key with it. A validator holding no key for an epoch takes no work for it rather than falling back to its own seeds.

Epochs run for **14 days** from epoch 19 onward, anchored Monday 16:00 UTC (epochs 1–18 were 7 days). The key's fingerprint is published as soon as the key exists, and the key itself once the epoch closes, so anyone can rebuild that epoch's seeds and confirm they were fixed before they were flown. Both appear on the epoch endpoints and on [swarm124.com](https://swarm124.com).

## 🔧 Troubleshooting

### Docker Issues

**Docker not installed:**

```text
docker: command not found
```

Follow the Docker installation section above.

**Docker permission denied:**

```text
Permission denied while trying to connect to Docker daemon
```

```bash
sudo usermod -aG docker swarm-validator   # the dedicated validator user
# Log out and back in
```

**Docker service not running:**

```text
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
mkdir -p swarm/state/miner_models
chmod 755 swarm/state/miner_models
```

**Docker container issues:**

```bash
docker system df
docker system prune -f
```

## 🆘 Support

- **Discord**: [discord.gg/8dPqPDw7GC](https://discord.gg/8dPqPDw7GC) (ping @Miguelikk or @AliSaaf)
- **GitHub Issues**: open a ticket with logs & error trace
- **Website**: [swarm124.com](https://swarm124.com)

Happy validating!
