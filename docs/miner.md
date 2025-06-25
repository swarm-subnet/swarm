# ⛏️ Swarm Miner Guide
*(Swarm subnet)*

The Swarm subnet tasks your miner with planning safe, energy‑efficient flight paths for a simulated drone across a procedurally generated world. 
This guide shows how to install, configure and run a Swarm miner that answers MapTask requests with a FlightPlan.

## 💻 System requirements to run the default miner code

| Component | Minimal | Recommended | Notes                                         |
|-----------|---------|-------------|-----------------------------------------------|
| CPU       | 2 cores  | 4 cores      | Path‑planning is light‑weight                 |
| RAM       | 2 GB     | 4 GB         |                                               |
| Disk      | 1 GB     | 50 GB         | Repository + virtual‑env                      |
| GPU       | none     | Optional     | Only if you integrate ML planners             |
| OS        | Linux / macOS / WSL2 | —           | Scripts are written for Ubuntu 22.04          |

ℹ️ The existing miner code just plans an straight line between the spawn point and the objective. Objects might be places randomly in that trajectory and the drone will crash! Improve the flying_strategy.py file to give better flight plans

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

pm2 start neurons/miner.py \
     --name "swarm_miner" \
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

## ✈️ How does the miner work?

1. Validator sends a MapTask (JSON with seed, world limits, time‑step, horizon, etc.)
2. Your miner calls `flying_strategy(task)` – a function you implement inside `neurons/miner.py`. It must return a FlightPlan, i.e. a time_stamped list of rotor RPMs. So, what power per propeller per t the drone needs to have to reach the objective.
3. Validator re‑simulates the plan inside PyBullet to verify:
   - Reaches goal inside horizon
   - Energy consumption below battery budget
4. A reward ∈ [0, 1] is computed from success, time and energy, then broadcast back to the miner.

The template miner shipped in the repo implements a naïve straight‑line planner that:

```text
spawn → (0,0,SAFE_Z) → (goal.x, goal.y, SAFE_Z) → goal
```

It is enough to start earning small rewards, but you are encouraged to replace `flying_strategy()` with smarter algorithms (A*, RRT*, PRM, NeRF, ML policies …).

## 🏆 Reward formula (overview)

| Term            | Weight | Description                                      |
|-----------------|--------|--------------------------------------------------|
| Mission success | 70 %   | 1.0 if goal reached, else 0                      |
| Time factor     | 15 %   | 1 − t_goal / horizon, clamped to [0,1]           |
| Energy factor   | 15 %   | 1 − E_used / E_budget, clamped to [0,1]          |

Full logic: `swarm/validator/reward.py`.

## 🆘 Need help?

- Discord  #swarm-dev channel – ping @Miguelikk
- GitHub issues – open a ticket with logs & error trace

Happy mining, and may your drones fly far 🚀!