⛏️ Swarm Miner Guide
(Drone‑Navigation / MapTask → FlightPlan subnet)

The Swarm subnet tasks your miner with planning safe, energy‑efficient
flight paths for a simulated drone across a procedurally generated
world.
This guide shows how to install, configure and run a Swarm miner
that answers MapTask requests with a FlightPlan.

💻 System requirements
Component	Minimal	Recommended	Notes
CPU	2 cores	4 cores    	Path‑planning is light‑weight
RAM	2 GB   	4 GB       	
Disk	1 GB   	5 GB       	Repository + virtual‑env
GPU	none	Optional   	Only if you integrate ML planners
OS	Linux / macOS / WSL2	—	Scripts are written for Ubuntu 22.04

ℹ️ All validation runs happen on the validator side.
Your miner only computes a path, no physics simulation is
performed locally.

🚀 Installation
bash
Copy
# 1) clone the repo (no sub‑modules required)
git clone https://github.com/miguelik2/swarm
cd swarm

# 2) create & activate a Python 3.11 virtual‑env
python3.11 -m venv miner_env
source miner_env/bin/activate

# 3) install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt        # core Swarm libs
pip install gym_pybullet_drones        # task helper (pure‑python wheels)

# ⚠️ if you plan to use a ML planner that needs PyTorch / JAX, install it here
Optional system packages
bash
Copy
sudo apt update && sudo apt install -y \
     build-essential git pkg-config libgl1-mesa-glx
🔧 Configuration
All runtime parameters are passed via CLI flags; nothing needs editing
inside the repo.

Flag	Description	Example
--netuid	Subnet netuid on-chain	--netuid 124
--wallet.name	Your coldkey name	--wallet.name my_cold
--wallet.hotkey	Hotkey used for mining	--wallet.hotkey my_hot
--subtensor.network	Network (finney, test)	--subtensor.network finney
--axon.port	TCP port your miner listens on	--axon.port 8091

Create the keys first if you have not:

bash
Copy
btcli wallet new_coldkey --wallet.name my_cold
btcli wallet new_hotkey  --wallet.name my_cold --wallet.hotkey my_hot
🏃‍♂️ Running the miner (PM2 example)
bash
Copy
source miner_env/bin/activate      # if not already active

pm2 start neurons/miner.py \
     --name "swarm_miner" \
     --interpreter python3.11 \
     -- \
     --netuid 124 \
     --subtensor.network finney \
     --wallet.name my_cold \
     --wallet.hotkey my_hot \
     --axon.port 8091
Check logs:

bash
Copy
pm2 logs swarm_miner
Stop / restart:

bash
Copy
pm2 restart swarm_miner
pm2 stop     swarm_miner
✈️ How does the miner work?
Validator sends a MapTask
( JSON with obstacle list, world limits, time‑step, horizon, etc. )

Your miner calls flying_strategy(task)
– a function you implement inside neurons/miner.py.
It must return a FlightPlan, i.e. an ordered list of way‑points
and thrust set‑points that theoretically bring the drone from spawn
to the goal.

Validator re‑simulates the plan inside PyBullet to verify:

Reaches goal inside horizon seconds

No collision with obstacles

Energy consumption below battery budget

A reward ∈ [0, 1] is computed from success, time and energy,
then broadcast back to the miner.

The template miner shipped in the repo implements a naïve
straight‑line planner that:

text
Copy
spawn → (0,0,SAFE_Z) → (goal.x, goal.y, SAFE_Z) → goal
It is enough to start earning small rewards, but you are encouraged to
replace flying_strategy() with smarter algorithms (A*, RRT*, PRM,
NeRF, ML policies …).

🏆 Reward formula (overview)
Term	Weight	Description
Mission success	70 %	1.0 if goal reached, else 0
Time factor	15 %	1 − t_goal / horizon, clamped to [0,1]
Energy factor	15 %	1 − E_used / E_budget, clamped to [0,1]

Full logic: swarm/validator/reward.py.

🔄 Auto‑update & auto‑deploy (optional)
The repo includes scripts/auto_update_deploy.sh which:

Checks origin/main every n minutes.

If swarm/__init__.py::__version__ increases, it pulls, then calls
scripts/update_deploy.sh to rebuild and restart your PM2 process.

bash
Copy
# make both scripts executable
chmod +x scripts/auto_update_deploy.sh scripts/update_deploy.sh

# run watcher (tmux / systemd / pm2)
bash scripts/auto_update_deploy.sh
Adjust process name, wallet keys and subtensor flags at the top of the
script.

🆘 Need help?
Discord  #swarm-dev channel – ping @Miguelikk

GitHub issues – open a ticket with logs & error trace

Happy mining, and may your drones fly far 🚀!