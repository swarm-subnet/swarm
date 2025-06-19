🚀 Swarm Validator Guide
(Drone‑Navigation subnet – netuid 124)

This document shows how to install and operate the Swarm validator
that drives the MapTask → FlightPlan competition.
The code‑base has zero external sub‑modules and runs on any recent
CPU‑only server.

🖥️ System requirements
Resource	Minimal	Notes
CPU	2 cores	Map generation & scoring are light‑weight
RAM	4 GB	Python + PyBullet
Disk	2 GB	Repo + virtual‑env
GPU	none	Validation replay uses head‑less PyBullet

Supported & tested Linux distros:

Ubuntu 22.04 LTS (Jammy)

Ubuntu 24.04 LTS (Noble)

Other distros should work – install equivalent packages manually.

📦 1 · Clone & install
bash
Copy
git clone https://github.com/miguelik2/swarm.git
cd swarm

# create Python 3.11 virtual‑env
python3.11 -m venv validator_env
source validator_env/bin/activate

# install requirements
pip install --upgrade pip
pip install -r requirements.txt          # bittensor + pybullet + numpy …
No git submodule commands are needed – repository is self‑contained.

🔑 2 · Create wallet keys (once)
bash
Copy
btcli wallet new_coldkey --wallet.name my_cold
btcli wallet new_hotkey  --wallet.name my_cold --wallet.hotkey my_validator
Fund the coldkey with enough TAO to register (≈ 0.002 TAO at time of
writing) on the Finney test‑network.

⚙️ 3 · Run the validator
PM2 launch example
bash
Copy
source validator_env/bin/activate   # if not already

pm2 start neurons/validator.py \
     --name "swarm_validator" \
     --interpreter python3.11 \
     -- \
     --netuid 124 \
     --subtensor.network finney \
     --wallet.name my_cold \
     --wallet.hotkey my_validator \
     --logging.debug
Flags explained

Flag	Value	Description
--netuid	124	Swarm subnet id
--subtensor.network	finney	Test‑net; use local for devnet
--wallet.name	coldkey	Your cold key name
--wallet.hotkey	hotkey	Hotkey that will validate
--logging.debug	–	Verbose logs

Logs:

bash
Copy
pm2 logs swarm_validator
Stop / restart:

bash
Copy
pm2 restart swarm_validator
pm2 stop     swarm_validator
🔄 4 · Automatic update & deploy (optional)
scripts/auto_update_deploy.sh

What it does

Every n minutes checks origin/main for a higher
swarm/__init__.py::__version__.

Pulls, resets to the new commit and calls
scripts/update_deploy.sh to rebuild & restart the PM2 validator
process – zero downtime.

How to use

bash
Copy
chmod +x scripts/auto_update_deploy.sh scripts/update_deploy.sh

# edit the variables at the top of auto_update_deploy.sh
nano scripts/auto_update_deploy.sh
#   PROCESS_NAME="swarm_validator"
#   WALLET_NAME="my_cold"
#   WALLET_HOTKEY="my_validator"
#   SUBTENSOR_PARAM="--subtensor.network finney"

# then run it under pm2 / tmux / systemd
pm2 start --name auto_update_validator \
          --interpreter /bin/bash \
          scripts/auto_update_deploy.sh
🧩 5 · What the validator actually does
Generate a map
Random obstacles, world limits, physics time‑step & horizon are
packed into a MapTask (see swarm/validator/task_gen.py).

Broadcast to miners
Sends the task to N randomly sampled miners (SAMPLE_K, default = 128).

Re‑simulate each returned FlightPlan in PyBullet (replay_once)
measuring:

Goal reached?

Time to goal

Energy used

Score → update weights
Reward is computed (swarm/validator/reward.py), moving‑average
weights are updated and pushed on‑chain.

Sleep a couple seconds and repeat.

Everything happens inside the easy‑to‑read loop in
swarm/validator/forward.py.

🆘 Support
Discord: #swarm-dev – ping @Miguelikk

Happy validating! 🚀