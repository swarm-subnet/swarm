<a id="miner-top"></a>

# Swarm Miner Guide

Train an autonomous drone pilot, benchmark it against 1,000 procedurally generated worlds, and compete on the [leaderboard](https://swarm124.com/benchmark).

---

<details>
  <summary><b>Table of Contents</b></summary>
  <ol>
    <li><a href="#system-requirements">System Requirements</a></li>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#workflow">Workflow</a></li>
    <li><a href="#creating-your-agent">Creating Your Agent</a></li>
    <li><a href="#observations--actions">Observations & Actions</a></li>
    <li><a href="#cli">CLI</a></li>
    <li><a href="#github-repo-setup">GitHub Repo Setup</a></li>
    <li><a href="#running-the-miner">Running the Miner</a></li>
    <li><a href="#scoring">Scoring</a></li>
    <li><a href="#benchmark-system">Benchmark System</a></li>
    <li><a href="#docker-whitelist">Docker Whitelist</a></li>
    <li><a href="#troubleshooting">Troubleshooting</a></li>
    <li><a href="#support">Support</a></li>
  </ol>
</details>

---

## System Requirements

Mining is extremely lightweight — your miner submits a GitHub URL to the backend and goes offline. Any machine with **Python 3.11+** and a network connection will do. Training hardware depends entirely on your approach (SB3, PyTorch, custom RL — your choice).

<p align="right">(<a href="#miner-top">back to top</a>)</p>

---

## Installation

```bash
git clone https://github.com/swarm-subnet/swarm
cd swarm

chmod +x scripts/miner/install_dependencies.sh
./scripts/miner/install_dependencies.sh

chmod +x scripts/miner/setup.sh
./scripts/miner/setup.sh

source miner_env/bin/activate
pip install -e .
```

<p align="right">(<a href="#miner-top">back to top</a>)</p>

---

## Workflow

The full miner workflow — from first install to competing on the leaderboard:

```
1. swarm doctor              ← Check environment readiness
2. Train your model           ← SB3, PyTorch, custom — your choice
3. swarm model test           ← Validate source folder before packaging
4. swarm model package        ← Bundle into Submission/submission.zip
5. swarm model verify         ← Verify compliance (size, structure)
6. swarm benchmark            ← Run local benchmark
7. Push to GitHub             ← Public repo with submission.zip + README
8. Submit model               ← One-shot submit, then go offline
```

<p align="right">(<a href="#miner-top">back to top</a>)</p>

---

## Creating Your Agent

### Start from the Template

```bash
cp -r swarm/submission_template/ my_agent/
cd my_agent/
# Edit drone_agent.py with your controller
```

### Agent Structure

Your agent must implement a `DroneFlightController` class:

```python
class DroneFlightController:
    def __init__(self):
        # Load your model (SB3, PyTorch, ONNX, etc.)
        from stable_baselines3 import PPO
        self.model = PPO.load("./my_model.zip")

    def act(self, observation):
        # observation: dict with "depth" (128,128,1) and "state" (N,)
        # Return action array [vx, vy, vz, speed, yaw]
        action, _ = self.model.predict(observation, deterministic=True)
        return action

    def reset(self):
        # Reset internal state between missions
        pass
```

**Required files:**
- `drone_agent.py` — Your controller class (REQUIRED)
- `requirements.txt` — Additional pip packages (optional, must be on the [whitelist](#docker-whitelist))
- Model files — weights, configs, etc.

**Auto-injected (do not include):**
- `main.py`, `agent.capnp`, `agent_server.py` — provided by the evaluation system

Submissions must be ≤ **50 MiB** compressed.

<p align="right">(<a href="#miner-top">back to top</a>)</p>

---

## Observations & Actions

### Observation Space

| Field | Shape | Description |
|-------|-------|-------------|
| `depth` | (128, 128, 1) | Normalized depth map (0.5m – 20m range) |
| `state` | (N,) | Position, velocity, orientation, action history, altitude, search area direction |

The search area provides a direction toward the goal with up to ±10m noise. The drone must use its depth sensor to find the actual landing platform.

### Action Space

| Index | Name | Range | Description |
|-------|------|-------|-------------|
| 0 | vx | [-1, 1] | Velocity X direction |
| 1 | vy | [-1, 1] | Velocity Y direction |
| 2 | vz | [-1, 1] | Velocity Z direction |
| 3 | speed | [0, 1] | Thrust multiplier |
| 4 | yaw | [-1, 1] | Target yaw angle (maps to [-π, π]) |

**Constraints:**
- Max velocity: 3.0 m/s
- Max yaw rate: 3.141 rad/s (180°/s)
- Simulation rate: 50 Hz (dt = 1/50)
- Episode horizon: 60 seconds

<p align="right">(<a href="#miner-top">back to top</a>)</p>

---

## CLI

Swarm includes a CLI for the full development workflow. Install with `pip install -e .`, then use `swarm <command>`.

### Check Environment

```bash
swarm doctor
```

Verifies Python version, Docker, required dependencies, writable directories, and environment setup.

### Test Your Agent

```bash
swarm model test --source my_agent/
```

Validates your source folder — checks `drone_agent.py` exists and compiles, `requirements.txt` format, and estimated package size.

### Package Your Agent

```bash
swarm model package --source my_agent/
```

Bundles your `drone_agent.py`, model files, and optional `requirements.txt` into `Submission/submission.zip` (default path).

### Verify Submission

```bash
swarm model verify --model Submission/submission.zip
```

Checks structure, file sizes, and compliance before uploading.

### Run Benchmark

```bash
# Default benchmark (3 seeds per environment group)
swarm benchmark --model Submission/submission.zip --workers 4

# Quick test (1 seed per environment type)
swarm benchmark --model Submission/submission.zip --seeds-per-group 1
```

The `--seeds-per-group` flag controls how many seeds run per environment type. Validators run 1,000 seeds total (200 screening + 800 full).

### View Results

```bash
swarm report
```

`doctor`, `model verify`, `model test`, and `report` support `--json` for machine-readable output.

<p align="right">(<a href="#miner-top">back to top</a>)</p>

---

## GitHub Repo Setup

Validators download models from your **public GitHub repository**. You must set up a repo with the correct structure.

### 1. Create Your Repo

Create a public GitHub repository (e.g., `github.com/YOUR_USER/any-name`). Each repo is bound to a single hotkey — validators reject a repo already claimed by a different miner.

### 2. Copy the Template README

Your repo **must** contain the exact Swarm template README. This is enforced by SHA-256 hash — any modification will cause validators to reject your model.

```bash
cp swarm/templates/README.md YOUR_REPO/README.md
```

Do not edit this file. The hash is checked on every download.

### 3. Add Your Submission

```bash
cp Submission/submission.zip YOUR_REPO/submission.zip
git add README.md submission.zip
git commit -m "Add submission"
git push
```

### 4. Submit

The backend will download your `submission.zip` and verify your `README.md` automatically.

<p align="right">(<a href="#miner-top">back to top</a>)</p>

---

## Running the Miner

### Configuration

| Flag | Description | Example |
|------|-------------|---------|
| `--github_url` | **Required.** Public GitHub repo URL | `--github_url https://github.com/user/repo` |
| `--netuid` | Subnet netuid | `--netuid 124` |
| `--wallet.name` | Your coldkey name | `--wallet.name my_cold` |
| `--wallet.hotkey` | Hotkey used for mining | `--wallet.hotkey my_hot` |
| `--subtensor.network` | Network (finney, test) | `--subtensor.network finney` |

### Create Keys

```bash
btcli wallet new_coldkey --wallet.name my_cold
btcli wallet new_hotkey  --wallet.name my_cold --wallet.hotkey my_hot
```

### Submit Your Model

```bash
source miner_env/bin/activate

python neurons/miner.py \
     --netuid 124 \
     --subtensor.network finney \
     --wallet.name my_cold \
     --wallet.hotkey my_hot \
     --github_url "https://github.com/YOUR_USER/YOUR_REPO"
```

The miner submits your model and exits. You do **not** need to stay online — validators discover your model automatically.

<p align="right">(<a href="#miner-top">back to top</a>)</p>

---

## Scoring

```
score = 0.45 × success + 0.45 × time + 0.10 × safety
```

| Term | Weight | Description |
|------|--------|-------------|
| **Success** | 0.45 | 1.0 if valid landing, 0.0 otherwise |
| **Time** | 0.45 | 1.0 if within target time, decays linearly to 0.0 at horizon |
| **Safety** | 0.10 | 1.0 if min clearance ≥ safe distance, 0.0 if ≤ 0.2m, linear between (safe = 1.0m default, 0.6m for Forest) |

Collision with any obstacle sets the score to **0.01** (grace for legitimate models).

### Landing Requirements

**Static platforms** — hold stable for 0.5 seconds:

| Condition | Threshold |
|-----------|-----------|
| Vertical velocity | ≤ 0.5 m/s |
| Horizontal velocity | ≤ 0.6 m/s (relative to platform) |
| Tilt (roll / pitch) | ≤ 15° |

**Moving platforms** — City, Open, Mountain, and Village maps have a chance of spawning a moving landing platform (circular, linear, or figure-8 patterns). Warehouse and Forest maps always use static platforms. For moving targets, contact with the platform counts as success (no stable hold needed). Your model should be prepared for both.

<p align="right">(<a href="#miner-top">back to top</a>)</p>

---

## Benchmark System

### How Your Model Is Evaluated

1. **Miner** submits GitHub URL to backend (one-shot, goes offline)
2. **Backend** downloads `submission.zip`, computes SHA-256, verifies README hash
3. **Validator** syncs with backend, downloads your model from GitHub, verifies hash
4. **Validator** runs your agent in a sandboxed Docker container:
   - **Screening** (200 seeds) — quick filter, must score >= **101%** of the current champion (or >= 0.1 if no champion)
   - **Full benchmark** (800 seeds) — complete evaluation across all 6 environment types (City, Open, Mountain, Village, Warehouse, Forest)
5. **Final score** = average of all 1,000 seed results
6. **Winner-take-all** — #1 scorer receives emissions

### Epoch Rotation

Seeds rotate every **7 days** (Monday 16:00 UTC). Each validator independently generates 1,000 cryptographically random seeds per epoch using `random.SystemRandom()` — there is no shared secret.

Per-epoch seeds are published on [swarm124.com](https://swarm124.com) for full transparency.

### Key Numbers

| Parameter | Value |
|-----------|-------|
| Total seeds per epoch | 1,000 |
| Screening seeds | 200 |
| Full benchmark seeds | 800 |
| Screening threshold | >= 101% of champion score (or >= 0.1 bootstrap) |
| Max submission size | 50 MiB (compressed) |

<p align="right">(<a href="#miner-top">back to top</a>)</p>

---

## Docker Whitelist

Your `requirements.txt` can only include packages from the approved whitelist. Anything else will be rejected.

**Approved packages:**

```
torch, torchvision, torchaudio, onnx, onnxruntime, onnxruntime-gpu,
stable-baselines3, sb3-contrib, gymnasium, gym, numpy, scipy,
scikit-learn, opencv-python, opencv-python-headless, pillow, imageio,
matplotlib, pyyaml, tqdm, einops, tensorboard, h5py, msgpack
```

Need a package not on this list? Ask in [Discord](https://discord.gg/8dPqPDw7GC).

<p align="right">(<a href="#miner-top">back to top</a>)</p>

---

## Troubleshooting

**"Missing drone_agent.py"** — Your ZIP must contain `drone_agent.py` at the root level. Template files are auto-injected.

**"Dangerous executable files detected"** — Remove `.exe`, `.so`, `.dll` files. Only Python code and model files allowed.

**"Agent too large"** — Submissions must be ≤ 50 MiB compressed.

**"RPC connection failed"** — Ensure your agent starts correctly and responds to ping requests.

**"README hash mismatch"** — Your GitHub repo's README.md must be the exact copy from `swarm/templates/README.md`. Any edit will cause rejection.

**Environment issues** — Run `swarm doctor` to diagnose.

<p align="right">(<a href="#miner-top">back to top</a>)</p>

---

## Support

- **Discord** — [discord.gg/8dPqPDw7GC](https://discord.gg/8dPqPDw7GC) (ping @Miguelikk or @AliSaaf)
- **GitHub Issues** — open a ticket with logs & error trace
- **Website** — [swarm124.com](https://swarm124.com)

<p align="right">(<a href="#miner-top">back to top</a>)</p>
