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
    <li><a href="#faq">FAQ</a></li>
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
4. swarm model package        ← Bundle one family into Submission/submission.zip
5. swarm model verify         ← Verify local artifact compliance
6. swarm benchmark            ← Run local benchmark
7. swarm repo package         ← Build repo-root artifacts/ + submission_manifest.json
8. swarm repo verify          ← Verify full GitHub submission layout
9. Push to GitHub             ← Public repo with README + manifest + family artifacts
10. Submit model              ← One-shot submit, then go offline
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
        # Return action array [dir_x, dir_y, dir_z, speed, yaw]
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

The search clue is a 2D (Δx, Δy) offset sampled uniformly inside a 30m circle around the victim's true XY (no Z component in V5). The drone must use its depth sensor to find the humanoid victim on the ground, then hover steadily overhead.

### Action Space

| Index | Name | Range | Description |
|-------|------|-------|-------------|
| 0 | dir_x | [-1, 1] | Direction X component |
| 1 | dir_y | [-1, 1] | Direction Y component |
| 2 | dir_z | [-1, 1] | Direction Z component |
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

# Or select a specific family contract
swarm model package --source my_agent/ --family-id cf_autopilot
```

Bundles your `drone_agent.py`, model files, optional `requirements.txt`, and a generated `swarm_policy_contract.json` into `Submission/submission.zip` (default path).

### Verify Submission

```bash
swarm model verify --model Submission/submission.zip
```

Checks structure, file sizes, family policy-contract compatibility, and a local runtime smoke test before uploading.

### Build Repo Submission Layout

```bash
swarm repo package \
  --repo-root YOUR_REPO \
  --family-source cf_search_and_rescue=./sar_agent \
  --family-source cf_autopilot=./autopilot_agent

# Or update a single family later
swarm repo package \
  --repo-root YOUR_REPO \
  --source ./autopilot_agent_v2 \
  --family-id cf_autopilot \
  --overwrite
```

This writes family artifacts under `artifacts/<family_id>/submission.zip` and updates the repo-root `submission_manifest.json`.

### Verify Repo Submission Layout

```bash
swarm repo verify --repo-root YOUR_REPO --strict-manifest
```

Checks manifest structure, artifact hashes, policy-contract compatibility, and a local runtime smoke test for every published family artifact in the repo.

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

### 3. Add `submission_manifest.json`

Multi-family repos now declare published artifacts through a repo-root
`submission_manifest.json`.

Repo layout rules for manifest v1:

- `submission_manifest.json` lives at the repo root.
- `README.md` lives at the repo root.
- Family artifacts live under `artifacts/<family_id>/`.
- Each artifact entry points to one `.zip` file and one `family_id`.
- A repo can publish at most one artifact per `family_id`.

Minimal example:

```json
{
  "manifest_version": "submission_manifest.v1",
  "repo_layout_rules": {
    "manifest_path": "submission_manifest.json",
    "readme_path": "README.md",
    "artifacts_dir": "artifacts",
    "artifact_extension": ".zip"
  },
  "artifacts": [
    {
      "family_id": "cf_search_and_rescue",
      "interface_version": "submission_zip.v1",
      "artifact_path": "artifacts/cf_search_and_rescue/submission.zip",
      "sha256": "<artifact sha256>",
      "metadata": {
        "notes": "baseline SAR agent"
      }
    }
  ]
}
```

Legacy single-artifact repos that only publish a root `submission.zip` remain
temporarily compatible, but new multi-family work should use the manifest.

### 3. Package Family Artifacts Into The Repo

```bash
swarm repo package \
  --repo-root YOUR_REPO \
  --family-source cf_search_and_rescue=./sar_agent \
  --family-source cf_autopilot=./autopilot_agent

swarm repo verify --repo-root YOUR_REPO --strict-manifest

git add README.md submission_manifest.json artifacts/
git commit -m "Add submission"
git push
```

### 4. Submit

> **One-shot.** Each hotkey can commit one model, lifetime. The chain scanner ignores any later commitment from the same hotkey, and the database enforces a one-model-per-hotkey constraint. Benchmark locally and only commit when the model consistently beats **champion + 0.015**. To try a different model, a new hotkey and subnet registration are required.

To protect your model from front-running (someone copying your submission before you commit), follow this order:

1. Keep your GitHub repo **private**
2. Run the miner command below to commit the URL to chain
3. Wait for the commit to finalize (~30 seconds)
4. Make the repo **public**

The system processes commitments in block order — the earliest committer always wins, and duplicate model hashes are automatically rejected. Newer versions of the Bittensor SDK also encrypt your URL on-chain until the reveal round, adding another layer of protection.

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
| **Success** | 0.45 | 1.0 if CONFIRMED, 0.0 otherwise |
| **Time** | 0.45 | 1.0 if within target time, decays linearly to 0.0 at horizon |
| **Safety** | 0.10 | 1.0 if min clearance ≥ safe distance, 0.0 if ≤ 0.2m, linear between (safe = 1.0m default, 0.6m for Forest) |

Non-success failures (collision, no-touch sphere, infeasible, spawn-failure, tilt, timeout) score **0.01** participation for legitimate models. Evaluator errors and illegitimate models score 0.0.

### CONFIRMED Requirements (V5 Search-and-Rescue)

All four conditions must hold continuously for 2.0 seconds:

| Condition | Threshold |
|-----------|-----------|
| Drone speed | < 1.0 m/s |
| Horizontal distance to victim | ≤ 2.0 m |
| Height above victim's AABB top | 2.0 - 4.0 m |
| Distance from 0.8 m no-touch sphere | strictly outside |

Boundaries get a 10 cm / 0.1 m·s⁻¹ hysteresis grace once the predicate is already active, except the no-touch sphere which stays strict.

<p align="right">(<a href="#miner-top">back to top</a>)</p>

---

## Benchmark System

### How Your Model Is Evaluated

1. **Miner** commits the GitHub URL on-chain (one-shot, then goes offline)
2. **Backend** detects the commit. The chain scanner polls every ~3 minutes, so registration completes within **3–5 minutes** of chain-commit finalization. The backend downloads `submission.zip`, verifies the SHA-256 and README hashes, and adds the model to the pending queue
3. **Validators** sync the pending queue, download the model directly from the miner's GitHub repo, and verify the hash against the backend record
4. Each validator runs the agent in a sandboxed Docker container:
   - **Screening** (200 seeds) — the model advances to full benchmark only once validator scores clear **champion + 0.015** (or >= 0.01 if no champion)
   - **Full benchmark** (800 seeds) — remaining seeds across all 6 environment types (City, Open, Mountain, Village, Warehouse, Forest)
5. Final score and pass/fail are determined by **stake-weighted consensus across all active validators**. No single validator can block or advance a model on its own
6. **Winner-take-all** — the highest-scoring model receives emissions

### Epoch Rotation

Seeds rotate every **7 days** (Monday 16:00 UTC). Each validator independently generates 1,000 cryptographically random seeds per epoch using `random.SystemRandom()` — there is no shared secret.

Per-epoch seeds are published on [swarm124.com](https://swarm124.com) for full transparency.

### Key Numbers

| Parameter | Value |
|-----------|-------|
| Total seeds per epoch | 1,000 |
| Screening seeds | 200 |
| Full benchmark seeds | 800 |
| Screening threshold | >= champion score + 0.015 (or >= 0.01 bootstrap) |
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
matplotlib, pyyaml, tqdm, einops, tensorboard, h5py, msgpack,
swarm-bullet3, swarm-drone-gym
```

Need a package not on this list? Ask in [Discord](https://discord.gg/8dPqPDw7GC).

<p align="right">(<a href="#miner-top">back to top</a>)</p>

---

## FAQ

### When will my score show up on the leaderboard?

Validators process the pending model queue in submission order, one item at a time per validator. Expected latency depends on queue depth:

- **Short queue** — first score reported within 10–30 minutes of submission.
- **Busy queue** — add the screening + benchmark time for each model ahead.
- **Epoch rollover freeze** — new-model work is paused briefly before each epoch transition and resumes after seed rotation.

The leaderboard reshuffles once stake-weighted consensus settles across validators.

### What happens if my model fails screening?

The hotkey is done. Each hotkey can commit **one model, lifetime** — any later commitment from the same hotkey is ignored. A failed model stays recorded against that hotkey; to compete again, register a new hotkey and submit from it.

Treat every submission as one-shot. Run the full CLI benchmark locally, tune aggressively, and only commit on-chain once the model consistently beats **champion + 0.015** across all environment types.

### Can I update my submission after committing?

No. A hotkey gets a single commitment. Validators (and the backend) see only the first registered model for that hotkey — subsequent chain commitments are ignored. To submit a different model, use a different hotkey.

### I submitted, but I don't see a score yet. What should I check?

Common causes:

- **Repo still private** — validators cannot download it. Make the repo public after the chain commit finalizes.
- **README hash mismatch** — `README.md` in the repo must be a byte-exact copy of `swarm/templates/README.md`. Any modification causes rejection.
- **Non-whitelisted package in `requirements.txt`** — see the [Docker Whitelist](#docker-whitelist).
- **Submission exceeds 50 MiB** — compressed size limit is enforced.
- **No validators online** — check [swarm124.com](https://swarm124.com) for validator activity.

If none apply, contact the team on [Discord](https://discord.gg/8dPqPDw7GC).

### How often are weights updated on-chain?

Validators set weights each forward cycle (a few minutes). Once consensus settles on a new champion, the next weight-set propagates the change across the subnet.

### What if two miners submit the same model?

Duplicate submission hashes are rejected and duplicate GitHub URLs are rejected. The earliest on-chain committer is accepted. To guard against front-running, follow: **private repo → chain commit → wait for finalization → make repo public** (see [GitHub Repo Setup](#github-repo-setup)).

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
