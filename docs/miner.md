<a id="miner-top"></a>

# Swarm Miner Guide

Train an autonomous drone pilot, benchmark it against 1,100 procedurally generated worlds, and compete on the [leaderboard](https://swarm124.com/benchmark).

---

<details>
  <summary><b>Table of Contents</b></summary>
  <ol>
    <li><a href="#system-requirements">System Requirements</a></li>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#challenge-families">Challenge Families</a></li>
    <li><a href="#workflow">Workflow</a></li>
    <li><a href="#creating-your-agent">Creating Your Agent</a></li>
    <li><a href="#observations--actions">Observations & Actions</a></li>
    <li><a href="#cli">CLI</a></li>
    <li><a href="#github-repo-setup">GitHub Repo Setup</a></li>
    <li><a href="#running-the-miner">Running the Miner</a></li>
    <li><a href="#scoring">Scoring</a></li>
    <li><a href="#emissions-king-of-the-hill">Emissions: King of the Hill</a></li>
    <li><a href="#benchmark-system">Benchmark System</a></li>
    <li><a href="#docker-whitelist">Docker Whitelist</a></li>
    <li><a href="#faq">FAQ</a></li>
    <li><a href="#troubleshooting">Troubleshooting</a></li>
    <li><a href="#support">Support</a></li>
  </ol>
</details>

---

## System Requirements

The miner process is lightweight: it commits a GitHub repository URL on-chain, then exits. The provided setup scripts target **Ubuntu**, require `sudo`, and install **Python 3.11** specifically. Other platforms can run the CLI after installing the equivalent Python and system dependencies manually. Training hardware depends on your approach (SB3, PyTorch, custom RL, or another compatible stack).

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
```

<p align="right">(<a href="#miner-top">back to top</a>)</p>

---

## Challenge Families

Swarm runs **six challenge families**. Five are active, public, and paying; Interceptor is completed with archived emissions. Each family is its own competition: its own queue, its own champion lineage, and its own slice of subnet emissions. One hotkey holds **one model in one family**; to compete in another family, register another hotkey.

| Family | ID | Drones | Maps | Emission slice | Guide |
|--------|----|--------|------|----------------|-------|
| Autopilot / Navigation | `cf_autopilot` | 1 | City, Open, Mountain, Village, Warehouse, Forest | 15% | [families/autopilot.md](families/autopilot.md) |
| Search and Rescue | `cf_search_and_rescue` | 1 | City, Open, Mountain, Village, Warehouse, Forest | 15% | [families/search_and_rescue.md](families/search_and_rescue.md) |
| Swarm Autopilot | `cf_swarm_autopilot` | 2–8 | City, Open, Mountain, Village, Forest | 20% | [families/swarm_autopilot.md](families/swarm_autopilot.md) |
| Swarm Search and Rescue | `cf_swarm_sar` | 2–8 | City, Open, Mountain, Village, Forest | 20% | [families/swarm_sar.md](families/swarm_sar.md) |
| Interceptor | `cf_interceptor` | 1 (vs. a validator-flown target) | Open | 0% (completed; historical 30%) | [families/interceptor.md](families/interceptor.md) |
| Office Interceptor | `cf_interceptor_office` | 1 (vs. a validator-flown target) | Office (fixed indoor map) | 30% | [families/office_interceptor.md](families/office_interceptor.md) |


The swarm families fly 2–8 drones per seed, all under one policy. Each active family holds a fixed slice of subnet emissions, and the five active slices add up to the whole pool. A slice still burns if its own family stops paying out — no kings, or archived. How a slice is split among a family's kings is covered in [Emissions](#emissions-king-of-the-hill).

<p align="right">(<a href="#miner-top">back to top</a>)</p>

---

## Workflow

The full miner workflow, from first install to competing on the leaderboard:

```
1. swarm doctor              ← Check environment readiness
2. Train your model           ← SB3, PyTorch, or custom
3. swarm model test           ← Validate source folder before packaging
4. swarm model package        ← Bundle one family into Submission/submission.zip
5. swarm model verify         ← Verify local artifact compliance
6. swarm benchmark            ← Run local benchmark
7. swarm repo package         ← Build repo-root artifacts/ + submission_manifest.json
8. swarm repo verify          ← Verify full GitHub submission layout
9. Push to GitHub             ← Public repo with README + manifest + the family artifact
10. Submit model              ← Commit the repo URL on-chain, then go offline
```

> Run `swarm model package` without `--family-id` and it asks which family you trained for, so you never bundle the wrong one. Pass `--family-id` to skip the prompt (required in scripts); a mismatched policy contract fails verification.

<p align="right">(<a href="#miner-top">back to top</a>)</p>

---

## Creating Your Agent

### Start from the Template

```bash
cp -r swarm/submission_template/ my_agent/
cd my_agent/
# Edit drone_agent.py with your controller
```

For Office Interceptor, copy the office starter to the packaged entry point:

```bash
mkdir -p my_agent/
cp swarm/submission_template/office_drone_agent.py my_agent/drone_agent.py
```

Test an Office Interceptor agent with its required family ID:

```bash
swarm model test --source my_agent/ --family-id cf_interceptor_office
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
        # observation: dict with "depth" (256,256,1), "rgb" (256,256,3) and "state" (N,)
        # Return action array [dir_x, dir_y, dir_z, speed, yaw, rgb_request]
        # (shapes shown are the Search-and-Rescue contract -- see families/<family>.md for yours)
        action, _ = self.model.predict(observation, deterministic=True)
        return action

    def reset(self):
        # Reset internal state between missions
        pass
```

**Required files:**
- `drone_agent.py`: Your controller class, at the zip root (REQUIRED)
- `requirements.txt`: Additional pip packages (optional, must be on the [whitelist](#docker-whitelist))
- Model artifacts using a supported extension: `.bin`, `.ckpt`, `.h5`, `.json`, `.npy`, `.npz`, `.onnx`, `.pb`, `.pkl`, `.pt`, `.pth`, `.safetensors`, `.tflite`, `.weights`, or `.zip`

`swarm model package` recursively includes only those artifacts plus root-level `drone_agent.py` and `requirements.txt`. Arbitrary helper `.py`, YAML, TOML, and other files are not packaged; keep required logic in `drone_agent.py` and required configuration in a supported model artifact.

**Auto-injected (do not include):**
- `main.py`, `agent.capnp`, `agent_server.py`, `runtime_caps.py`: provided by the evaluation system

**Hard limits, enforced at intake:**
- Compressed artifact download ≤ **50 MiB**
- Total **uncompressed** content ≤ **50 MiB** (summed across zip entries: a zip-bomb guard, so squeezing the archive harder does not help)
- No `.exe`, `.so`, `.dll`, `.sh`, `.bat`, or `.pyc` entries
- No path traversal, absolute paths, or symlinks inside the zip

<p align="right">(<a href="#miner-top">back to top</a>)</p>

---

## Observations & Actions

The interface below is the **Search and Rescue** one. Each family defines its own observation/action contract. Check the [family guide](#challenge-families) for the family you target before training.

### Observation Space

| Field | Shape | Description |
|-------|-------|-------------|
| `depth` | (256, 256, 1) | Normalized depth map (0.5 m – 30 m range) |
| `rgb` | (256, 256, 3) | On-demand colour frame in [0,1]; all zeros unless your previous action requested it (max 40 requests per episode) |
| `state` | (N,) | Position, velocity, orientation, action history, altitude, search area direction |

The search clue is an offset sampled inside a **30 m** circle around the victim (the swarm SAR family shares one clue over a disk that scales with team size: 80·√(n/8) m, i.e. 40 m for 2 drones up to 80 m for 8). The drone must use its depth sensor to find the humanoid victim on the ground, then hover steadily overhead.

For Office Interceptor, the contract is `rgb` (256, 256, 3) plus a 127-float `state` vector, with four RC-stick actions `[lr, fb, ud, yaw]`. Its speed cap is 3 m/s and its episode horizon is 60 seconds. See the [Office Interceptor guide](families/office_interceptor.md) for the full contract.

### Action Space

| Index | Name | Range | Description |
|-------|------|-------|-------------|
| 0 | dir_x | [-1, 1] | Direction X component |
| 1 | dir_y | [-1, 1] | Direction Y component |
| 2 | dir_z | [-1, 1] | Direction Z component |
| 3 | speed | [0, 1] | Thrust multiplier |
| 4 | yaw | [-1, 1] | Target yaw angle (maps to [-π, π]) |
| 5 | rgb_request | [0, 1] | Set above 0.5 to receive a colour frame in the next observation's `rgb` (max 40 per episode) |

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
swarm model test --source my_agent/ --family-id cf_autopilot
```

Packages the source against the selected family's policy contract, applies the submission ZIP structure/safety checks, and runs the local runtime smoke test. The requirements whitelist is enforced later by the validator, so review it separately before publishing.

### Package Your Agent

```bash
swarm model package --source my_agent/ --family-id cf_autopilot
```

Bundles your `drone_agent.py`, model files, optional `requirements.txt`, and a generated `swarm_policy_contract.json` into `Submission/submission.zip` (default path). Omit `--family-id` in a terminal and it prompts you to pick the family; it is required (and errors without it) for non-interactive runs.

### Verify Submission

```bash
swarm model verify --model Submission/submission.zip
```

Checks the compressed ZIP against the 50 MiB download cap, inspects ZIP safety and structure, verifies the family policy contract, and runs a local runtime smoke test. Its local uncompressed-safety limit defaults to 300 MiB; pass `--max-uncompressed-mb 50` to mirror intake exactly.

### Build Repo Submission Layout

```bash
swarm repo package \
  --repo-root YOUR_REPO \
  --family-source cf_autopilot=./autopilot_agent

# Or update the artifact later
swarm repo package \
  --repo-root YOUR_REPO \
  --source ./autopilot_agent_v2 \
  --family-id cf_autopilot \
  --overwrite
```

This writes the family artifact under `artifacts/<family_id>/submission.zip` and updates the repo-root `submission_manifest.json`. A repo packages exactly **one family**: passing a second `--family-source`, or a family different from the one already in the manifest, is rejected.

### Verify Repo Submission Layout

```bash
swarm repo verify --repo-root YOUR_REPO
```

Checks manifest structure, the artifact hash, policy-contract compatibility, and a local runtime smoke test for the published family artifact in the repo.

### Run Benchmark

```bash
# Default benchmark (3 seeds per environment group)
swarm benchmark --model Submission/submission.zip --workers 4

# Quick test (1 seed per environment type)
swarm benchmark --model Submission/submission.zip --seeds-per-group 1
```

The `--seeds-per-group` flag controls how many seeds run per environment type. Validators run 1,100 seeds total.

### View Results

```bash
swarm report
```

<p align="right">(<a href="#miner-top">back to top</a>)</p>

---

## GitHub Repo Setup

Validators download models from your **public GitHub repository**. You must set up a repo with the correct structure.

### 1. Create Your Repo

Create a GitHub repository. Commit its repository-root URL (`https://github.com/YOUR_USER/YOUR_REPO`; a trailing slash or `.git` suffix is normalized, but subpaths and other hosts are rejected). Files are fetched from `main`, falling back to `master`. Keep the repo private only until the chain commitment finalizes, then make it public so the scanner can read it. Each repo is bound to a single hotkey, and a URL already registered to another miner is skipped.

### 2. The Template README

Your repo **must** contain the exact Swarm template README, enforced by SHA-256 hash. You do not copy it by hand: `swarm repo package` writes the correct `README.md` into your repo for you, and `swarm repo verify` re-checks it before you publish.

> **Why this matters: the failure is silent.** If `README.md` is not a byte-identical copy of the template, the chain scanner never registers your submission, and no error reaches you: the commitment simply produces nothing on the leaderboard. The tooling prevents this, so the rule is simple: let `swarm repo package` write the README and do not hand-edit it. If a submission never appears, run `swarm repo verify` and check the `README.md` line **first**.

### 3. Add `submission_manifest.json`

Repos declare their published artifact through a repo-root `submission_manifest.json`.

Repo layout rules for manifest v1:

- `submission_manifest.json` lives at the repo root.
- `README.md` lives at the repo root.
- The family artifact lives under `artifacts/<family_id>/` with a `.zip` extension.
- The artifact entry declares `family_id`, `interface_version`, `artifact_path`, `sha256`, `size_bytes`, and `metadata`.
- A repo publishes exactly **one artifact for one family**; a manifest listing more is rejected.
- The artifact's `sha256` is verified against the downloaded file.

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
      "family_id": "cf_autopilot",
      "interface_version": "submission_zip.v1",
      "artifact_path": "artifacts/cf_autopilot/submission.zip",
      "sha256": "<artifact sha256>",
      "size_bytes": 123456,
      "metadata": {
        "notes": "baseline autopilot agent"
      }
    }
  ]
}
```

All families use the `submission_zip.v1` interface. A hand-written manifest must include `size_bytes`, set to the ZIP byte size; the generated manifest already includes it. For an Office Interceptor submission, the manifest targets `cf_interceptor_office` and its artifact path is under `artifacts/cf_interceptor_office/`. A repo without `submission_manifest.json` is rejected.

### 4. Package The Family Artifact Into The Repo

```bash
swarm repo package \
  --repo-root YOUR_REPO \
  --family-source cf_autopilot=./autopilot_agent

swarm repo verify --repo-root YOUR_REPO

git -C YOUR_REPO add README.md submission_manifest.json artifacts/
git -C YOUR_REPO commit -m "Add submission"
git -C YOUR_REPO push
```

### 5. Submit

> **One model per hotkey.** A hotkey enters exactly one family with exactly one model, published from its repo's manifest. To compete in more families, register more hotkeys, one per family.
>
> Treat every submission as final. Once your model is evaluated, the hotkey's slot is **locked**: pushing a new artifact does not replace it and does not re-run the benchmark. To compete again, register a **new hotkey** and submit from it. See the [FAQ](#faq) for more.
>
> After the backend accepts a model, that hotkey cannot submit another one. An epoch change, a failed evaluation, or a version update does not free it. If the repository is rejected before the model is accepted, fix the problem and submit again with the same hotkey. Test carefully before committing because an accepted artifact cannot be replaced.

To protect your model from front-running (someone copying your submission before you commit), follow this order:

1. Keep your GitHub repo **private**
2. Run the miner command below to commit the URL to chain
3. Wait for the commit to finalize (~30 seconds)
4. Make the repo **public**

Commitments are processed in block order: the earliest committer wins. A model hash already registered to another miner is rejected, and so is a repo URL owned by a different hotkey.

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

The miner commits your repo URL on-chain and exits. You do **not** need to stay online: validators discover your model automatically.

The repo URL carries your single family artifact. A manifest declaring more than one family is rejected outright, and a commitment naming a different family while your hotkey already holds a live model is ignored.

<p align="right">(<a href="#miner-top">back to top</a>)</p>

---

## Scoring

Per-seed reward for most families:

```
score = 0.45 × success + 0.45 × time + 0.10 × safety
```

| Term | Weight | Description |
|------|--------|-------------|
| **Success** | 0.45 | 1.0 if the mission objective is met, 0.0 otherwise |
| **Time** | 0.45 | 1.0 if within target time, decays to 0.0 at the horizon |
| **Safety** | 0.10 | 1.0 if min clearance ≥ 1.0 m (0.6 m in Forest), 0.0 at ≤ 0.2 m, linear between |

The Interceptor and Office Interceptor families override the weights to 0.5 success / 0.5 time, with no safety term.

Non-success failures (collision, timeout, etc.) score **0.01** participation for legitimate models; evaluator errors and illegitimate models score 0.0.

Your **model score** is the mean of the eligible recorded seed scores across the 1,100-seed range, stitched together from whichever validators ran each seed (the earliest accepted report per seed counts, so re-runs never double-count). Deterministic environment failures and validator-infrastructure failures satisfy coverage but are excluded from the mean.

### CONFIRMED Requirements (Search and Rescue)

All four conditions must hold continuously for 2.0 seconds:

| Condition | Threshold |
|-----------|-----------|
| Drone speed | < 1.0 m/s |
| Horizontal distance to victim | ≤ 2.0 m |
| Height above victim's AABB top | 2.0 – 4.0 m |
| Distance from victim center | ≥ 0.8 m |

The speed, horizontal-distance, and height-band bounds get a 0.1 m / 0.1 m·s⁻¹ hysteresis grace once the predicate is already active; the 0.8 m no-touch sphere gets no grace.

### Becoming Champion

The first evaluated model in a family becomes champion unconditionally. After that, a challenger must beat the reigning champion's score by a **dynamic improvement floor**: flat while the champion score is ≤ 0.5, then decaying toward a minimum as the champion approaches a perfect 1.0: late-game inches are cheaper to require than early-game leaps.

| Families | Floor (champion ≤ 0.5) | Floor minimum (champion → 1.0) |
|----------|------------------------|--------------------------------|
| All families | +0.015 | +0.005 |

<p align="right">(<a href="#miner-top">back to top</a>)</p>

---

## Emissions: King of the Hill

Emissions are **not** winner-take-all, and there is no single global leaderboard payout. Each family runs its own King-of-the-Hill lineage: the family's emission slice is split across its **last five crowned kings**. A seat's share is set first by its rank in the window — the reigning champion holds the full weight and every step back keeps **70%** of the seat above it — and fine-tuned by how much that king improved on the score it beat. Being dethroned doesn't stop your earnings. Only five newer crownings pushing you out of the window does.

The practical consequences:

- A copycat that barely clears the floor earns almost nothing; a real jump earns a dominant share and keeps paying through the next several dethronings.
- Your seat's share is frozen at crowning; champion re-evaluations on fresh seeds don't change it.
- Payout requires your repo to stay intact and reachable. A deleted, private, or changed repo stops the seat paying; restoring the exact accepted repo and artifact restores eligibility while the seat remains in the window.

The exact formula, window mechanics, and edge cases are in [king_of_the_hill.md](king_of_the_hill.md).

<p align="right">(<a href="#miner-top">back to top</a>)</p>

---

## Benchmark System

### How Your Model Is Evaluated

1. **Miner** commits the GitHub URL on-chain, then goes offline
2. **Backend** detects the commit: the chain scanner polls every 3 minutes, so registration lands within minutes of finalization. It verifies the README hash, downloads the manifest and the declared artifact, checks its SHA-256, and creates one **Pending Benchmark** row
3. Each family is a **queue lane**: champion epoch re-evals run first, then any queued re-evals, then the oldest pending model; a rotation cursor cycles across families so no lane starves
4. **Validators** lease the model's seeds individually from a shared pool and run the agent in a sandboxed Docker container: the full **1,100 seeds** per family, spread over the family's environment types
5. When the whole seed range [0, 1100) is covered (by any mix of validators' completed seeds), the stitched mean becomes the model's score and the status flips to **Evaluated**; the champion check then runs

Every submission runs the full 1,100-seed benchmark directly. (A 300-seed screening pre-gate exists in the code behind a hardcoded `SCREENING_ENABLED = False`; it is off, and validators offering screening work are refused.)

A transient timeout or RPC-transport failure is retried once for that seed, subject to run-wide retry budgets. Deterministic environment failures and validator-infrastructure failures are excluded from the score; failures caused by the submitted agent still count. Smoke-test with `swarm model verify` before committing.

### Epoch Rotation

Epochs run for **14 days** from epoch 19 onward, anchored Monday 16:00 UTC (epochs 1–18 were 7 days). Each validator independently generates its own 1,100 seeds per family per epoch using `random.SystemRandom()`: there is no shared secret. Validators publish each epoch's seed sets to the backend **after** the epoch ends, where they are publicly readable.

At rollover, pending models keep their queue position, discard partial results, and restart evaluation on the new epoch's seeds. Every champion is also queued for re-evaluation. For the final **1.5 hours** of an epoch the scanner stops registering new commitments; commitments made then are picked up after rollover.

### Key Numbers

| Parameter | Value |
|-----------|-------|
| Seeds per family per epoch | 1,100 |
| Seed claim size | Dynamic: up to the validator's free worker slots (API cap 64) |
| Chain scanner interval | 3 minutes |
| Epoch length | 14 days from epoch 19 (Monday 16:00 UTC anchor) |
| Pre-rollover registration freeze | 1.5 hours |
| Max artifact size | 50 MiB compressed download and 50 MiB uncompressed content |
| Models per hotkey | 1 (one family per hotkey) |
| Chain commit cooldown | ~20 minutes |

<p align="right">(<a href="#miner-top">back to top</a>)</p>

---

## Docker Whitelist

Your `requirements.txt` can only include packages from the approved whitelist (`DOCKER_PIP_WHITELIST` in `swarm/constants.py`). It is enforced on the validator host at evaluation time: a non-whitelisted package fails your run there, not at submission.

**Approved packages:**

```
torch, torchvision, torchaudio, onnx, onnxruntime, onnxruntime-gpu,
stable-baselines3, sb3-contrib, gymnasium, gym, numpy, scipy,
scikit-learn, opencv-python, opencv-python-headless, pillow, imageio,
matplotlib, pyyaml, tqdm, einops, tensorboard, h5py, msgpack,
swarm-bullet3, swarm-drone-gym
```

Version pins are fine; pip option lines, URL/path installs (`git+`, `http://`, `https://`, `file:`, `./`, or an absolute path), and PEP 508 `@` direct references are rejected.

Need a package not on this list? Ask in [Discord](https://discord.gg/8dPqPDw7GC).

<p align="right">(<a href="#miner-top">back to top</a>)</p>

---

## FAQ

### When will my score show up on the leaderboard?

Registration takes minutes (the scanner polls every 3 minutes). Evaluation time depends on how many models sit ahead of yours across the family lanes: champion epoch re-evals take priority, and the queue rotates one family at a time. During the 1.5-hour pre-rollover freeze new commitments wait until the next epoch.

### Can I update my submission after committing?

**No, once your model has been evaluated.** A hotkey gets one submission. A changed artifact on an evaluated model is ignored, not re-run, so to try a better model you need a new hotkey.

The only exception is when the backend rejects the repository before accepting the model. In that case, fix the problem and submit again with the same hotkey. Once accepted, the hotkey stays used through epoch changes and version updates. Never replace a champion's artifact; the new artifact is rejected, and payment stops until the exact accepted repository and artifact are restored.

The chain rate-limits commits to roughly one per 20 minutes.

### What happens if my model fails evaluation?

The hotkey is used up. A model that was evaluated and failed (or whose repo went dead) keeps its slot locked, so pushing a new artifact will not re-run it. To try again, register a new hotkey. The exception is a submission rejected for a fixable packaging problem, which leaves the slot open to fix and re-commit.

### Can I compete in more than one family?

Not on the same hotkey: every hotkey enters exactly one family. To compete in several families, register one hotkey per family, each publishing from its own repo via `submission_manifest.json`. Each entry is evaluated and championed independently.

### I submitted, but I don't see a score yet. What should I check?

In order of likelihood:

- **README hash mismatch**: the number-one silent killer. `README.md` must be a byte-exact copy of the template; if it isn't, the scanner drops the submission without any visible error. Run `swarm repo verify` to catch it, or `swarm repo package` to rewrite the correct README. Check this first.
- **Repo still private**: the backend cannot fetch it. Make it public after the chain commit finalizes; the scanner re-reads all commitments every pass.
- **Wrong URL shape**: use a repository-root GitHub URL such as `https://github.com/owner/repo`; subpaths and non-GitHub hosts are rejected.
- **Manifest problems**: artifact path/hash mismatch, the artifact missing from `artifacts/<family_id>/`, or a manifest declaring more than one family.
- **Freeze window**: commits during the last 1.5 hours of an epoch register after rollover.
- **Non-whitelisted package or oversized artifact**: see [Docker Whitelist](#docker-whitelist) and the 50 MiB compressed/uncompressed caps.

If none apply, contact the team on [Discord](https://discord.gg/8dPqPDw7GC).

### How often are weights updated on-chain?

Validators refresh the king windows from the backend, recompute the weights locally, and set them on-chain on a periodic cadence, so a new champion's effect on rewards shows up within the epoch, not instantly.

### What if two miners submit the same model?

Model hashes are globally unique: a hash already registered to any miner is rejected, and a repo URL owned by a different hotkey is skipped. The earliest on-chain committer wins. To guard against front-running, follow: **private repo → chain commit → wait for finalization → make repo public** (see [GitHub Repo Setup](#github-repo-setup)).

<p align="right">(<a href="#miner-top">back to top</a>)</p>

---

## Troubleshooting

**"Missing drone_agent.py"**: Your ZIP must contain `drone_agent.py` at the root level. Template files are auto-injected.

**"Dangerous executable files detected"**: Remove `.exe`, `.so`, `.dll`, `.sh`, `.bat`, and `.pyc` files. Only Python code and model files are allowed.

**"Agent too large"**: Both the downloaded ZIP and its total uncompressed content must be ≤ 50 MiB. Shrink the weights rather than relying on compression.

**"RPC connection failed"**: Ensure your agent starts correctly and responds to ping requests. Transient RPC-transport failures are retried once per seed, subject to the run-wide retry budget; persistent agent failures still count against the submission.

**"README hash mismatch"**: Your repo's `README.md` must be the exact template copy. Any edit (including whitespace or line-ending changes) makes the scanner silently ignore your submission. Re-run `swarm repo package` to rewrite the correct file, then `swarm repo verify` to confirm.

**Wrong family packaged**: repackage with `swarm model package` and pick the right family at the prompt (or pass the correct `--family-id`).

**Environment issues**: Run `swarm doctor` to diagnose.

<p align="right">(<a href="#miner-top">back to top</a>)</p>

---

## Support

- **Discord**: [discord.gg/8dPqPDw7GC](https://discord.gg/8dPqPDw7GC) (ping @Miguelikk or @AliSaaf)
- **GitHub Issues**: open a ticket with logs & error trace
- **Website**: [swarm124.com](https://swarm124.com)

<p align="right">(<a href="#miner-top">back to top</a>)</p>
