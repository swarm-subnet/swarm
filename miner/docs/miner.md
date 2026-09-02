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
    <li><a href="#submitting-your-model">Submitting Your Model</a></li>
    <li><a href="#private-until-crowned">Private Until Crowned</a></li>
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

The miner process is lightweight: it commits your model's digest on-chain, uploads the archive to the Swarm backend, and exits. The provided setup scripts target **Ubuntu**, require `sudo`, and install **Python 3.11** specifically. Other platforms can run the CLI after installing the equivalent Python and system dependencies manually. Training hardware depends on your approach (SB3, PyTorch, custom RL, or another compatible stack).

<p align="right">(<a href="#miner-top">back to top</a>)</p>

---

## Installation

```bash
git clone https://github.com/swarm-subnet/swarm
cd swarm

chmod +x miner/src/scripts/install_dependencies.sh
./miner/src/scripts/install_dependencies.sh

chmod +x miner/src/scripts/setup.sh
./miner/src/scripts/setup.sh

source miner_env/bin/activate
```

<p align="right">(<a href="#miner-top">back to top</a>)</p>

---

## Challenge Families

Swarm runs **six challenge families**. Five are active and paying; Interceptor is completed with archived emissions. Each family is its own competition: its own queue, its own champion lineage, and its own slice of subnet emissions. One hotkey holds **one model in one family**; to compete in another family, register another hotkey.

| Family | ID | Drones | Maps | Emission slice | Guide |
|--------|----|--------|------|----------------|-------|
| Autopilot / Navigation | `cf_autopilot` | 1 | City, Open, Mountain, Village, Warehouse, Forest | 15% | [families/autopilot.md](../../docs/families/autopilot.md) |
| Search and Rescue | `cf_search_and_rescue` | 1 | City, Open, Mountain, Village, Warehouse, Forest | 15% | [families/search_and_rescue.md](../../docs/families/search_and_rescue.md) |
| Swarm Autopilot | `cf_swarm_autopilot` | 2–8 | City, Open, Mountain, Village, Forest | 20% | [families/swarm_autopilot.md](../../docs/families/swarm_autopilot.md) |
| Swarm Search and Rescue | `cf_swarm_sar` | 2–8 | City, Open, Mountain, Village, Forest | 20% | [families/swarm_sar.md](../../docs/families/swarm_sar.md) |
| Interceptor | `cf_interceptor` | 1 (vs. a validator-flown target) | Open | 0% (completed; historical 30%) | [families/interceptor.md](../../docs/families/interceptor.md) |
| Office Interceptor | `cf_interceptor_office` | 1 (vs. a validator-flown target) | Office (fixed indoor map) | 30% | [families/office_interceptor.md](../../docs/families/office_interceptor.md) |


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
7. swarm model submit         ← Package, verify, commit on-chain and upload, in one step
```

> Run `swarm model package` or `swarm model submit` without `--family-id` and it asks which family you trained for, so you never bundle the wrong one. Pass `--family-id` to skip the prompt (required in scripts); a mismatched policy contract fails verification.

<p align="right">(<a href="#miner-top">back to top</a>)</p>

---

## Creating Your Agent

### Start from the Template

```bash
mkdir -p my_agent/
cp miner/src/submission_template/drone_agent.py my_agent/
cd my_agent/
# Edit drone_agent.py with your controller
```

For Office Interceptor, copy the office starter to the packaged entry point:

```bash
mkdir -p my_agent/
cp miner/src/submission_template/office_drone_agent.py my_agent/drone_agent.py
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
- `requirements.txt`: Additional packages (optional, must be on the [whitelist](#docker-whitelist))
- Model artifacts using a supported extension: `.bin`, `.ckpt`, `.h5`, `.json`, `.npy`, `.npz`, `.onnx`, `.pb`, `.pkl`, `.pt`, `.pth`, `.safetensors`, `.tflite`, `.weights`, or `.zip`

`swarm model package` recursively includes only those artifacts plus root-level `drone_agent.py` and `requirements.txt`. Arbitrary helper `.py`, YAML, TOML, and other files are not packaged; keep required logic in `drone_agent.py` and required configuration in a supported model artifact.

**Auto-injected (do not include):**
- `main.py`, `agent.capnp`, `agent_server.py`, `runtime_caps.py`: provided by the evaluation system

**Hard limits, enforced before the commit and again at intake:**
- Compressed archive ≤ **50 MiB**
- Total **uncompressed** content ≤ **50 MiB** (summed across zip entries: a zip-bomb guard, so squeezing the archive harder does not help)
- No `.exe`, `.so`, `.dll`, `.sh`, `.bat`, or `.pyc` entries
- No path traversal, absolute paths, or symlinks inside the zip
- Every line of `requirements.txt` on the [whitelist](#docker-whitelist)

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

For Office Interceptor, the contract is `rgb` (256, 256, 3) plus a 127-float `state` vector, with four RC-stick actions `[lr, fb, ud, yaw]`. Its speed cap is 3 m/s and its episode horizon is 60 seconds. See the [Office Interceptor guide](../../docs/families/office_interceptor.md) for the full contract.

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

Swarm includes a CLI for the full development workflow. Install it from the repository root, then use `swarm <command>`.

```bash
cd "$(git rev-parse --show-toplevel)"
uv venv
source .venv/bin/activate
uv pip install -e .
```

### Check Environment

```bash
swarm doctor
```

Verifies Python version, Docker, required dependencies, writable directories, and environment setup.

### Test Your Agent

```bash
swarm model test --source my_agent/ --family-id cf_autopilot
```

Packages the source against the selected family's policy contract, applies the submission ZIP structure/safety checks, and runs the local runtime smoke test.

### Package Your Agent

```bash
swarm model package --source my_agent/ --family-id cf_autopilot
```

Bundles your `drone_agent.py`, model files, optional `requirements.txt`, and a generated `swarm_policy_contract.json` into `Submission/submission.zip` (default path). Omit `--family-id` in a terminal and it prompts you to pick the family; it is required (and errors without it) for non-interactive runs.

### Verify Submission

```bash
swarm model verify --model Submission/submission.zip
```

Checks the compressed ZIP against the 50 MiB cap, inspects ZIP safety and structure, verifies the family policy contract, and runs a local runtime smoke test. Its local uncompressed-safety limit defaults to 300 MiB; pass `--max-uncompressed-mb 50` to mirror intake exactly.

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

### Submit

```bash
swarm model submit --source my_agent/ --family-id cf_autopilot \
  --wallet.name my_cold --wallet.hotkey my_hot
```

See [Submitting Your Model](#submitting-your-model).

<p align="right">(<a href="#miner-top">back to top</a>)</p>

---

## Submitting Your Model

### Create Keys

```bash
btcli wallet new_coldkey --wallet.name my_cold
btcli wallet new_hotkey  --wallet.name my_cold --wallet.hotkey my_hot
```

### One command

```bash
source miner_env/bin/activate

swarm model submit \
     --source my_agent/ \
     --family-id cf_autopilot \
     --wallet.name my_cold \
     --wallet.hotkey my_hot
```

Already have a packaged archive? Pass it instead of a source folder:

```bash
swarm model submit --artifact Submission/submission.zip --family-id cf_autopilot \
     --wallet.name my_cold --wallet.hotkey my_hot
```

What the command does, in order:

1. **Packages** the source folder into `Submission/submission.zip` (skipped with `--artifact`).
2. **Verifies** the archive locally: size caps, zip safety, policy contract, runtime smoke test, and every `requirements.txt` line against the whitelist. A failure stops here, before anything touches the chain.
3. **Checks the backend**: it is reachable, the submission window is open, and this exact archive is not already registered.
4. **Commits** the archive's SHA-256 and the family on-chain from your hotkey.
5. **Uploads** the archive to the Swarm backend, signed by the same hotkey. The backend checks that the bytes hash to the committed digest and stores them in a private vault.

The upload waits for the backend's chain scanner (it runs every 3 minutes) and retries with backoff for up to 30 minutes. If it still cannot land, the command prints the exact `--upload-only` line to run later; the commitment is already on-chain and holds for 6 hours, and a late upload of the same digest is always accepted:

```bash
swarm model submit --artifact Submission/submission.zip --family-id cf_autopilot --upload-only \
     --wallet.name my_cold --wallet.hotkey my_hot
```

### Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--source` / `--artifact` | Source folder to package, or an archive to submit as is (one of the two) | required |
| `--family-id` | The family this model competes in | prompted in a terminal |
| `--wallet.name` / `--wallet.hotkey` | The coldkey and the mining hotkey | `default` |
| `--netuid` | Subnet netuid | `124` |
| `--subtensor.network` | Network (finney, test) | `finney` |
| `--backend-url` | Backend API URL | the public API |
| `--upload-only` | Re-run only the upload for a digest already committed | off |

### When something is refused

Every refusal comes with the reason and what to do about it. The ones you may meet:

| Message | Meaning |
|---------|---------|
| `requirements_rejected` | A `requirements.txt` line is not on the [whitelist](#docker-whitelist). Nothing was committed. |
| `artifact_too_large` | The archive is over 50 MiB compressed. Nothing was committed. |
| already registered to another miner | Someone committed the same archive first (earliest on-chain commit wins). Change the model and package again. |
| this hotkey already holds a submission | One submission per hotkey, ever. Register a new hotkey. |
| same model as an existing submission | The archive is the same code and weights as a model the backend already holds. Comments, formatting and zip packaging do not make it a new model; change the code or the weights. |
| submission window is closed | You are inside the 1.5-hour pre-epoch freeze. Wait for rollover and run the command again. |

> **One model per hotkey.** A hotkey enters exactly one family with exactly one model. To compete in more families, register more hotkeys, one per family.
>
> Treat every submission as final. Once your model is evaluated, the hotkey's slot is **locked**: uploading a new archive does not replace it and does not re-run the benchmark. To compete again, register a **new hotkey** and submit from it. See the [FAQ](#faq) for more.

Commitments are processed in block order: the earliest committer wins a digest. The chain rate-limits commits to roughly one per 20 minutes.

<p align="right">(<a href="#miner-top">back to top</a>)</p>

---

## Private Until Crowned

Your archive is never public while it competes. It travels from your machine to the Swarm backend, and from there only to the trusted validators that evaluate it, each of which verifies its SHA-256 against your on-chain commitment before running it in a sandbox and deletes it once the task is done.

- **Did not win**: the model stays in the backend's vault, forever. No repository, no download, no source. Your own copy is the only one outside the team.
- **Took the crown**: the model is published, so that everyone can fork and improve it. The download opens on its leaderboard page, and the code and weights are pushed to the [champions repository](https://github.com/swarm-subnet/swarm-champions) under `<family>/<crown date>_uid-<uid>_<hash>/`, with the archive attached to a GitHub Release. Dethroned champions stay published.

Forking and improving a published champion is allowed and encouraged. Re-submitting a champion with only cosmetic changes is refused at upload, and a real change still has to clear the [crowning floor](#becoming-champion).

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

A new champion is published within minutes of its crowning: see [Private Until Crowned](#private-until-crowned).

<p align="right">(<a href="#miner-top">back to top</a>)</p>

---

## Emissions: King of the Hill

Emissions are **not** winner-take-all, and there is no single global leaderboard payout. Each family runs its own King-of-the-Hill lineage: the family's emission slice is split across its **last five crowned kings**. A seat's share is set first by its rank in the window — the reigning champion holds the full weight and every step back keeps **70%** of the seat above it — and fine-tuned by how much that king improved on the score it beat. Being dethroned doesn't stop your earnings. Only five newer crownings pushing you out of the window does.

The practical consequences:

- A copycat that barely clears the floor earns almost nothing; a real jump earns a dominant share and keeps paying through the next several dethronings.
- Your seat's share is frozen at crowning; champion re-evaluations on fresh seeds don't change it.
- A seat pays as long as the backend still holds the exact archive it was crowned with; there is nothing for you to keep online.

The exact formula, window mechanics, and edge cases are in [king_of_the_hill.md](../../docs/king_of_the_hill.md).

<p align="right">(<a href="#miner-top">back to top</a>)</p>

---

## Benchmark System

### How Your Model Is Evaluated

1. **Miner** runs `swarm model submit`: the digest goes on-chain, the archive goes to the backend, then the miner goes offline
2. **Backend** detects the commit: the chain scanner polls every 3 minutes, so registration lands within minutes of finalization. Once the uploaded bytes match the committed digest and pass the intake checks, it creates one **Pending Benchmark** row
3. Each family is a **queue lane**: champion epoch re-evals run first, then any queued re-evals, then the oldest pending model; a rotation cursor cycles across families so no lane starves
4. **Validators** lease the model's seeds individually from a shared pool, fetch the archive from the backend, verify its hash, and run the agent in a sandboxed Docker container: the full **1,100 seeds** per family, spread over the family's environment types
5. When the whole seed range [0, 1100) is covered (by any mix of validators' completed seeds), the stitched mean becomes the model's score and the status flips to **Evaluated**; the champion check then runs

Every submission runs the full 1,100-seed benchmark directly. (A 300-seed screening pre-gate exists in the code behind a hardcoded `SCREENING_ENABLED = False`; it is off, and validators offering screening work are refused.)

A transient timeout or RPC-transport failure is retried once for that seed, subject to run-wide retry budgets. Deterministic environment failures and validator-infrastructure failures are excluded from the score; failures caused by the submitted agent still count. Smoke-test with `swarm model verify` before submitting.

### Epoch Rotation

Epochs run for **14 days** from epoch 19 onward, anchored Monday 16:00 UTC (epochs 1–18 were 7 days). Each validator independently generates its own 1,100 seeds per family per epoch using `random.SystemRandom()`: there is no shared secret. Validators publish each epoch's seed sets to the backend **after** the epoch ends, where they are publicly readable.

At rollover, pending models keep their queue position, discard partial results, and restart evaluation on the new epoch's seeds. Every champion is also queued for re-evaluation. For the final **1.5 hours** of an epoch the scanner stops registering new commitments; `swarm model submit` refuses to commit in that window and tells you when it reopens.

### Key Numbers

| Parameter | Value |
|-----------|-------|
| Seeds per family per epoch | 1,100 |
| Seed claim size | Dynamic: up to the validator's free worker slots (API cap 64) |
| Chain scanner interval | 3 minutes |
| Upload window after a commit | 6 hours (a late upload of the same digest is still accepted afterwards) |
| Epoch length | 14 days from epoch 19 (Monday 16:00 UTC anchor) |
| Pre-rollover registration freeze | 1.5 hours |
| Max artifact size | 50 MiB compressed and 50 MiB uncompressed content |
| Models per hotkey | 1 (one family per hotkey) |
| Chain commit cooldown | ~20 minutes |

<p align="right">(<a href="#miner-top">back to top</a>)</p>

---

## Docker Whitelist

Your `requirements.txt` can only include packages from the approved whitelist (`DOCKER_PIP_WHITELIST` in `swarm/constants.py`). `swarm model submit` checks every line before the commit, and validators enforce the same list when they install your dependencies.

**Approved packages:**

```
torch, torchvision, torchaudio, onnx, onnxruntime, onnxruntime-gpu,
stable-baselines3, sb3-contrib, gymnasium, gym, numpy, scipy,
scikit-learn, opencv-python, opencv-python-headless, pillow, imageio,
matplotlib, pyyaml, tqdm, einops, tensorboard, h5py, msgpack,
swarm-bullet3, swarm-drone-gym
```

Version pins are fine; installer option lines, URL/path installs (`git+`, `http://`, `https://`, `file:`, `./`, or an absolute path), and PEP 508 `@` direct references are rejected.

Need a package not on this list? Ask in [Discord](https://discord.gg/8dPqPDw7GC).

<p align="right">(<a href="#miner-top">back to top</a>)</p>

---

## FAQ

### When will my score show up on the leaderboard?

Registration takes minutes (the scanner polls every 3 minutes, and your upload lands right after). Evaluation time depends on how many models sit ahead of yours across the family lanes: champion epoch re-evals take priority, and the queue rotates one family at a time.

### Can I update my submission after committing?

**No, once your model has been evaluated.** A hotkey gets one submission. To try a better model you need a new hotkey.

The only exception is a submission refused before it was accepted: an archive that failed the local checks was never committed, so fix it and run `swarm model submit` again with the same hotkey. Once accepted, the hotkey stays used through epoch changes and version updates.

### What happens if my model fails evaluation?

The hotkey is used up. A model that was evaluated and failed keeps its slot locked. To try again, register a new hotkey. Your archive stays private in the backend's vault; nobody else can see it.

### Can I compete in more than one family?

Not on the same hotkey: every hotkey enters exactly one family. To compete in several families, register one hotkey per family and submit to each. Each entry is evaluated and championed independently.

### Can I see other miners' models?

Only champions. Every crowned model is published to the [champions repository](https://github.com/swarm-subnet/swarm-champions) and downloadable from its leaderboard page, so you can study and improve on it. A model that never won stays private.

### I submitted, but I don't see a score yet. What should I check?

In order of likelihood:

- **The upload never landed**: the command ends with the `--upload-only` line to run; run it. The commitment holds, and a late upload of the same digest is accepted.
- **Freeze window**: the command refuses to commit during the last 1.5 hours of an epoch and tells you when it reopens.
- **Wrong hotkey**: the upload must be signed by the hotkey that committed. Pass the same `--wallet.hotkey` to `--upload-only`.
- **Refused at upload**: the command prints the backend's reason (see [When something is refused](#when-something-is-refused)).

If none apply, contact the team on [Discord](https://discord.gg/8dPqPDw7GC).

### How often are weights updated on-chain?

Validators refresh the king windows from the backend, recompute the weights locally, and set them on-chain on a periodic cadence, so a new champion's effect on rewards shows up within the epoch, not instantly.

### What if two miners submit the same model?

Archives are globally unique: a digest already registered to any miner is refused, and so is an archive that is the same code and weights as an existing submission under a different digest. The earliest on-chain committer wins. Since nobody can see a model that has not won, there is nothing to copy before it is published.

<p align="right">(<a href="#miner-top">back to top</a>)</p>

---

## Troubleshooting

**"Missing drone_agent.py"**: Your ZIP must contain `drone_agent.py` at the root level. Template files are auto-injected.

**"Dangerous executable files detected"**: Remove `.exe`, `.so`, `.dll`, `.sh`, `.bat`, and `.pyc` files. Only Python code and model files are allowed.

**"Agent too large"**: Both the archive and its total uncompressed content must be ≤ 50 MiB. Shrink the weights rather than relying on compression.

**"requirements_rejected"**: A line of `requirements.txt` is not on the [whitelist](#docker-whitelist) or uses a URL, path or installer option. Fix the line and submit again; nothing was committed.

**"RPC connection failed"**: Ensure your agent starts correctly and responds to ping requests. Transient RPC-transport failures are retried once per seed, subject to the run-wide retry budget; persistent agent failures still count against the submission.

**"Upload gave up"**: Run the printed `--upload-only` command once the backend is reachable again. The on-chain commitment is already in place.

**Wrong family packaged**: repackage with `swarm model package` and pick the right family at the prompt (or pass the correct `--family-id`).

**Environment issues**: Run `swarm doctor` to diagnose.

<p align="right">(<a href="#miner-top">back to top</a>)</p>

---

## Support

- **Discord**: [discord.gg/8dPqPDw7GC](https://discord.gg/8dPqPDw7GC) (ping @Miguelikk or @AliSaaf)
- **GitHub Issues**: open a ticket with logs & error trace
- **Website**: [swarm124.com](https://swarm124.com)

<p align="right">(<a href="#miner-top">back to top</a>)</p>
