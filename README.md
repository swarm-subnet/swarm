# Conceptual Summary – “Drone-Nav” Bittensor Subnet
*(what we learned, why we chose it, and how every piece fits together)*

## 1 · Problem framing
We want a Bittensor subnet whose miners receive a “fly from A → B” task, return a deterministic instruction list (thrust/RPM profile or way-point stream), and are judged automatically by a validator that re-simulates the flight.

### Key constraints

| Need                                   | Implications                                                               |
|----------------------------------------|-----------------------------------------------------------------------------|
| Deterministic physics & sensor read-back | Validator must bit-reproduce miner trajectory on different hardware.       |
| Zero-setup for miners                  | ❌ large game engines; ✅ pure-Python / pip-install.                       |
| Procedural map generation              | Task JSON must carry a single random seed that recreates the world inside both miner and validator. |
| Fast, headless, CPU-only option        | Enables hundreds of parallel miner instances.                               |

## 2 · Simulator choice → gym-pybullet-drones (PyBullet)

### Criterion & Why PyBullet wins

| Criterion                           | Why PyBullet wins                                                                                         |
|-------------------------------------|-----------------------------------------------------------------------------------------------------------|
| Deterministic (same CPU/FPU order)  | `resetSimulation` → `loadURDF(...)` in identical order + fixed-step loop ⇒ bit-level repeatability.       |
| Lightweight                         | Python package ≈ 40 MB; no Unreal/Unity, no GPU compile.                                                  |
| Procedural meshes on the fly        | Add boxes, gates, etc. through `p.createCollisionShape` at runtime.                                       |
| Headless or GUI                     | Miners → `gui=False`; validator can switch GUI on for debugging.                                          |
| Standard Gym API                    | Integrates with RL libs, but we need it mainly for deterministic physics + sensor suite (RGB, depth, IMU, GPS stub). |

## 3 · Subnet architecture sketch

```text
          ┌──────────────┐        publishes TASK JSON        ┌──────────────┐
          │  Validator   │───────────────────────────────────▶│    Miner     │
          └──────────────┘                                   └──────────────┘
                 ▲                                                   │
                 │ 4. re-plays instructions & scores                 │ 3. plan path
                 │                                                   ▼
          1. seed, A, B, config                         Gym-PyBullet-Drones sim
                 │                                                   │
                 └───────────────────────────────────────────────────┘
                            both sides call build_world(seed)
Deterministic round-trip
```

```python
# in both roles
np.random.seed(task["map_seed"])
build_world()                 # same order ⇒ same bodyIds
```

### Instruction format (example)

```json
{
  "commands":[
     {"t":0.00,"rpm":[2100,2100,2100,2100]},
     {"t":0.01,"rpm":[2112,2095, ... ]},
     ...
  ],
  "sha256":"0xDEADBEEF…"  # integrity check
}
```

### Validator rule of thumb

```python
success = (dist(goal, pos[-1]) < ε) and no_collision and T < horizon
score   = w_time*dt + w_energy*Σrpm² + ...
```

## 4 · Practical playbook

| Action                         | One-liner                                                                                       |
|--------------------------------|-------------------------------------------------------------------------------------------------|
| Install                        | `pip install -e gym-pybullet-drones` (inside conda env, Python 3.10).                            |
| Hover demo                     | `python -m hover_demo` opens GUI, random RPMs for 4 s.                                           |
| Training demo (hover PPO)      | `python examples/learn.py --timesteps 20000 --gui False`                                        |
| Tune episode length            | In `HoverAviary._computeTruncated`: change positional box (±1.5 m), tilt limit (0.4 rad ≈ 23°), or `EPISODE_LEN_SEC`. |
| Generate your own env          | Subclass `BaseRLAviary`, override `_addObstacles()` and reward.                                 |

## 5 · Code base orientation

```text
envs/
 ├─ BaseAviary.py      # low-level: PyBullet hook, physics variants, sensors
 ├─ BaseRLAviary.py    # adds Gym spaces, PID helper, action buffer
 ├─ HoverAviary.py     # simple hover task (reward & termination)
 ├─ MultiHoverAviary.py … CtrlAviary.py … BetaAviary.py … CFAviary.py
examples/
 ├─ pid.py             # PID hover test
 ├─ learn.py           # PPO template
```

### Termination logic we saw:

```python
def _computeTruncated(self):
    too_far = abs(x) > 1.5 or abs(y) > 1.5 or z > 2.0
    too_tilt= abs(roll) > .4 or abs(pitch) > .4
    timeout = sim_time > 8.0
    return too_far or too_tilt or timeout
```

They keep training bites short (≈240 control steps), prevent physics blow-ups, and keep the drone visible in default camera.

## 6 · Determinism checklist (miner & validator)

- Pin package versions (`bullet3==3.27`, `gym-pybullet-drones` commit hash).
- `p.setPhysicsEngineParameter(deterministicOverlappingPairs=1)`
- Fixed-step loop: `for _ in range(PYB_STEPS_PER_CTRL): stepSimulation()`
- No wall-clock sleeps in replay.
- Include seed + drone model + physics flags in task JSON.

## 7 · What’s left to build

- Task seeder smart enough to vary map complexity yet keep it solvable.
- Reference scorer (time, energy, smoothness) to mint TAO rewards.
- Miner SDK that wraps `Env` → returns JSON instructions, hides PyBullet details.
- Rich telemetry (optional): attach RGB/depth for perception-driven sub-tasks later.

**tl;dr**
Use gym-pybullet-drones as the simulation kernel; drive it with a seed-based map builder; let miners output deterministic control traces; validators replay & score. Lightweight, reproducible, and perfectly fits Bittensor’s miner/validator pattern.
