# =============================================================================
# SWARM SUBNET CONSTANTS
# =============================================================================
# Centralized constants for the Swarm Bittensor subnet. This file contains all
# configuration values, limits, and parameters used throughout the system.
# =============================================================================

import os
from datetime import datetime, timezone
from pathlib import Path

# =============================================================================
# EPOCH
# =============================================================================

EPOCH_FREEZE_SECONDS = 5400                # 1.5 hours before epoch end — no new evaluations

# =============================================================================
# NETWORK & COMMUNICATION
# =============================================================================

FORWARD_SLEEP_SEC = 2.0                 # Pause between validator forward passes (seconds)
FORWARD_IDLE_SEC = 300                  # Pause when no models to evaluate (seconds)
BACKEND_GRACE_PERIOD_SEC = 3600         # Use cached weights for 1h after last successful sync
WANDB_IDLE_RESTART_SEC = 5 * 3600      # Restart W&B run every 5h when idle

# =============================================================================
# SIMULATION & PHYSICS
# =============================================================================

# Core simulation parameters
SIM_DT = 1/50                           # Physics simulation timestep (50 Hz)
SOLVER_ITERATIONS = 4                   # PyBullet constraint solver iterations (default 50, reduced for speed)
SOLVER_MIN_ISLAND_SIZE = 128            # Minimum solver island size (reduces per-island overhead)
HORIZON_SEC = 60                       # Maximum simulated flight duration (seconds)
# World generation parameters
HEIGHT_SCALE = 1.5                      # Obstacle height scale factor
RANDOM_START = True                     # Toggle random starting point generation
# Camera and rendering settings
CAMERA_FOV_BASE = 90.0                  # Base field of view (degrees)
CAMERA_FOV_VARIANCE = 2.0               # FOV randomization range (±degrees)
# Depth sensor parameters
DEPTH_NEAR = 0.05                       # PyBullet camera near plane (meters)
DEPTH_FAR = 30.0                        # PyBullet camera far plane (meters)
DEPTH_MIN_M = 0.5                       # Minimum useful depth range (meters)
DEPTH_MAX_M = 20.0                      # Maximum useful depth range (meters)
# Search area parameters
SEARCH_AREA_NOISE_Z = 5.0               # ±5m vertical noise — forces real altitude search
SEARCH_RADIUS_MIN = 5.0                 # Minimum per-seed search radius (meters)
SEARCH_RADIUS_MAX = 20.0                # Maximum per-seed search radius (meters), clamped per-seed to what fits the horizon
# Search-aware time scoring — budget the time to sweep the search disk so a good
# searcher can still reach a perfect time term. See swarm/validator/reward.py.
SEARCH_SWEEP_ALPHA = 0.75               # Coverage-overhead factor (~70-80th percentile area search)
SEARCH_DETECT_WIDTH = 5.1               # Effective downward detection swath (meters): cruise footprint 2*SAFE_Z*tan(FOV/2)=6.0 x 0.85 overlap loss
SEARCH_LAND_SEC = 2.0                   # Time budgeted to settle/land once the pad is found (seconds)
SEARCH_TIME_BUFFER = 1.06               # Slack multiplier on the search-aware target time
SEARCH_FEASIBILITY_MARGIN_SEC = 1.0     # Keep target time this far under the horizon when clamping radius
# Light randomization parameters
LIGHT_RANDOMIZATION_ENABLED = True      # Enable random light direction (time of day)
# Propulsion efficiency
PROP_EFF = 0.60                         # Propeller efficiency coefficient

# =============================================================================
# MODEL & AI EVALUATION
# =============================================================================

# Model size and validation limits — sourced from submission_policy so the
# backend and validator agree on the same ceiling.
from swarm.core.submission_policy import MAX_UNCOMPRESSED_BYTES as _POLICY_MAX_BYTES

MAX_MODEL_BYTES = _POLICY_MAX_BYTES
EVAL_TIMEOUT_SEC = 120.0                # Model evaluation subprocess timeout (seconds)

# Docker worker auto-sizing
DOCKER_WORKER_MEMORY = "6g"             # Memory limit per Docker worker container
DOCKER_WORKER_CPUS = "2"                # CPU limit per Docker worker container


def available_vcpu_count() -> int:
    try:
        if hasattr(os, "sched_getaffinity"):
            count = len(os.sched_getaffinity(0))
            if count > 0:
                return int(count)
    except Exception:
        pass
    try:
        count = os.cpu_count()
        if count and int(count) > 0:
            return int(count)
    except Exception:
        pass
    return 1


def cpus_per_docker_worker() -> int:
    """Integer CPUs each docker worker is sized for, derived from DOCKER_WORKER_CPUS."""
    try:
        return max(1, int(float(DOCKER_WORKER_CPUS)))
    except (TypeError, ValueError):
        return 1


def default_docker_worker_count(*, maximum: int = 12) -> int:
    """Number of docker workers that fit on this host without CPU oversubscription."""
    return max(1, min(int(maximum), available_vcpu_count() // cpus_per_docker_worker()))


# Docker parallel workers for validator and benchmark evaluation.
# One worker per `DOCKER_WORKER_CPUS` vCPUs so each worker can be pinned to a
# dedicated CPU group; capped at 12 workers.
N_DOCKER_WORKERS = default_docker_worker_count(maximum=12)

# Docker pip package whitelist (approved packages for miner requirements.txt)
DOCKER_PIP_WHITELIST = {
    "torch", "torchvision", "torchaudio",
    "onnx", "onnxruntime", "onnxruntime-gpu",
    "stable-baselines3", "sb3-contrib",
    "gymnasium", "gym",
    "swarm-bullet3", "swarm-drone-gym",
    "numpy", "scipy", "scikit-learn",
    "opencv-python", "opencv-python-headless",
    "pillow", "imageio",
    "matplotlib",
    "pyyaml",
    "tqdm",
    "einops",
    "tensorboard",
    "h5py",
    "msgpack",
}

# Per-step RPC timing (miner inference fairness)
RPC_STEP_TIMEOUT_SEC = 0.500            # Per agent.act() call fallback (seconds)
RPC_FIRST_STEP_TIMEOUT_SEC = 2.0        # First step grace for model warmup/JIT (seconds)
RPC_RESET_TIMEOUT_SEC = 5.0             # Max wall-clock for agent.reset() between seeds (seconds)
RPC_PING_TIMEOUT_SEC = 2.0              # Max wall-clock for agent.ping() health check (seconds)
RPC_MAX_STRIKES_PER_SEED = 15           # Timeouts before failing a seed
GLOBAL_EVAL_BASE_SEC = 600.0            # Base overhead for global worker timeout (seconds); one-seed validator batches get ~600s wall-clock
GLOBAL_EVAL_PER_SEED_SEC = 15.0         # Per-seed budget in global worker timeout (seconds)
GLOBAL_EVAL_CAP_SEC = 600.0             # Hard upper bound for global worker timeout (seconds)

# Hardware-fair calibrated timing
MINER_COMPUTE_BUDGET_SEC = 0.500        # Guaranteed pure-compute budget per step (seconds)
# Reference-time scoring: the validator measures itself on a committed baseline model
# and judges each act() in baseline-equivalent time (speed_factor = local_p90 / owner_p90).
SPEED_FACTOR_MIN = 0.25                  # Lower guard against an invalid calibration measurement
SPEED_FACTOR_MAX_ELIGIBLE = 3.0          # Beyond this the host is too slow to score fairly; it self-excludes
RPC_CONNECT_MAX_WAIT_SEC = 60.0          # Total budget to reach a serving RPC agent (covers slow cold model loads)
HARD_CAP_REF_SEC = 1.25                  # Per-act liveness ceiling in baseline-equivalent seconds
HARD_CAP_MARGIN_SEC = 0.050              # Transport-jitter margin added to the per-act hard cap (seconds)
HARD_CAP_STRIKES_PER_SEED = 3            # Hard-cap timeouts allowed before failing the seed
FIRST_STEP_BUDGET_REF_SEC = 2.0          # Baseline-equivalent compute budget for the first act (warmup/JIT)
FIRST_STEP_HARD_CAP_REF_SEC = 3.0        # Per-act hard cap for the first act in baseline-equivalent seconds
# Reference-time scoring is OFF by default: the light baseline model mis-predicts
# heavy models' speed, over-correcting their act() time and falsely striking them.
# Legacy per-step timing is used until a representative baseline is calibrated.
# Set SWARM_USE_REFERENCE_TIMING=1 to opt back in.
USE_REFERENCE_TIMING = os.getenv("SWARM_USE_REFERENCE_TIMING", "0").strip().lower() not in ("0", "false", "no")
CALIBRATION_ROUNDS = 10                 # Number of round-trips to measure RPC overhead
CALIBRATION_OVERHEAD_CAP_SEC = 0.100    # Max acceptable pipeline overhead (seconds)
CALIBRATION_TIMEOUT_SEC = 5.0           # Per-round calibration timeout (seconds)
CALIBRATION_BENCHMARK_REF_NS = 15_000_000 # Reference CPU benchmark time (ns) — 3×(512×512) matmul, single-thread
CALIBRATION_CPU_FACTOR_CAP = 2.0        # Max CPU scaling factor (prevents abuse on very slow HW)
CALIBRATION_MARGIN_SEC = 0.015          # Safety margin for response deserialization jitter (seconds)
CALIBRATION_RECAL_INTERVAL = 100        # Re-calibrate every N seeds to catch thermal throttling
CALIBRATION_WARN_OVERHEAD_MS = 30.0     # Log warning when calibrated overhead exceeds this (ms)
CALIBRATION_WARN_CPU_FACTOR = 1.5       # Log warning when CPU factor exceeds this
EVAL_SUMMARY_INTERVAL_SEC = 60          # Periodic evaluation progress summary interval (seconds)

# Model storage and processing
MODEL_DIR = Path("miner_models_v2")     # Directory for storing miner model files
BLACKLIST_FILE = MODEL_DIR / "fake_models_blacklist.txt"  # Blacklisted model hashes file
CHUNK_SIZE = 2 * 1024 * 1024            # File transfer chunk size (2 MiB)
SUBPROC_MEM_MB = 8192                   # Memory limit per evaluation subprocess (MB)

# =============================================================================
# DRONE & FLIGHT CONTROL
# =============================================================================

# Drone physical specifications
DRONE_MASS = 0.027                          # Drone mass (kg) - CF2X Crazyflie
DRONE_HULL_RADIUS = 0.12                    # Drone hull radius from center to edge (meters)
ALTITUDE_RAY_INSET = 0.09                   # Inset from hull edge for altitude ray origin (meters)
MAX_RAY_DISTANCE = 20.0                     # Downward LiDAR maximum detection range (meters)

# Landing and positioning parameters
LANDING_PLATFORM_RADIUS = 0.6          # Landing platform acceptance radius (meters)
PLATFORM = True                         # Enable landing platform rendering
START_PLATFORM = True                  # Enable solid start platform spawn
START_PLATFORM_RADIUS = 0.6
START_PLATFORM_HEIGHT = 0.2            # Physical height of the start platform (meters)
START_PLATFORM_SURFACE_Z = 0.2         # Absolute Z height of the platform surface (meters)
START_PLATFORM_TAKEOFF_BUFFER = 0.121   # Initial clearance above platform surface (meters)
START_PLATFORM_RANDOMIZE = True        # Enable random platform heights when random start is used
START_PLATFORM_MIN_Z = 0.2             # Minimum platform surface height when randomizing (meters)
START_PLATFORM_MAX_Z = 10             # Maximum platform surface height when randomizing (meters)
HOVER_SEC = 0                           # Required hover duration for mission success (seconds)
SAFE_Z = 3                              # Default cruise altitude (meters)
GOAL_TOL = LANDING_PLATFORM_RADIUS * 0.8 * 1.06  # TAO badge radius for precision landing (0.5088m)
SPEED_LIMIT = 3.0                       # Maximum drone velocity limit (m/s)
MAX_YAW_RATE = 3.141                    # Maximum yaw rotation rate (rad/s) - 180 degrees per second

# Landing detection parameters
LANDING_MAX_VZ = 0.5                    # Maximum vertical velocity for valid landing (m/s)
LANDING_MAX_VXY_REL = 0.6               # Maximum horizontal velocity relative to platform (m/s)
LANDING_MAX_TILT_RAD = 0.26             # Maximum roll/pitch for valid landing (~15 degrees)
LANDING_STABLE_SEC = 0.5                # Required stable contact duration for success (seconds)
# Goal generation ranges (legacy defaults)
SAFE_ZONE_RADIUS = 2.0                  # Minimum clearance around obstacles (meters)
MAX_ATTEMPTS_PER_OBS = 100              # Maximum retry attempts when placing obstacles
# Goal platform colors
GOAL_COLOR_PALETTE = [
    [0.0, 0.8, 0.0, 1.0],               # Green (original)
    [0.0, 0.0, 0.9, 1.0],               # Blue
    [0.9, 0.0, 0.0, 1.0],               # Red
    [0.9, 0.9, 0.0, 1.0],               # Yellow
    [0.6, 0.0, 0.8, 1.0],               # Purple
    [0.0, 0.8, 0.8, 1.0],               # Cyan
    [0.9, 0.5, 0.0, 1.0],               # Orange
]
# City variant distribution
CITY_VARIANT_DISTRIBUTION = {
    1: 0.10,  # Residential
    2: 0.25,  # Mixed
    3: 0.35,  # Urban
    4: 0.30,  # Hard Mode (city_type=3, difficulty=3)
}

assert abs(sum(CITY_VARIANT_DISTRIBUTION.values()) - 1.0) < 0.001, "City variant probabilities must sum to 1.0"

# =============================================================================
# SCORING & REWARDS
# =============================================================================

# Miner sampling and evaluation
SAMPLE_K = 256                          # Number of miners sampled per forward pass
# Emission burning mechanism
BURN_EMISSIONS = False                   # Burn a fraction to UID 0; the rest is split among the kings
BURN_FRACTION = 0.0                    # Fraction of emissions to burn
KEEP_FRACTION = 1.0 - BURN_FRACTION     # Fraction of emissions to distribute
UID_ZERO = 0                            # Special UID for burning emissions

# Reward distribution mechanism
WINNER_TAKE_ALL = True                  # Enable winner-take-all rewards (winner gets all available emissions)

# Safety metric parameters
REWARD_W_SUCCESS = 0.45                 # Weight for success term in reward calculation
REWARD_W_TIME = 0.45                    # Weight for time efficiency term in reward calculation
REWARD_W_SAFETY = 0.10                  # Weight for safety term in reward calculation
SAFETY_DISTANCE_SAFE = 1.0              # Full safety score at this clearance (meters)
SAFETY_DISTANCE_DANGER = 0.2            # Zero safety score at this clearance (meters)

# Landing-zone floor suppression: ignore the supporting floor right under a
# legitimately low landing platform so the final descent is not penalized for
# unavoidable proximity to the ground.
LANDING_FLOOR_MAX_HEIGHT = 0.15         # Max AABB z-extent treated as floor (meters)
LANDING_COLUMN_PADDING = 0.10           # XY padding around landing radius (meters)
LANDING_ALTITUDE_BUFFER = 0.10          # Vertical slack above safe distance (meters)

# =============================================================================
# BENCHMARK SYSTEM
# =============================================================================

from swarm import version_split as _vs
BENCHMARK_VERSION = ".".join(_vs[:3])
BENCHMARK_TOTAL_SEED_COUNT = 1100       # Total seeds per epoch
BENCHMARK_SCREENING_SEED_COUNT = 300    # Seeds used for screening phase
BENCHMARK_FULL_SEED_COUNT = 800         # Seeds used for full benchmark phase
SCREENING_BOOTSTRAP_THRESHOLD = 0.01    # Minimum score threshold during bootstrap
SEED_SCORE_BATCH_MAX = 300              # Backend max per POST /validators/seed-scores

# Epoch system — seeds rotate every 7 days (Monday 16:00 UTC)
EPOCH_DURATION_SECONDS = 7 * 86400
EPOCH_ANCHOR_UTC = datetime(2026, 3, 30, 16, 0, 0, tzinfo=timezone.utc)
SCREENING_MIN_IMPROVEMENT = 0.015       # Must score above top model + this margin to pass

# Early screening termination — abort screening when outcome is statistically certain
SCREENING_CHECKPOINT_SIZE = 50                              # Seeds evaluated per checkpoint
SCREENING_EARLY_FAIL_FACTORS = {50: 0.50, 100: 0.70, 150: 0.85}

# Unified streaming chunk size used by screening, benchmark, and reeval phases.
# Smaller chunks give fresher UI updates at the cost of more seed-score uploads.
UNIFIED_CHUNK_SIZE = 10
MAX_INFLIGHT_SEED_UPLOADS = 3

# Stratified templates: every model is graded on the same challenge composition, not random seeds.
def _banded_pool(
    challenge_type: int,
    distance: tuple[float, float],
    *,
    n_slots: int,
    n_bands: int,
    moving_prob: float,
    goal_height_range=None,
) -> list[dict]:
    lo, hi = distance
    width = (hi - lo) / n_bands
    n_moving = round(n_slots * moving_prob)
    pool: list[dict] = []
    for i in range(n_slots):
        band = i % n_bands
        pool.append(dict(
            challenge_type=challenge_type,
            distance_range=(round(lo + band * width, 1), round(lo + (band + 1) * width, 1)),
            goal_height_range=goal_height_range,
            moving_platform=(i < n_moving),
        ))
    return pool


def _interleave(pools: list[list[dict]], expected: int) -> list[dict]:
    slots: list[dict] = []
    for i in range(max(len(p) for p in pools)):
        for pool in pools:
            if i < len(pool):
                slots.append(pool[i])
    if len(slots) != expected:
        raise RuntimeError(f"Template must have {expected} entries, got {len(slots)}")
    return slots


# Screening template — 50 entries, cycled 6× for 300 screening seeds (two distance bands per type).
def _build_screening_template() -> list[dict]:
    return _interleave([
        _banded_pool(1, (22, 40), n_slots=8, n_bands=2, moving_prob=0.25, goal_height_range=(0.3, 0.8)),
        _banded_pool(2, (28, 65), n_slots=8, n_bands=2, moving_prob=0.75, goal_height_range=(5.0, 12.0)),
        _banded_pool(3, (65, 95), n_slots=8, n_bands=2, moving_prob=0.25),
        _banded_pool(4, (28, 50), n_slots=9, n_bands=2, moving_prob=0.25),
        _banded_pool(5, (18, 30), n_slots=9, n_bands=2, moving_prob=0.0, goal_height_range=(1.0, 6.0)),
        _banded_pool(6, (22, 40), n_slots=8, n_bands=2, moving_prob=0.0, goal_height_range=(0.5, 2.0)),
    ], expected=50)


# Benchmark template — 100 entries, cycled 8× for 800 benchmark seeds (three distance bands per type).
def _build_benchmark_template() -> list[dict]:
    return _interleave([
        _banded_pool(1, (22, 45),  n_slots=17, n_bands=3, moving_prob=0.25, goal_height_range=(0.2, 1.0)),
        _banded_pool(2, (28, 72),  n_slots=17, n_bands=3, moving_prob=0.80, goal_height_range=(4.0, 14.0)),
        _banded_pool(3, (65, 100), n_slots=17, n_bands=3, moving_prob=0.25),
        _banded_pool(4, (28, 56),  n_slots=17, n_bands=3, moving_prob=0.25),
        _banded_pool(5, (18, 35),  n_slots=16, n_bands=3, moving_prob=0.0, goal_height_range=(0.2, 10.0)),
        _banded_pool(6, (22, 45),  n_slots=16, n_bands=3, moving_prob=0.0, goal_height_range=(0.2, 3.0)),
    ], expected=100)


SCREENING_TEMPLATE: list[dict] = _build_screening_template()
BENCHMARK_TEMPLATE: list[dict] = _build_benchmark_template()

# =============================================================================
# CHALLENGE TYPE DISTRIBUTION
# =============================================================================

CHALLENGE_TYPE_DISTRIBUTION = {
    1: 1 / 6,  # City navigation (procedural roads)
    2: 1 / 6,  # Open flight (no obstacles)
    3: 1 / 6,  # Mountain navigation
    4: 1 / 6,  # Village navigation
    5: 1 / 6,  # Warehouse navigation
    6: 1 / 6,  # Forest navigation
}

assert abs(sum(CHALLENGE_TYPE_DISTRIBUTION.values()) - 1.0) < 0.001, "Challenge probabilities must sum to 1.0"

# =============================================================================
# CHALLENGE TYPE PARAMETERS
# =============================================================================

# Type 1: City Navigation
TYPE_1_WORLD_RANGE = 75
TYPE_1_SAFE_ZONE = 1.0
TYPE_1_R_MIN, TYPE_1_R_MAX = 22, 45
TYPE_1_H_MIN, TYPE_1_H_MAX = 0.2, 1
TYPE_1_START_H_MIN, TYPE_1_START_H_MAX = 0.2, 5
TYPE_1_HORIZON = HORIZON_SEC

# Type 2: Open Flight (No Obstacles)
TYPE_2_WORLD_RANGE = 60
TYPE_2_N_OBSTACLES = 0
TYPE_2_HEIGHT_SCALE = 1.0
TYPE_2_SAFE_ZONE = 0.0
TYPE_2_R_MIN, TYPE_2_R_MAX = 28, 72
TYPE_2_H_MIN, TYPE_2_H_MAX = 4, 14
TYPE_2_START_H_MIN, TYPE_2_START_H_MAX = 0.05, 10
TYPE_2_HORIZON = HORIZON_SEC

# Type 3: Mountain Navigation
TYPE_3_SAFE_ZONE = 1.0
TYPE_3_R_MIN, TYPE_3_R_MAX = 65, 100
TYPE_3_H_MIN, TYPE_3_H_MAX = 0, 0
TYPE_3_START_H_MIN, TYPE_3_START_H_MAX = 0, 0
TYPE_3_HORIZON = HORIZON_SEC
TYPE_3_SCALE_MIN = 0.6
TYPE_3_SCALE_MAX = 0.8
TYPE_3_SCALE_SEED_OFFSET = 777777
TYPE_3_WORLD_RANGE_RATIO = 0.60
TYPE_3_VILLAGE_RANGE = 40.0
# Village (challenge_type 4) keeps its own far-goal band — its ±40m world box
# caps reachable distance near 56m, so it must NOT inherit the mountain 50-100 band.
VILLAGE_R_MIN, VILLAGE_R_MAX = 28, 56

# Legacy split kept for compatibility utilities. Internal task schema now uses:
# type=3 mountain, type=4 village.
MOUNTAIN_SUBTYPE_DISTRIBUTION = {
    1: 0.75,  # Mountains Only
    2: 0.25,  # Ski Village
}

# Type 5: Warehouse Navigation (rectangular: 80.2m × 50.6m floor, 12m ceiling)
# Constants retain the TYPE_4_* prefix for backward import compatibility.
TYPE_4_WORLD_RANGE_X = 38                           # ±38m X (floor_spawn_half_x=40.1m, 2m wall margin)
TYPE_4_WORLD_RANGE_Y = 23                           # ±23m Y (floor_spawn_half_y=25.3m, 2m wall margin)
TYPE_4_R_MIN, TYPE_4_R_MAX = 18, 35
TYPE_4_H_MIN, TYPE_4_H_MAX = 0.2, 10.0             # Floor to roof(12m) minus 2m ceiling clearance
TYPE_4_START_H_MIN, TYPE_4_START_H_MAX = 0.2, 10.0
TYPE_4_HORIZON = HORIZON_SEC
TYPE_4_PLATFORM_CLEARANCE = 1.0                     # Minimum clearance from warehouse structures (meters)
TYPE_4_PLATFORM_MAX_ATTEMPTS = 200                  # Max attempts to find collision-free platform position
TYPE_4_MIN_PLATFORM_DISTANCE = 10.0                 # Minimum 3D distance between start and goal platforms (meters)

# Type 6: Forest Navigation (100×100m ground, 96×96m playable with 2m edge margin)
TYPE_6_WORLD_RANGE = 42                             # ±42m playable XY (96m total with margin)
TYPE_6_R_MIN, TYPE_6_R_MAX = 22, 45
TYPE_6_H_MIN, TYPE_6_H_MAX = 0.2, 3.0
TYPE_6_START_H_MIN, TYPE_6_START_H_MAX = 0.2, 3.0
TYPE_6_HORIZON = HORIZON_SEC
TYPE_6_SAFETY_DISTANCE_SAFE = 0.6                   # Tighter safety for dense forest (meters)

# Per-challenge override for SAFETY_DISTANCE_SAFE; types not present fall back
# to the global value.
SAFETY_DISTANCE_SAFE_BY_TYPE = {
    6: TYPE_6_SAFETY_DISTANCE_SAFE,
}

FOREST_MODE_DISTRIBUTION = {
    1: 0.25,   # Normal (green foliage)
    2: 0.25,   # Autumn (orange/yellow)
    3: 0.25,   # Snow (white, bare + snow-covered)
    4: 0.25,   # Dead (no leaves, bare branches)
}
FOREST_DIFFICULTY_DISTRIBUTION = {
    1: 0.45,   # Easy  (130 trees, loose spacing)
    2: 0.35,   # Normal (170 trees, medium spacing)
    3: 0.20,   # Hard  (210 trees, tight spacing)
}
assert abs(sum(FOREST_MODE_DISTRIBUTION.values()) - 1.0) < 0.001
assert abs(sum(FOREST_DIFFICULTY_DISTRIBUTION.values()) - 1.0) < 0.001

# =============================================================================
# MOVING PLATFORM (challenge variant, applies to any map type)
# =============================================================================

MOVING_PLATFORM_PROB = {
    1: 0.25,
    2: 0.80,
    3: 0.25,
    4: 0.25,
    5: 0.00,
    6: 0.00,
}
MOVING_PLATFORM_SEED_OFFSET = 555555

PLATFORM_MOVEMENT_PATTERNS = ["circular", "linear", "figure8"]
PLATFORM_SPEED_MIN, PLATFORM_SPEED_MAX = 0.6, 1.2
PLATFORM_RADIUS_MIN, PLATFORM_RADIUS_MAX = 2.0, 4.0
PLATFORM_DELAY_MIN, PLATFORM_DELAY_MAX = 0.0, 2.0
PLATFORM_TRANSITION_MIN, PLATFORM_TRANSITION_MAX = 2.5, 3.5
PLATFORM_LINEAR_DIRECTIONS = ["x", "y", "xy"]

PLATFORM_AVOIDANCE_ENABLED = True
PLATFORM_STEER_ANGLES = [20, -20, 40, -40, 60, -60, 80, -80, 120, -120, 160, -160, 180]
PLATFORM_MIN_STEP_M = 0.05

# =============================================================================
# DISTANCE-BASED CULLING
# =============================================================================

CULL_VISUAL_RADIUS = 35.0               # Hide visuals beyond this distance (meters)
CULL_PHYSICS_RADIUS = 50.0              # Disable collision beyond this distance (meters)
CULL_INTERVAL_STEPS = 5                 # Re-evaluate cull state every N steps
CULL_MIN_AABB_SPAN = 5.0                # Minimum AABB XY span to be a cull target (meters)
CULL_MIN_FACES = 100                    # Minimum mesh face count to be a cull target
CULL_MIN_TOTAL_FACES = 100_000          # Auto-enable threshold (total faces across targets)
