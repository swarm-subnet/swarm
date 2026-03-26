# =============================================================================
# SWARM SUBNET CONSTANTS
# =============================================================================
# Centralized constants for the Swarm Bittensor subnet. This file contains all
# configuration values, limits, and parameters used throughout the system.
# =============================================================================

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
FORWARD_IDLE_SEC = 30                   # Pause when no models to evaluate (seconds)

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
SEARCH_AREA_NOISE_Z = 2.0               # ±2m vertical noise
SEARCH_RADIUS_MIN = 0.0                 # Minimum per-seed search radius (meters)
SEARCH_RADIUS_MAX = 10.0                # Maximum per-seed search radius (meters)
# Light randomization parameters
LIGHT_RANDOMIZATION_ENABLED = True      # Enable random light direction (time of day)
# Propulsion efficiency
PROP_EFF = 0.60                         # Propeller efficiency coefficient

# =============================================================================
# MODEL & AI EVALUATION
# =============================================================================

# Model size and validation limits
MAX_MODEL_BYTES = 50 * 1024 * 1024      # Maximum compressed model size (50 MiB)
EVAL_TIMEOUT_SEC = 120.0                # Model evaluation subprocess timeout (seconds)

# Docker parallel workers for validator and benchmark evaluation
N_DOCKER_WORKERS = 8                    # Bumped from 3→8 for faster parallel seed evaluation
DOCKER_WORKER_MEMORY = "6g"             # Memory limit per Docker worker container
DOCKER_WORKER_CPUS = "2"                # CPU limit per Docker worker container

# Docker pip package whitelist (approved packages for miner requirements.txt)
DOCKER_PIP_WHITELIST = {
    "torch", "torchvision", "torchaudio",
    "onnx", "onnxruntime", "onnxruntime-gpu",
    "stable-baselines3", "sb3-contrib",
    "gymnasium", "gym",
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
GLOBAL_EVAL_BASE_SEC = 105.0            # Base overhead for global worker timeout (seconds); with one-seed validator batches and 15s/seed this yields a 120s wall-clock cap
GLOBAL_EVAL_PER_SEED_SEC = 15.0         # Per-seed budget in global worker timeout (seconds)
GLOBAL_EVAL_CAP_SEC = 590.0             # Hard upper bound for global worker timeout (seconds)

# Hardware-fair calibrated timing
MINER_COMPUTE_BUDGET_SEC = 0.500        # Guaranteed pure-compute budget per step (seconds)
CALIBRATION_ROUNDS = 10                 # Number of round-trips to measure RPC overhead
CALIBRATION_OVERHEAD_CAP_SEC = 0.100    # Max acceptable pipeline overhead (seconds)
CALIBRATION_TIMEOUT_SEC = 5.0           # Per-round calibration timeout (seconds)
CALIBRATION_BENCHMARK_REF_NS = 15_000_000 # Reference CPU benchmark time (ns) — 3×(512×512) matmul, single-thread
CALIBRATION_CPU_FACTOR_CAP = 2.0        # Max CPU scaling factor (prevents abuse on very slow HW)
CALIBRATION_MARGIN_SEC = 0.015          # Safety margin for response deserialization jitter (seconds)
CALIBRATION_RECAL_INTERVAL = 100        # Re-calibrate every N seeds to catch thermal throttling

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
BURN_EMISSIONS = True                   # Enable emission burning to UID 0
BURN_FRACTION = 0.95                    # Fraction of emissions to burn
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

# =============================================================================
# BENCHMARK SYSTEM
# =============================================================================

BENCHMARK_VERSION = "SB1.1"             # Current benchmark version identifier
BENCHMARK_TOTAL_SEED_COUNT = 1000       # Total seeds per epoch
BENCHMARK_SCREENING_SEED_COUNT = 200    # Seeds used for screening phase
BENCHMARK_FULL_SEED_COUNT = 800         # Seeds used for full benchmark phase
SCREENING_BOOTSTRAP_THRESHOLD = 0.1     # Minimum score threshold during bootstrap

# Epoch system — seeds rotate every 7 days (Monday 16:00 UTC)
EPOCH_DURATION_SECONDS = 7 * 86400
EPOCH_ANCHOR_UTC = datetime(2026, 3, 23, 16, 0, 0, tzinfo=timezone.utc)
SCREENING_TOP_MODEL_FACTOR = 1.01       # Must score above 101% of top model to pass

# Early screening termination — abort screening when outcome is statistically certain
SCREENING_CHECKPOINT_SIZE = 50                              # Seeds evaluated per checkpoint
SCREENING_EARLY_FAIL_FACTORS = {50: 0.60, 100: 0.80, 150: 0.90}
SCREENING_EARLY_PASS_FACTORS = {50: 1.30, 100: 1.15}

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
TYPE_1_R_MIN, TYPE_1_R_MAX = 5, 45
TYPE_1_H_MIN, TYPE_1_H_MAX = 0.2, 1
TYPE_1_START_H_MIN, TYPE_1_START_H_MAX = 0.2, 5
TYPE_1_HORIZON = HORIZON_SEC

# Type 2: Open Flight (No Obstacles)
TYPE_2_WORLD_RANGE = 20
TYPE_2_N_OBSTACLES = 0
TYPE_2_HEIGHT_SCALE = 1.0
TYPE_2_SAFE_ZONE = 0.0
TYPE_2_R_MIN, TYPE_2_R_MAX = 10, 25
TYPE_2_H_MIN, TYPE_2_H_MAX = 3, 10
TYPE_2_START_H_MIN, TYPE_2_START_H_MAX = 0.05, 10
TYPE_2_HORIZON = HORIZON_SEC

# Type 3: Mountain Navigation
TYPE_3_SAFE_ZONE = 1.0
TYPE_3_R_MIN, TYPE_3_R_MAX = 20, 100
TYPE_3_H_MIN, TYPE_3_H_MAX = 0, 0
TYPE_3_START_H_MIN, TYPE_3_START_H_MAX = 0, 0
TYPE_3_HORIZON = HORIZON_SEC
TYPE_3_SCALE_MIN = 0.6
TYPE_3_SCALE_MAX = 0.8
TYPE_3_SCALE_SEED_OFFSET = 777777
TYPE_3_WORLD_RANGE_RATIO = 0.60
TYPE_3_VILLAGE_RANGE = 40.0

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
TYPE_4_R_MIN, TYPE_4_R_MAX = 5, 35
TYPE_4_H_MIN, TYPE_4_H_MAX = 0.2, 10.0             # Floor to roof(12m) minus 2m ceiling clearance
TYPE_4_START_H_MIN, TYPE_4_START_H_MAX = 0.2, 10.0
TYPE_4_HORIZON = HORIZON_SEC
TYPE_4_PLATFORM_CLEARANCE = 1.0                     # Minimum clearance from warehouse structures (meters)
TYPE_4_PLATFORM_MAX_ATTEMPTS = 200                  # Max attempts to find collision-free platform position
TYPE_4_MIN_PLATFORM_DISTANCE = 10.0                 # Minimum 3D distance between start and goal platforms (meters)

# Type 6: Forest Navigation (100×100m ground, 96×96m playable with 2m edge margin)
TYPE_6_WORLD_RANGE = 42                             # ±42m playable XY (96m total with margin)
TYPE_6_R_MIN, TYPE_6_R_MAX = 10, 45
TYPE_6_H_MIN, TYPE_6_H_MAX = 0.2, 3.0
TYPE_6_START_H_MIN, TYPE_6_START_H_MAX = 0.2, 3.0
TYPE_6_HORIZON = HORIZON_SEC
TYPE_6_SAFETY_DISTANCE_SAFE = 0.6                   # Tighter safety for dense forest (meters)

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
