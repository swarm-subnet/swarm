# =============================================================================
# SWARM SUBNET CONSTANTS
# =============================================================================
# Centralized constants for the Swarm Bittensor subnet. This file contains all
# configuration values, limits, and parameters used throughout the system.
# =============================================================================

from pathlib import Path

# =============================================================================
# NETWORK & COMMUNICATION
# =============================================================================

QUERY_REF_TIMEOUT = 3.0                 # PolicyRef request timeout (seconds)
QUERY_BLOB_TIMEOUT = 30.0               # Model blob download timeout (seconds)
FORWARD_SLEEP_SEC = 2.0                 # Pause between validator forward passes (seconds)

# =============================================================================
# SIMULATION & PHYSICS
# =============================================================================

# Core simulation parameters
SIM_DT = 1/50                           # Physics simulation timestep (50 Hz)
HORIZON_SEC = 30                        # Maximum simulated flight duration (seconds)
# World generation parameters
WORLD_RANGE = 30                        # Random scenery placement range (±meters)
HEIGHT_SCALE = 2                        # Obstacle height scale factor
N_OBSTACLES = 100                       # Number of random obstacles in simulation world
RANDOM_START = True                    # Toggle random starting point generation
# Camera and rendering settings
CAM_HZ = 60                             # Camera update frequency (Hz)
# Propulsion efficiency
PROP_EFF = 0.60                         # Propeller efficiency coefficient

# =============================================================================
# MODEL & AI EVALUATION
# =============================================================================

# Model size and validation limits
MAX_MODEL_BYTES = 10 * 1024 * 1024      # Maximum compressed model size (10 MiB)
EVAL_TIMEOUT_SEC = 120.0                # Model evaluation subprocess timeout (seconds)
# Model storage and processing
MODEL_DIR = Path("miner_models_v2")     # Directory for storing miner model files
BLACKLIST_FILE = MODEL_DIR / "fake_models_blacklist.txt"  # Blacklisted model hashes file
CHUNK_SIZE = 2 * 1024 * 1024            # File transfer chunk size (2 MiB)
SUBPROC_MEM_MB = 8192                   # Memory limit per evaluation subprocess (MB)
# Security metadata requirements
SAFE_META_FILENAME = "safe_policy_meta.json"  # Required metadata file in model archives

# =============================================================================
# DRONE & FLIGHT CONTROL
# =============================================================================

# Drone physical specifications
DRONE_MASS = 0.027                          # Drone mass (kg) - CF2X Crazyflie
DRONE_HULL_RADIUS = 0.12                    # Drone hull radius from center to edge (meters)
MAX_RAY_DISTANCE = 20.0                     # Maximum obstacle detection range (meters)

# Sensor Noise Configuration
ENABLE_NOISE = True                        # Enable sensor noise simulation by default
DEFAULT_NOISE_LEVEL = "good"              # Default noise difficulty: "good", "medium", or "hard"

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
HOVER_SEC = 3                           # Required hover duration for mission success (seconds)
SAFE_Z = 3                              # Default cruise altitude (meters)
GOAL_TOL = LANDING_PLATFORM_RADIUS * 0.8 * 1.06  # TAO badge radius for precision landing (0.5088m)
SPEED_LIMIT = 3.0                       # Maximum drone velocity limit (m/s)
# Goal generation ranges
R_MIN, R_MAX = 10, 30                    # Radial goal distance range (meters)
H_MIN, H_MAX = 1, 10                    # Height variation range for goals (meters)
START_H_MIN, START_H_MAX = 0.05, 10     # Random start height range (meters)
# Environment building limits
SAFE_ZONE_RADIUS = 2.0                  # Minimum clearance around obstacles (meters)
MAX_ATTEMPTS_PER_OBS = 100              # Maximum retry attempts when placing obstacles

# =============================================================================
# SCORING & REWARDS
# =============================================================================

# Miner sampling and evaluation
SAMPLE_K = 256                          # Number of miners sampled per forward pass
EMA_ALPHA = 0.20                        # Exponential moving average coefficient for weights
# Emission burning mechanism
BURN_EMISSIONS = True                   # Enable emission burning to UID 0
BURN_FRACTION = 0.75                    # Fraction of emissions to burn 
KEEP_FRACTION = 1.0 - BURN_FRACTION     # Fraction of emissions to distribute 
UID_ZERO = 0                            # Special UID for burning emissions

# Reward distribution mechanism
WINNER_TAKE_ALL = True                  # Enable winner-take-all rewards (winner gets all available emissions)
N_RUNS_HISTORY = 10                     # Number of runs to track for victory average

# =============================================================================
# CHALLENGE TYPE DISTRIBUTION
# =============================================================================

CHALLENGE_TYPE_DISTRIBUTION = {
    1: 0.30,
    2: 0.25,
    3: 0.25,
    4: 0.15,
    5: 0.05,
}

assert abs(sum(CHALLENGE_TYPE_DISTRIBUTION.values()) - 1.0) < 0.001, "Challenge probabilities must sum to 1.0"

# =============================================================================
# CHALLENGE TYPE PARAMETERS
# =============================================================================

TYPE_1_N_OBSTACLES = 100
TYPE_1_HEIGHT_SCALE = 2
TYPE_1_SAFE_ZONE = 2.0
TYPE_1_R_MIN = 10
TYPE_1_R_MAX = 30
TYPE_1_WORLD_RANGE = 30

TYPE_2_N_OBSTACLES = 125
TYPE_2_HEIGHT_SCALE = 6
TYPE_2_SAFE_ZONE = 2.0
TYPE_2_R_MIN = 10
TYPE_2_R_MAX = 30
TYPE_2_WORLD_RANGE = 30

TYPE_3_N_OBSTACLES = 75
TYPE_3_HEIGHT_SCALE = 1
TYPE_3_SAFE_ZONE = 2.0
TYPE_3_R_MIN = 10
TYPE_3_R_MAX = 30
TYPE_3_WORLD_RANGE = 30

TYPE_4_N_OBSTACLES = 100
TYPE_4_HEIGHT_SCALE = 2
TYPE_4_SAFE_ZONE = 2.0
TYPE_4_R_MIN = 20
TYPE_4_R_MAX = 30
TYPE_4_WORLD_RANGE = 30

TYPE_5_N_OBSTACLES = 0
TYPE_5_HEIGHT_SCALE = 0
TYPE_5_SAFE_ZONE = 2.0
TYPE_5_R_MIN = 30
TYPE_5_R_MAX = 50
TYPE_5_WORLD_RANGE = 55


