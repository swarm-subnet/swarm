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
# SEED SYNCHRONIZATION
# =============================================================================

USE_SYNCHRONIZED_SEEDS = True           # Enable synchronized seed generation across validators
SEED_WINDOW_MINUTES = 10                # Time window duration for seed synchronization (minutes)

# =============================================================================
# SIMULATION & PHYSICS
# =============================================================================

# Core simulation parameters
SIM_DT = 1/50                           # Physics simulation timestep (50 Hz)
HORIZON_SEC = 60                        # Maximum simulated flight duration (seconds)
# World generation parameters
WORLD_RANGE = 20                        # Random scenery placement range (±meters)
HEIGHT_SCALE = 1.5                      # Obstacle height scale factor
N_OBSTACLES = 40                        # Number of random obstacles in simulation world
RANDOM_START = True                    # Toggle random starting point generation
# Camera and rendering settings
CAM_HZ = 60                             # Camera update frequency (Hz)
CAMERA_FOV_BASE = 90.0                  # Base field of view (degrees)
CAMERA_FOV_VARIANCE = 2.0               # FOV randomization range (±degrees)
# Depth sensor parameters
DEPTH_NEAR = 0.05                       # PyBullet camera near plane (meters)
DEPTH_FAR = 1000.0                      # PyBullet camera far plane (meters)
DEPTH_MIN_M = 0.5                       # Minimum useful depth range (meters)
DEPTH_MAX_M = 20.0                      # Maximum useful depth range (meters)
# Search area parameters
SEARCH_AREA_NOISE_XY = 10.0             # ±10m horizontal noise = 20m total search zone
SEARCH_AREA_NOISE_Z = 2.0               # ±2m vertical noise
# Sensor noise parameters
SENSOR_NOISE_ENABLED = True             # Enable camera sensor noise
SENSOR_NOISE_STD = 5.0                  # Gaussian noise standard deviation
SENSOR_EXPOSURE_MIN = 0.95              # Minimum exposure multiplier
SENSOR_EXPOSURE_MAX = 1.05              # Maximum exposure multiplier
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
# Goal generation ranges
R_MIN, R_MAX = 5, 15                    # Radial goal distance range (meters)
H_MIN, H_MAX = 1, 5                     # Height variation range for goals (meters)
START_H_MIN, START_H_MAX = 0.05, 10     # Random start height range (meters)
# Environment building limits
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
# Distant scenery parameters
DISTANT_SCENERY_ENABLED = True          # Enable distant visual objects
DISTANT_SCENERY_MIN_RANGE = 75.0        # Minimum distance from origin (meters)
DISTANT_SCENERY_MAX_RANGE = 100.0        # Maximum distance from origin (meters)
DISTANT_SCENERY_COUNT = 18              # Number of distant objects

# =============================================================================
# SCORING & REWARDS
# =============================================================================

# Miner sampling and evaluation
SAMPLE_K = 256                          # Number of miners sampled per forward pass
EMA_ALPHA = 0.20                        # Exponential moving average coefficient for weights
# Emission burning mechanism
BURN_EMISSIONS = True                   # Enable emission burning to UID 0
BURN_FRACTION = 0.95                    # Fraction of emissions to burn 
KEEP_FRACTION = 1.0 - BURN_FRACTION     # Fraction of emissions to distribute 
UID_ZERO = 0                            # Special UID for burning emissions

# Reward distribution mechanism
WINNER_TAKE_ALL = True                  # Enable winner-take-all rewards (winner gets all available emissions)
N_RUNS_HISTORY = 100                     # Number of runs to track for victory average
MIN_RUNS_FOR_WEIGHTS = 25               # Minimum runs required before miner is eligible for weights

# =============================================================================
# LOW-PERFORMER FILTERING
# =============================================================================

LOW_PERFORMER_FILTER_ENABLED = True     # Enable filtering of consistently low-scoring models
MIN_AVG_SCORE_THRESHOLD = 0.2          # Minimum average score to remain in active evaluation pool
MIN_EVALUATION_RUNS = 20                # Check interval and minimum runs before filtering
EVALUATION_WINDOW = 20                 # Number of recent runs to evaluate for low-performer detection
# =============================================================================
# CHALLENGE TYPE DISTRIBUTION
# =============================================================================

CHALLENGE_TYPE_DISTRIBUTION = {
    1: 0.30,  # Standard navigation
    2: 0.25,  # higher obstacles challenge
    3: 0.25,  # Easy navigation
    4: 0.20,  # No obstacles (open flight)
}

assert abs(sum(CHALLENGE_TYPE_DISTRIBUTION.values()) - 1.0) < 0.001, "Challenge probabilities must sum to 1.0"

# =============================================================================
# CHALLENGE TYPE PARAMETERS
# =============================================================================

TYPE_1_N_OBSTACLES = 40
TYPE_1_HEIGHT_SCALE = 1.5
TYPE_1_SAFE_ZONE = 2.0

TYPE_2_N_OBSTACLES = 50
TYPE_2_HEIGHT_SCALE = 3
TYPE_2_SAFE_ZONE = 2.0

TYPE_3_N_OBSTACLES = 25
TYPE_3_HEIGHT_SCALE = 0.8
TYPE_3_SAFE_ZONE = 2.0

TYPE_4_N_OBSTACLES = 0
TYPE_4_HEIGHT_SCALE = 1.0
TYPE_4_SAFE_ZONE = 0.0

# =============================================================================
# PER-TYPE NORMALIZATION SYSTEM
# =============================================================================

AVGS_DIR = Path("avgs")
ENABLE_PER_TYPE_NORMALIZATION = True
