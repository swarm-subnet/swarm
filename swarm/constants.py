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

QUERY_TIMEOUT = 30.0                    # Dendrite query timeout (seconds)
FORWARD_SLEEP_SEC = 2.0                 # Pause between validator forward passes (seconds)

# =============================================================================
# SIMULATION & PHYSICS
# =============================================================================

# Core simulation parameters
SIM_DT = 1/50                           # Physics simulation timestep (50 Hz)
HORIZON_SEC = 30                        # Maximum simulated flight duration (seconds)
# World generation parameters
WORLD_RANGE = 30                        # Random scenery placement range (±meters)
HEIGHT_SCALE = 2                        # Obstacle height scale factor (lower = easier maps)
N_OBSTACLES = 100                       # Number of random obstacles in simulation world
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

# Landing and positioning parameters
LANDING_PLATFORM_RADIUS = 0.6          # Landing platform acceptance radius (meters)
PLATFORM = True                         # Enable landing platform rendering
STABLE_LANDING_SEC = 1.0                # Required stable landing duration for success (seconds)
HOVER_SEC = 3                           # Required hover duration for mission success (seconds)
SAFE_Z = 3                              # Default cruise altitude (meters)
GOAL_TOL = LANDING_PLATFORM_RADIUS * 0.8 * 1.06  # TAO badge radius for precision landing (0.5088m)
SPEED_LIMIT = 3.0                       # Maximum drone velocity limit (m/s)
# Goal generation ranges
R_MIN, R_MAX = 3, 30                    # Radial goal distance range (meters)
H_MIN, H_MAX = 1, 10                    # Height variation range for goals (meters)
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


