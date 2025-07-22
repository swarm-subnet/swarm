# Constants for the Swarm project

WORLD_RANGE = 30        # random scenery is placed in ±WORLD_RANGE (m)
HEIGHT_SCALE = 1  # scale factor for the height of obstacles, lower tends to create easier maps
N_OBSTACLES = 100  # number of random obstacles in the world

# ────────── Validator constants ─────────────────────────────────
SIM_DT              = 1/50      # 50 Hz physics step sent to miners
HORIZON_SEC         = 30      # max simulated flight time
SAMPLE_K            = 256       # miners sampled per forward
QUERY_TIMEOUT       = 30.0      # dendrite timeout (s)
FORWARD_SLEEP_SEC   = 2.0       # pause between forwards
EMA_ALPHA           = 0.20      # weights EMA coefficient
FORWARD_SLEEP_SEC = 300  # pause between forwards (s)

HOVER_SEC    = 3      # legacy constant (no longer used for landing)
CAM_HZ       = 60     # camera update rate (Hz)
PROP_EFF     = 0.60   # propeller efficiency 

LANDING_PLATFORM_RADIUS = 0.6  # Landing platform radius (m)
STABLE_LANDING_SEC = 1.0       # Required stable landing duration for success (s)
# ───────── parameters & constants ─────────
SAFE_Z: float   = 3     # cruise altitude (m)
GOAL_TOL: float = 1    # waypoint acceptance sphere (m)
CAM_HZ:  int    = 60
# ───────────────────────────────────────────
R_MIN, R_MAX = 1, 5          # radial goal distance (m)
H_MIN, H_MAX = 1, 5          # radial goal distance (m)
# ───────────────────────────────────────────
SAVE_FLIGHTPLANS = False  # save flight plans to disk
BURN_EMISSIONS = True  # burn emissions in the validator

# ────────── Platform mode toggle ─────────────────────────────────
PLATFORM = False           # Toggle for solid platform (True) / visual-only (False)