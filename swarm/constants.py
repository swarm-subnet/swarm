# Constants for the Swarm project

WORLD_RANGE = 30.0        # random scenery is placed in ±WORLD_RANGE (m)
HEIGHT_SCALE = 1.8  # scale factor for the height of obstacles, lower tends to create easier maps
N_OBSTACLES = 100  # number of random obstacles in the world

# ────────── Validator constants ─────────────────────────────────
SIM_DT              = 1/50      # 50 Hz physics step sent to miners
HORIZON_SEC         = 30      # max simulated flight time
SAMPLE_K            = 20       # miners sampled per forward
QUERY_TIMEOUT       = 30.0      # dendrite timeout (s)
FORWARD_SLEEP_SEC   = 2.0       # pause between forwards
EMA_ALPHA           = 0.20      # weights EMA coefficient

WAYPOINT_TOL = 1      # success sphere
HOVER_SEC    = 3      # time to hover at the goal (s)
CAM_HZ       = 60     # camera update rate (Hz)
PROP_EFF     = 0.60   # propeller efficiency 

# ───────── parameters & constants ─────────
SAFE_Z: float   = 2     # cruise altitude (m)
GOAL_TOL: float = 1    # waypoint acceptance sphere (m)
CAM_HZ:  int    = 60
# ───────────────────────────────────────────
R_MIN, R_MAX = 10.0, 30.0          # radial goal distance (m)
H_MIN, H_MAX = 2, 10          # radial goal distance (m)