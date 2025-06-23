# Constants for the Swarm project

WORLD_RANGE = 40.0        # random scenery is placed in ±WORLD_RANGE (m)
HEIGHT_SCALE = 1.0  # scale factor for the height of obstacles, lower tends to create easier maps

# ────────── Validator constants ─────────────────────────────────
SIM_DT              = 0.02      # 50 Hz physics step sent to miners
HORIZON_SEC         = 30      # max simulated flight time
SAMPLE_K            = 20       # miners sampled per forward
QUERY_TIMEOUT       = 30.0      # dendrite timeout (s)
FORWARD_SLEEP_SEC   = 2.0       # pause between forwards
EMA_ALPHA           = 0.20      # weights EMA coefficient

WAYPOINT_TOL = 1      # success sphere
HOVER_SEC    = 3
CAM_HZ       = 60
PROP_EFF     = 0.60

# ───────── parameters & constants ─────────
SAFE_Z: float   = 2     # cruise altitude (m)
GOAL_TOL: float = 1    # waypoint acceptance sphere (m)
CAM_HZ:  int    = 60
# ───────────────────────────────────────────