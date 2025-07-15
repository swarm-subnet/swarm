# Constants for the Swarm project

WORLD_RANGE = 30.0        # random scenery is placed in ±WORLD_RANGE (m)
HEIGHT_SCALE = 1.8  # scale factor for the height of obstacles, lower tends to create easier maps
N_OBSTACLES = 100  # number of random obstacles in the world

# ────────── Validator constants ─────────────────────────────────
SIM_DT              = 1/50      # 50 Hz physics step sent to miners
HORIZON_SEC         = 30      # max simulated flight time
SAMPLE_K            = 256       # miners sampled per forward
QUERY_TIMEOUT       = 30.0      # dendrite timeout (s)
FORWARD_SLEEP_SEC   = 2.0       # pause between forwards
EMA_ALPHA           = 0.20      # weights EMA coefficient
FORWARD_SLEEP_SEC = 300  # pause between forwards (s)

WAYPOINT_TOL = 1      # landing success radius  
HOVER_SEC    = 3      # legacy constant (no longer used for landing)
CAM_HZ       = 60     # camera update rate (Hz)
PROP_EFF     = 0.60   # propeller efficiency 

LANDING_PLATFORM_RADIUS = 0.6  # Landing platform radius (m)
STABLE_LANDING_SEC = 1.0       # Required stable landing duration for success (s)
# ───────── parameters & constants ─────────
SAFE_Z: float   = 2     # cruise altitude (m)
GOAL_TOL: float = 1    # waypoint acceptance sphere (m)
CAM_HZ:  int    = 60
# ───────────────────────────────────────────
R_MIN, R_MAX = 10.0, 30          # radial goal distance (m)
H_MIN, H_MAX = 2, 10          # radial goal distance (m)
# ───────────────────────────────────────────
SAVE_FLIGHTPLANS = False  # save flight plans to disk

# ────────── Platform mode toggle ─────────────────────────────────
PLATFORM = True           # Toggle for solid platform (True) / visual-only (False)

# ────────── Bird Simulation System ─────────────────────────────────
ENABLE_BIRDS = True        # Enable/disable avian simulation system
N_BIRDS = 25               # Total avian entities spawned per simulation map
BIRD_SIZE = 0.3            # Avian collision detection radius (meters)
BIRD_SPEED_MIN = 2.5       # Minimum avian flight velocity (m/s)
BIRD_SPEED_MAX = 5.0       # Maximum avian flight velocity (m/s)

# ────────── Wind Simulation System ─────────────────────────────────
ENABLE_WIND = True         # Enable/disable atmospheric wind simulation
WIND_SPEED_MIN = 0.5       # Minimum wind velocity magnitude (m/s)
WIND_SPEED_MAX = 3.0       # Maximum wind velocity magnitude (m/s)
WIND_DIRECTION_CHANGE_INTERVAL = 10.0  # Wind direction change interval (seconds)