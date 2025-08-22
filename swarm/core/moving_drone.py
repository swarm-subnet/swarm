# swarm/envs/moving_drone.py
from __future__ import annotations

from typing import cast
import numpy as np
from numpy.typing import NDArray
import gymnasium.spaces as spaces
import pybullet as p

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import (
    DroneModel, Physics, ActionType, ObservationType,
)

# ── project‑level utilities ────────────────────────────────────────────────
from swarm.validator.reward import flight_reward          # 3‑term scorer
from swarm.constants        import GOAL_TOL, HOVER_SEC


class MovingDroneAviary(BaseRLAviary):
    """
    Single‑drone environment whose *start*, *goal* and *horizon* are supplied
    via an external `MapTask`.

    The per‑step reward is the **increment** of `flight_reward`, so it can be
    fed directly to PPO/TD3/etc. without extra shaping.
    """
    MAX_TILT_RAD: float = 1.047         # safety cut‑off for roll / pitch (rad)

    # --------------------------------------------------------------------- #
    # 1. constructor
    # --------------------------------------------------------------------- #
    def __init__(
        self,
        task,
        drone_model : DroneModel   = DroneModel.CF2X,
        physics     : Physics      = Physics.PYB,
        pyb_freq    : int          = 240,
        ctrl_freq   : int          = 30,
        gui         : bool         = False,
        record      : bool         = False,
        obs         : ObservationType = ObservationType.KIN,
        act         : ActionType      = ActionType.RPM,
    ):
        """
        Parameters
        ----------
        task : MapTask
            Must expose `.start`, `.goal`, `.horizon`, `.sim_dt`.
        Remaining arguments are forwarded to ``BaseRLAviary`` unchanged.
        """
        self.task       = task
        self.GOAL_POS   = np.asarray(task.goal, dtype=float)
        self.EP_LEN_SEC = float(task.horizon)

        # internal book‑keeping
        self._time_alive   = 0.0
        self._hover_sec    = 0.0
        self._success      = False
        self._collision    = False
        self._t_to_goal    = None
        self._prev_score   = 0.0

        # ‑‑‑ define 16 ray directions for obstacle detection ‑‑‑
        self._init_ray_directions()

        # Let BaseRLAviary set up the PyBullet world
        super().__init__(
            drone_model  = drone_model,
            num_drones   = 1,
            initial_xyzs = np.asarray([task.start]),
            initial_rpys = None,
            physics      = physics,
            pyb_freq     = pyb_freq,
            ctrl_freq    = ctrl_freq,
            gui          = gui,
            record       = record,
            obs          = obs,
            act          = act,
        )

        # ‑‑‑ extend observation with obstacle distances (16-D) + goal vector (3-D) ‑‑‑
        obs_space = cast(spaces.Box, self.observation_space)
        old_low,  old_high  = obs_space.low, obs_space.high

        # Distance sensors: 16 dimensions, range [0.0, 1.0] (scaled from meters)
        dist_low = np.zeros((old_low.shape[0], 16), dtype=np.float32)
        dist_high = np.ones((old_high.shape[0], 16), dtype=np.float32)  # scaled to [0.0, 1.0]

        # Goal vector: 3 dimensions, unlimited range
        goal_low = -np.ones((old_low.shape[0], 3), dtype=np.float32) * np.inf
        goal_high = +np.ones((old_high.shape[0], 3), dtype=np.float32) * np.inf

        self.observation_space = spaces.Box(
            low   = np.concatenate([old_low, dist_low, goal_low], axis=1),
            high  = np.concatenate([old_high, dist_high, goal_high], axis=1),
            dtype = np.float32,
        )

    # --------------------------------------------------------------------- #
    # 2. low‑level helpers
    # --------------------------------------------------------------------- #
    @property
    def _sim_dt(self) -> float:
        """Physics step in seconds (1 / CTRL_FREQ)."""
        return 1.0 / self.CTRL_FREQ

    def _init_ray_directions(self):
        """Initialize 16 ray directions for obstacle detection - balanced with side coverage."""
        # Mathematical constants
        cos_45 = np.cos(np.radians(45))      # √2/2 ≈ 0.707107
        sin_45 = np.sin(np.radians(45))      # √2/2 ≈ 0.707107
        cos_30 = np.cos(np.radians(30))      # √3/2 ≈ 0.866025
        sin_30 = np.sin(np.radians(30))      # 1/2 = 0.500000

        self.ray_directions = np.array([
            # ═══ 2 PURE VERTICAL (essential for landing/overhead) ═══
            [0, 0, 1],                       # 1: Pure Up
            [0, 0, -1],                      # 2: Pure Down

            # ═══ 8 HORIZONTAL - BALANCED COVERAGE ═══
            [1, 0, 0],                       # 3: Forward (0°)
            [cos_45, sin_45, 0],             # 4: Forward-Right (45°)
            [0, 1, 0],                       # 5: Right (90°)
            [-cos_45, sin_45, 0],            # 6: Back-Right (135°)
            [-1, 0, 0],                      # 7: Back (180°)
            [-cos_45, -sin_45, 0],           # 8: Back-Left (225°)
            [0, -1, 0],                      # 9: Left (270°)
            [cos_45, -sin_45, 0],            # 10: Forward-Left (315°)

            # ═══ 6 DIAGONAL RAYS (3D coverage at ±30° elevation) ═══
            [cos_30, 0, sin_30],             # 11: Forward-Up (30°)
            [cos_30, 0, -sin_30],            # 12: Forward-Down (-30°)
            [-cos_30, 0, sin_30],            # 13: Back-Up (30°)
            [-cos_30, 0, -sin_30],           # 14: Back-Down (-30°)
            [0, cos_30, sin_30],             # 15: Right-Up (30°)
            [0, -cos_30, sin_30],            # 16: Left-Up (30°)
        ], dtype=np.float32)

        # Detection range and small origin offset to avoid self-hits
        self.max_ray_distance: float = 10.0
        self._ray_origin_offset: float = 0.12   # ~12 cm outside the hull

    def _get_obstacle_distances(self, drone_position: np.ndarray, drone_orientation: np.ndarray) -> np.ndarray:
        """
        Perform 16-ray casting for obstacle detection using batch processing.

        Parameters
        ----------
        drone_position : np.ndarray
            Current drone position [x, y, z] in world frame
        drone_orientation : np.ndarray
            Current drone orientation as 3x3 rotation matrix (body->world)

        Returns
        -------
        np.ndarray
            Array of 16 distances in meters [0.0 - 10.0].
            Distances are measured from the drone COM and clamped to max range.
        """
        rot_matrix = drone_orientation

        start_positions = []
        end_positions = []

        # Length of the ray segment when starting offset outside the hull
        seg_len = self.max_ray_distance - self._ray_origin_offset
        if seg_len <= 0:
            seg_len = 1e-6

        for direction in self.ray_directions:
            # Transform direction from body frame to world frame and normalize
            world_dir = rot_matrix @ direction
            n = float(np.linalg.norm(world_dir))
            if n < 1e-9:
                world_dir = np.array([0.0, 0.0, 1.0], dtype=float)  # fallback
            else:
                world_dir = world_dir / n

            # Start just outside the drone to avoid self-hit, end at max range from COM
            start_pos = drone_position + world_dir * self._ray_origin_offset
            end_pos   = drone_position + world_dir * self.max_ray_distance

            start_positions.append(start_pos.tolist())
            end_positions.append(end_pos.tolist())

        # Batch ray test - pass the correct physics client
        results = p.rayTestBatch(
            start_positions, end_positions,
            physicsClientId=getattr(self, "CLIENT", 0)
        )

        # Extract distances from results, converting "from start" to "from COM"
        distances = []
        for r in results:
            hit_uid = r[0]          # -1 means no hit
            hit_frac = float(r[2])  # [0,1] along the segment (start->end)
            if hit_uid != -1:
                # distance from COM = offset + fraction*segment_length, clamped
                d = self._ray_origin_offset + hit_frac * seg_len
                distances.append(float(min(self.max_ray_distance, max(0.0, d))))
            else:
                distances.append(self.max_ray_distance)

        return np.array(distances, dtype=np.float32)

    def _euler_to_rotation_matrix(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """
        Convert Euler angles directly to rotation matrix (body->world).
        """
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        return np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp,   cp*sr,            cp*cr           ]
        ])

    def _check_collision(self) -> bool:
        """
        Check if drone has collided with any obstacle using PyBullet contact points.
        Returns True if collision detected, False otherwise.
        
       Platform contact near goal is landing (good), not collision (bad).
        """
        drone_id = self.DRONE_IDS[0]
        contact_points = p.getContactPoints(
            bodyA=drone_id,
            physicsClientId=getattr(self, "CLIENT", 0)
        )
        
        # Get current drone position for goal distance check
        state = self._getDroneStateVector(0)
        drone_pos = state[0:3]
        dist_to_goal = float(np.linalg.norm(drone_pos - self.GOAL_POS))
        
        # Check if drone is in contact with any body other than ground plane
        for contact in contact_points:
            body_b = contact[2]  # Second body in contact
            if body_b != -1:  # -1 means no contact or ground plane
                # Additional check: ensure meaningful contact force
                normal_force = contact[9]
                if normal_force > 0.01:  # Minimum threshold to avoid numerical noise
                    
                    # If drone is near goal, platform contact is landing, NOT collision
                    from swarm.constants import GOAL_TOL
                    if dist_to_goal <= GOAL_TOL:
                        # Drone is in success zone - platform contact is landing, not collision
                        continue
                    
                    # Drone is outside success zone - any contact is collision
                    return True
        
        return False

    # --------------------------------------------------------------------- #
    # 3. OpenAI‑Gym API overrides
    # --------------------------------------------------------------------- #
    def reset(self, **kwargs):
        """
        Resets the underlying simulator and internal counters,
        returns initial observation and info as usual.
        """
        obs, info = super().reset(**kwargs)

        self._time_alive = 0.0
        self._hover_sec  = 0.0
        self._success    = False
        self._collision  = False
        self._t_to_goal  = None

        # baseline score (t = 0, e = 0)
        self._prev_score = flight_reward(
            success = False,
            t       = 0.0,
            horizon = self.EP_LEN_SEC,
            task    = None,
        )

        return obs, info

    # -------- reward ----------------------------------------------------- #
    def _computeReward(self) -> float:
        """
        **Incremental** reward based on the three‑term `flight_reward`.
        """
        # current distance to goal
        state = self._getDroneStateVector(0)
        dist  = float(np.linalg.norm(state[0:3] - self.GOAL_POS))

        # ── success detection: remain inside GOAL_TOL for HOVER_SEC seconds ──
        reached = dist < GOAL_TOL
        if reached:
            self._hover_sec += self._sim_dt
            if self._hover_sec >= HOVER_SEC and not self._success:
                self._success   = True
                self._t_to_goal = self._time_alive
        else:
            self._hover_sec = 0.0

        # ── clock update ────────────────────────────────────────────────────
        self._time_alive += self._sim_dt

        # ── call new reward function ───────────────────────────────────────
        # If collision detected, force score to 0
        if self._collision:
            score = 0.0
        else:
            score = flight_reward(
                success = self._success,
                t       = (self._t_to_goal if self._success else self._time_alive),
                horizon = self.EP_LEN_SEC,
                task    = None,
            )

        r_t              = score - self._prev_score
        self._prev_score = score
        return float(r_t)

    # -------- termination ------------------------------------------------ #
    def _computeTerminated(self) -> bool:
        """
        Episode ends when success condition is met OR collision detected.
        """
        # Check for collision first
        if self._check_collision():
            self._collision = True
            return True
            
        # Check for success
        return bool(self._success)

    # -------- truncation (timeout / safety) ------------------------------ #
    def _computeTruncated(self) -> bool:
        """
        Early termination on excessive tilt or elapsed horizon.
        """
        # safety cut‑off
        state = self._getDroneStateVector(0)
        roll, pitch = state[7], state[8]
        if abs(roll) > self.MAX_TILT_RAD or abs(pitch) > self.MAX_TILT_RAD:
            return True

        # timeout
        return self._time_alive >= self.EP_LEN_SEC

    # -------- extra logging --------------------------------------------- #
    def _computeInfo(self):
        state = self._getDroneStateVector(0)
        dist  = float(np.linalg.norm(state[0:3] - self.GOAL_POS))
        return {
            "distance_to_goal": dist,
            "score"           : self._prev_score,
            "success"         : self._success,
            "collision"       : self._collision,
            "t_to_goal"       : self._t_to_goal,
        }

    # -------- observation extension -------------------------------------- #
    def _computeObs(self) -> np.ndarray:
        """
        Full base observation (112-D) + obstacle distances (16-D) + goal vector (3-D) → 131-D.

        Distances are computed from the **current PyBullet pose** and
        then scaled to [0,1] by dividing by `max_ray_distance` (10 m).
        """
        base_obs: NDArray[np.float32] | None = super()._computeObs()  # shape (1, 112) in your setup
        if base_obs is None:
            return np.zeros((1, 131), dtype=np.float32)

        # --- Get exact pose from PyBullet to stay in sync with physics ---
        uid = self.DRONE_IDS[0]
        pos_w, orn_w = p.getBasePositionAndOrientation(uid, physicsClientId=getattr(self, "CLIENT", 0))
        pos_w = np.asarray(pos_w, dtype=float)
        rot_m = np.asarray(p.getMatrixFromQuaternion(orn_w), dtype=float).reshape(3, 3)

        # --- Cast rays from that pose ---
        distances_m = self._get_obstacle_distances(pos_w, rot_m).reshape(1, 16)

        # Scale to [0,1] for the observation (10 m range)
        distances_scaled = distances_m / self.max_ray_distance

        # Goal vector relative to current position (scaled by 10 for consistency with your setup)
        rel = ((self.GOAL_POS - pos_w) / 10.0).reshape(1, 3)

        return np.concatenate([base_obs, distances_scaled, rel], axis=1).astype(np.float32)
