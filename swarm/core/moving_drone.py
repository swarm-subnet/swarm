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
    MAX_TILT_RAD: float = 0.7          # safety cut‑off for roll / pitch (rad)

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
        self._t_to_goal    = None
        self._prev_score   = 0.0

        # ‑‑‑ define 18 ray directions for obstacle detection ‑‑‑
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

        # ‑‑‑ extend observation with obstacle distances (18-D) + goal vector (3-D) ‑‑‑
        obs_space = cast(spaces.Box, self.observation_space)
        old_low,  old_high  = obs_space.low, obs_space.high
        
        # Distance sensors: 18 dimensions, range [0.0, 1.0] (scaled from meters)
        dist_low = np.zeros((old_low.shape[0], 18), dtype=np.float32)
        dist_high = np.ones((old_high.shape[0], 18), dtype=np.float32)  # scaled to [0.0, 1.0]
        
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
        """Initialize the 18 ray directions for obstacle detection with exact mathematical expressions."""
        # Mathematical constants
        cos_45 = np.cos(np.radians(45))  # √2/2 ≈ 0.707107
        sin_45 = np.sin(np.radians(45))  # √2/2 ≈ 0.707107
        cos_30 = np.cos(np.radians(30))  # √3/2 ≈ 0.866025
        sin_30 = np.sin(np.radians(30))  # 1/2 = 0.500000
        
        self.ray_directions = np.array([
            # 8 horizontal directions (every 45°)
            [1, 0, 0],                    # 1: Forward (0°)
            [cos_45, sin_45, 0],          # 2: Forward-Right (45°)  
            [0, 1, 0],                    # 3: Right (90°)
            [-cos_45, sin_45, 0],         # 4: Back-Right (135°)
            [-1, 0, 0],                   # 5: Back (180°)
            [-cos_45, -sin_45, 0],        # 6: Back-Left (225°)
            [0, -1, 0],                   # 7: Left (270°)
            [cos_45, -sin_45, 0],         # 8: Forward-Left (315°)
            
            # 2 pure vertical
            [0, 0, 1],                    # 9: Up (90° elevation)
            [0, 0, -1],                   # 10: Down (-90° elevation)
            
            # 8 diagonal (30° elevation angles)
            [cos_30, 0, sin_30],          # 11: Forward-Up (30° elevation)
            [cos_30, 0, -sin_30],         # 12: Forward-Down (-30° elevation)
            [-cos_30, 0, sin_30],         # 13: Back-Up (30° elevation)
            [-cos_30, 0, -sin_30],        # 14: Back-Down (-30° elevation)
            [0, cos_30, sin_30],          # 15: Right-Up (30° elevation)
            [0, cos_30, -sin_30],         # 16: Right-Down (-30° elevation)
            [0, -cos_30, sin_30],         # 17: Left-Up (30° elevation)
            [0, -cos_30, -sin_30],        # 18: Left-Down (-30° elevation)
        ], dtype=np.float32)
        
        self.max_ray_distance = 10.0  # Maximum detection range in meters

    def _get_obstacle_distances(self, drone_position: np.ndarray, drone_orientation: np.ndarray) -> np.ndarray:
        """
        Perform 18-ray casting for obstacle detection using batch processing.
        
        Parameters
        ----------
        drone_position : np.ndarray
            Current drone position [x, y, z]
        drone_orientation : np.ndarray
            Current drone orientation as 3x3 rotation matrix
            
        Returns
        -------
        np.ndarray
            Array of 18 distances in meters [0.0 - 10.0]
            Note: These are scaled by 10 in _computeObs() to match goal vector scaling
        """
        # Use the rotation matrix directly (no conversion needed!)
        rot_matrix = drone_orientation
        
        # Transform all ray directions to world coordinates
        start_positions = []
        end_positions = []
        
        for direction in self.ray_directions:
            # Transform direction from drone body frame to world frame
            world_direction = rot_matrix @ direction
            end_position = drone_position + world_direction * self.max_ray_distance
            
            start_positions.append(drone_position.tolist())
            end_positions.append(end_position.tolist())
        
        # Batch ray test - much faster than individual rays
        results = p.rayTestBatch(start_positions, end_positions)
        
        # Extract distances from results
        distances = []
        for result in results:
            hit_object_id = result[0]
            if hit_object_id != -1:  # Hit detected
                hit_distance = result[2] * self.max_ray_distance
                distances.append(float(hit_distance))
            else:  # No hit - max distance
                distances.append(self.max_ray_distance)
        
        return np.array(distances, dtype=np.float32)

    def _euler_to_rotation_matrix(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """
        Convert Euler angles directly to rotation matrix
        
        Parameters
        ----------
        roll : float
            Roll angle in radians
        pitch : float
            Pitch angle in radians  
        yaw : float
            Yaw angle in radians
            
        Returns
        -------
        np.ndarray
            3x3 rotation matrix for transforming from body frame to world frame
        """
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        
        return np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp,   cp*sr,            cp*cr           ]
        ])

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
        self._t_to_goal  = None

        # baseline score (t = 0, e = 0)
        self._prev_score = flight_reward(
            success = False,
            t       = 0.0,
            e       = 0.0,
            horizon = self.EP_LEN_SEC,
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
        score = flight_reward(
            success = self._success,
            t       = (self._t_to_goal if self._success else self._time_alive),
            e       = 0.0,                        # energy not tracked inside env
            horizon = self.EP_LEN_SEC,
        )

        r_t              = score - self._prev_score
        self._prev_score = score
        return float(r_t)

    # -------- termination ------------------------------------------------ #
    def _computeTerminated(self) -> bool:
        """
        Episode ends only when the success condition is definitely met *or*
        PyBullet flags a fatal collision (handled upstream).
        """
        # TODO: re‑enable collision handling (if desired)
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
            "t_to_goal"       : self._t_to_goal,
        }

    # -------- observation extension -------------------------------------- #
    def _computeObs(self) -> np.ndarray:
        """
        Full base observation (112-D) + obstacle distances (18-D) + goal vector (3-D) → 133-D.
        """
        base_obs: NDArray[np.float32] | None = super()._computeObs()                  # shape (1, 112)
        if base_obs is None:
            return np.zeros((1, 133), dtype=np.float32)
        
        # Get current drone state for ray casting (single drone at index 0)
        drone_position = base_obs[0, 0:3]  # Extract position from base_obs directly
        
        roll, pitch, yaw = base_obs[0, 6], base_obs[0, 7], base_obs[0, 8]
        rotation_matrix = self._euler_to_rotation_matrix(roll, pitch, yaw)
        distances = self._get_obstacle_distances(drone_position, rotation_matrix).reshape(1, 18)
        
        distances_scaled = distances / 10.0
        rel = ((self.GOAL_POS - drone_position) / 10.0).reshape(1, 3)
        
        return np.concatenate([base_obs, distances_scaled, rel], axis=1).astype(np.float32)
