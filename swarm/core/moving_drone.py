from __future__ import annotations
from typing import Optional, Tuple, Iterable
import numpy as np

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import (
    DroneModel, Physics, ActionType, ObservationType
)

# ---- task‑level utilities ----------------------------------------------------
from swarm.validator.reward   import flight_reward          # new 5‑term scorer
from swarm.constants          import GOAL_TOL, HOVER_SEC
import gymnasium.spaces as spaces

class MovingDroneAviary(BaseRLAviary):
    """
    Single‑drone environment whose *goal position and horizon* are
    provided by an external MapTask (start, goal, horizon).

    The episode reward is the *increment* of `flight_reward`, so you can
    feed it straight into PPO/TD3/… without additional shaping.
    """
    MAX_TILT_RAD: float = 0.7

    # --------------------------------------------------------------------- #
    # 1. constructor
    # --------------------------------------------------------------------- #
    def __init__(
        self,
        task,                                  
        drone_model : DroneModel = DroneModel.CF2X,
        physics     : Physics     = Physics.PYB,
        pyb_freq    : int         = 240,
        ctrl_freq   : int         = 30,
        gui         : bool        = False,
        record      : bool        = False,
        obs         : ObservationType = ObservationType.KIN,
        act         : ActionType      = ActionType.RPM,
    ):
        """
        Parameters
        ----------
        task : MapTask
            Supplies `.start`, `.goal`, `.horizon`, `.sim_dt`.
        *remaining args* : forwarded to BaseRLAviary
        """
        self.task             = task
        self.GOAL_POS         = np.asarray(task.goal, dtype=float)
        self.EP_LEN_SEC       = float(task.horizon)

        # bookkeeping for reward shaping *inside* the aviary
        self._time_alive      = 0.0
        self._hover_sec       = 0.0
        self._d_start         = 1.0          
        self._prev_score      = 0.0
        self._success         = False
        self._t_to_goal       = None         
        self.observation_space = None

        super().__init__(
            drone_model   = drone_model,
            num_drones    = 1,
            initial_xyzs  = np.asarray([task.start]),
            initial_rpys  = None,
            physics       = physics,
            pyb_freq      = pyb_freq,
            ctrl_freq     = ctrl_freq,
            gui           = gui,
            record        = record,
            obs           = obs,
            act           = act,
        )

        old_low, old_high = self.observation_space.low, self.observation_space.high
        pad_low  = -np.ones((old_low.shape[0], 3), dtype=np.float32) * np.inf
        pad_high = +np.ones((old_high.shape[0], 3), dtype=np.float32) * np.inf
        self.observation_space = spaces.Box(
            low  = np.concatenate([old_low,  pad_low ], axis=1),
            high = np.concatenate([old_high, pad_high], axis=1),
            dtype=np.float32,
        )
    # --------------------------------------------------------------------- #
    # 2. low‑level helpers
    # --------------------------------------------------------------------- #
    @property
    def _sim_dt(self) -> float:
        """Physics step in seconds (PyBullet freq / ctrl freq already set)."""
        return 1.0 / self.CTRL_FREQ

    # --------------------------------------------------------------------- #
    # 3. Drone‑gym API overrides
    # --------------------------------------------------------------------- #
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)   # BaseRLAviary does the heavy lift

        self._time_alive = 0.0
        self._hover_sec  = 0.0
        self._success    = False
        self._t_to_goal  = None

        pos0             = obs[0, :3]
        self._d_start    = float(np.linalg.norm(pos0 - self.GOAL_POS))
        if self._d_start <= 0.0:
            self._d_start = 1e-9

        # baseline score at t = 0
        self._prev_score = flight_reward(
            success   = False,
            t_alive   = 0.0,
            d_start   = self._d_start,
            d_final   = self._d_start,
            horizon   = self.EP_LEN_SEC,
            goal_tol  = GOAL_TOL,
        )

        return obs, info

    # -------- reward ----------------------------------------------------- #
    def _computeReward(self) -> float:
        """
        Incremental shaped reward based on the new five‑term `flight_reward`.
        """
        state   = self._getDroneStateVector(0)
        dist    = float(np.linalg.norm(state[0:3] - self.GOAL_POS))

        # ---------- success bookkeeping (hover inside tolerance) ----------
        reached = dist < GOAL_TOL
        if reached:
            self._hover_sec += self._sim_dt
            if self._hover_sec >= HOVER_SEC and not self._success:
                # First time we are *certain* the goal is reached
                self._success   = True
                self._t_to_goal = self._time_alive
        else:
            self._hover_sec = 0.0

        # ---------- clock update ----------
        self._time_alive += self._sim_dt

        # ---------- reward computation ----------
        rw_args = dict(
            success = self._success,
            t_alive = self._time_alive,
            d_start = self._d_start,
            d_final = dist,
            horizon = self.EP_LEN_SEC,
            goal_tol = GOAL_TOL,
        )
        if self._success:
            # `flight_reward` requires these when success == True
            rw_args["t_to_goal"] = self._t_to_goal
            rw_args["e_used"]    = 0.0   

        score = flight_reward(**rw_args)

        r_t              = score - self._prev_score
        self._prev_score = score
        return float(r_t)

    # -------- termination ------------------------------------------------ #
    def _computeTerminated(self) -> bool:
        """
        Episode ends *only* when the success condition is met or PyBullet
        flags a fatal collision (handled by BaseAviary internally).
        """
        #TODO - READD collision handling
        return bool(self._success)

    # -------- truncation (timeout / safety) ------------------------------ #
    def _computeTruncated(self) -> bool:
        """
        Early‑terminate the episode when
        1) the drone is tilted too much (safety), **or**
        2) the configured horizon has elapsed (timeout).
        """
        # ---------- safety: roll / pitch limits ----------
        state = self._getDroneStateVector(0)
        roll, pitch = state[7], state[8]
        if abs(roll) > self.MAX_TILT_RAD or abs(pitch) > self.MAX_TILT_RAD:
            return True                           # stop immediately

        # ---------- timeout: task horizon ----------
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
    
    def _computeObs(self) -> np.ndarray:
        """
        Kinematics + relative goal vector  (15‑D).
          • index 0‑11 : default KIN observation
          • index 12‑14: (goal_x − x, goal_y − y, goal_z − z)
        """
        kin  = super()._computeObs()           # shape (12,)
        rel = (self.GOAL_POS - kin[0, :3]).reshape(1, 3)/10.0
        return np.concatenate([kin, rel], axis=1).astype(np.float32)
