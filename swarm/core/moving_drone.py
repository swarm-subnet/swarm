# swarm/envs/moving_drone.py
from __future__ import annotations

import numpy as np
import gymnasium.spaces as spaces
import pybullet as p

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import (
    DroneModel, Physics, ActionType, ObservationType,
)

# ── project‑level utilities ────────────────────────────────────────────────
from swarm.validator.reward import flight_reward          # 3-term scorer


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
        obs         : ObservationType = ObservationType.RGB,
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
        self._success      = False
        self._collision    = False
        self._t_to_goal    = None
        self._prev_score   = 0.0

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

        if self.OBS_TYPE != ObservationType.RGB:
            raise ValueError("MovingDroneAviary only supports ObservationType.RGB observations.")

        enhanced_width, enhanced_height = 480, 360
        self.IMG_RES = np.array([enhanced_width, enhanced_height])
        self.rgb = np.zeros((self.NUM_DRONES, enhanced_height, enhanced_width, 4), dtype=np.uint8)
        self.dep = np.ones((self.NUM_DRONES, enhanced_height, enhanced_width), dtype=np.float32)
        self.seg = np.zeros((self.NUM_DRONES, enhanced_height, enhanced_width), dtype=np.uint8)

        img_shape = (enhanced_height, enhanced_width, 4)
        state_dim = 12 + self.ACTION_BUFFER_SIZE * 4
        self._state_dim = state_dim

        self.observation_space = spaces.Dict({
            "rgb": spaces.Box(
                low=0,
                high=255,
                shape=img_shape,
                dtype=np.uint8
            ),
            "state": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(state_dim,),
                dtype=np.float32
            ),
        })

    # --------------------------------------------------------------------- #
    # 2. low‑level helpers
    # --------------------------------------------------------------------- #
    @property
    def _sim_dt(self) -> float:
        """Physics step in seconds (1 / CTRL_FREQ)."""
        return 1.0 / self.CTRL_FREQ

    def _getDroneImages(self, nth_drone, segmentation: bool = True):
        if self.OBS_TYPE != ObservationType.RGB:
            return super()._getDroneImages(nth_drone, segmentation)
        
        if self.IMG_RES is None:
            print("[ERROR] in MovingDroneAviary._getDroneImages(), IMG_RES not set")
            exit()
        
        cli = getattr(self, "CLIENT", 0)
        drone_pos = self.pos[nth_drone, :]
        rot_mat = np.array(p.getMatrixFromQuaternion(self.quat[nth_drone, :])).reshape(3, 3)
        
        forward = rot_mat @ np.array([1.0, 0.0, 0.0])
        forward = forward / np.linalg.norm(forward)
        up = rot_mat @ np.array([0.0, 0.0, 1.0])
        
        camera_offset = 0.35
        camera_pos = drone_pos + forward * camera_offset + up * 0.05
        
        target = camera_pos + forward * 20.0
        
        DRONE_CAM_VIEW = p.computeViewMatrix(
            cameraEyePosition=camera_pos,
            cameraTargetPosition=target,
            cameraUpVector=[0, 0, 1],
            physicsClientId=cli
        )
        
        aspect = self.IMG_RES[0] / self.IMG_RES[1]
        DRONE_CAM_PRO = p.computeProjectionMatrixFOV(
            fov=90.0,
            aspect=aspect,
            nearVal=0.05,
            farVal=1000.0,
            physicsClientId=cli
        )
        
        SEG_FLAG = p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX if segmentation else p.ER_NO_SEGMENTATION_MASK
        [w, h, rgb, dep, seg] = p.getCameraImage(
            width=self.IMG_RES[0],
            height=self.IMG_RES[1],
            shadow=1,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            viewMatrix=DRONE_CAM_VIEW,
            projectionMatrix=DRONE_CAM_PRO,
            flags=SEG_FLAG,
            physicsClientId=cli
        )
        
        rgb = np.reshape(rgb, (h, w, 4))
        dep = np.reshape(dep, (h, w))
        seg = np.reshape(seg, (h, w))
        return rgb, dep, seg

    def _check_collision(self) -> bool:
        """
        Inspect contact points and update success/collision flags.
        Returns True only when an obstacle collision occurred.
        """
        drone_id = self.DRONE_IDS[0]
        contact_points = p.getContactPoints(
            bodyA=drone_id,
            physicsClientId=getattr(self, "CLIENT", 0)
        )

        if not contact_points:
            return False

        landing_surface_uid = getattr(self, '_landing_surface_uid', None)
        platform_support_uid = getattr(self, '_platform_support_uid', None)

        platform_hit = False
        obstacle_hit = False

        for contact in contact_points:
            body_b = contact[2]
            if body_b == -1:
                continue

            normal_force = contact[9]
            if normal_force <= 0.01:
                continue

            if (
                (landing_surface_uid is not None and body_b == landing_surface_uid)
                or (platform_support_uid is not None and body_b == platform_support_uid)
            ):
                platform_hit = True
                continue

            obstacle_hit = True
            break

        if platform_hit and not self._success:
            self._success = True
            self._t_to_goal = self._time_alive

        if obstacle_hit:
            self._collision = True

        return obstacle_hit

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
        self._success    = False
        self._collision  = False
        self._t_to_goal  = None

        self._prev_score = flight_reward(
            success = False,
            t       = 0.0,
            horizon = self.EP_LEN_SEC,
            task    = None,
        )

        self._spawn_task_world()

        return obs, info

    def _spawn_task_world(self):
        """Rebuild the procedural world defined by self.task."""
        from swarm.core.env_builder import build_world
        
        cli = getattr(self, "CLIENT", 0)
        platform_support_uid, landing_surface_uid = build_world(
            seed=self.task.map_seed,
            cli=cli,
            start=self.task.start,
            goal=self.task.goal,
            challenge_type=self.task.challenge_type,
        )
        
        self._platform_support_uid = platform_support_uid
        self._landing_surface_uid = landing_surface_uid
        
        start_xyz = np.asarray(self.task.start, dtype=float)
        start_quat = p.getQuaternionFromEuler([0.0, 0.0, 0.0])
        
        p.resetBasePositionAndOrientation(
            self.DRONE_IDS[0],
            start_xyz,
            start_quat,
            physicsClientId=cli,
        )

    # -------- reward ----------------------------------------------------- #
    def _computeReward(self) -> float:
        """
        **Incremental** reward based on the three-term `flight_reward`.
        """
        # Update contact flags before awarding reward
        self._check_collision()

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
    def _computeObs(self):
        """
        Build RGB + state observation for the single-drone task.
        """
        rgb_obs = super()._computeObs()
        if rgb_obs is None:
            h, w = (self.IMG_RES[1], self.IMG_RES[0]) if self.IMG_RES is not None else (48, 64)
            state_dim = getattr(self, "_state_dim", 112)
            return {
                "rgb": np.zeros((h, w, 4), dtype=np.uint8),
                "state": np.zeros((state_dim,), dtype=np.float32),
            }

        img = rgb_obs[0].astype(np.uint8)

        state_vec = self._getDroneStateVector(0)
        obs_12 = np.hstack([
            state_vec[0:3],
            state_vec[7:10],
            state_vec[10:13],
            state_vec[13:16]
        ]).astype(np.float32)

        state_full = np.array([obs_12], dtype=np.float32)
        for i in range(self.ACTION_BUFFER_SIZE):
            state_full = np.hstack([state_full, np.array([self.action_buffer[i][0, :]])])
        state_full = state_full.flatten().astype(np.float32)

        return {
            "rgb": img,
            "state": state_full,
        }
