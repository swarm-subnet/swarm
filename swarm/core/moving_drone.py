# swarm/envs/moving_drone.py
from __future__ import annotations

import math
import numpy as np
import gymnasium.spaces as spaces
import pybullet as p

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import (
    DroneModel, Physics, ActionType, ObservationType,
)

# ── project‑level utilities ────────────────────────────────────────────────
from swarm.core.env_builder import build_world
from swarm.validator.reward import flight_reward
from swarm.constants import (
    DRONE_HULL_RADIUS, ALTITUDE_RAY_INSET, MAX_RAY_DISTANCE,
    DEPTH_NEAR, DEPTH_FAR, DEPTH_MIN_M, DEPTH_MAX_M,
    SEARCH_AREA_NOISE_Z,
    CAMERA_FOV_BASE, CAMERA_FOV_VARIANCE,
    LIGHT_RANDOMIZATION_ENABLED,
    PLATFORM_MOVEMENT_PATTERNS,
    PLATFORM_SPEED_MIN, PLATFORM_SPEED_MAX,
    PLATFORM_RADIUS_MIN, PLATFORM_RADIUS_MAX,
    PLATFORM_DELAY_MIN, PLATFORM_DELAY_MAX,
    PLATFORM_TRANSITION_MIN, PLATFORM_TRANSITION_MAX,
    PLATFORM_LINEAR_DIRECTIONS,
    PLATFORM_AVOIDANCE_ENABLED, PLATFORM_STEER_ANGLES, PLATFORM_MIN_STEP_M,
    LANDING_PLATFORM_RADIUS,
    SAFETY_DISTANCE_SAFE,
    START_PLATFORM_TAKEOFF_BUFFER,
    LANDING_MAX_VZ, LANDING_MAX_VXY_REL, LANDING_MAX_TILT_RAD, LANDING_STABLE_SEC,
)


class MovingDroneAviary(BaseRLAviary):
    """
    Single‑drone environment whose *start*, *goal* and *horizon* are supplied
    via an external `MapTask`.

    The per‑step reward is the **increment** of `flight_reward`, so it can be
    fed directly to PPO/TD3/etc. without extra shaping.
    """
    MAX_TILT_RAD: float = 1.047         # safety cut‑off for roll / pitch (rad)
    _fov: float = 90.0

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
        self._moving    = getattr(task, 'moving_platform', False)

        self._time_alive = 0.0
        self._success = False
        self._collision = False
        self._t_to_goal = None
        self._prev_score = 0.0
        self._step_processed = False
        self._min_clearance_episode = SAFETY_DISTANCE_SAFE
        self._landing_stable_time = 0.0
        self._prev_platform_pos = None
        self._platform_velocity = np.zeros(3, dtype=np.float32)
        
        seed = getattr(task, 'map_seed', 0)
        
        self._platform_orbit_center = self.GOAL_POS.copy()
        self._current_platform_pos = self.GOAL_POS.copy()
        self._movement_pattern = self._get_movement_pattern_from_seed(seed)
        self._platform_offsets = []

        self._init_platform_randomization(seed)
        search_radius = getattr(task, 'search_radius', 10.0)
        rng = np.random.RandomState(seed)
        noise_xy = rng.uniform(-search_radius, search_radius, size=2)
        noise_z = rng.uniform(-SEARCH_AREA_NOISE_Z, SEARCH_AREA_NOISE_Z)
        self._search_area_center = self.GOAL_POS.copy()
        self._search_area_center[0] += noise_xy[0]
        self._search_area_center[1] += noise_xy[1]
        self._search_area_center[2] += noise_z
        
        fov_rng = np.random.RandomState(seed)
        fov_rng.rand()
        self._fov = CAMERA_FOV_BASE + fov_rng.uniform(-CAMERA_FOV_VARIANCE, CAMERA_FOV_VARIANCE)

        if LIGHT_RANDOMIZATION_ENABLED:
            light_rng = np.random.RandomState(seed)
            light_rng.rand()
            light_rng.rand()
            light_rng.rand()
            angle = light_rng.uniform(0, 2 * np.pi)
            self._light_direction = [
                -np.cos(angle),
                0.1 * np.sin(angle * 3),
                np.sin(angle)
            ]
        else:
            self._light_direction = [0, 0, 1]

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

        enhanced_width, enhanced_height = 128, 128
        self.IMG_RES = np.array([enhanced_width, enhanced_height])
        self.dep = np.ones((self.NUM_DRONES, enhanced_height, enhanced_width), dtype=np.float32)

        action_dim = self.action_space.shape[-1]
        state_dim = 12 + self.ACTION_BUFFER_SIZE * action_dim + 1 + 3
        self._state_dim = state_dim

        depth_shape = (enhanced_height, enhanced_width, 1)
        self.observation_space = spaces.Dict({
            "depth": spaces.Box(
                low=0.0,
                high=1.0,
                shape=depth_shape,
                dtype=np.float32
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
    
    def _get_movement_pattern_from_seed(self, seed: int) -> str:
        """Deterministically select movement pattern based on seed."""
        if not self._moving:
            return "static"
        rng = np.random.RandomState(seed)
        rng.rand()
        rng.rand()
        rng.rand()
        rng.rand()
        pattern_idx = rng.randint(0, len(PLATFORM_MOVEMENT_PATTERNS))
        return PLATFORM_MOVEMENT_PATTERNS[pattern_idx]

    def _init_platform_randomization(self, seed: int) -> None:
        """Initialize randomized platform movement parameters."""
        if not self._moving:
            self._platform_speed = 0.0
            self._platform_radius = 0.0
            self._platform_delay = 0.0
            self._platform_transition_time = 0.0
            self._platform_phase = 0.0
            self._platform_linear_dir = "x"
            self._platform_linear_angle = 0.0
            return

        rng = np.random.RandomState((seed + 77777) & 0xFFFFFFFF)
        self._platform_speed = rng.uniform(PLATFORM_SPEED_MIN, PLATFORM_SPEED_MAX)
        self._platform_radius = rng.uniform(PLATFORM_RADIUS_MIN, PLATFORM_RADIUS_MAX)
        self._platform_delay = rng.uniform(PLATFORM_DELAY_MIN, PLATFORM_DELAY_MAX)
        self._platform_transition_time = rng.uniform(PLATFORM_TRANSITION_MIN, PLATFORM_TRANSITION_MAX)
        self._platform_phase = rng.uniform(0, 2 * np.pi)
        dir_idx = rng.randint(0, len(PLATFORM_LINEAR_DIRECTIONS))
        self._platform_linear_dir = PLATFORM_LINEAR_DIRECTIONS[dir_idx]
        self._platform_linear_angle = rng.uniform(0, 2 * np.pi)

    def _get_orbit_position(self, t_eff: float) -> np.ndarray:
        """Calculate orbit position for a given effective time."""
        center = self._platform_orbit_center
        speed = self._platform_speed
        radius = self._platform_radius
        phase = self._platform_phase
        pattern = self._movement_pattern

        if pattern == "circular":
            angle = t_eff * speed * 0.3 + phase
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            return np.array([x, y, center[2]], dtype=np.float32)

        elif pattern == "linear":
            offset = radius * math.sin(t_eff * speed * 0.5 + phase)
            if self._platform_linear_dir == "x":
                x = center[0] + offset
                y = center[1]
            elif self._platform_linear_dir == "y":
                x = center[0]
                y = center[1] + offset
            else:
                x = center[0] + offset * math.cos(self._platform_linear_angle)
                y = center[1] + offset * math.sin(self._platform_linear_angle)
            return np.array([x, y, center[2]], dtype=np.float32)

        elif pattern == "figure8":
            angle = t_eff * speed * 0.3 + phase
            x = center[0] + radius * math.sin(angle)
            y = center[1] + radius * math.sin(2 * angle) / 2
            return np.array([x, y, center[2]], dtype=np.float32)

        return np.array(center, dtype=np.float32)

    def _calculate_platform_position(self, t: float) -> np.ndarray:
        """Calculate platform position at time t with smooth transition."""
        if not self._moving:
            return self._platform_orbit_center.copy()

        delay = self._platform_delay
        transition = self._platform_transition_time
        center = self._platform_orbit_center

        if t < delay:
            return np.array(center, dtype=np.float32)

        orbit_start = self._get_orbit_position(0.0)

        if t < delay + transition:
            t_ratio = (t - delay) / transition
            t_smooth = t_ratio * t_ratio * (3.0 - 2.0 * t_ratio)
            return center + t_smooth * (orbit_start - center)

        t_eff = t - delay - transition
        return self._get_orbit_position(t_eff)
    
    def _platform_path_blocked(self, current_pos, target_pos):
        """Check if path or destination is blocked by obstacles."""
        cli = getattr(self, "CLIENT", 0)
        direction = target_pos - current_pos
        dist = np.linalg.norm(direction[:2])
        if dist < 0.001:
            return False, None

        excluded = set(getattr(self, '_end_platform_uids', []))
        excluded |= set(getattr(self, '_start_platform_uids', []))
        excluded.add(self.DRONE_IDS[0])
        excluded.add(getattr(self, 'PLANE_ID', 0))

        offsets = [
            np.array([0, 0, 0], dtype=np.float32),
            np.array([LANDING_PLATFORM_RADIUS, 0, 0], dtype=np.float32),
            np.array([-LANDING_PLATFORM_RADIUS, 0, 0], dtype=np.float32),
            np.array([0, LANDING_PLATFORM_RADIUS, 0], dtype=np.float32),
            np.array([0, -LANDING_PLATFORM_RADIUS, 0], dtype=np.float32),
        ]

        for offset in offsets:
            ray_from = (current_pos + offset).tolist()
            ray_to = (target_pos + offset).tolist()
            result = p.rayTest(ray_from, ray_to, physicsClientId=cli)
            if result and result[0][0] != -1 and result[0][0] not in excluded:
                return True, np.array(result[0][3], dtype=np.float32)

        end_uids = getattr(self, '_end_platform_uids', [])
        if end_uids:
            plat_uid = end_uids[0]
            saved_pos, saved_orn = p.getBasePositionAndOrientation(plat_uid, physicsClientId=cli)
            p.resetBasePositionAndOrientation(plat_uid, target_pos.tolist(), [0, 0, 0, 1], physicsClientId=cli)
            num_bodies = p.getNumBodies(physicsClientId=cli)
            for body_idx in range(num_bodies):
                body_uid = p.getBodyUniqueId(body_idx, physicsClientId=cli)
                if body_uid in excluded:
                    continue
                mn, mx = p.getAABB(body_uid, physicsClientId=cli)
                if (mx[0] - mn[0]) > 50.0 or (mx[1] - mn[1]) > 50.0:
                    continue
                contacts = p.getClosestPoints(plat_uid, body_uid, distance=0.15, physicsClientId=cli)
                if contacts:
                    for c in contacts:
                        if c[8] < 0.15:
                            p.resetBasePositionAndOrientation(plat_uid, list(saved_pos), list(saved_orn), physicsClientId=cli)
                            return True, target_pos.copy()
            p.resetBasePositionAndOrientation(plat_uid, list(saved_pos), list(saved_orn), physicsClientId=cli)

        return False, None

    def _update_moving_platform(self):
        """Update platform position with obstacle avoidance."""
        nominal_pos = self._calculate_platform_position(self._time_alive)

        if not self._moving:
            self._prev_platform_pos = nominal_pos.copy()
            self._current_platform_pos = nominal_pos
            self._platform_velocity = np.zeros(3, dtype=np.float32)
            return

        current = self._current_platform_pos
        if current is None:
            current = nominal_pos

        blocked, _ = self._platform_path_blocked(current, nominal_pos)

        if not blocked or not PLATFORM_AVOIDANCE_ENABLED:
            new_pos = nominal_pos
        else:
            direction = nominal_pos - current
            raw_dist = np.linalg.norm(direction[:2])
            step = np.clip(raw_dist * 0.3, PLATFORM_MIN_STEP_M, 0.10)
            base_angle = math.atan2(direction[1], direction[0])

            new_pos = current.copy()
            for angle_deg in PLATFORM_STEER_ANGLES:
                angle = base_angle + math.radians(angle_deg)
                candidate = current.copy()
                candidate[0] += step * math.cos(angle)
                candidate[1] += step * math.sin(angle)
                candidate[2] = nominal_pos[2]

                candidate_blocked, _ = self._platform_path_blocked(current, candidate)
                if not candidate_blocked:
                    new_pos = candidate
                    break

        center = self._platform_orbit_center
        rel = new_pos[:2] - center[:2]
        r = np.linalg.norm(rel)
        r_max = self._platform_radius + 0.3
        if r > r_max:
            new_pos[:2] = center[:2] + rel * (r_max / max(r, 1e-6))

        if self._prev_platform_pos is not None:
            dt = self._sim_dt
            if dt > 0:
                self._platform_velocity = (new_pos - self._prev_platform_pos) / dt

        self._prev_platform_pos = new_pos.copy()
        self._current_platform_pos = new_pos

        if not hasattr(self, '_end_platform_uids') or not self._end_platform_uids:
            return

        cli = getattr(self, "CLIENT", 0)

        if not self._platform_offsets and self._end_platform_uids:
            initial_pos = self._platform_orbit_center
            for uid in self._end_platform_uids:
                pos, _ = p.getBasePositionAndOrientation(uid, physicsClientId=cli)
                offset = np.array(pos, dtype=np.float32) - initial_pos
                self._platform_offsets.append(offset)

        for i, uid in enumerate(self._end_platform_uids):
            offset = self._platform_offsets[i] if i < len(self._platform_offsets) else np.zeros(3)
            final_pos = new_pos + offset
            p.resetBasePositionAndOrientation(
                uid,
                final_pos.tolist(),
                [0, 0, 0, 1],
                physicsClientId=cli
            )

    def _getDroneImages(self, nth_drone, segmentation: bool = False):
        """Get camera images from drone. Returns (rgb, depth, seg) but we only use depth."""
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
            cameraUpVector=up.tolist(),
            physicsClientId=cli
        )
        
        aspect = self.IMG_RES[0] / self.IMG_RES[1]
        DRONE_CAM_PRO = p.computeProjectionMatrixFOV(
            fov=self._fov,
            aspect=aspect,
            nearVal=0.05,
            farVal=1000.0,
            physicsClientId=cli
        )
        
        SEG_FLAG = p.ER_NO_SEGMENTATION_MASK | p.ER_DEPTH_ONLY
        [w, h, _rgb, dep, _seg] = p.getCameraImage(
            width=self.IMG_RES[0],
            height=self.IMG_RES[1],
            shadow=0,
            renderer=p.ER_TINY_RENDERER,
            viewMatrix=DRONE_CAM_VIEW,
            projectionMatrix=DRONE_CAM_PRO,
            lightDirection=self._light_direction,
            flags=SEG_FLAG,
            physicsClientId=cli
        )
        
        dep = np.reshape(dep, (h, w))
        return None, dep, None

    def _get_altitude_distance(self) -> float:
        """Cast single ray downward for ground/altitude detection."""
        cli = getattr(self, "CLIENT", 0)
        uid = self.DRONE_IDS[0]
        pos, _ = p.getBasePositionAndOrientation(uid, physicsClientId=cli)
        pos = np.asarray(pos, dtype=float)

        ray_origin_offset = DRONE_HULL_RADIUS - ALTITUDE_RAY_INSET
        start = [pos[0], pos[1], pos[2] - ray_origin_offset]
        end = [pos[0], pos[1], pos[2] - MAX_RAY_DISTANCE]

        result = p.rayTest(start, end, physicsClientId=cli)
        hit_uid, _, hit_frac, _, _ = result[0]

        if hit_uid != -1:
            seg_len = MAX_RAY_DISTANCE - ray_origin_offset
            return min(MAX_RAY_DISTANCE, ray_origin_offset + hit_frac * seg_len)
        return MAX_RAY_DISTANCE

    def _process_depth(self, depth_buffer: np.ndarray) -> np.ndarray:
        """Convert PyBullet depth buffer to normalized depth map [0,1] for 0.5-20m range."""
        depth_buffer = np.clip(depth_buffer, 0.0, 1.0)
        
        denominator = DEPTH_FAR - (DEPTH_FAR - DEPTH_NEAR) * depth_buffer
        denominator = np.maximum(denominator, DEPTH_NEAR * 1e-6)
        
        depth_meters = DEPTH_FAR * DEPTH_NEAR / denominator
        depth_clipped = np.clip(depth_meters, DEPTH_MIN_M, DEPTH_MAX_M)
        depth_normalized = (depth_clipped - DEPTH_MIN_M) / (DEPTH_MAX_M - DEPTH_MIN_M)
        return depth_normalized.astype(np.float32)[..., np.newaxis]

    def _generate_search_area_center(self, seed: int = None) -> np.ndarray:
        """Generate search area center position with noise for GPS simulation."""
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = self.np_random
        search_radius = getattr(self.task, 'search_radius', 10.0)
        noise_xy = rng.uniform(-search_radius, search_radius, size=2)
        noise_z = rng.uniform(-SEARCH_AREA_NOISE_Z, SEARCH_AREA_NOISE_Z)
        center = self.GOAL_POS.copy()
        center[0] += noise_xy[0]
        center[1] += noise_xy[1]
        center[2] += noise_z
        return center

    def _check_collision(self) -> tuple:
        """
        Inspect contact points and detect collisions.
        Returns (platform_hit, obstacle_hit) tuple.
        """
        drone_id = self.DRONE_IDS[0]
        contact_points = p.getContactPoints(
            bodyA=drone_id,
            physicsClientId=getattr(self, "CLIENT", 0)
        )

        if not contact_points:
            return False, False

        end_platform_uids = getattr(self, '_end_platform_uids', [])
        start_platform_uids = getattr(self, '_start_platform_uids', [])

        platform_hit = False
        obstacle_hit = False

        for contact in contact_points:
            body_b = contact[2]
            if body_b == -1:
                continue

            normal_force = contact[9]
            if normal_force <= 0.01:
                continue

            if body_b in end_platform_uids:
                platform_hit = True
                continue

            if body_b in start_platform_uids:
                continue

            obstacle_hit = True
            break

        if obstacle_hit:
            self._collision = True

        return platform_hit, obstacle_hit

    def _update_landing_state(self, platform_contact: bool) -> None:
        """Update landing state machine based on contact and drone state."""
        if self._success or self._collision:
            return

        if not platform_contact:
            self._landing_stable_time = 0.0
            return

        if self._moving:
            self._success = True
            self._t_to_goal = self._time_alive
            return

        state = self._getDroneStateVector(0)
        roll, pitch = state[7], state[8]
        vel = state[10:13]

        vz = abs(vel[2])
        drone_vxy = vel[0:2]
        platform_vxy = self._platform_velocity[0:2]
        rel_vxy = np.linalg.norm(drone_vxy - platform_vxy)

        velocity_ok = vz <= LANDING_MAX_VZ and rel_vxy <= LANDING_MAX_VXY_REL
        upright_ok = abs(roll) <= LANDING_MAX_TILT_RAD and abs(pitch) <= LANDING_MAX_TILT_RAD

        if velocity_ok and upright_ok:
            self._landing_stable_time += self._sim_dt
            if self._landing_stable_time >= LANDING_STABLE_SEC:
                self._success = True
                self._t_to_goal = self._time_alive
        else:
            self._landing_stable_time = 0.0

    def _update_min_clearance(self) -> None:
        """Update minimum obstacle clearance for the episode."""
        if self._collision:
            self._min_clearance_episode = 0.0
            return

        cli = getattr(self, "CLIENT", 0)
        drone_id = self.DRONE_IDS[0]
        end_platform_uids = getattr(self, '_end_platform_uids', [])
        start_platform_uids = getattr(self, '_start_platform_uids', [])
        ground_id = getattr(self, 'PLANE_ID', 0)
        excluded = {drone_id, -1, ground_id} | set(end_platform_uids) | set(start_platform_uids)

        num_bodies = p.getNumBodies(physicsClientId=cli)
        min_dist = SAFETY_DISTANCE_SAFE

        for body_idx in range(num_bodies):
            body_uid = p.getBodyUniqueId(body_idx, physicsClientId=cli)
            if body_uid in excluded:
                continue

            closest = p.getClosestPoints(
                bodyA=drone_id,
                bodyB=body_uid,
                distance=SAFETY_DISTANCE_SAFE,
                physicsClientId=cli
            )

            for point in closest:
                dist = point[8]
                if dist < min_dist:
                    min_dist = dist

        if min_dist < self._min_clearance_episode:
            self._min_clearance_episode = min_dist

    # --------------------------------------------------------------------- #
    # 3. OpenAI‑Gym API overrides
    # --------------------------------------------------------------------- #
    def reset(self, **kwargs):
        """Reset environment and internal state for a new episode."""
        seed = kwargs.get('seed', None)
        if seed is None:
            seed = getattr(self.task, 'map_seed', None)
        self._search_area_center = self._generate_search_area_center(seed=seed)
        
        obs, info = super().reset(**kwargs)

        self._time_alive = 0.0
        self._success = False
        self._collision = False
        self._t_to_goal = None
        self._step_processed = False
        self._min_clearance_episode = SAFETY_DISTANCE_SAFE
        self._landing_stable_time = 0.0
        self._prev_platform_pos = None
        self._platform_velocity = np.zeros(3, dtype=np.float32)
        self._platform_offsets = []

        self._prev_score = flight_reward(
            success=False,
            t=0.0,
            horizon=self.EP_LEN_SEC,
            task=None,
            min_clearance=self._min_clearance_episode,
            collision=self._collision,
        )

        self._spawn_task_world()
        
        obs_after = self._computeObs()
        if obs_after is not None and "state" in obs_after:
            actual_state_dim = obs_after["state"].shape[0]
            if actual_state_dim != self._state_dim:
                self._state_dim = actual_state_dim
                self.observation_space = spaces.Dict({
                    "depth": self.observation_space["depth"],
                    "state": spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(actual_state_dim,),
                        dtype=np.float32
                    ),
                })
        else:
            obs_after = obs

        return obs_after, info

    def step(self, action):
        """Execute one step. Per-step updates run exactly once here."""
        self._step_processed = False
        self._process_step_updates()
        return super().step(action)

    def _process_step_updates(self):
        """Handle platform movement, time increment, collision, landing, and clearance check."""
        if self._step_processed:
            return
        self._step_processed = True
        self._update_moving_platform()
        self._time_alive += self._sim_dt
        platform_hit, _ = self._check_collision()
        self._update_landing_state(platform_hit)
        self._update_min_clearance()

    def _spawn_task_world(self):
        """Rebuild the procedural world defined by self.task."""

        cli = getattr(self, "CLIENT", 0)
        result = build_world(
            seed=self.task.map_seed,
            cli=cli,
            start=self.task.start,
            goal=self.task.goal,
            challenge_type=self.task.challenge_type,
        )

        if len(result) == 4:
            end_platform_uids, start_platform_uids, start_surface_z, goal_surface_z = result
        else:
            end_platform_uids, start_platform_uids = result
            start_surface_z = None
            goal_surface_z = None

        self._end_platform_uids = end_platform_uids if end_platform_uids else []
        self._start_platform_uids = start_platform_uids if start_platform_uids else []

        start_xyz = np.array(self.task.start, dtype=float)

        if start_surface_z is not None:
            self.task.start = (
                self.task.start[0],
                self.task.start[1],
                start_surface_z + START_PLATFORM_TAKEOFF_BUFFER
            )
            start_xyz[2] = start_surface_z + START_PLATFORM_TAKEOFF_BUFFER
            self.GOAL_POS = np.array(self.task.goal, dtype=float)

        if goal_surface_z is not None:
            self.task.goal = (
                self.task.goal[0],
                self.task.goal[1],
                goal_surface_z
            )
            self.GOAL_POS[2] = goal_surface_z

        self._platform_orbit_center = self.GOAL_POS.copy()
        self._current_platform_pos = self.GOAL_POS.copy()
        self._prev_platform_pos = None
        self._platform_offsets = []

        start_quat = p.getQuaternionFromEuler([0.0, 0.0, 0.0])

        p.resetBasePositionAndOrientation(
            self.DRONE_IDS[0],
            start_xyz,
            start_quat,
            physicsClientId=cli,
        )

    # -------- reward ----------------------------------------------------- #
    def _computeReward(self) -> float:
        """Compute incremental reward based on current state."""
        score = flight_reward(
            success=self._success,
            t=(self._t_to_goal if self._success else self._time_alive),
            horizon=self.EP_LEN_SEC,
            task=None,
            min_clearance=self._min_clearance_episode,
            collision=self._collision,
        )

        r_t = score - self._prev_score
        self._prev_score = score
        return float(r_t)

    # -------- termination ------------------------------------------------ #
    def _computeTerminated(self) -> bool:
        """Return True if episode ended via collision or goal reached."""
        return self._collision or self._success

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
            "distance_to_goal"    : dist,
            "score"               : self._prev_score,
            "success"             : self._success,
            "collision"           : self._collision,
            "t_to_goal"           : self._t_to_goal,
            "min_clearance"       : self._min_clearance_episode,
            "landing_stable_time" : self._landing_stable_time,
        }

    # -------- observation extension -------------------------------------- #
    def _computeObs(self):
        """
        Build depth + state observation for the single-drone task.
        Optimized: calls _getDroneImages directly, skips parent class overhead.
        """
        # Get depth directly (skip parent class which would also store rgb/seg)
        _, depth_raw, _ = self._getDroneImages(0)

        if depth_raw is None:
            h, w = (self.IMG_RES[1], self.IMG_RES[0]) if self.IMG_RES is not None else (128, 128)
            state_dim = getattr(self, "_state_dim", 115)
            return {
                "depth": np.zeros((h, w, 1), dtype=np.float32),
                "state": np.zeros((state_dim,), dtype=np.float32),
            }

        # Store in self.dep for compatibility
        self.dep[0] = depth_raw
        depth = self._process_depth(depth_raw)

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

        altitude = self._get_altitude_distance() / MAX_RAY_DISTANCE
        state_full = np.append(state_full, altitude).astype(np.float32)

        drone_pos = state_vec[0:3]
        search_area_vector = (self._search_area_center - drone_pos).astype(np.float32)
        state_full = np.append(state_full, search_area_vector).astype(np.float32)

        actual_state_dim = state_full.shape[0]
        if actual_state_dim != self._state_dim:
            self._state_dim = actual_state_dim
            self.observation_space = spaces.Dict({
                "depth": self.observation_space["depth"],
                "state": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(actual_state_dim,),
                    dtype=np.float32
                ),
            })

        return {
            "depth": depth,
            "state": state_full,
        }
