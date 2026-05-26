# swarm/envs/moving_drone.py
from __future__ import annotations

import functools
import math
import os
import numpy as np
import gymnasium.spaces as spaces
import pybullet as p
from PIL import Image

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import (
    DroneModel, Physics, ActionType, ObservationType, ImageType,
)

from swarm.challenge_families import evaluate_rollout, runtime_family_for_task
from swarm.constants import (
    DRONE_HULL_RADIUS, ALTITUDE_RAY_INSET, MAX_RAY_DISTANCE,
    DEPTH_NEAR, DEPTH_FAR, DEPTH_MIN_M, DEPTH_MAX_M,
    CAMERA_FOV_BASE, CAMERA_FOV_VARIANCE,
    LIGHT_RANDOMIZATION_ENABLED,
    SAFETY_DISTANCE_SAFE,
    SAFETY_DISTANCE_SAFE_BY_TYPE,
    START_PLATFORM_TAKEOFF_BUFFER,
    CULL_VISUAL_RADIUS, CULL_PHYSICS_RADIUS, CULL_INTERVAL_STEPS,
    CULL_MIN_AABB_SPAN, CULL_MIN_FACES, CULL_MIN_TOTAL_FACES,
    SOLVER_ITERATIONS, SOLVER_MIN_ISLAND_SIZE,
)


@functools.lru_cache(maxsize=4096)
def _count_obj_faces_cached(path: str, mtime_ns: int, size: int) -> int:
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        return 0
    return data.count(b"\nf ") + (1 if data.startswith(b"f ") else 0)


def _inside_safety_patch(contact_point, safety_patch) -> bool:
    cx, cy = safety_patch.xy
    dx = float(contact_point[0]) - float(cx)
    dy = float(contact_point[1]) - float(cy)
    horiz = math.hypot(dx, dy)
    if horiz > safety_patch.radius:
        return False
    cz = float(contact_point[2])
    z_low = float(safety_patch.surface_z) - float(safety_patch.z_below)
    z_high = float(safety_patch.surface_z) + float(safety_patch.z_above)
    return z_low <= cz <= z_high


class MovingDroneAviary(BaseRLAviary):
    """
    Single‑drone environment whose *start*, *goal* and *horizon* are supplied
    via an external `MapTask`.

    The per-step reward is the incremental change in the family-owned rollout
    score so training loops can consume it without additional shaping.
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
        sar_mode    : bool          = False,
    ):
        """
        Parameters
        ----------
        task : MapTask
            Must expose `.start`, `.goal`, `.horizon`, `.sim_dt`.
        sar_mode : bool
            Backward-compatible family runtime hint. The active challenge
            family may normalize or ignore it.
        """
        self.task       = task
        self._original_start = tuple(task.start)
        self._original_goal = tuple(task.goal)
        self.GOAL_POS   = np.asarray(task.goal, dtype=float)
        self.EP_LEN_SEC = float(task.horizon)
        self.family_runtime = runtime_family_for_task(task)
        self.sar_mode = bool(sar_mode)

        self._time_alive = 0.0
        self._success = False
        self._collision = False
        self._t_to_goal = None
        self._prev_score = 0.0
        self._step_processed = False
        self._min_clearance_episode = SAFETY_DISTANCE_SAFE

        from swarm.protocol import FailureReason
        self._failure_reason = FailureReason.NONE.value

        seed = getattr(task, 'map_seed', 0)
        self._search_area_center = self.GOAL_POS.copy()
        self.family_runtime.initialise_env_state(
            self,
            requested_mode=bool(sar_mode),
        )

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
        clue_dim = int(self.family_runtime.state_clue_dim(task))
        state_dim = 12 + self.ACTION_BUFFER_SIZE * action_dim + 1 + clue_dim
        self._state_dim = state_dim
        self._clue_dim = clue_dim

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

        self._cull_targets = []
        self._cull_vis_hidden = set()
        self._cull_phys_disabled = set()
        self._cull_step_counter = 0
        self._cull_enabled = False

        self._cached_proj_matrix = None

    # --------------------------------------------------------------------- #
    # 2. low‑level helpers
    # --------------------------------------------------------------------- #
    @property
    def _sim_dt(self) -> float:
        """Physics step in seconds (1 / CTRL_FREQ)."""
        return 1.0 / self.CTRL_FREQ
    
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
        
        camera_offset = 0.13
        camera_pos = drone_pos + forward * camera_offset + up * 0.05
        
        target = camera_pos + forward * 20.0
        
        DRONE_CAM_VIEW = p.computeViewMatrix(
            cameraEyePosition=camera_pos,
            cameraTargetPosition=target,
            cameraUpVector=up.tolist(),
            physicsClientId=cli
        )
        
        DRONE_CAM_PRO = self._cached_proj_matrix
        if DRONE_CAM_PRO is None:
            aspect = self.IMG_RES[0] / self.IMG_RES[1]
            DRONE_CAM_PRO = p.computeProjectionMatrixFOV(
                fov=self._fov,
                aspect=aspect,
                nearVal=0.05,
                farVal=DEPTH_FAR,
                physicsClientId=cli
            )
            self._cached_proj_matrix = DRONE_CAM_PRO

        seg_flag = p.ER_NO_SEGMENTATION_MASK
        depth_only_flag = getattr(p, "ER_DEPTH_ONLY", None)
        if depth_only_flag is not None:
            seg_flag |= depth_only_flag
        [w, h, _rgb, dep, _seg] = p.getCameraImage(
            width=self.IMG_RES[0],
            height=self.IMG_RES[1],
            shadow=0,
            renderer=p.ER_TINY_RENDERER,
            viewMatrix=DRONE_CAM_VIEW,
            projectionMatrix=DRONE_CAM_PRO,
            lightDirection=self._light_direction,
            flags=seg_flag,
            physicsClientId=cli
        )
        
        dep = np.reshape(dep, (h, w))
        return None, dep, None

    def _get_altitude_distance(self) -> float:
        """Cast single ray downward for ground/altitude detection."""
        cli = getattr(self, "CLIENT", 0)
        pos = self.pos[0]

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

    def _check_collision(self) -> tuple:
        """Inspect contact points; sets ``_collision`` on any non-ground impact.

        Returns a ``(False, obstacle_hit)`` tuple. Touching a mannequin part
        (victim) is treated as an obstacle hit — the no-touch sphere governs
        the CONFIRMED predicate, not contact handling.
        """
        drone_id = self.DRONE_IDS[0]
        contact_points = p.getContactPoints(
            bodyA=drone_id,
            physicsClientId=getattr(self, "CLIENT", 0)
        )

        if not contact_points:
            return False, False

        obstacle_hit = False

        for contact in contact_points:
            body_b = contact[2]
            if body_b == -1:
                continue

            normal_force = contact[9]
            if normal_force <= 0.01:
                continue

            obstacle_hit = True
            break

        if obstacle_hit:
            self._collision = True

        return False, obstacle_hit


    @staticmethod
    def _count_mesh_faces(path: str) -> int:
        try:
            st = os.stat(path)
        except OSError:
            return 0
        return _count_obj_faces_cached(path, st.st_mtime_ns, st.st_size)

    def _build_cull_targets(self) -> None:
        """Scan scene bodies and build the cull-target list."""
        cli = getattr(self, "CLIENT", 0)
        drone_id = self.DRONE_IDS[0]
        ground_id = getattr(self, "PLANE_ID", 0)
        protected = (
            {drone_id, ground_id}
            | set(self.family_runtime.protected_body_uids(self))
        )

        targets = []
        total_faces = 0
        n = p.getNumBodies(physicsClientId=cli)

        for i in range(n):
            uid = p.getBodyUniqueId(i, physicsClientId=cli)
            if uid in protected:
                continue
            mn, mx = p.getAABB(uid, physicsClientId=cli)
            span = max(mx[0] - mn[0], mx[1] - mn[1])
            if span < CULL_MIN_AABB_SPAN:
                continue
            vdata = p.getVisualShapeData(uid, physicsClientId=cli)
            if not vdata:
                continue
            faces = 0
            for v in vdata:
                if v[2] == p.GEOM_MESH:
                    fname = v[4].decode() if isinstance(v[4], bytes) else str(v[4])
                    faces += self._count_mesh_faces(fname)
            if faces < CULL_MIN_FACES:
                continue
            cx = (mn[0] + mx[0]) * 0.5
            cy = (mn[1] + mx[1]) * 0.5
            rgba_orig = list(vdata[0][7])
            targets.append((uid, cx, cy, span / 2.0, rgba_orig))
            total_faces += faces

        self._cull_targets = targets
        self._cull_vis_hidden = set()
        self._cull_phys_disabled = set()
        self._cull_step_counter = 0
        self._cull_enabled = (not getattr(self, "GUI", False)) and total_faces >= CULL_MIN_TOTAL_FACES

    def _apply_distance_cull(self) -> None:
        """Toggle visual/physics state for bodies beyond camera range."""
        if getattr(self, "GUI", False):
            if self._cull_vis_hidden or self._cull_phys_disabled:
                self._restore_culled_bodies()
            return
        if not self._cull_enabled:
            return
        self._cull_step_counter += 1
        if self._cull_step_counter % CULL_INTERVAL_STEPS != 0:
            return

        cli = getattr(self, "CLIENT", 0)
        dp = p.getBasePositionAndOrientation(self.DRONE_IDS[0], physicsClientId=cli)[0]
        dx, dy = dp[0], dp[1]
        vis_hidden = self._cull_vis_hidden
        phys_disabled = self._cull_phys_disabled

        for uid, cx, cy, hs, rgba in self._cull_targets:
            dist = math.sqrt((cx - dx) ** 2 + (cy - dy) ** 2)
            surface_dist = dist - hs

            if surface_dist > CULL_VISUAL_RADIUS:
                if uid not in vis_hidden:
                    p.changeVisualShape(uid, -1, rgbaColor=[0, 0, 0, 0], physicsClientId=cli)
                    vis_hidden.add(uid)
            elif uid in vis_hidden:
                p.changeVisualShape(uid, -1, rgbaColor=rgba, physicsClientId=cli)
                vis_hidden.discard(uid)

            if surface_dist > CULL_PHYSICS_RADIUS:
                if uid not in phys_disabled:
                    p.setCollisionFilterGroupMask(uid, -1, 0, 0, physicsClientId=cli)
                    phys_disabled.add(uid)
            elif uid in phys_disabled:
                p.setCollisionFilterGroupMask(uid, -1, 1, 0xFF, physicsClientId=cli)
                phys_disabled.discard(uid)

    def _restore_culled_bodies(self) -> None:
        """Restore all culled bodies to their original state."""
        cli = getattr(self, "CLIENT", 0)
        for uid, _, _, _, rgba in self._cull_targets:
            if uid in self._cull_vis_hidden:
                p.changeVisualShape(uid, -1, rgbaColor=rgba, physicsClientId=cli)
            if uid in self._cull_phys_disabled:
                p.setCollisionFilterGroupMask(uid, -1, 1, 0xFF, physicsClientId=cli)
        self._cull_vis_hidden.clear()
        self._cull_phys_disabled.clear()


    def _update_min_clearance(self) -> None:
        """Update minimum obstacle clearance for the episode."""
        if self._collision:
            self._min_clearance_episode = 0.0
            return

        cli = getattr(self, "CLIENT", 0)
        drone_id = self.DRONE_IDS[0]
        ground_id = getattr(self, 'PLANE_ID', 0)
        excluded = {drone_id, -1, ground_id}

        excluded |= set(self.family_runtime.protected_body_uids(self))
        safety_patch = self.family_runtime.safety_patch(self)

        min_dist = SAFETY_DISTANCE_SAFE

        d_min, d_max = p.getAABB(drone_id, physicsClientId=cli)
        search_min = [d_min[0] - SAFETY_DISTANCE_SAFE, d_min[1] - SAFETY_DISTANCE_SAFE, d_min[2] - SAFETY_DISTANCE_SAFE]
        search_max = [d_max[0] + SAFETY_DISTANCE_SAFE, d_max[1] + SAFETY_DISTANCE_SAFE, d_max[2] + SAFETY_DISTANCE_SAFE]
        overlapping = p.getOverlappingObjects(search_min, search_max, physicsClientId=cli)

        if overlapping:
            drone_pos = self.pos[0, :]
            checked = set()
            for body_uid, _link_idx in overlapping:
                if body_uid in excluded or body_uid in checked:
                    continue
                checked.add(body_uid)
                closest = p.getClosestPoints(
                    bodyA=drone_id,
                    bodyB=body_uid,
                    distance=SAFETY_DISTANCE_SAFE,
                    physicsClientId=cli
                )

                for point in closest:
                    contact = point[6]
                    if (
                        safety_patch is not None
                        and body_uid == safety_patch.support_uid
                        and _inside_safety_patch(contact, safety_patch)
                    ):
                        continue
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

        p.resetSimulation(physicsClientId=self.CLIENT)
        self._housekeeping()
        self._updateAndStoreKinematicInformation()
        self._startVideoRecording()

        self._time_alive = 0.0
        self._success = False
        self._collision = False
        self._t_to_goal = None
        self._step_processed = False
        self._min_clearance_episode = SAFETY_DISTANCE_SAFE

        from swarm.protocol import FailureReason
        self._failure_reason = FailureReason.NONE.value
        self.family_runtime.reset_env_state(self)

        self._reset_action_buffer()

        self._prev_score = evaluate_rollout(
            task=self.task,
            success=False,
            t=0.0,
            horizon=self.EP_LEN_SEC,
            min_clearance=self._min_clearance_episode,
            collision=self._collision,
            legitimate_model=True,
            failure_reason=self._failure_reason,
        ).score

        self.family_runtime.spawn_task_world(self)
        self._updateAndStoreKinematicInformation()

        cli = getattr(self, "CLIENT", 0)
        p.setPhysicsEngineParameter(
            numSolverIterations=SOLVER_ITERATIONS,
            minimumSolverIslandSize=SOLVER_MIN_ISLAND_SIZE,
            physicsClientId=cli,
        )

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
        info_after = self._computeInfo()
        return obs_after, info_after

    def step(self, action):
        """Execute one control step with post-physics bookkeeping."""
        self._step_processed = False
        if self.RECORD and not self.GUI and self.step_counter % self.CAPTURE_FREQ == 0:
            [w, h, rgb, dep, seg] = p.getCameraImage(
                width=self.VID_WIDTH,
                height=self.VID_HEIGHT,
                shadow=1,
                viewMatrix=self.CAM_VIEW,
                projectionMatrix=self.CAM_PRO,
                renderer=p.ER_TINY_RENDERER,
                flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                physicsClientId=self.CLIENT,
            )
            (Image.fromarray(np.reshape(rgb, (h, w, 4)), 'RGBA')).save(
                os.path.join(self.IMG_PATH, "frame_" + str(self.FRAME_NUM) + ".png")
            )
            self.FRAME_NUM += 1
            if self.VISION_ATTR:
                for i in range(self.NUM_DRONES):
                    self.rgb[i], self.dep[i], self.seg[i] = self._getDroneImages(i)
                    self._exportImage(
                        img_type=ImageType.RGB,
                        img_input=self.rgb[i],
                        path=self.ONBOARD_IMG_PATH + "/drone_" + str(i) + "/",
                        frame_num=int(self.step_counter / self.IMG_CAPTURE_FREQ),
                    )
        if self.GUI and self.USER_DEBUG:
            current_input_switch = p.readUserDebugParameter(
                self.INPUT_SWITCH,
                physicsClientId=self.CLIENT,
            )
            if current_input_switch > self.last_input_switch:
                self.last_input_switch = current_input_switch
                self.USE_GUI_RPM = not self.USE_GUI_RPM
        if self.USE_GUI_RPM:
            for i in range(4):
                self.gui_input[i] = p.readUserDebugParameter(
                    int(self.SLIDERS[i]),
                    physicsClientId=self.CLIENT,
                )
            clipped_action = np.tile(self.gui_input, (self.NUM_DRONES, 1))
            if self.step_counter % (self.PYB_FREQ / 2) == 0:
                self.GUI_INPUT_TEXT = [
                    p.addUserDebugText(
                        "Using GUI RPM",
                        textPosition=[0, 0, 0],
                        textColorRGB=[1, 0, 0],
                        lifeTime=1,
                        textSize=2,
                        parentObjectUniqueId=self.DRONE_IDS[i],
                        parentLinkIndex=-1,
                        replaceItemUniqueId=int(self.GUI_INPUT_TEXT[i]),
                        physicsClientId=self.CLIENT,
                    ) for i in range(self.NUM_DRONES)
                ]
        else:
            clipped_action = np.reshape(
                self._preprocessAction(action),
                (self.NUM_DRONES, 4),
            )
        for _ in range(self.PYB_STEPS_PER_CTRL):
            if (
                self.PYB_STEPS_PER_CTRL > 1
                and self.PHYSICS in [
                    Physics.DYN,
                    Physics.PYB_GND,
                    Physics.PYB_DRAG,
                    Physics.PYB_DW,
                    Physics.PYB_GND_DRAG_DW,
                ]
            ):
                self._updateAndStoreKinematicInformation()
            for i in range(self.NUM_DRONES):
                if self.PHYSICS == Physics.PYB:
                    self._physics(clipped_action[i, :], i)
                elif self.PHYSICS == Physics.DYN:
                    self._dynamics(clipped_action[i, :], i)
                elif self.PHYSICS == Physics.PYB_GND:
                    self._physics(clipped_action[i, :], i)
                    self._groundEffect(clipped_action[i, :], i)
                elif self.PHYSICS == Physics.PYB_DRAG:
                    self._physics(clipped_action[i, :], i)
                    self._drag(self.last_clipped_action[i, :], i)
                elif self.PHYSICS == Physics.PYB_DW:
                    self._physics(clipped_action[i, :], i)
                    self._downwash(i)
                elif self.PHYSICS == Physics.PYB_GND_DRAG_DW:
                    self._physics(clipped_action[i, :], i)
                    self._groundEffect(clipped_action[i, :], i)
                    self._drag(self.last_clipped_action[i, :], i)
                    self._downwash(i)
            if self.PHYSICS != Physics.DYN:
                p.stepSimulation(physicsClientId=self.CLIENT)
            self.last_clipped_action = clipped_action
        self._updateAndStoreKinematicInformation()
        self._process_step_updates()
        obs = self._computeObs()
        reward = self._computeReward()
        terminated = self._computeTerminated()
        truncated = self._computeTruncated()
        info = self._computeInfo()
        self.step_counter = self.step_counter + (1 * self.PYB_STEPS_PER_CTRL)
        return obs, reward, terminated, truncated, info

    def _process_step_updates(self):
        """Handle post-physics episode bookkeeping exactly once per control step."""
        if self._step_processed:
            return
        self._step_processed = True
        self._time_alive += self._sim_dt
        self._check_collision()
        self._family_post_step_update()
        self._update_min_clearance()
        self._apply_distance_cull()

    def _family_post_step_update(self) -> None:
        self.family_runtime.post_step_update(self)

    def _legacy_sar_runtime(self):
        return self.family_runtime

    def _sar_drone_state(self):
        return self._legacy_sar_runtime().legacy_sar_drone_state(self)

    def _sar_check_predicate(self) -> bool:
        return self._legacy_sar_runtime().legacy_sar_check_predicate(self)

    def _sar_step_update(self) -> None:
        self._legacy_sar_runtime().legacy_sar_step_update(self)

    def _reset_action_buffer(self) -> None:
        """Zero the action history so reset observations do not leak prior episodes."""
        action_dim = int(self.action_space.shape[-1])
        self.action_buffer.clear()
        for _ in range(self.ACTION_BUFFER_SIZE):
            self.action_buffer.append(
                np.zeros((self.NUM_DRONES, action_dim), dtype=np.float32)
            )

    def _spawn_task_world(self):
        """Backward-compatible SAR wrapper retained for legacy tests."""
        self.family_runtime.spawn_task_world(self)

    # -------- reward ----------------------------------------------------- #
    def _computeReward(self) -> float:
        """Compute incremental reward based on current state."""
        evaluation = evaluate_rollout(
            task=self.task,
            success=self._success,
            t=(self._t_to_goal if self._success else self._time_alive),
            horizon=self.EP_LEN_SEC,
            min_clearance=self._min_clearance_episode,
            collision=self._collision,
            legitimate_model=True,
            failure_reason=getattr(self, "_failure_reason", "NONE"),
        )

        reward = self.family_runtime.compute_training_reward(
            env=self,
            evaluation=evaluation,
            previous_score=float(self._prev_score),
        )
        self._prev_score = float(evaluation.score)
        return float(reward)

    # -------- termination ------------------------------------------------ #
    def _computeTerminated(self) -> bool:
        """Return True if episode ended via collision or goal reached."""
        if self.family_runtime.compute_terminated(self):
            return True
        return self._collision or self._success

    # -------- truncation (timeout / safety) ------------------------------ #
    def _computeTruncated(self) -> bool:
        """Early termination through the active family runtime."""
        from swarm.protocol import FailureReason

        terminal_already = (
            self._collision
            or self._success
            or self._failure_reason != FailureReason.NONE.value
        )

        state = self._getDroneStateVector(0)
        roll, pitch = state[7], state[8]
        return self.family_runtime.compute_truncated(
            self,
            terminal_already=terminal_already,
            roll=float(roll),
            pitch=float(pitch),
        )

    def _sar_infeasible(self) -> bool:
        return self._legacy_sar_runtime().legacy_sar_infeasible(self)

    # -------- extra logging --------------------------------------------- #
    def _computeInfo(self):
        state = self._getDroneStateVector(0)
        dist  = float(np.linalg.norm(state[0:3] - self.GOAL_POS))
        info = {
            "distance_to_goal"    : dist,
            "score"               : self._prev_score,
            "success"             : self._success,
            "collision"           : self._collision,
            "t_to_goal"           : self._t_to_goal,
            "min_clearance"       : self._min_clearance_episode,
            "failure_reason"      : getattr(self, "_failure_reason", "NONE"),
        }
        info.update(self.family_runtime.build_info(self))
        return info

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

        action_dim = self.action_buffer[0].shape[1] if self.ACTION_BUFFER_SIZE > 0 else 0
        base_len = 12 + self.ACTION_BUFFER_SIZE * action_dim
        clue_dim = getattr(self, "_clue_dim", 3)
        state_full = np.empty(base_len + 1 + clue_dim, dtype=np.float32)

        state_full[0:3] = state_vec[0:3]
        state_full[3:6] = state_vec[7:10]
        state_full[6:9] = state_vec[10:13]
        state_full[9:12] = state_vec[13:16]

        offset = 12
        for i in range(self.ACTION_BUFFER_SIZE):
            state_full[offset:offset + action_dim] = self.action_buffer[i][0, :]
            offset += action_dim

        state_full[base_len] = self._get_altitude_distance() / MAX_RAY_DISTANCE
        clue_offset = np.asarray(
            self.family_runtime.clue_offset(self, state_vec),
            dtype=np.float32,
        )
        state_full[base_len + 1:base_len + 1 + clue_dim] = clue_offset[:clue_dim]

        return {
            "depth": depth,
            "state": state_full,
        }
