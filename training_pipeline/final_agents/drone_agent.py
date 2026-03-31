"""
Swarm Subnet 124 - Autonomous Drone Flight Controller (standalone).
Self-contained: all logic inlined (movement, depth, search area, goal detection). No external modules.
"""

import numpy as np
from collections import deque
from scipy.ndimage import minimum_filter, maximum_filter


class DroneFlightController:
    """
    Flight controller: state machine (0=move to search area, 1=search, 3=goal).
    All observation parsing, depth, movement, search area, and goal detection
    are inlined.
    """

    # Movement / speed
    MAX_SPEED_M_S = 1
    ACCELERATION_M_S2 = 1.5
    DECELERATION_START_DISTANCE = 1.5
    AT_TARGET_TOLERANCE = 0.5
    AT_TARGET_SPEED_THRESHOLD = 0.5
    SIM_DT = 1.0 / 50.0
    MAX_VELOCITY_CHANGE_NORM_M_S = 1.0
    RISE_HEIGHT_M = 0
    HORIZONTAL_M = 1.2
    MAX_SAFE_ROLL_PITCH_RAD = 0.3
    ACCEL_SCALE_AT_HIGH_TILT = 0.7
    MAX_SAFE_ANGULAR_VEL_RAD_S = 0.2
    ACCEL_SCALE_AT_HIGH_ANG_VEL = 0.7
    YAW_FACE_DIRECTION_MIN_DISTANCE_M = 0.5
    MAX_APPROACH_ANGLE_RAD = 0.15
    SPEED_SCALE_AT_LARGE_ANGLE = 0.6
    # Search area
    SEARCH_AREA_RADIUS_X_M = 10.0
    SEARCH_AREA_RADIUS_Y_M = 10.0
    SEARCH_AREA_RADIUS_Z_M = 2.0
    # Camera / depth
    CAMERA_FOV_RAD = 0.5 * 3.141592
    FRONT_PATCH_ROWS = 3
    FRONT_PATCH_COLS = 3
    FRONT_PATCH_RADIUS_PX = 3
    FRONT_OBSTACLE_MIN_DEPTH_M = 5.0
    DETOUR_WAYPOINT_SMOOTHING_PREV_WEIGHT = 0.5
    DANGER_DISTANCE_M = 1
    STATE1_DANGER_STEPS_WINDOW = 20
    GOAL_PLATFORM_MAX_EXTENT_M = 0.8  # max distance from component mean to any pixel
    DETOUR_MIN_CLEAR_DISTANCE_PX = 5
    CLEAR_DIRECTION_MIN_DEPTH_M = 1.0
    # 1m height error interval when keeping height for open path
    MAX_VERTICAL_COMPONENT_HORIZONTAL = 1.0 / (2.0 ** 0.5)  # ~0.707, ~45°
    CLEAR_DIRECTION_MIN_MARGIN_M = 0.5
    # Goal detection
    DEPTH_EDGE_JUMP = 0.08
    DEPTH_SIMILARITY_THRESHOLD_M = 0.1
    DEPTH_RANGE_NEAR_M = 0.5
    DEPTH_RANGE_FAR_M = 20.0
    GOAL_PLATFORM_MAX_HEIGHT_DIFF_M = 0.5
    MIN_COMPONENT_SIZE = 4
    BAD_PIXEL_DEPTH_THRESHOLD_M = 0.5
    GOOD_PIXEL_WINDOW_SIZE = 5
    GOAL_AVERAGE_POSITION_DISTANCE_M = 3.0
    GOAL_NEAR_FIXED_DISTANCE_M = 2.0
    NEAR_GOAL_DEPTH_MAX_M = 1.0
    # State machine
    STATE1_TURN_YAW_STEP_NORM = 0.12
    STATE1_TURN_270_TARGET_RAD = 1.5  # 270° in rad/pi units
    STATE2_YAW_STEP_RAD = 0.08

    def __init__(self):
        self.state = 0
        self.target = []
        self.search_area_center = None
        self.initial_position = None
        self.detected_goal_position = None
        self.detected_goal_direction = None
        self.state1_turning_270 = False
        self.state1_turn_270_prev_yaw = None
        self.state1_turn_270_swept_rad = 0.0
        self.state1_turn_direction = None
        self.state1_seeking_clear_direction = False
        self.state1_chosen_clear_direction = None
        self.state1_has_detour_waypoint = False
        self.state1_danger_steps_remaining = 0
        self.state3_no_goal_steps = 0
        self.STATE3_NO_GOAL_LOOK_STEPS = 50
        self.act_step = 0
        self.MIN_STEPS_BEFORE_STATE3 = 0
        self.goal_detection_count = 0
        self.MIN_GOAL_DETECTIONS_BEFORE_STATE3 = 10

    # ---------- Observation / state (inlined state_utils) ----------
    def _get_current_position(self, observation):
        if isinstance(observation, dict) and "state" in observation:
            state = observation["state"]
            return np.asarray(state[0:3], dtype=np.float32)
        if hasattr(observation, "shape") and len(observation.shape) == 1 and observation.shape[0] >= 3:
            return np.asarray(observation[0:3], dtype=np.float32)
        return np.zeros(3, dtype=np.float32)

    def _get_current_velocity(self, observation):
        if isinstance(observation, dict) and "state" in observation:
            state = observation["state"]
            if len(state) >= 9:
                return np.asarray(state[6:9], dtype=np.float32)
        return np.zeros(3, dtype=np.float32)

    def _get_roll_pitch(self, observation):
        if isinstance(observation, dict) and "state" in observation:
            state = observation["state"]
            if len(state) >= 5:
                return float(state[3]), float(state[4])
        return 0.0, 0.0

    def _get_yaw(self, observation):
        if isinstance(observation, dict) and "state" in observation:
            state = observation["state"]
            if len(state) >= 6:
                return float(state[5])
        return None

    def _get_angular_velocity(self, observation):
        if isinstance(observation, dict) and "state" in observation:
            state = observation["state"]
            if len(state) >= 12:
                return np.asarray(state[9:12], dtype=np.float32)
        return np.zeros(3, dtype=np.float32)

    def _get_search_area_vector(self, observation):
        if isinstance(observation, dict) and "state" in observation:
            state = observation["state"]
            if len(state) >= 3:
                return np.asarray(state[-3:], dtype=np.float32)
        return np.zeros(3, dtype=np.float32)

    # ---------- Depth / camera (inlined depth_utils) ----------
    def _direction_world_to_pixel(self, observation, direction_world, camera_fov_rad=None):
        if camera_fov_rad is None:
            camera_fov_rad = self.CAMERA_FOV_RAD
        if not isinstance(observation, dict) or "depth" not in observation:
            return None
        depth = np.squeeze(np.asarray(observation["depth"], dtype=np.float32))
        if depth.ndim != 2:
            return None
        H, W = depth.shape
        roll, pitch = self._get_roll_pitch(observation)
        yaw = self._get_yaw(observation)
        if yaw is None:
            yaw = 0.0
        dir_w = np.asarray(direction_world, dtype=np.float32).reshape(3)
        dn = np.linalg.norm(dir_w)
        if dn < 1e-9:
            return None
        dir_w = dir_w / dn
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float32)
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float32)
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float32)
        R = Rz @ Ry @ Rx
        ray_body = (R.T @ dir_w).astype(np.float32)
        if ray_body[0] <= 1e-6:
            return None
        half_fov = camera_fov_rad / 2.0
        horz = float(np.arctan2(-ray_body[1], ray_body[0]))
        vert = float(np.arctan2(-ray_body[2], ray_body[0]))
        c = (W - 1) / 2.0 + (W / 2.0) * (horz / half_fov)
        r = (H - 1) / 2.0 + (H / 2.0) * (vert / half_fov)
        r = int(np.clip(r, 0, H - 1))
        c = int(np.clip(c, 0, W - 1))
        return (r, c)

    def _pixel_to_direction_world(self, observation, r, c, camera_fov_rad=None):
        if camera_fov_rad is None:
            camera_fov_rad = self.CAMERA_FOV_RAD
        if not isinstance(observation, dict) or "depth" not in observation:
            return None
        depth = np.squeeze(np.asarray(observation["depth"], dtype=np.float32))
        if depth.ndim != 2:
            return None
        H, W = depth.shape
        roll, pitch = self._get_roll_pitch(observation)
        yaw = self._get_yaw(observation)
        if yaw is None:
            yaw = 0.0
        half_fov = camera_fov_rad / 2.0
        horz = (c - (W - 1) / 2.0) / (W / 2.0) * half_fov
        vert = (r - (H - 1) / 2.0) / (H / 2.0) * half_fov
        ray_body = np.array([1.0, -np.tan(horz), -np.tan(vert)], dtype=np.float32)
        n = np.linalg.norm(ray_body)
        if n < 1e-9:
            ray_body = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            ray_body = ray_body / n
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float32)
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float32)
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float32)
        R = Rz @ Ry @ Rx
        direction_world = (R @ ray_body).astype(np.float32)
        dn = float(np.linalg.norm(direction_world))
        if dn < 1e-9:
            return None
        return (direction_world / dn).astype(np.float32)

    def _get_look_direction_world(self, observation, camera_fov_rad=None):
        if camera_fov_rad is None:
            camera_fov_rad = self.CAMERA_FOV_RAD
        roll, pitch = self._get_roll_pitch(observation)
        yaw = self._get_yaw(observation)
        if yaw is None:
            yaw = 0.0
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float32)
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float32)
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float32)
        R = Rz @ Ry @ Rx
        direction_world = (R @ np.array([1.0, 0.0, 0.0], dtype=np.float32)).astype(np.float32)
        dn = float(np.linalg.norm(direction_world))
        if dn < 1e-9:
            return None
        return (direction_world / dn).astype(np.float32)

    def _depth_meters_for_direction(self, observation, direction_world, camera_fov_rad=None):
        if camera_fov_rad is None:
            camera_fov_rad = self.CAMERA_FOV_RAD
        pixel = self._direction_world_to_pixel(observation, direction_world, camera_fov_rad=camera_fov_rad)
        if pixel is None:
            return None
        if not isinstance(observation, dict) or "depth" not in observation:
            return None
        depth = np.squeeze(np.asarray(observation["depth"], dtype=np.float32))
        if depth.ndim != 2:
            return None
        r, c = pixel
        r = int(np.clip(r, 0, depth.shape[0] - 1))
        c = int(np.clip(c, 0, depth.shape[1] - 1))
        raw = float(depth[r, c])
        depth_m = 0.5 + 19.5 * np.clip(raw, 0.0, 1.0)
        return depth_m

    def _center_depth_clear_at_least_m(self, observation, min_clear_m, front_patch_radius_px=None):
        if front_patch_radius_px is None:
            front_patch_radius_px = self.FRONT_PATCH_RADIUS_PX
        if not isinstance(observation, dict) or "depth" not in observation:
            return False
        depth = np.squeeze(np.asarray(observation["depth"], dtype=np.float32))
        if depth.ndim != 2:
            return False
        H, W = depth.shape
        c0 = max(0, W // 2 - front_patch_radius_px)
        c1 = min(W, W // 2 + front_patch_radius_px + 1)
        min_raw = float(np.min(depth[H // 2, c0:c1]))
        return (0.5 + 19.5 * min_raw) >= min_clear_m

    def _front_obstacle_detected(
        self,
        observation,
        current_velocity,
        roll,
        pitch,
        yaw,
        front_obstacle_min_depth_m=None,
        front_patch_rows=None,
        front_patch_cols=None,
        camera_fov_rad=None,
    ):
        if front_obstacle_min_depth_m is None:
            front_obstacle_min_depth_m = self.FRONT_OBSTACLE_MIN_DEPTH_M
        if front_patch_rows is None:
            front_patch_rows = self.FRONT_PATCH_ROWS
        if front_patch_cols is None:
            front_patch_cols = self.FRONT_PATCH_COLS
        if camera_fov_rad is None:
            camera_fov_rad = self.CAMERA_FOV_RAD
        if not isinstance(observation, dict) or "depth" not in observation:
            return False
        depth = np.squeeze(np.asarray(observation["depth"], dtype=np.float32))
        if depth.ndim != 2:
            return False
        H, W = depth.shape
        vel = np.asarray(current_velocity, dtype=np.float32).reshape(3)
        speed = float(np.linalg.norm(vel))
        if speed > 0.1:
            direction_world = vel / speed
        else:
            cy, sy = np.cos(yaw), np.sin(yaw)
            cp, sp = np.cos(pitch), np.sin(pitch)
            cr, sr = np.cos(roll), np.sin(roll)
            Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float32)
            Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float32)
            Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float32)
            R = Rz @ Ry @ Rx
            direction_world = (R @ np.array([1.0, 0.0, 0.0], dtype=np.float32))
            dn = np.linalg.norm(direction_world)
            if dn < 1e-9:
                return False
            direction_world = direction_world / dn
        pixel = self._direction_world_to_pixel(observation, direction_world, camera_fov_rad=camera_fov_rad)
        if pixel is None:
            return False
        pr, pc = pixel
        n, m = front_patch_rows, front_patch_cols
        r0 = max(0, pr - (n - 1) // 2)
        r1 = min(H, pr + (n - 1) // 2 + 1)
        c0 = max(0, pc - (m - 1) // 2)
        c1 = min(W, pc + (m - 1) // 2 + 1)
        min_raw = float(np.min(depth[r0:r1, c0:c1]))
        return (0.5 + 19.5 * min_raw) < front_obstacle_min_depth_m

    def _get_front_patch_min_depth_m(
        self,
        observation,
        current_velocity,
        roll,
        pitch,
        yaw,
        front_patch_rows=None,
        front_patch_cols=None,
        camera_fov_rad=None,
    ):
        if front_patch_rows is None:
            front_patch_rows = self.FRONT_PATCH_ROWS
        if front_patch_cols is None:
            front_patch_cols = self.FRONT_PATCH_COLS
        if camera_fov_rad is None:
            camera_fov_rad = self.CAMERA_FOV_RAD
        if not isinstance(observation, dict) or "depth" not in observation:
            return None
        depth = np.squeeze(np.asarray(observation["depth"], dtype=np.float32))
        if depth.ndim != 2:
            return None
        H, W = depth.shape
        vel = np.asarray(current_velocity, dtype=np.float32).reshape(3)
        speed = float(np.linalg.norm(vel))
        if speed > 0.1:
            direction_world = vel / speed
        else:
            cy, sy = np.cos(yaw), np.sin(yaw)
            cp, sp = np.cos(pitch), np.sin(pitch)
            cr, sr = np.cos(roll), np.sin(roll)
            Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float32)
            Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float32)
            Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float32)
            R = Rz @ Ry @ Rx
            direction_world = (R @ np.array([1.0, 0.0, 0.0], dtype=np.float32))
            dn = np.linalg.norm(direction_world)
            if dn < 1e-9:
                return None
            direction_world = direction_world / dn
        pixel = self._direction_world_to_pixel(observation, direction_world, camera_fov_rad=camera_fov_rad)
        if pixel is None:
            return None
        pr, pc = pixel
        n, m = front_patch_rows, front_patch_cols
        r0 = max(0, pr - (n - 1) // 2)
        r1 = min(H, pr + (n - 1) // 2 + 1)
        c0 = max(0, pc - (m - 1) // 2)
        c1 = min(W, pc + (m - 1) // 2 + 1)
        return float(0.5 + 19.5 * np.min(depth[r0:r1, c0:c1]))

    def _obstacle_close_on_sides_not_front(self, observation, danger_distance_m=None):
        """True only when front is clear (no close obstacle) AND left or right has close obstacle."""
        if danger_distance_m is None:
            danger_distance_m = self.DANGER_DISTANCE_M
        if not isinstance(observation, dict) or "depth" not in observation:
            return False
        depth = np.squeeze(np.asarray(observation["depth"], dtype=np.float32))
        if depth.ndim != 2:
            return False
        H, W = depth.shape
        c1, c2 = W // 3, 2 * W // 3
        raw_to_m = lambda raw: 0.5 + 19.5 * float(raw)
        front_min = raw_to_m(np.min(depth[:, c1:c2])) if c2 > c1 else raw_to_m(np.min(depth))
        left_min = raw_to_m(np.min(depth[:, :c1])) if c1 > 0 else float("inf")
        right_min = raw_to_m(np.min(depth[:, c2:])) if c2 < W else float("inf")
        return front_min >= danger_distance_m and (left_min < danger_distance_m or right_min < danger_distance_m)

    def _get_front_pixel(self, observation, current_velocity, roll, pitch, yaw, camera_fov_rad=None):
        """Return (r, c) of the pixel in the movement direction (world frame), or None."""
        if camera_fov_rad is None:
            camera_fov_rad = self.CAMERA_FOV_RAD
        if not isinstance(observation, dict) or "depth" not in observation:
            return None
        vel = np.asarray(current_velocity, dtype=np.float32).reshape(3)
        speed = float(np.linalg.norm(vel))
        if speed > 0.1:
            direction_world = vel / speed
        else:
            cy, sy = np.cos(yaw), np.sin(yaw)
            cp, sp = np.cos(pitch), np.sin(pitch)
            cr, sr = np.cos(roll), np.sin(roll)
            Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float32)
            Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float32)
            Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float32)
            R = Rz @ Ry @ Rx
            direction_world = (R @ np.array([1.0, 0.0, 0.0], dtype=np.float32))
            dn = np.linalg.norm(direction_world)
            if dn < 1e-9:
                return None
            direction_world = direction_world / dn
        return self._direction_world_to_pixel(observation, direction_world, camera_fov_rad=camera_fov_rad)

    def _choose_detour_waypoint_connected_component(
        self,
        observation,
        current_pos,
        final_target,
        obstacle_depth_m,
        current_velocity,
        roll,
        pitch,
        yaw,
        camera_fov_rad=None,
        obstacle_margin_m=0.5,
        height_tolerance_m=1.0,
        min_clear_distance_px=None,
    ):
        """Find obstacle CC containing front pixel; forbidden = CC+10px and depth<obstacle+10px; pick closest to final_target."""
        if camera_fov_rad is None:
            camera_fov_rad = self.CAMERA_FOV_RAD
        if min_clear_distance_px is None:
            min_clear_distance_px = self.DETOUR_MIN_CLEAR_DISTANCE_PX
        if not isinstance(observation, dict) or "depth" not in observation:
            return None
        depth = np.squeeze(np.asarray(observation["depth"], dtype=np.float32))
        if depth.ndim != 2:
            return None
        H, W = depth.shape
        depth_meters = 0.5 + 19.5 * np.clip(depth, 0.0, 1.0)
        cur = np.asarray(current_pos, dtype=np.float32).reshape(3)
        final = np.asarray(final_target, dtype=np.float32).reshape(3)
        target_z = float(final[2])
        z_lo = target_z - height_tolerance_m
        z_hi = target_z + height_tolerance_m

        threshold = float(obstacle_depth_m) + obstacle_margin_m
        obstacle_mask = (depth_meters < threshold).astype(np.uint8)

        front_pixel = self._get_front_pixel(observation, current_velocity, roll, pitch, yaw, camera_fov_rad=camera_fov_rad)
        if front_pixel is None:
            return None
        pr, pc = front_pixel
        pr, pc = int(np.clip(pr, 0, H - 1)), int(np.clip(pc, 0, W - 1))

        component_mask = np.zeros((H, W), dtype=bool)
        queue = [(pr, pc)]
        while queue:
            r, c = queue.pop()
            if r < 0 or r >= H or c < 0 or c >= W:
                continue
            if component_mask[r, c]:
                continue
            if not obstacle_mask[r, c]:
                continue
            component_mask[r, c] = True
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                queue.append((r + dr, c + dc))

        k = min_clear_distance_px
        near_obstacle_mask = (depth_meters < float(obstacle_depth_m))
        bad_mask = component_mask | near_obstacle_mask
        forbidden = np.zeros((H, W), dtype=bool)
        q = deque()
        bad_r, bad_c = np.nonzero(bad_mask)
        forbidden[bad_r, bad_c] = True
        q = deque(zip(bad_r.tolist(), bad_c.tolist(), [0]*len(bad_r)))
        neighbors_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        while q:
            r, c, d = q.popleft()
            if d + 1 >= k:
                continue
            for dr, dc in neighbors_8:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and not forbidden[nr, nc]:
                    forbidden[nr, nc] = True
                    q.append((nr, nc, d + 1))

        candidate_mask = ~forbidden

        roll_v, pitch_v = self._get_roll_pitch(observation)
        yaw_v = self._get_yaw(observation)
        if yaw_v is None:
            yaw_v = 0.0
        half_fov = camera_fov_rad / 2.0
        cr_v, sr_v = np.cos(roll_v), np.sin(roll_v)
        cp_v, sp_v = np.cos(pitch_v), np.sin(pitch_v)
        cy_v, sy_v = np.cos(yaw_v), np.sin(yaw_v)
        Rx = np.array([[1, 0, 0], [0, cr_v, -sr_v], [0, sr_v, cr_v]], dtype=np.float32)
        Ry = np.array([[cp_v, 0, sp_v], [0, 1, 0], [-sp_v, 0, cp_v]], dtype=np.float32)
        Rz = np.array([[cy_v, -sy_v, 0], [sy_v, cy_v, 0], [0, 0, 1]], dtype=np.float32)
        R = Rz @ Ry @ Rx

        cand_r, cand_c = np.nonzero(candidate_mask)
        if len(cand_r) == 0:
            return None
        horz = (cand_c.astype(np.float32) - (W - 1) / 2.0) / (W / 2.0) * half_fov
        vert = (cand_r.astype(np.float32) - (H - 1) / 2.0) / (H / 2.0) * half_fov
        rx = np.ones(len(cand_r), dtype=np.float32)
        ry = -np.tan(horz); rz_arr = -np.tan(vert)
        rn = np.sqrt(rx**2 + ry**2 + rz_arr**2)
        rn = np.maximum(rn, 1e-9)
        rx /= rn; ry /= rn; rz_arr /= rn
        dw_x = R[0, 0]*rx + R[0, 1]*ry + R[0, 2]*rz_arr
        dw_y = R[1, 0]*rx + R[1, 1]*ry + R[1, 2]*rz_arr
        dw_z = R[2, 0]*rx + R[2, 1]*ry + R[2, 2]*rz_arr
        dn_arr = np.sqrt(dw_x**2 + dw_y**2 + dw_z**2)
        dn_arr = np.maximum(dn_arr, 1e-9)
        dw_x /= dn_arr; dw_y /= dn_arr; dw_z /= dn_arr

        dm = depth_meters[cand_r, cand_c]
        pos_x = cur[0] + dm * dw_x
        pos_y = cur[1] + dm * dw_y
        pos_z = cur[2] + dm * dw_z

        in_band = (pos_z >= z_lo) & (pos_z <= z_hi)
        small_dz = np.abs(dw_z) < 1e-9
        cur_in_band = (z_lo <= cur[2] <= z_hi)
        case2 = small_dz & (~in_band) & cur_in_band

        t_lo_arr = np.where(np.abs(dw_z) > 1e-9, (z_lo - cur[2]) / dw_z, 0.0)
        t_hi_arr = np.where(np.abs(dw_z) > 1e-9, (z_hi - cur[2]) / dw_z, 0.0)
        t_min_arr = np.minimum(t_lo_arr, t_hi_arr)
        t_max_arr = np.maximum(t_lo_arr, t_hi_arr)
        tv_lo = np.maximum(0.0, t_min_arr)
        tv_hi = np.minimum(dm, t_max_arr)
        case3 = (~in_band) & (~small_dz) & (tv_lo <= tv_hi)

        valid = in_band | case2 | case3
        if not np.any(valid):
            return None

        wp_x = np.where(in_band, pos_x, np.where(case3, cur[0] + tv_hi * dw_x, pos_x))
        wp_y = np.where(in_band, pos_y, np.where(case3, cur[1] + tv_hi * dw_y, pos_y))
        wp_z = np.where(in_band, pos_z, np.where(case3, cur[2] + tv_hi * dw_z, pos_z))

        vi = np.nonzero(valid)[0]
        wp_x = wp_x[vi]; wp_y = wp_y[vi]; wp_z = wp_z[vi]
        dist_to_final = np.sqrt((wp_x - final[0])**2 + (wp_y - final[1])**2 + (wp_z - final[2])**2)
        best_idx = int(np.argmin(dist_to_final))
        return np.array([wp_x[best_idx], wp_y[best_idx], wp_z[best_idx]], dtype=np.float32)

    def _choose_clearest_direction_toward_center(
        self,
        observation,
        direction_to_center,
        top_k=5,
        patch_size=5,
        camera_fov_rad=None,
        max_vertical_component=None,
        min_clear_depth_m=None,
        clear_margin_m=None,
    ):
        if camera_fov_rad is None:
            camera_fov_rad = self.CAMERA_FOV_RAD
        if max_vertical_component is None:
            max_vertical_component = self.MAX_VERTICAL_COMPONENT_HORIZONTAL
        if clear_margin_m is None:
            clear_margin_m = self.CLEAR_DIRECTION_MIN_MARGIN_M
        if not isinstance(observation, dict) or "depth" not in observation:
            return None
        depth = np.squeeze(np.asarray(observation["depth"], dtype=np.float32))
        if depth.ndim != 2:
            return None
        H, W = depth.shape
        depth_meters = 0.5 + 19.5 * depth
        to_center = np.asarray(direction_to_center, dtype=np.float32).reshape(3)
        dn = float(np.linalg.norm(to_center))
        if dn < 1e-9:
            to_center = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            to_center = to_center / dn
        half = patch_size // 2
        roll_v, pitch_v = self._get_roll_pitch(observation)
        yaw_v = self._get_yaw(observation)
        if yaw_v is None:
            yaw_v = 0.0
        half_fov = camera_fov_rad / 2.0
        cr, sr = np.cos(roll_v), np.sin(roll_v)
        cp, sp = np.cos(pitch_v), np.sin(pitch_v)
        cy, sy = np.cos(yaw_v), np.sin(yaw_v)
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float32)
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float32)
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float32)
        R = Rz @ Ry @ Rx
        rows = np.arange(H, dtype=np.float32)
        cols = np.arange(W, dtype=np.float32)
        cc, rr = np.meshgrid(cols, rows)
        horz = (cc - (W - 1) / 2.0) / (W / 2.0) * half_fov
        vert = (rr - (H - 1) / 2.0) / (H / 2.0) * half_fov
        ray_x = np.ones((H, W), dtype=np.float32)
        ray_y = -np.tan(horz)
        ray_z = -np.tan(vert)
        ray_norm = np.sqrt(ray_x**2 + ray_y**2 + ray_z**2)
        ray_norm = np.maximum(ray_norm, 1e-9)
        ray_x /= ray_norm; ray_y /= ray_norm; ray_z /= ray_norm
        dw_x = R[0, 0]*ray_x + R[0, 1]*ray_y + R[0, 2]*ray_z
        dw_y = R[1, 0]*ray_x + R[1, 1]*ray_y + R[1, 2]*ray_z
        dw_z = R[2, 0]*ray_x + R[2, 1]*ray_y + R[2, 2]*ray_z
        dn = np.sqrt(dw_x**2 + dw_y**2 + dw_z**2)
        dn = np.maximum(dn, 1e-9)
        dw_x /= dn; dw_y /= dn; dw_z /= dn
        vert_ok = np.abs(dw_z) <= max_vertical_component
        if min_clear_depth_m is not None:
            min_depth_map = minimum_filter(depth_meters, size=patch_size, mode='constant', cval=0.0)
            depth_ok = min_depth_map >= min_clear_depth_m
        else:
            depth_ok = np.ones((H, W), dtype=bool)
        valid = vert_ok & depth_ok
        if not np.any(valid):
            return None
        to_center_xy = np.array([to_center[0], to_center[1]], dtype=np.float32)
        to_center_xy_norm = float(np.linalg.norm(to_center_xy))
        if to_center_xy_norm < 1e-9:
            to_center_xy = np.array([1.0, 0.0], dtype=np.float32)
        else:
            to_center_xy = to_center_xy / to_center_xy_norm
        dir_xy_norm = np.sqrt(dw_x**2 + dw_y**2)
        dir_xy_norm = np.maximum(dir_xy_norm, 1e-9)
        cos_map = (dw_x * to_center_xy[0] + dw_y * to_center_xy[1]) / dir_xy_norm
        cos_map = np.clip(cos_map, -1.0, 1.0)
        angle_map = np.arccos(cos_map)
        angle_map[~valid] = 1e9
        best_idx = np.argmin(angle_map)
        br, bc = divmod(int(best_idx), W)
        if angle_map[br, bc] > 1e8:
            return None
        out = np.array([dw_x[br, bc], dw_y[br, bc], 0.0], dtype=np.float32)
        n = float(np.linalg.norm(out))
        if n < 1e-9:
            return None
        return (out / n).astype(np.float32)

    # ---------- Movement (inlined; logic aligned with movement.py) ----------
    def _accel_scale_for_tilt(self, roll, pitch):
        """Scale in (0, 1]: 1 when level, lower when roll/pitch high."""
        tilt = max(abs(roll), abs(pitch))
        if tilt >= self.MAX_SAFE_ROLL_PITCH_RAD:
            return self.ACCEL_SCALE_AT_HIGH_TILT
        return 1.0 - (1.0 - self.ACCEL_SCALE_AT_HIGH_TILT) * (tilt / self.MAX_SAFE_ROLL_PITCH_RAD)

    def _accel_scale_for_angular_velocity(self, roll_rate, pitch_rate):
        """Scale in (0, 1]: 1 when stable, lower when angular velocity high."""
        ang_vel_magnitude = max(abs(roll_rate), abs(pitch_rate))
        if ang_vel_magnitude >= self.MAX_SAFE_ANGULAR_VEL_RAD_S:
            return self.ACCEL_SCALE_AT_HIGH_ANG_VEL
        return 1.0 - (1.0 - self.ACCEL_SCALE_AT_HIGH_ANG_VEL) * (
            ang_vel_magnitude / self.MAX_SAFE_ANGULAR_VEL_RAD_S
        )

    def _speed_scale_for_approach_angle(self, current_velocity, desired_direction):
        """Scale in (0, 1] for speed based on angle between current velocity and desired direction."""
        vel = np.asarray(current_velocity, dtype=np.float32).reshape(3)
        dir_vec = np.asarray(desired_direction, dtype=np.float32).reshape(3)
        nv = float(np.linalg.norm(vel))
        if nv < 1e-6:
            return 1.0
        cos_angle = float(np.dot(vel, dir_vec)) / (nv + 1e-9)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = float(np.arccos(cos_angle))
        scale = 1.0 - (1.0 - self.SPEED_SCALE_AT_LARGE_ANGLE) * min(
            1.0, angle / self.MAX_APPROACH_ANGLE_RAD
        )
        return float(np.clip(scale, self.SPEED_SCALE_AT_LARGE_ANGLE, 1.0))

    def _move3d(
        self,
        current_pos,
        target_point,
        current_velocity,
        roll=None,
        pitch=None,
        roll_rate=None,
        pitch_rate=None,
        max_speed_m_s=None,
        deceleration_start_distance=None,
        at_target_tolerance=None,
        at_target_speed_threshold=None,
        yaw_face_direction_min_distance_m=None,
        max_velocity_change_norm_m_s=None,
        **kwargs,
    ):
        """Compute action [vx, vy, vz, speed, yaw] to move toward target_point (logic from movement.py)."""
        if max_speed_m_s is None:
            max_speed_m_s = self.MAX_SPEED_M_S
        if deceleration_start_distance is None:
            deceleration_start_distance = self.DECELERATION_START_DISTANCE
        if at_target_tolerance is None:
            at_target_tolerance = self.AT_TARGET_TOLERANCE
        if at_target_speed_threshold is None:
            at_target_speed_threshold = self.AT_TARGET_SPEED_THRESHOLD
        if yaw_face_direction_min_distance_m is None:
            yaw_face_direction_min_distance_m = self.YAW_FACE_DIRECTION_MIN_DISTANCE_M
        if max_velocity_change_norm_m_s is None:
            max_velocity_change_norm_m_s = self.MAX_VELOCITY_CHANGE_NORM_M_S
        current = np.asarray(current_pos, dtype=np.float32).reshape(3)
        target = np.asarray(target_point, dtype=np.float32).reshape(3)
        vel = np.asarray(current_velocity, dtype=np.float32).reshape(3)
        diff = target - current
        distance = float(np.linalg.norm(diff))
        if distance < at_target_tolerance and float(np.linalg.norm(vel)) < at_target_speed_threshold:
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        if distance < 1e-6:
            direction = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        else:
            direction = diff / distance
        if distance < deceleration_start_distance:
            decel_range = deceleration_start_distance
            if decel_range > 0:
                speed_factor = distance / decel_range
                desired_speed_cap = max_speed_m_s * max(0.0, min(1.0, speed_factor))
            else:
                desired_speed_cap = 0.0
        else:
            desired_speed_cap = max_speed_m_s
        scale = 1.0
        if roll is not None and pitch is not None:
            scale = min(scale, self._accel_scale_for_tilt(roll, pitch))
        if roll_rate is not None and pitch_rate is not None:
            scale = min(scale, self._accel_scale_for_angular_velocity(roll_rate, pitch_rate))
        desired_speed_cap *= scale
        if distance >= 1e-6:
            angle_scale = self._speed_scale_for_approach_angle(vel, direction)
            desired_speed_cap *= angle_scale
        goal_velocity = direction * desired_speed_cap
        goal_norm = float(np.linalg.norm(goal_velocity))
        if goal_norm > 1e-6:
            goal_velocity_unit = goal_velocity / goal_norm
        else:
            goal_velocity_unit = direction
        current_speed = float(np.linalg.norm(vel))
        if current_speed < 1e-6:
            current_speed_in_goal_direction = 0.0
        else:
            cos_angle = float(np.dot(vel, goal_velocity_unit)) / (current_speed + 1e-9)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            current_speed_in_goal_direction = current_speed * cos_angle
        desired_speed_from_max_change = current_speed_in_goal_direction + max_velocity_change_norm_m_s
        desired_speed = min(
            desired_speed_cap,
            max(0.0, desired_speed_from_max_change),
            max_speed_m_s,
        )
        velocity_direction_vector = direction
        velocity = velocity_direction_vector * desired_speed
        speed = float(np.linalg.norm(velocity))
        if speed > max_speed_m_s and speed > 1e-6:
            velocity = velocity * (max_speed_m_s / speed)
            speed = max_speed_m_s
        desired_yaw_norm = None
        horiz_dist = float(np.sqrt(diff[0] * diff[0] + diff[1] * diff[1]))
        if horiz_dist >= yaw_face_direction_min_distance_m:
            dir_xy_norm = float(np.sqrt(direction[0] ** 2 + direction[1] ** 2))
            if dir_xy_norm > 1e-6:
                desired_yaw = float(np.arctan2(direction[1], direction[0]))
                desired_yaw_norm = float(np.clip(desired_yaw / np.pi, -1.0, 1.0))
        action = np.zeros(5, dtype=np.float32)
        action[0] = velocity[0]
        action[1] = velocity[1]
        action[2] = velocity[2]
        action[3] = speed
        action[4] = desired_yaw_norm if desired_yaw_norm is not None else 0.0
        return action

    # ---------- Search area (inlined search_area) ----------
    def _compute_search_area_center(self, current_pos, search_area_vector):
        current = np.asarray(current_pos, dtype=np.float32).reshape(3)
        vector = np.asarray(search_area_vector, dtype=np.float32).reshape(3)
        return current + vector

    def _build_height_match_waypoints(self, current_pos, center, horizontal_m=None, rise_height_m=None):
        if horizontal_m is None:
            horizontal_m = self.HORIZONTAL_M
        if rise_height_m is None:
            rise_height_m = self.RISE_HEIGHT_M
        cur = np.asarray(current_pos, dtype=np.float32).reshape(3)
        cen = np.asarray(center, dtype=np.float32).reshape(3)
        waypoints = []
        if cen[2] - 1 > cur[2]:
            waypoints.append(np.array([cur[0], cur[1], max(0.6, cen[2] - 1)], dtype=np.float32))
        else:
            dx = cen[0] - cur[0]
            dy = cen[1] - cur[1]
            hor_norm = np.sqrt(dx * dx + dy * dy)
            if hor_norm > 1e-6:
                dx, dy = dx / hor_norm, dy / hor_norm
            else:
                dx, dy = 1.0, 0.0
            mid_x = cur[0] + dx * horizontal_m
            mid_y = cur[1] + dy * horizontal_m
            mid_z = cur[2] + rise_height_m
            waypoints.append(np.array([mid_x, mid_y, mid_z], dtype=np.float32))
            waypoints.append(np.array([mid_x, mid_y, max(0.6, cen[2] - 1)], dtype=np.float32))
        return waypoints

    def _generate_random_point_in_search_area(
        self, search_area_center, radius_x=None, radius_y=None, radius_z=None
    ):
        if radius_x is None:
            radius_x = self.SEARCH_AREA_RADIUS_X_M
        if radius_y is None:
            radius_y = self.SEARCH_AREA_RADIUS_Y_M
        if radius_z is None:
            radius_z = self.SEARCH_AREA_RADIUS_Z_M
        center = np.asarray(search_area_center, dtype=np.float32).reshape(3)
        offset_x = np.random.uniform(-radius_x, radius_x)
        offset_y = np.random.uniform(-radius_y, radius_y)
        offset_z = np.random.uniform(-radius_z, radius_z)
        point = center + np.array([offset_x, offset_y, offset_z], dtype=np.float32)
        point[2] = np.clip(point[2], 1.0, 10.0)
        return point

    # ---------- Goal detection (inlined goal_detection) ----------
    def _get_near_depth_average_direction(
        self,
        observation,
        max_depth_m=None,
        depth_range_near_m=None,
        depth_range_far_m=None,
        camera_fov_rad=None,
    ):
        if max_depth_m is None:
            max_depth_m = self.NEAR_GOAL_DEPTH_MAX_M
        if depth_range_near_m is None:
            depth_range_near_m = self.DEPTH_RANGE_NEAR_M
        if depth_range_far_m is None:
            depth_range_far_m = self.DEPTH_RANGE_FAR_M
        if camera_fov_rad is None:
            camera_fov_rad = self.CAMERA_FOV_RAD
        if not isinstance(observation, dict) or "depth" not in observation or "state" not in observation:
            return (None,)
        depth = np.squeeze(np.asarray(observation["depth"], dtype=np.float32))
        if depth.ndim != 2:
            return (None,)
        H, W = depth.shape
        depth_meters = 0.5 + 19.5 * depth
        current_pos = self._get_current_position(observation)
        roll, pitch = self._get_roll_pitch(observation)
        yaw = self._get_yaw(observation)
        if yaw is None:
            yaw = 0.0
        half_fov = camera_fov_rad / 2.0
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float32)
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float32)
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float32)
        R = Rz @ Ry @ Rx
        near_mask = depth_meters < max_depth_m
        near_r, near_c = np.nonzero(near_mask)
        if len(near_r) == 0:
            return (None,)
        cur_f = np.asarray(current_pos, dtype=np.float32).reshape(3)
        dm = np.clip(depth_meters[near_r, near_c], depth_range_near_m, depth_range_far_m)
        h = (near_c.astype(np.float32) - (W - 1) / 2.0) / (W / 2.0) * half_fov
        v = (near_r.astype(np.float32) - (H - 1) / 2.0) / (H / 2.0) * half_fov
        rx = np.ones(len(near_r), dtype=np.float32)
        ry = -np.tan(h); rz = -np.tan(v)
        rn = np.sqrt(rx**2 + ry**2 + rz**2)
        rn = np.maximum(rn, 1e-9)
        rx /= rn; ry /= rn; rz /= rn
        dx = R[0, 0]*rx + R[0, 1]*ry + R[0, 2]*rz
        dy = R[1, 0]*rx + R[1, 1]*ry + R[1, 2]*rz
        dz = R[2, 0]*rx + R[2, 1]*ry + R[2, 2]*rz
        dn_arr = np.sqrt(dx**2 + dy**2 + dz**2)
        dn_arr = np.maximum(dn_arr, 1e-9)
        dx /= dn_arr; dy /= dn_arr; dz /= dn_arr
        avg_x = float(np.mean(cur_f[0] + dm * dx))
        avg_y = float(np.mean(cur_f[1] + dm * dy))
        avg_z = float(np.mean(cur_f[2] + dm * dz))
        avg_position = np.array([avg_x, avg_y, avg_z], dtype=np.float32)
        to_avg = avg_position - cur_f
        dn = float(np.linalg.norm(to_avg))
        if dn < 1e-6:
            return (None,)
        direction_world = (to_avg / dn).astype(np.float32)
        return (direction_world,)

    def _detect_goal_platform(
        self,
        observation,
        initial_position=None,
        depth_edge_jump=None,
        depth_similarity_threshold_m=None,
        depth_range_near_m=None,
        depth_range_far_m=None,
        camera_fov_rad=None,
        goal_platform_max_height_diff_m=None,
        start_platform_position_threshold_m=5.0,
        min_component_size=None,
    ):
        if depth_edge_jump is None:
            depth_edge_jump = self.DEPTH_EDGE_JUMP
        if depth_similarity_threshold_m is None:
            depth_similarity_threshold_m = self.DEPTH_SIMILARITY_THRESHOLD_M
        if depth_range_near_m is None:
            depth_range_near_m = self.DEPTH_RANGE_NEAR_M
        if depth_range_far_m is None:
            depth_range_far_m = self.DEPTH_RANGE_FAR_M
        if camera_fov_rad is None:
            camera_fov_rad = self.CAMERA_FOV_RAD
        if goal_platform_max_height_diff_m is None:
            goal_platform_max_height_diff_m = self.GOAL_PLATFORM_MAX_HEIGHT_DIFF_M
        if min_component_size is None:
            min_component_size = self.MIN_COMPONENT_SIZE
        if not isinstance(observation, dict) or "depth" not in observation or "state" not in observation:
            return False, None, None, False, None
        depth = np.squeeze(np.asarray(observation["depth"], dtype=np.float32))
        if depth.ndim != 2:
            return False, None, None, False, None
        H, W = depth.shape
        edge = np.zeros((H, W), dtype=bool)
        edge[:-1, :] = (depth[1:, :] - depth[:-1, :]) >= depth_edge_jump
        depth_meters = 0.5 + 19.5 * depth
        visited = np.zeros_like(edge, dtype=bool)
        best_size = 0
        best_pixel = None
        best_component_pixels = []
        current_pos = self._get_current_position(observation)
        roll, pitch = self._get_roll_pitch(observation)
        yaw = self._get_yaw(observation)
        if yaw is None:
            yaw = 0.0
        half_fov = camera_fov_rad / 2.0
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float32)
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float32)
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float32)
        R = Rz @ Ry @ Rx

        cur_f = np.asarray(current_pos, dtype=np.float32).reshape(3)

        def _batch_world_positions(pixels):
            arr = np.array(pixels, dtype=np.float32)
            prs = arr[:, 0]; pcs = arr[:, 1]
            dm = np.clip(depth_meters[prs.astype(int), pcs.astype(int)], depth_range_near_m, depth_range_far_m)
            h = (pcs - (W - 1) / 2.0) / (W / 2.0) * half_fov
            v = (prs - (H - 1) / 2.0) / (H / 2.0) * half_fov
            rx = np.ones(len(prs), dtype=np.float32)
            ry = -np.tan(h); rz = -np.tan(v)
            rn = np.sqrt(rx**2 + ry**2 + rz**2)
            rn = np.maximum(rn, 1e-9)
            rx /= rn; ry /= rn; rz /= rn
            dx = R[0, 0]*rx + R[0, 1]*ry + R[0, 2]*rz
            dy = R[1, 0]*rx + R[1, 1]*ry + R[1, 2]*rz
            dz = R[2, 0]*rx + R[2, 1]*ry + R[2, 2]*rz
            dn = np.sqrt(dx**2 + dy**2 + dz**2)
            dn = np.maximum(dn, 1e-9)
            dx /= dn; dy /= dn; dz /= dn
            return np.stack([cur_f[0] + dm*dx, cur_f[1] + dm*dy, cur_f[2] + dm*dz], axis=-1)

        def direction_world_for_pixel(pr, pc):
            horz = (pc - (W - 1) / 2.0) / (W / 2.0) * half_fov
            vert = (pr - (H - 1) / 2.0) / (H / 2.0) * half_fov
            ray_body = np.array([1.0, -np.tan(horz), -np.tan(vert)], dtype=np.float32)
            n = np.linalg.norm(ray_body)
            if n < 1e-9:
                ray_body = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            else:
                ray_body = ray_body / n
            direction_world = (R @ ray_body).astype(np.float32)
            dn = np.linalg.norm(direction_world)
            if dn > 1e-9:
                direction_world = direction_world / dn
            return direction_world

        def bfs_collect_component(sr, sc):
            queue = deque([(sr, sc)])
            visited[sr, sc] = True
            pixels = [(sr, sc)]
            while queue:
                r, c = queue.popleft()
                current_depth_m = depth_meters[r, c]
                for dr, dc in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and edge[nr, nc] and not visited[nr, nc]:
                        if abs(depth_meters[nr, nc] - current_depth_m) < depth_similarity_threshold_m:
                            visited[nr, nc] = True
                            queue.append((nr, nc))
                            pixels.append((nr, nc))
            if len(pixels) > 0:
                pixels_sorted = sorted(pixels, key=lambda p: p[1])
                mid = len(pixels_sorted) // 2
                best_r, best_c = pixels_sorted[mid]
                return len(pixels), (best_r, best_c), pixels
            return 0, None, []

        def expand_component_with_similar_depth(edge_pixels):
            visited_expand = np.zeros((H, W), dtype=bool)
            for pr, pc in edge_pixels:
                visited_expand[pr, pc] = True
            region_pixels = list(edge_pixels)
            queue = deque(edge_pixels)
            while queue:
                r, c = queue.popleft()
                current_depth_m = depth_meters[r, c]
                for dr, dc in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and not visited_expand[nr, nc]:
                        if abs(depth_meters[nr, nc] - current_depth_m) < depth_similarity_threshold_m:
                            visited_expand[nr, nc] = True
                            queue.append((nr, nc))
                            region_pixels.append((nr, nc))
            return region_pixels

        seed_mask = edge & (~visited)
        seed_r, seed_c = np.nonzero(seed_mask)
        for idx in range(len(seed_r)):
            r, c = int(seed_r[idx]), int(seed_c[idx])
            if visited[r, c]:
                continue
            size, centroid, edge_pixels = bfs_collect_component(r, c)
            if size <= 0 or centroid is None or size <= min_component_size:
                continue
            region_pixels = expand_component_with_similar_depth(edge_pixels)
            if not region_pixels:
                continue
            positions = _batch_world_positions(region_pixels)
            z_vals = positions[:, 2]
            z_span = float(z_vals.max() - z_vals.min())
            if z_span > goal_platform_max_height_diff_m:
                continue
            mean_pos = positions.mean(axis=0)
            max_dist_from_mean = float(np.max(np.linalg.norm(positions - mean_pos, axis=1)))
            if max_dist_from_mean > self.GOAL_PLATFORM_MAX_EXTENT_M:
                continue
            best_size = size
            best_pixel = centroid
            best_component_pixels = region_pixels
        if best_pixel is None or best_size <= min_component_size:
            return False, None, None, False, None
        r, c = best_pixel
        distance_to_goal = float(depth_meters[r, c])
        distance_to_goal = np.clip(distance_to_goal, depth_range_near_m, depth_range_far_m)
        # Bad pixels: depth < 0.5 m
        bad_pixels = (depth_meters < self.BAD_PIXEL_DEPTH_THRESHOLD_M)
        half_w = self.GOOD_PIXEL_WINDOW_SIZE // 2
        bad_dilated = maximum_filter(bad_pixels.astype(np.uint8), size=2*half_w+1, mode='constant', cval=0)
        good_mask = bad_dilated == 0
        if not good_mask[r, c]:
            gr, gc = np.nonzero(good_mask)
            if len(gr) > 0:
                dists = np.abs(gr - r) + np.abs(gc - c)
                best_i = int(np.argmin(dists))
                r, c = int(gr[best_i]), int(gc[best_i])
        d_norm = float(np.clip(depth[r, c], 0.0, 1.0))
        depth_m = 0.5 + 19.5 * d_norm
        depth_m = np.clip(depth_m, depth_range_near_m, depth_range_far_m)
        horz = (c - (W - 1) / 2.0) / (W / 2.0) * half_fov
        vert = (r - (H - 1) / 2.0) / (H / 2.0) * half_fov
        ray_body = np.array([1.0, -np.tan(horz), -np.tan(vert)], dtype=np.float32)
        n = np.linalg.norm(ray_body)
        if n < 1e-9:
            ray_body = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            ray_body = ray_body / n
        direction_world = (R @ ray_body).astype(np.float32)
        dn = np.linalg.norm(direction_world)
        if dn > 1e-9:
            direction_world = direction_world / dn
        cur = np.asarray(current_pos, dtype=np.float32).reshape(3)
        if distance_to_goal > self.GOAL_AVERAGE_POSITION_DISTANCE_M:
            position_world = cur + depth_m * direction_world
        elif distance_to_goal >= 0.5 and best_component_pixels:
            bcp = np.array(best_component_pixels, dtype=int)
            avg_depth = float(np.mean(np.clip(depth_meters[bcp[:, 0], bcp[:, 1]], depth_range_near_m, depth_range_far_m)))
            position_world = cur + avg_depth * direction_world
        elif distance_to_goal < 0.5 and best_component_pixels:
            bcp2 = np.array(best_component_pixels, dtype=int)
            r_avg = float(np.mean(bcp2[:, 0]))
            c_avg = float(np.mean(bcp2[:, 1]))
            r_avg = int(np.clip(r_avg, 0, H - 1))
            c_avg = int(np.clip(c_avg, 0, W - 1))
            direction_world = direction_world_for_pixel(r_avg, c_avg)
            position_world = cur + self.GOAL_NEAR_FIXED_DISTANCE_M * direction_world
        else:
            position_world = cur + depth_m * direction_world
        is_start = False
        if initial_position is not None:
            dist_to_start_platform = float(np.linalg.norm(
                position_world - np.asarray(initial_position, dtype=np.float32).reshape(3)
            ))
            if dist_to_start_platform < start_platform_position_threshold_m:
                is_start = True
        return True, direction_world, position_world, is_start, (r, c)

    # ---------- State 1 helpers ----------
    def _state1_move3d_kwargs(self):
        return dict(
            max_speed_m_s=self.MAX_SPEED_M_S,
            deceleration_start_distance=self.DECELERATION_START_DISTANCE,
            at_target_tolerance=self.AT_TARGET_TOLERANCE,
            at_target_speed_threshold=self.AT_TARGET_SPEED_THRESHOLD,
            yaw_face_direction_min_distance_m=self.YAW_FACE_DIRECTION_MIN_DISTANCE_M,
        )

    def _state1_next_random_target(self):
        if self.search_area_center is None:
            return None
        center = np.asarray(self.search_area_center, dtype=np.float32).reshape(3)
        pt = self._generate_random_point_in_search_area(
            center, radius_x=self.SEARCH_AREA_RADIUS_X_M,
            radius_y=self.SEARCH_AREA_RADIUS_Y_M, radius_z=0.0,
        )
        pt[2] = float(center[2])
        return np.asarray(pt, dtype=np.float32)

    def act(self, observation):
        current_pos = self._get_current_position(observation)
        current_velocity = self._get_current_velocity(observation)
        roll, pitch = self._get_roll_pitch(observation)
        angular_velocity = self._get_angular_velocity(observation)
        roll_rate = float(angular_velocity[0]) if len(angular_velocity) >= 1 else None
        pitch_rate = float(angular_velocity[1]) if len(angular_velocity) >= 2 else None
        yaw = self._get_yaw(observation)
        self.act_step += 1
        # print(self.state)
        if self.initial_position is None:
            self.initial_position = np.asarray(current_pos, dtype=np.float32).reshape(3).copy()

        if self.state != 3:
            detected, direction_world, position_world, is_start, _ = self._detect_goal_platform(
                observation,
                initial_position=self.initial_position,
                start_platform_position_threshold_m=5.0,
            )
            if detected and not is_start:
                self.goal_detection_count += 1
                pos_world = np.asarray(position_world, dtype=np.float32).reshape(3)
                dist_to_goal = float(np.linalg.norm(pos_world - np.asarray(current_pos, dtype=np.float32).reshape(3)))
                if (
                    dist_to_goal > 0.5
                    and self.act_step >= self.MIN_STEPS_BEFORE_STATE3
                    and self.state1_danger_steps_remaining == 0
                    and self.goal_detection_count >= self.MIN_GOAL_DETECTIONS_BEFORE_STATE3
                ):
                    self.state = 3
                    self.detected_goal_position = pos_world.copy()
                    self.detected_goal_direction = np.asarray(direction_world, dtype=np.float32).reshape(3).copy()
            else:
                self.goal_detection_count = 0

        if self.state == 3:
            detected, direction_world, position_world, is_start, _ = self._detect_goal_platform(
                observation, initial_position=self.initial_position, start_platform_position_threshold_m=5.0
            )
            # print(detected)
            if detected and not is_start:
                self.state3_no_goal_steps = 0
                pos_world = np.asarray(position_world, dtype=np.float32).reshape(3)
                dist_to_goal = float(np.linalg.norm(pos_world - np.asarray(current_pos, dtype=np.float32).reshape(3)))
                if dist_to_goal > 0.5:
                    self.detected_goal_position = pos_world.copy()
                    self.detected_goal_direction = np.asarray(direction_world, dtype=np.float32).reshape(3).copy()
            else:
                self.state3_no_goal_steps += 1

            if self.state3_no_goal_steps >= self.STATE3_NO_GOAL_LOOK_STEPS:
                hold_pos = np.asarray(current_pos, dtype=np.float32).reshape(3)
                action = self._move3d(
                    current_pos, hold_pos, current_velocity,
                    roll=roll, pitch=pitch, roll_rate=roll_rate, pitch_rate=pitch_rate,
                    max_speed_m_s=self.MAX_SPEED_M_S,
                    deceleration_start_distance=0,  # state 3: no decel, max speed
                    at_target_tolerance=self.AT_TARGET_TOLERANCE,
                    at_target_speed_threshold=self.AT_TARGET_SPEED_THRESHOLD,
                    yaw_face_direction_min_distance_m=self.YAW_FACE_DIRECTION_MIN_DISTANCE_M,
                )
                yaw_rad = self._get_yaw(observation) or 0.0
                yaw_step = self.STATE2_YAW_STEP_RAD / np.pi
                action[4] = float(np.clip((yaw_rad / np.pi + yaw_step + 1) % 2 - 1, -1.0, 1.0))
                return action

            if not detected and self.detected_goal_position is not None:
                stored = np.asarray(self.detected_goal_position, dtype=np.float32).reshape(3)
                current = np.asarray(current_pos, dtype=np.float32).reshape(3)
                to_stored = stored - current
                dist_to_stored = float(np.linalg.norm(to_stored))
                reached = dist_to_stored <= self.AT_TARGET_TOLERANCE
                passed = False
                if self.detected_goal_direction is not None:
                    dir_vec = np.asarray(self.detected_goal_direction, dtype=np.float32).reshape(3)
                    dn = float(np.linalg.norm(dir_vec))
                    if dn > 1e-6:
                        dir_vec = dir_vec / dn
                        passed = float(np.dot(to_stored, dir_vec)) < -0.1
                if (reached or passed) and self.detected_goal_direction is not None:
                    dir_vec = np.asarray(self.detected_goal_direction, dtype=np.float32).reshape(3)
                    dn = float(np.linalg.norm(dir_vec))
                    if dn > 1e-6:
                        dir_vec = dir_vec / dn
                        move_target = stored + dir_vec * 10.0
                    else:
                        move_target = stored
                else:
                    move_target = stored
                action = self._move3d(
                    current_pos, move_target, current_velocity,
                    roll=roll, pitch=pitch, roll_rate=roll_rate, pitch_rate=pitch_rate,
                    max_speed_m_s=self.MAX_SPEED_M_S,
                    deceleration_start_distance=0,  # state 3: no decel, max speed
                    at_target_tolerance=self.AT_TARGET_TOLERANCE,
                    at_target_speed_threshold=self.AT_TARGET_SPEED_THRESHOLD,
                    yaw_face_direction_min_distance_m=self.YAW_FACE_DIRECTION_MIN_DISTANCE_M,
                )
                return action

            dist_to_goal = float("inf")
            if self.detected_goal_position is not None:
                dist_to_goal = float(np.linalg.norm(np.asarray(current_pos, dtype=np.float32).reshape(3) - np.asarray(self.detected_goal_position, dtype=np.float32).reshape(3)))
            if dist_to_goal < 0.5:
                near_dir, = self._get_near_depth_average_direction(observation, max_depth_m=1.0)
                if near_dir is not None:
                    dir_vec = np.asarray(near_dir, dtype=np.float32).reshape(3)
                else:
                    dir_vec = np.asarray(self.detected_goal_direction, dtype=np.float32).reshape(3) if self.detected_goal_direction is not None else None
            else:
                dir_vec = np.asarray(self.detected_goal_direction, dtype=np.float32).reshape(3) if self.detected_goal_direction is not None else None
            if dir_vec is not None:
                n = float(np.linalg.norm(dir_vec))
                if n > 1e-6:
                    dir_vec = dir_vec / n
                    current_speed = float(np.linalg.norm(current_velocity))
                    target_speed = min(self.MAX_SPEED_M_S, current_speed + self.ACCELERATION_M_S2 * self.SIM_DT)
                    angle_scale = self._speed_scale_for_approach_angle(current_velocity, dir_vec)
                    target_speed = target_speed * angle_scale
                    velocity = dir_vec * target_speed
                else:
                    velocity = np.asarray(current_velocity, dtype=np.float32).reshape(3)
                yaw_to_direction = None
                dir_xy_norm = float(np.sqrt(dir_vec[0]**2 + dir_vec[1]**2))
                if dir_xy_norm > 1e-6:
                    desired_yaw = float(np.arctan2(dir_vec[1], dir_vec[0]))
                    yaw_to_direction = float(np.clip(desired_yaw / np.pi, -1.0, 1.0))
                current_yaw_rad = self._get_yaw(observation)
                current_yaw = float(np.clip((current_yaw_rad if current_yaw_rad is not None else 0.0) / np.pi, -1.0, 1.0))
                action = np.zeros(5, dtype=np.float32)
                action[0] = velocity[0]
                action[1] = velocity[1]
                action[2] = velocity[2]
                action[3] = float(np.linalg.norm(velocity))
                action[4] = yaw_to_direction if yaw_to_direction is not None else current_yaw
                return action

        if self.state == 0:
            if self.search_area_center is None:
                search_area_vector = self._get_search_area_vector(observation)
                if np.linalg.norm(search_area_vector) > 0.01:
                    self.search_area_center = self._compute_search_area_center(current_pos, search_area_vector)
                    self.search_area_center = np.asarray(self.search_area_center, dtype=np.float32).reshape(3)

            if self.search_area_center is not None:
                if not self.target:
                    self.target = self._build_height_match_waypoints(
                        current_pos, self.search_area_center,
                        horizontal_m=self.HORIZONTAL_M, rise_height_m=self.RISE_HEIGHT_M,
                    )
                if self.target:
                    current_waypoint = self.target[0]
                    action = self._move3d(
                        current_pos, current_waypoint, current_velocity,
                        roll=roll, pitch=pitch, roll_rate=roll_rate, pitch_rate=pitch_rate,
                        max_speed_m_s=self.MAX_SPEED_M_S,
                        deceleration_start_distance=self.DECELERATION_START_DISTANCE,
                        at_target_tolerance=self.AT_TARGET_TOLERANCE,
                        at_target_speed_threshold=self.AT_TARGET_SPEED_THRESHOLD,
                        yaw_face_direction_min_distance_m=self.YAW_FACE_DIRECTION_MIN_DISTANCE_M,
                    )
                    center = np.asarray(self.search_area_center, dtype=np.float32).reshape(3)
                    to_center_xy = np.array([center[0] - current_pos[0], center[1] - current_pos[1]], dtype=np.float32)
                    horiz = float(np.linalg.norm(to_center_xy))
                    if horiz > 1e-6:
                        desired_yaw = float(np.arctan2(to_center_xy[1], to_center_xy[0]))
                        action[4] = float(np.clip(desired_yaw / np.pi, -1.0, 1.0))
                    dist = float(np.linalg.norm(np.asarray(current_waypoint, dtype=np.float32).reshape(3) - np.asarray(current_pos, dtype=np.float32).reshape(3)))
                    speed = float(np.linalg.norm(current_velocity))
                    if dist < self.AT_TARGET_TOLERANCE and speed < self.AT_TARGET_SPEED_THRESHOLD:
                        self.target.pop(0)
                        if not self.target:
                            self.target = [np.asarray(self.search_area_center, dtype=np.float32)]
                            self.state = 1
                    return action
                return np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        if self.state == 1:
            if self.search_area_center is None:
                return np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            cur = np.asarray(current_pos, dtype=np.float32).reshape(3)
            center = np.asarray(self.search_area_center, dtype=np.float32).reshape(3)
            mk = self._state1_move3d_kwargs()
            yaw_rad = yaw if yaw is not None else 0.0

            # ----- Danger zone: only when front is clear but left/right has close obstacle -----
            if self._obstacle_close_on_sides_not_front(observation, danger_distance_m=self.DANGER_DISTANCE_M):
                self.state1_danger_steps_remaining = self.STATE1_DANGER_STEPS_WINDOW
            if self.state1_danger_steps_remaining > 0:
                self.state1_danger_steps_remaining -= 1

            # ----- Front obstacle while going to target (or while seeking): Find open path, set detour -----
            if (
                self.target
                and not self.state1_turning_270
                and self._front_obstacle_detected(
                    observation, current_velocity, roll, pitch, yaw_rad,
                    front_obstacle_min_depth_m=self.FRONT_OBSTACLE_MIN_DEPTH_M,
                    front_patch_rows=self.FRONT_PATCH_ROWS,
                    front_patch_cols=self.FRONT_PATCH_COLS,
                    camera_fov_rad=self.CAMERA_FOV_RAD,
                )
            ):
                already_seeking = self.state1_seeking_clear_direction
                obstacle_depth_m = self._get_front_patch_min_depth_m(
                    observation, current_velocity, roll, pitch, yaw_rad,
                    front_patch_rows=self.FRONT_PATCH_ROWS,
                    front_patch_cols=self.FRONT_PATCH_COLS,
                    camera_fov_rad=self.CAMERA_FOV_RAD,
                )
                final_target = self.target[-1] if self.target else center
                waypoint = self._choose_detour_waypoint_connected_component(
                    observation, cur, final_target, obstacle_depth_m or 5.0,
                    current_velocity, roll, pitch, yaw_rad,
                    camera_fov_rad=self.CAMERA_FOV_RAD,
                    height_tolerance_m=1.0,
                    min_clear_distance_px=5,
                )
                self.state1_chosen_clear_direction = None
                if waypoint is not None and self.state1_danger_steps_remaining == 0:
                    prev_wp = np.asarray(self.target[0], dtype=np.float32).reshape(3) if self.target else cur
                    w_prev = self.DETOUR_WAYPOINT_SMOOTHING_PREV_WEIGHT
                    blended = w_prev * prev_wp + (1.0 - w_prev) * np.asarray(waypoint, dtype=np.float32).reshape(3)
                    self.target.insert(0, blended.astype(np.float32))
                    self.state1_has_detour_waypoint = True
                else:
                    to_center = center - cur
                    n = float(np.linalg.norm(to_center))
                    to_center = to_center / n if n > 1e-6 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
                    chosen = self._choose_clearest_direction_toward_center(
                        observation, to_center, top_k=5, patch_size=5, camera_fov_rad=self.CAMERA_FOV_RAD,
                        min_clear_depth_m=(obstacle_depth_m + 0.5) if obstacle_depth_m is not None else None,
                    )
                    self.state1_chosen_clear_direction = chosen
                    if chosen is not None and (self.state1_danger_steps_remaining == 0 or waypoint is None):
                        depth_m = self._depth_meters_for_direction(observation, chosen, camera_fov_rad=self.CAMERA_FOV_RAD)
                        dist = min(max(0.0, (float(depth_m) - 5.0) if depth_m else 0.0), 5.0) if depth_m is not None else 1.0
                        new_wp = np.asarray(cur + dist * chosen, dtype=np.float32)
                        prev_wp = np.asarray(self.target[0], dtype=np.float32).reshape(3) if self.target else cur
                        w_prev = self.DETOUR_WAYPOINT_SMOOTHING_PREV_WEIGHT
                        blended = w_prev * prev_wp + (1.0 - w_prev) * new_wp
                        self.target.insert(0, blended.astype(np.float32))
                        self.state1_has_detour_waypoint = True
                    elif not already_seeking:
                        to_xy = to_center[:2] / (np.linalg.norm(to_center[:2]) or 1e-6)
                        look_x, look_y = float(np.cos(yaw_rad)), float(np.sin(yaw_rad))
                        self.state1_turn_direction = 1 if (-look_y * to_xy[0] + look_x * to_xy[1]) >= 0 else -1
                        self.state1_has_detour_waypoint = False
                if not already_seeking:
                    self.state1_seeking_clear_direction = True

            # ----- Going to detour (open path found); go to first waypoint -----
            if self.state1_seeking_clear_direction and self.target and self.state1_has_detour_waypoint:
                wp = np.asarray(self.target[0], dtype=np.float32).reshape(3).copy()
                # Keep current height only when using direction-based detour (chosen_clear_direction)
                if self.state1_chosen_clear_direction is not None:
                    wp[2] = cur[2]
                action = self._move3d(
                    current_pos, wp, current_velocity,
                    roll=roll, pitch=pitch, roll_rate=roll_rate, pitch_rate=pitch_rate, **mk,
                )
                if self.state1_chosen_clear_direction is not None:
                    d = np.array([self.state1_chosen_clear_direction[0], self.state1_chosen_clear_direction[1]], dtype=np.float32)
                    if np.linalg.norm(d) > 1e-6:
                        action[4] = float(np.clip(np.arctan2(d[1], d[0]) / np.pi, -1.0, 1.0))
                if np.linalg.norm(wp - cur) < self.AT_TARGET_TOLERANCE and np.linalg.norm(current_velocity) < self.AT_TARGET_SPEED_THRESHOLD:
                    self.target.pop(0)
                    self.state1_seeking_clear_direction = False
                    self.state1_chosen_clear_direction = None
                    self.state1_has_detour_waypoint = False
                    if not self.target:
                        yaw_rad = self._get_yaw(observation) or 0.0
                        self.state1_turn_270_prev_yaw = yaw_rad
                        self.state1_turn_270_swept_rad = 0.0
                        self.state1_turning_270 = True
            elif self.state1_seeking_clear_direction:
                action = self._move3d(
                    current_pos, cur, current_velocity,
                    roll=roll, pitch=pitch, roll_rate=roll_rate, pitch_rate=pitch_rate, **mk,
                )
                yaw_rad = self._get_yaw(observation) or 0.0
                turn_dir = self.state1_turn_direction if self.state1_turn_direction is not None else (1 if np.random.rand() < 0.5 else -1)
                action[4] = float(np.clip(
                    np.clip(yaw_rad / np.pi, -1.0, 1.0) + self.STATE1_TURN_YAW_STEP_NORM * turn_dir,
                    -1.0, 1.0,
                ))
                if self.state1_danger_steps_remaining == 0 and self._center_depth_clear_at_least_m(observation, self.CLEAR_DIRECTION_MIN_DEPTH_M, front_patch_radius_px=self.FRONT_PATCH_RADIUS_PX):
                    look_dir = self._get_look_direction_world(observation, camera_fov_rad=self.CAMERA_FOV_RAD)
                    if look_dir is not None:
                        depth_m = self._depth_meters_for_direction(observation, look_dir, camera_fov_rad=self.CAMERA_FOV_RAD)
                        dist = min(max(0.0, (float(depth_m) - 3.0) if depth_m else 0.0), 1.0) if depth_m is not None else 1.0
                        milestone = np.asarray(cur + dist * look_dir, dtype=np.float32)
                        self.target.insert(0, milestone)
                        self.state1_has_detour_waypoint = True
                        self.state1_seeking_clear_direction = False
                return action

            if self.state1_turning_270:
                action = self._move3d(
                    current_pos, cur, current_velocity,
                    roll=roll, pitch=pitch, roll_rate=roll_rate, pitch_rate=pitch_rate, **mk,
                )
                yaw_rad = self._get_yaw(observation) or 0.0
                if self.state1_turn_270_prev_yaw is not None:
                    delta = (yaw_rad - self.state1_turn_270_prev_yaw + np.pi) % (2 * np.pi) - np.pi
                    self.state1_turn_270_swept_rad += delta / np.pi
                self.state1_turn_270_prev_yaw = yaw_rad
                action[4] = float(np.clip((yaw_rad / np.pi + self.STATE2_YAW_STEP_RAD + 1) % 2 - 1, -1.0, 1.0))
                if self.state1_turn_270_swept_rad >= self.STATE1_TURN_270_TARGET_RAD:
                    self.state1_turning_270 = False
                    self.state1_turn_270_prev_yaw = None
                    self.state1_turn_270_swept_rad = 0.0
                    next_pt = self._state1_next_random_target()
                    if next_pt is not None:
                        self.target = [next_pt]
                return action

            if self.target:
                wp = np.asarray(self.target[0], dtype=np.float32).reshape(3).copy()
                wp[2] = cur[2]
                action = self._move3d(
                    current_pos, wp, current_velocity,
                    roll=roll, pitch=pitch, roll_rate=roll_rate, pitch_rate=pitch_rate, **mk,
                )
                if np.linalg.norm(wp - cur) < self.AT_TARGET_TOLERANCE and np.linalg.norm(current_velocity) < self.AT_TARGET_SPEED_THRESHOLD:
                    self.target.pop(0)
                    yaw_rad = self._get_yaw(observation) or 0.0
                    self.state1_turn_270_prev_yaw = yaw_rad
                    self.state1_turn_270_swept_rad = 0.0
                    self.state1_turning_270 = True
                return action

            return np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        action = np.random.uniform(-1, 1, size=5)
        action[3] = np.clip(action[3], 0, 1)
        return action

    def reset(self):
        self.state = 0
        self.target = []
        self.search_area_center = None
        self.initial_position = None
        self.detected_goal_position = None
        self.detected_goal_direction = None
        self.state1_turning_270 = False
        self.state1_turn_270_prev_yaw = None
        self.state1_turn_270_swept_rad = 0.0
        self.state1_turn_direction = None
        self.state1_seeking_clear_direction = False
        self.state1_chosen_clear_direction = None
        self.state1_has_detour_waypoint = False
        self.state1_danger_steps_remaining = 0
        self.state3_no_goal_steps = 0
        self.act_step = 0
        self.goal_detection_count = 0
