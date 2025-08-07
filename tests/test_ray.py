#!/usr/bin/env python3
"""
Visual demonstration of MovingDroneAviary obstacle detection.

Shows that ray sensors correctly ignore visual-only objects and detect real collision obstacles.
Includes smooth camera controls and real-time ray visualization.
"""

import time
import numpy as np
import pybullet as p
import math
from gym_pybullet_drones.utils.enums import ActionType, ObservationType

from swarm.constants import HORIZON_SEC, SIM_DT
from swarm.core.moving_drone import MovingDroneAviary
from swarm.validator.task_gen import MapTask

class CameraController:
    """Smooth camera controller with keyboard and mouse support."""
    
    def __init__(self):
        self.camera_distance = 8.0
        self.camera_yaw = 45.0
        self.camera_pitch = -30.0
        self.camera_target = [0.0, 0.0, 1.0]
        
        self.rotation_speed = 2.0
        self.movement_speed = 0.5
        self.zoom_speed = 0.5
        self.mouse_sensitivity = 0.5
        
        self.last_mouse_x = None
        self.last_mouse_y = None
        
    def update_camera(self):
        """Update camera based on keyboard and mouse input."""
        keys = p.getKeyboardEvents()
        current_keys = set(keys.keys())
        dt = 0.016
        
        # Camera rotation with A/D
        if ord('a') in current_keys or ord('A') in current_keys:
            self.camera_yaw -= self.rotation_speed * 60 * dt
        if ord('d') in current_keys or ord('D') in current_keys:
            self.camera_yaw += self.rotation_speed * 60 * dt
            
        # Zoom with W/S
        if ord('w') in current_keys or ord('W') in current_keys:
            self.camera_distance = max(1.0, self.camera_distance - self.zoom_speed * 60 * dt)
        if ord('s') in current_keys or ord('S') in current_keys:
            self.camera_distance = min(50.0, self.camera_distance + self.zoom_speed * 60 * dt)
            
        # Target height with Q/E
        if ord('q') in current_keys or ord('Q') in current_keys:
            self.camera_target[2] -= self.movement_speed * 60 * dt
        if ord('e') in current_keys or ord('E') in current_keys:
            self.camera_target[2] += self.movement_speed * 60 * dt
            
        # Reset camera with R
        if ord('r') in current_keys or ord('R') in current_keys:
            self.reset_camera()
            
        # Handle mouse input (simplified for compatibility)
        try:
            mouse_events = p.getMouseEvents()
            for event in mouse_events:
                if len(event) >= 4 and event[0] == 2:  # Mouse move event
                    mouse_x, mouse_y = event[1], event[2]
                    
                    if len(event) > 3 and event[3] == 1:  # Left mouse button held
                        if self.last_mouse_x is not None and self.last_mouse_y is not None:
                            dx = mouse_x - self.last_mouse_x
                            dy = mouse_y - self.last_mouse_y
                            
                            self.camera_yaw += dx * self.mouse_sensitivity * 60 * dt
                            self.camera_pitch = max(-89, min(89, self.camera_pitch - dy * self.mouse_sensitivity * 60 * dt))
                    
                    self.last_mouse_x = mouse_x
                    self.last_mouse_y = mouse_y
                elif len(event) >= 1 and event[0] == 1:  # Mouse button released
                    self.last_mouse_x = None
                    self.last_mouse_y = None
        except:
            pass
        
        self.apply_camera()
        
    def reset_camera(self):
        """Reset camera to default position."""
        self.camera_distance = 8.0
        self.camera_yaw = 45.0
        self.camera_pitch = -30.0
        self.camera_target = [0.0, 0.0, 1.0]
        
    def apply_camera(self):
        """Apply current camera settings to PyBullet."""
        p.resetDebugVisualizerCamera(
            cameraDistance=self.camera_distance,
            cameraYaw=self.camera_yaw,
            cameraPitch=self.camera_pitch,
            cameraTargetPosition=self.camera_target
        )

def create_demo_scene():
    """Create test scene with visual-only and collision obstacles."""
    drone_x, drone_y, drone_z = 0.0, 0.0, 1.0
    
    # Visual-only sphere (should be ignored)
    visual_sphere = p.createVisualShape(
        p.GEOM_SPHERE, 
        radius=0.6,
        rgbaColor=[1.0, 1.0, 0.0, 0.8]
    )
    sphere_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=visual_sphere,
        basePosition=[drone_x + 2.5, drone_y, drone_z]
    )
    
    # Real collision box (should be detected)
    collision_box = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.4, 0.4, 0.4])
    visual_box = p.createVisualShape(
        p.GEOM_BOX, 
        halfExtents=[0.4, 0.4, 0.4],
        rgbaColor=[0.0, 0.0, 1.0, 1.0]
    )
    box_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision_box,
        baseVisualShapeIndex=visual_box,
        basePosition=[drone_x + 5.0, drone_y, drone_z]
    )
    
    # Additional collision obstacles around the scene
    obstacles = [
        # Green cylinder - right
        {
            'collision': p.createCollisionShape(p.GEOM_CYLINDER, radius=0.3, height=1.5),
            'visual': p.createVisualShape(p.GEOM_CYLINDER, radius=0.3, length=1.5, rgbaColor=[0.0, 1.0, 0.0, 1.0]),
            'position': [drone_x, drone_y + 4.0, drone_z]
        },
        # Blue wall - left
        {
            'collision': p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 1.0, 1.0]),
            'visual': p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 1.0, 1.0], rgbaColor=[0.0, 0.0, 1.0, 1.0]),
            'position': [drone_x, drone_y - 3.5, drone_z]
        },
        # Orange block - behind
        {
            'collision': p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.8]),
            'visual': p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.8], rgbaColor=[0.8, 0.4, 0.0, 1.0]),
            'position': [drone_x - 4.5, drone_y, drone_z]
        },
        # Purple pillar - diagonal front-right
        {
            'collision': p.createCollisionShape(p.GEOM_CYLINDER, radius=0.2, height=2.0),
            'visual': p.createVisualShape(p.GEOM_CYLINDER, radius=0.2, length=2.0, rgbaColor=[0.6, 0.0, 0.6, 1.0]),
            'position': [drone_x + 3.0, drone_y + 3.0, drone_z]
        },
        # Cyan block - diagonal back-left
        {
            'collision': p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.4, 0.4, 0.6]),
            'visual': p.createVisualShape(p.GEOM_BOX, halfExtents=[0.4, 0.4, 0.6], rgbaColor=[0.0, 0.8, 0.8, 1.0]),
            'position': [drone_x - 3.5, drone_y - 2.5, drone_z]
        },
        # Magenta cube - above
        {
            'collision': p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.3, 0.3, 0.3]),
            'visual': p.createVisualShape(p.GEOM_BOX, halfExtents=[0.3, 0.3, 0.3], rgbaColor=[1.0, 0.0, 1.0, 1.0]),
            'position': [drone_x + 1.0, drone_y + 1.0, drone_z + 2.5]
        },
        # Dark yellow block - below
        {
            'collision': p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.4, 0.4, 0.2]),
            'visual': p.createVisualShape(p.GEOM_BOX, halfExtents=[0.4, 0.4, 0.2], rgbaColor=[0.5, 0.5, 0.0, 1.0]),
            'position': [drone_x - 1.0, drone_y - 1.0, drone_z - 0.5]
        },
        # Dark red cylinder - diagonal front-left
        {
            'collision': p.createCollisionShape(p.GEOM_CYLINDER, radius=0.25, height=1.8),
            'visual': p.createVisualShape(p.GEOM_CYLINDER, radius=0.25, length=1.8, rgbaColor=[0.8, 0.2, 0.2, 1.0]),
            'position': [drone_x + 3.5, drone_y - 2.8, drone_z]
        },
        # Light green block - diagonal back-right
        {
            'collision': p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.35, 0.35, 0.7]),
            'visual': p.createVisualShape(p.GEOM_BOX, halfExtents=[0.35, 0.35, 0.7], rgbaColor=[0.2, 0.8, 0.2, 1.0]),
            'position': [drone_x - 2.8, drone_y + 3.2, drone_z]
        }
    ]
    
    # Create all obstacles
    for obstacle in obstacles:
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=obstacle['collision'],
            baseVisualShapeIndex=obstacle['visual'],
            basePosition=obstacle['position']
        )
    
    # Ground plane
    ground_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[20.0, 20.0, 0.1])
    ground_visual = p.createVisualShape(
        p.GEOM_BOX, 
        halfExtents=[20.0, 20.0, 0.1],
        rgbaColor=[0.7, 0.7, 0.7, 1.0]
    )
    ground_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=ground_collision,
        baseVisualShapeIndex=ground_visual,
        basePosition=[0, 0, -0.1]
    )
    
    return sphere_id, box_id, ground_id

def visualize_rays(env, obs):
    """Visualize ray casting from MovingDroneAviary observation."""
    drone_pos = np.array([0.0, 0.0, 1.0])
    drone_rotation = np.eye(3)
    
    # Extract distances from observation
    if len(obs.shape) > 1 and obs.shape[0] == 1:
        obs_flat = obs[0]
    else:
        obs_flat = obs
        
    if len(obs_flat) == 131:  # Updated: 112 base + 16 distances + 3 goal = 131
        distance_obs = obs_flat[112:128]  # 16 distance values (normalized)
        distances = distance_obs * 10.0  # Convert back to meters
    else:
        distances = env._get_obstacle_distances(drone_pos, drone_rotation)
    
    # Draw rays
    for i, (direction, distance) in enumerate(zip(env.ray_directions, distances)):
        world_direction = drone_rotation @ direction
        ray_end_point = drone_pos + world_direction * distance
        
        # Color: Red if hit, Blue if clear
        color = [1, 0, 0] if distance < env.max_ray_distance else [0, 0, 1]
            
        # Draw ray line
        p.addUserDebugLine(
            drone_pos,
            ray_end_point,
            lineColorRGB=color,
            lineWidth=1.0,
            lifeTime=0
        )
        
        # Draw hit point marker
        if distance < env.max_ray_distance:
            p.addUserDebugLine(
                ray_end_point,
                ray_end_point + np.array([0, 0, 0.2]),
                lineColorRGB=[1, 1, 0],
                lineWidth=3.0,
                lifeTime=0
            )

def main():
    """Main demonstration function."""    
    # Create task and environment
    task = MapTask(
        start=(0.0, 0.0, 1.0),
        goal=(10.0, 10.0, 1.0),
        horizon=HORIZON_SEC,
        map_seed=42,
        sim_dt=SIM_DT,
    )
    
    # Setup environment
    ctrl_freq = int(round(1.0 / task.sim_dt))
    common_kwargs = dict(
        gui=True,
        record=False,
        obs=ObservationType.KIN,
        ctrl_freq=ctrl_freq,
        pyb_freq=ctrl_freq,
    )
    env = MovingDroneAviary(task, act=ActionType.VEL, **common_kwargs)
    
    # Initialize camera controller
    camera = CameraController()
    
    # Configure visualization
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
    
    # Reset environment and create scene
    env.reset(seed=task.map_seed)
    create_demo_scene()
    
    # Get initial observation
    action = np.array([[0, 0, 0, 0]])
    obs, _, _, _, _ = env.step(action)
    
    # Initial visualization
    visualize_rays(env, obs)
    camera.apply_camera()
    
    try:
        while True:
            # Update camera
            camera.update_camera()
            
            # Step simulation
            action = np.array([[0, 0, 0, 0]])
            obs, _, _, _, _ = env.step(action)
            
            time.sleep(0.016)  # ~60 FPS
            
    except KeyboardInterrupt:
        print("Demo completed!")
    finally:
        env.close()

if __name__ == "__main__":
    main() 