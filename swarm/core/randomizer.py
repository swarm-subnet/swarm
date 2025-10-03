from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class SensorNoiseModule:
    """
    Adds realistic sensor noise to drone observations for sim-to-real transfer.
    
    Simulates noise characteristics of real hardware sensors:
    - GPS: position drift and signal dropouts
    - IMU: attitude estimation errors
    - Gyroscope: measurement noise with bias drift
    - LiDAR: range measurement errors, spikes, and dropouts
    """
    
    NOISE_CONFIGS = {
        "good": {
            "gps_h": 0.2, "gps_v": 0.1, "euler": 0.5, "vel": 0.05,
            "gyro": 0.02, "gyro_drift": 0.0001, "lidar": 0.05
        },
        "medium": {
            "gps_h": 0.6, "gps_v": 0.3, "euler": 1.0, "vel": 0.1,
            "gyro": 0.035, "gyro_drift": 0.0003, "lidar": 0.07
        },
        "hard": {
            "gps_h": 1.0, "gps_v": 0.5, "euler": 2.0, "vel": 0.2,
            "gyro": 0.05, "gyro_drift": 0.0005, "lidar": 0.10
        }
    }
    
    def __init__(self, noise_level: str = "medium", enable_noise: bool = True):
        """
        Initialize the SensorNoiseModule.

        Args:
            noise_level: Difficulty level ("good", "medium", "hard")
            enable_noise: Whether to apply noise (True) or pass through (False)
        """
        if noise_level not in self.NOISE_CONFIGS:
            raise ValueError(f"noise_level must be one of {list(self.NOISE_CONFIGS.keys())}")

        self.noise_level = noise_level
        self.enable_noise = enable_noise
        self.cfg = self.NOISE_CONFIGS[noise_level]

        self.last_valid_gps: NDArray[np.float32] | None = None
        self.gps_dropout_counter = 0
        self.gyro_bias = np.zeros(3, dtype=np.float32)
    
    def reset(self) -> None:
        """Reset the noise module state for a new episode."""
        self.last_valid_gps = None
        self.gps_dropout_counter = 0
        self.gyro_bias = np.zeros(3, dtype=np.float32)
    
    def _add_gps_noise(self, position: NDArray[np.float32]) -> NDArray[np.float32]:
        if np.random.random() < 0.02:
            self.gps_dropout_counter = np.random.randint(5, 50)
        
        if self.gps_dropout_counter > 0:
            self.gps_dropout_counter -= 1
            if self.last_valid_gps is not None:
                return self.last_valid_gps.copy()
            return position.copy()
        
        noise = np.array([
            np.random.normal(0, self.cfg["gps_h"]),
            np.random.normal(0, self.cfg["gps_h"]),
            np.random.normal(0, self.cfg["gps_v"])
        ], dtype=np.float32)
        
        noisy_pos = position + noise
        self.last_valid_gps = noisy_pos.copy()
        return noisy_pos
    
    def _add_euler_noise(self, euler: NDArray[np.float32]) -> NDArray[np.float32]:
        deg_to_rad = np.pi / 180.0
        noise = np.random.normal(0, self.cfg["euler"] * deg_to_rad, size=3).astype(np.float32)
        return euler + noise
    
    def _add_velocity_noise(self, velocity: NDArray[np.float32]) -> NDArray[np.float32]:
        noise = np.random.normal(0, self.cfg["vel"], size=3).astype(np.float32)
        return velocity + noise
    
    def _add_gyro_noise(self, angular_vel: NDArray[np.float32]) -> NDArray[np.float32]:
        noise = np.random.normal(0, self.cfg["gyro"], size=3).astype(np.float32)
        self.gyro_bias += np.random.normal(0, self.cfg["gyro_drift"], size=3).astype(np.float32)
        return angular_vel + noise + self.gyro_bias
    
    def _add_lidar_noise(self, distances: NDArray[np.float32]) -> NDArray[np.float32]:
        noisy = []
        for dist in distances:
            noise = np.random.normal(0, self.cfg["lidar"])
            d = dist + noise
            
            if np.random.random() < 0.01:
                d = 20.0
            if np.random.random() < 0.01:
                d = 20.0
            
            noisy.append(np.clip(d, 0.0, 20.0))
        
        return np.array(noisy, dtype=np.float32)
    
    def apply_noise(self, obs: NDArray[np.float32], goal_pos: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Apply sensor noise to observation vector.
        
        Observation structure (131-D):
            [0:3]     Position (x, y, z)
            [3:6]     Euler angles (roll, pitch, yaw)
            [6:9]     Linear velocity (vx, vy, vz)
            [9:12]    Angular velocity (wx, wy, wz)
            [12:112]  Action history buffer
            [112:128] LiDAR distances (16 rays, normalized)
            [128:131] Relative goal vector
        """
        if not self.enable_noise:
            return obs
        
        noisy_obs = obs.copy()
        
        noisy_obs[0, 0:3] = self._add_gps_noise(obs[0, 0:3])
        noisy_obs[0, 3:6] = self._add_euler_noise(obs[0, 3:6])
        noisy_obs[0, 6:9] = self._add_velocity_noise(obs[0, 6:9])
        noisy_obs[0, 9:12] = self._add_gyro_noise(obs[0, 9:12])
        
        lidar_norm = obs[0, 112:128]
        lidar_m = lidar_norm * 20.0
        noisy_lidar_m = self._add_lidar_noise(lidar_m)
        noisy_obs[0, 112:128] = noisy_lidar_m / 20.0
        
        noisy_pos = noisy_obs[0, 0:3]
        rel_goal = (goal_pos - noisy_pos) / 20.0
        noisy_obs[0, 128:131] = rel_goal
        
        return noisy_obs

