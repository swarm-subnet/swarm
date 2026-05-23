import numpy as np


class DroneFlightController:
    """
    Swarm Subnet 124 — Autonomous Drone Flight Controller (V5 Search-and-Rescue)

    Implement your navigation policy in this class. The controller receives
    observations from a simulated drone and must output flight commands to
    locate a humanoid victim on the ground and hover over it.

    Observation Space:
        Dictionary with two keys:
        - "depth": numpy array (128, 128, 1) — normalized depth map [0,1] for the
          0.5-20m range. Use this to find the victim with vision.
        - "state": numpy array (N,) — flight state vector. Index ranges:
            [0:3]   drone position (x, y, z) in meters
            [3:6]   orientation (roll, pitch, yaw)
            [6:9]   linear velocities (vx, vy, vz) m/s
            [9:12]  angular velocities (wx, wy, wz) rad/s
            [12:12+A] action history (A = ACTION_BUFFER_SIZE × action_dim)
            [12+A]  altitude (normalized)
            [12+A+1:12+A+3] search-clue offset (Δx, Δy) — 2D, victim XY is
              somewhere within a 30 m circle around this clue. No Z component.

    Mission (V5 Search-and-Rescue):
        - You receive a noisy 2D coordinate inside a 30 m circle around the
          victim's true XY. There is no Z in the clue.
        - Travel to the area, use depth to find the victim, then hover 2-4 m
          above the victim's AABB top, within 2 m horizontal distance, at less
          than 1.0 m/s speed, for 2 continuous seconds.
        - The victim has a 0.8 m no-touch sphere — entering it ends the
          episode as a terminal failure.

    Action Space:
        numpy array (5,) containing [dir_x, dir_y, dir_z, speed, yaw]
        - dir_x, dir_y, dir_z: direction components, range [-1, 1]
        - speed: thrust multiplier, range [0, 1]
        - yaw: target yaw normalized, range [-1, 1] maps to [-π, π]

    Scoring:
        score = 0.45 × success + 0.45 × time + 0.10 × safety
        Successful CONFIRMED returns the full scoring stack.
        Legitimate non-success failures (collision, no-touch-sphere,
        infeasible, spawn-failure, tilt, timeout) return 0.01 participation.

    Constraints:
        - Max velocity: 3.0 m/s (enforced by validator)
        - Max yaw rate: 3.141 rad/s (180°/s)
        - Simulation rate: 30 Hz control / 30 Hz physics
    """
    
    def __init__(self):
        """
        Initialize your flight controller.
        
        Load your trained model here using any ML framework
        
        Example:
            import torch
            self.model = torch.jit.load("policy.pt")
            self.model.eval()
        """
        pass
    
    def act(self, observation):
        """
        Compute flight action for current observation.
        
        Args:
            observation: dict with "depth" (128,128,1) and "state" (N,) arrays
        
        Returns:
            numpy array (5,) containing [dir_x, dir_y, dir_z, speed, yaw]
        """
        action = np.random.uniform(-1, 1, size=5)
        action[3] = np.clip(action[3], 0, 1)
        return action
    
    def reset(self):
        """
        Reset controller state at mission start.
        
        Called before each new mission. Use this to reset any internal
        state like hidden states, observation buffers, or counters.
        """
        pass
