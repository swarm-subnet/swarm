class RLAgent:
    """
    Miner's agent implementation.
    This is the ONLY file miners need to modify.
    """
    
    def __init__(self):
        """Load your trained model here using any framework (PyTorch, JAX, etc.)"""
        pass
    
    def act(self, observation):
        """
        Given observation, return action.
        
        Args:
            observation: dict with keys:
                - "rgb": numpy array shape (96, 96, 4) - RGBA camera image
                - "state": numpy array shape (N,) - state vector including position, orientation, velocities, and action history
        
        Returns:
            action: numpy array shape (5,) - [vx, vy, vz, speed, yaw]
                - vx, vy, vz: velocity direction components (normalized -1 to +1)
                - speed: speed multiplier (0 to 1)
                - yaw: target yaw angle (normalized -1 to +1, maps to -π to +π radians)
        """
        import numpy as np
        return np.random.uniform(-1, 1, size=5)
    
    def reset(self):
        """Called at episode start - optional state reset"""
        pass

