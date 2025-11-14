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
            observation: numpy array shape (12,) - [x, y, z, vx, vy, vz, ...]
        
        Returns:
            action: numpy array shape (4,) - RPM commands for 4 rotors
        """
        import numpy as np
        return np.random.uniform(-1, 1, size=4)
    
    def reset(self):
        """Called at episode start - optional state reset"""
        pass

