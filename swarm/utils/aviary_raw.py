# aviary_raw.py ----------------------------------------------------------
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
import numpy as np

class HoverAviaryRawRPM(HoverAviary):
    """Override HoverAviary so that it can receive raw RPMs"""
    def _preprocessAction(self, action):
        # action shape = (num_drones, 4)
        return np.clip(action, 0.0, self.MAX_RPM)