# aviary_raw.py ----------------------------------------------------------
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import ActionType
import numpy as np

class HoverAviaryRawRPM(HoverAviary):
    """ActionType.RPM but the value *is* the RPM (0 … MAX_RPM)."""
    def _preprocessAction(self, action):
        # action shape = (num_drones, 4)
        return np.clip(action, 0.0, self.MAX_RPM)