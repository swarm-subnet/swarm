# ---------------------------------------------------------------------
# Camera helper ─ follow the first drone at ~60 Hz
# ---------------------------------------------------------------------
import numpy as np, pybullet as p, pybullet_data

def track_drone(cli, drone_id) -> None:
    """Keep the PyBullet spectator camera locked on the drone."""
    pos, _ = p.getBasePositionAndOrientation(drone_id,
                                             physicsClientId=cli)
    tgt = np.add(pos, [0.0, 0.0, 0.4])                 # look ≈0.4 m above CG
    p.resetDebugVisualizerCamera(cameraDistance=1,   # zoom-out
                                 cameraYaw=0,
                                 cameraPitch=-25,       # slight downward tilt
                                 cameraTargetPosition=tgt,
                                 physicsClientId=cli)