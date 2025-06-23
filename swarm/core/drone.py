# ---------------------------------------------------------------------
# Camera helper ─ follow the first drone at ~60 Hz
# ---------------------------------------------------------------------
import numpy as np, pybullet as p, pybullet_data

def track_drone(cli, drone_id) -> None:
    """Keep the PyBullet spectator camera locked on the drone."""
    pos, _ = p.getBasePositionAndOrientation(drone_id,
                                             physicsClientId=cli)
    tgt = np.add(pos, [0.0, 0.0, 0.4])                 # look ≈0.4 m above CG
    p.resetDebugVisualizerCamera(cameraDistance=2,   # zoom-out
                                 cameraYaw=0,
                                 cameraPitch=-25,       # slight downward tilt
                                 cameraTargetPosition=tgt,
                                 physicsClientId=cli)
    
def safe_disconnect_gui(client_id: int) -> None:
    """
    Close a GUI connection that uses the ExampleBrowser without crashing
    Mesa/Intel drivers.

    1.  Disable visualiser ⇒ the render thread stops touching GL.
    2.  Sleep ~50 ms     ⇒ lets the ExampleBrowser message-loop finish
                           one full tick.
    3.  Finally call p.disconnect().
    """
    import pybullet as p, time

    try:
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0,
                                   physicsClientId=client_id)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,        0,
                                   physicsClientId=client_id)
        p.removeAllUserDebugItems(physicsClientId=client_id)
        time.sleep(0.5)                 # one browser tick (~16 ms) x3
    finally:
        p.disconnect(physicsClientId=client_id)