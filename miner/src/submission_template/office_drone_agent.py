"""Starter agent for the OFFICE INTERCEPTOR family (cf_interceptor_office).

Copy this file to drone_agent.py in your submission zip. The office contract
differs from the SAR families: 4 RC-stick actions and an rgb+state observation.

Observation (dict):
    - "rgb": numpy array (256, 256, 3), float32 in [0, 1] — the drone's forward
      camera. Fresh frames arrive at ~25 Hz and are held between captures, like
      the real Tello's 30 fps stream. Colors, lighting and camera imperfections
      are randomized every episode: geometry is the reliable signal, not color.
      The TARGET IS NOT VISIBLE in the image (and the ToF never returns it) —
      use the camera for navigating, the detector block for finding the target.
    - "state": numpy array (127,), float32. Index ranges:
        0:4    attitude — pitch, roll (rad), sin(yaw), cos(yaw); yaw is RELATIVE
               to your takeoff heading (seeded per episode) — no world compass
        4:7    body velocity — forward, right, down (m/s, SDK convention)
        7:10   body specific force — forward, right, down (m/s^2; hover reads
               ~-9.8 on the down axis, like the real IMU)
        10:15  altitude — downward ToF (m), fused height (m), barometer (m),
               packet age (s), valid flag
        15:27  detector — box count, age since the last frame that reported any
               box (s; false positives reset it too), then two
               [cx, cy, w, h, confidence] slots normalized to the rgb frame
               (same camera projection: a box marks where the invisible target
               would appear in obs["rgb"]); telling real boxes from ghosts and
               tracking the target are YOUR job — this block is the ONLY
               sighting channel
        27:127 action history — the last 25 actions, 4 values each

Action:
    numpy array (4,) of Tello RC sticks in [-1, 1]: [lr, fb, ud, yaw]
    - lr: strafe left (-1) / right (+1)
    - fb: backward (-1) / forward (+1) along the camera heading
    - ud: descend (-1) / ascend (+1)
    - yaw: turn rate, counterclockwise (-1) / clockwise (+1)
    [0, 0, 0, 0] hovers.

Mission: find the target drone and intercept it alongside. Success is physical
contact, or holding within 0.20 m of it horizontally while level with it (within
0.08 m of its height) for three control steps. Hovering above it is not an
interception and scores nothing. Crashing into the office ends the episode as a
failure.

Scoring:
    score = 0.5 x success + 0.5 x time
    A catch returns 0.5 plus up to 0.5 for speed; timeout, collision and other
    legitimate failures return 0.01 participation.

Constraints:
    - Control rate: 50 Hz (dt = 1/50 s), episode horizon 60 s
    - Full stick = about 3.0 m/s per axis (Tello slow mode); ceiling about 3 m
    - The airframe is dealt per episode: speed, dead zone, slew rate and motor
      lag all move by up to 18%, the room by 2-5% on each axis, and the target's
      apparent size by 10%. Nothing here is a constant worth memorising.
"""

import numpy as np


class DroneFlightController:
    """Contract smoke-test baseline: climbs off the floor, then yaws without intercepting."""

    def __init__(self):
        # Load your trained model here (any framework).
        pass

    def act(self, observation):
        state = observation["state"]
        fused_height = float(state[11])
        action = np.zeros(4, dtype=np.float32)
        if fused_height < 1.2:
            action[2] = 0.4          # take off gently (full-stick hits the 3 m ceiling)
        else:
            action[3] = 0.4          # turn in place without intercepting
        return action

    def reset(self):
        # Called before each new mission; clear any internal state here.
        pass
