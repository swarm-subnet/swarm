"""Starter agent for the OFFICE INTERCEPTOR family (cf_office_interceptor).

Copy this file to drone_agent.py in your submission zip. The office contract
differs from the SAR families: 4 RC-stick actions and an rgb+state observation.

Observation (dict):
    - "rgb": numpy array (256, 256, 3), float32 in [0, 1] — the drone's forward
      camera. Fresh frames arrive at ~25 Hz and are held between captures, like
      the real Tello's 30 fps stream. Colors, lighting and camera imperfections
      are randomized every episode: geometry is the reliable signal, not color.
    - "state": numpy array (127,), float32. Index ranges:
        0:4    attitude — pitch, roll (rad), sin(yaw), cos(yaw)
        4:7    body velocity — forward, right, down (m/s, SDK convention)
        7:10   body specific force — forward, right, down (m/s^2; hover reads
               ~-9.8 on the down axis, like the real IMU)
        10:15  altitude — downward ToF (m), fused height (m), barometer (m),
               packet age (s), valid flag
        15:27  detector — box count, age since the last frame that reported any
               box (s; false positives reset it too), then two
               [cx, cy, w, h, confidence] slots normalized to the rgb frame
               (boxes overlay obs["rgb"] directly); telling real boxes from
               ghosts and tracking the target are YOUR job
        27:127 action history — the last 25 actions, 4 values each

Action:
    numpy array (4,) of Tello RC sticks in [-1, 1]: [lr, fb, ud, yaw]
    - lr: strafe left (-1) / right (+1)
    - fb: backward (-1) / forward (+1) along the camera heading
    - ud: descend (-1) / ascend (+1)
    - yaw: turn rate, counterclockwise (-1) / clockwise (+1)
    [0, 0, 0, 0] hovers. While the detector has not truly seen the target for
    0.8 s, the safety interlock slows forward flight and strafing to a crawl
    (backing off stays full speed) — search with yaw and altitude, then commit.

Mission: find the target drone and physically hit it. Contact ends the episode
as a success; crashing into the office ends it as a failure.

Scoring:
    score = 0.5 x success + 0.5 x time
    A catch returns 0.5 plus up to 0.5 for speed; timeout, collision and other
    legitimate failures return 0.01 participation.

Constraints:
    - Control rate: 50 Hz (dt = 1/50 s), episode horizon 60 s
    - Full stick = 3.0 m/s per axis (Tello slow mode); ceiling at 3 m
"""

import numpy as np


class DroneFlightController:
    """Minimal correct office agent: climbs off the floor, then yaw-searches."""

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
            action[3] = 0.4          # search by turning until the detector reports a box
        return action

    def reset(self):
        # Called before each new mission; clear any internal state here.
        pass
