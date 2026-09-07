# The MIT License (MIT)
# Copyright © 2026 Swarm

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from pathlib import Path

import numpy as np
from stable_baselines3 import PPO


class DroneFlightController:
    """Baseline PPO controller: loads the ppo_policy.zip packaged next to this file.

    Works for every challenge family: depth frames are downsampled to the
    resolution the policy was trained on, extra observation keys (rgb) are
    ignored, and multi-drone observations run the same policy once per drone.
    """

    def __init__(self):
        """Load the packaged policy and cache the input size and action bounds it was trained with."""
        policy_path = Path(__file__).resolve().parent / "ppo_policy.zip"
        self._model = PPO.load(str(policy_path), device="cpu")
        self._depth_size = int(self._model.observation_space["depth"].shape[0])
        self._low = self._model.action_space.low
        self._high = self._model.action_space.high

    def _policy_obs(self, depth, state):
        """Downsample one depth frame to the policy resolution and pair it with the state vector.

        Args:
            depth: Depth image of shape (H, W, 1) from the validator observation.
            state: Flat state vector for the same drone.

        Returns:
            Observation dict with the "depth" and "state" keys the policy expects.
        """
        step = max(1, depth.shape[0] // self._depth_size)
        return {
            "depth": np.ascontiguousarray(depth[::step, ::step, :], dtype=np.float32),
            "state": np.asarray(state, dtype=np.float32),
        }

    def _predict(self, depth, state):
        """Run the policy on one drone and clip the action to the contract bounds."""
        action, _ = self._model.predict(self._policy_obs(depth, state), deterministic=True)
        return np.clip(action, self._low, self._high)

    def act(self, observation):
        """Return the action for one drone, or one action per drone in a multi-drone task.

        A 2-D state means one row per drone, in which case the policy runs once
        per row and the results are stacked in the same order.

        Args:
            observation: Validator observation dict with at least "depth" and "state".

        Returns:
            float32 action array, shape (action_dim,) or (num_drones, action_dim).
        """
        depth = np.asarray(observation["depth"])
        state = np.asarray(observation["state"])
        if state.ndim == 2:
            actions = [self._predict(depth[i], state[i]) for i in range(state.shape[0])]
            return np.stack(actions).astype(np.float32)
        return np.asarray(self._predict(depth, state), dtype=np.float32)

    def reset(self):
        """Called at the start of every episode; the baseline keeps no state between steps."""
        pass
