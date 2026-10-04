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

"""Solar Patrol challenge family: a DJI Matrice 4TD patrols a solar park out of its Dock 3.

One episode is one patrol: the dock opens, the drone takes off, searches the park, reports intruders, returns and
lands, all inside 390 s. The model decides ten times a second while physics keeps the shared 50 Hz.

This file only wires the parts together, in the order below. Each part lives in its own module and owns one
piece of the patrol; the contract module owns what crosses between the model and the simulator.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Optional

import numpy as np
from gymnasium import spaces

from swarm.domain_model import CHALLENGE_TYPE_TO_ENVIRONMENT_TYPE
from swarm.protocol import SCHEMA_VERSION, FailureReason

from ..base import ChallengeFamilyEvaluation, ChallengeFamilyRuntime, ChallengeFamilyRuntimeProfile
from . import (
    airframe,
    camera,
    coverage,
    decoys,
    dock,
    drone_state,
    flight_limit,
    ground_distance,
    laser,
    outputs,
    park,
    reports,
    score,
    seed_checks,
    sensor_noise,
    theft,
    wind,
    zoom,
)
from .contract import (
    ACTION_DIM,
    ACTION_HIGH,
    ACTION_LOW,
    CHALLENGE_TYPE,
    DECISION_STEPS,
    FAMILY_ID,
    Outcome,
    decode_action,
    new_site_map,
    new_state,
)
from .episode import SolarEpisode
from .task import solar_patrol_task

# Build order on reset: the world first, then what stands in it, then the sensors that look at it.
_PARTS = (park, dock, airframe, flight_limit, wind, theft, decoys, coverage, camera, zoom, laser, ground_distance,
          drone_state, sensor_noise, reports, score, outputs)

_FAILURE_BY_END = {
    "landed": FailureReason.NONE.value,
    "timeout": FailureReason.TIMEOUT.value,
    "collision": FailureReason.OBSTACLE_COLLISION.value,
    "flight_limit": FailureReason.NONE.value,
}


class SolarPatrolChallengeFamily(ChallengeFamilyRuntime):
    """The patrol runtime: the hooks the environment and the evaluator call, each handed to the parts in turn."""

    family_id = FAMILY_ID
    runtime_supported = True
    decision_steps = DECISION_STEPS
    physics_mode = "pyb_drag"            # rotor drag in still air too, the same law the seeded wind applies
    seeded_sun = park.SEEDED_SUN
    night_share = park.NIGHT_SHARE
    sky_from_sun = park.SKY_FROM_SUN
    daylight = park.DAYLIGHT
    render_backend = camera.RENDER_BACKEND
    prepares_seeds = True                # every seed is flown once by the seed checks' reference pilot
    # Off until the fairness of acts timed beside a frame is decided; on needs a wheel with CAMERA_RELEASES_GIL.
    observation_ahead = False
    clearance_metric = False             # the patrol's score reads no min_clearance

    # ------------------------------------------------------------------ #
    # runtime profile and task generation
    # ------------------------------------------------------------------ #
    def runtime_profile(self, task: Any) -> ChallengeFamilyRuntimeProfile:
        """Container settings for a patrol: the navigation class on the base image, with room for a 390 s flight."""
        return ChallengeFamilyRuntimeProfile(
            family_id=self.family_id,
            profile_name="solar_patrol",
            resource_class="navigation",
            image_key="base",
            env_bootstrap={"sar_mode": False},
            docker_env={
                "SWARM_CHALLENGE_FAMILY_ID": self.family_id,
                "SWARM_RUNTIME_PROFILE": "solar_patrol",
                "SWARM_RUNTIME_RESOURCE_CLASS": "navigation",
                "SWARM_RUNTIME_IMAGE_KEY": "base",
                "SWARM_RUNTIME_ENV_BOOTSTRAP": "sar_mode=false",
            },
            # Placeholders until the evaluation time limits are measured on full patrols.
            global_eval_base_sec=600.0,
            global_eval_per_seed_sec=900.0,
            global_eval_cap_sec=14400.0,
            # A worker keeps about 2.76 GB of build caches between patrols; the default 2,500 MiB would drop them every 3 seeds.
            worker_recycle_rss_mb=3200.0,
        )

    def env_kwargs_for_task(self, task: Any) -> dict[str, Any]:
        """Constructor arguments the patrol environment needs: search-and-rescue mode is always off."""
        return {"sar_mode": False}

    def build_random_task(self, *, sim_dt: float, seed: Optional[int]) -> Any:
        """The patrol for one seed, once the seed checks accept it."""
        return seed_checks.approve(solar_patrol_task(seed=int(seed or 0), sim_dt=sim_dt))

    def build_screening_tasks(self, *, sim_dt: float, seeds: list[int], offset: int = 0,
                              total_seed_count: Optional[int] = None) -> list[Any]:
        """Screening flies the same patrols as the benchmark: every seed is its own park."""
        return [self.build_random_task(sim_dt=sim_dt, seed=seed) for seed in seeds]

    def seeds_prepared(self, seeds: list[int]) -> bool:
        """Whether every seed's check verdict, and its replacements' where it failed, is already kept."""
        return all(seed_checks.replacement(seed) is not None for seed in seeds)

    def prepare_seeds(self, seeds: list[int]) -> None:
        """Fly the reference pilot over every seed not judged yet, on the seed checks' own workers."""
        seed_checks.prepare(seeds, seed_checks.workers())

    # ------------------------------------------------------------------ #
    # environment lifecycle
    # ------------------------------------------------------------------ #
    def drone_urdf(self, env: Any) -> Optional[str]:
        """The aircraft body the environment loads, or None for the default one."""
        return airframe.urdf(env)

    def initialise_env_state(self, env: Any, *, requested_mode: bool = False) -> None:
        """Attach an empty patrol to a fresh environment, so the observation path works before the first reset."""
        env.sar_mode = False
        env._solar = SolarEpisode(seed=int(getattr(env.task, "map_seed", 0)))

    def reset_env_state(self, env: Any) -> None:
        """Start a new patrol for the task's seed."""
        env._solar = SolarEpisode(seed=int(getattr(env.task, "map_seed", 0)))
        env._collision_exempt_uids = frozenset()

    def spawn_task_world(self, env: Any) -> None:
        """Build this seed's park and put everything in it, each part in the build order."""
        for part in _PARTS:
            part.reset(env, env._solar)

    def advance_world(self, env: Any) -> None:
        """Move the world one control step before physics: the park's movers, the thieves and the decoys."""
        ep = env._solar
        park.advance(env, ep)
        theft.advance(env, ep)
        decoys.advance(env, ep)

    def action_space(self, env: Any) -> spaces.Box:
        """The contract's action vector, one row per drone."""
        low = np.tile(np.asarray(ACTION_LOW, dtype=np.float32), (env.NUM_DRONES, 1))
        high = np.tile(np.asarray(ACTION_HIGH, dtype=np.float32), (env.NUM_DRONES, 1))
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def preprocess_action(self, env: Any, action: Any) -> np.ndarray:
        """Turn the model's action into rotor speeds for one control step.

        The same action arrives on every control step of a decision; a button counts only on the step its value
        rises, so holding it does not press it again. While the dock flies the drone, its own flight wins.
        """
        ep = env._solar
        raw = np.asarray(action, dtype=np.float32).reshape(-1)
        held = ep.held_action
        # A step repeating the last step's action, which is already the previous one, clips and decodes as it did.
        if held is not None and held[1] is ep.previous_action and held[0] == raw.tobytes():
            a, decoded = held[1], held[2]
        else:
            a = raw
            if a.size != ACTION_DIM or not np.all(np.isfinite(a)):
                a = np.zeros(ACTION_DIM, dtype=np.float32)
            a = outputs.clip(a)
            decoded = decode_action(a, ep.previous_action)
            ep.held_action = (raw.tobytes(), a, decode_action(a, a))
        env.action_buffer.append(a.reshape(1, ACTION_DIM).copy())
        command = sensor_noise.delay(env, ep, decoded)
        ep.previous_action = a
        ep.command = command
        outputs.apply(env, ep, command)
        dock.command(env, ep, command)
        setpoint = dock.autopilot(env, ep)
        if setpoint is None:
            setpoint = outputs.setpoint(env, ep, command)
        return outputs.fly(env, ep, setpoint)

    def post_step_update(self, env: Any) -> None:
        """Bookkeeping after one control step of physics: flight phase, limit, sensors, reports, coverage, score."""
        ep = env._solar
        ep.step += 1
        dock.update(env, ep)
        airframe.update(env, ep)
        drone_state.update(env, ep)
        wind.update(env, ep)
        flight_limit.update(env, ep)
        camera.update(env, ep)
        zoom.update(env, ep)
        laser.update(env, ep)
        ground_distance.update(env, ep)
        reports.update(env, ep)
        coverage.update(env, ep)
        score.update(env, ep)
        if env._collision:
            ep.end("collision")
        if sensor_noise.snapshot_due(ep):
            # The next decision is shown this snapshot, so its view is built now, before the window's last steps.
            ep.view = self._build_view(env, ep, self._clean_view(env, ep))
            ep.view_step = ep.step - ep.step % DECISION_STEPS + DECISION_STEPS
        if ep.step == ep.view_step:
            sensor_noise.show(ep)

    def observation_fixed(self, env: Any) -> bool:
        """True from the snapshot on: the view for the coming decision is built and nothing later changes it."""
        return env._solar.view_step > env._solar.step

    def compute_terminated(self, env: Any) -> bool:
        """True once the patrol has ended on landing, a collision or the flight limit's stop line."""
        ep = env._solar
        if ep.outcome.end_reason:
            env._success = ep.outcome.end_reason == "landed" and ep.outcome.returned_by_model
            if env._success and env._t_to_goal is None:
                env._t_to_goal = env._time_alive
            env._failure_reason = _FAILURE_BY_END[ep.outcome.end_reason]
            return True
        return False

    def compute_truncated(self, env: Any, *, terminal_already: bool, roll: float, pitch: float) -> bool:
        """End the patrol when the clock runs out, or as a collision when the drone tips past the tilt limit."""
        ep = env._solar
        if abs(float(roll)) > float(env.MAX_TILT_RAD) or abs(float(pitch)) > float(env.MAX_TILT_RAD):
            ep.end("collision")
        elif ep.time_s >= env.EP_LEN_SEC:
            ep.end("timeout")
        if ep.outcome.end_reason and not terminal_already:
            env._failure_reason = _FAILURE_BY_END[ep.outcome.end_reason]
        return bool(ep.outcome.end_reason)

    def protected_body_uids(self, env: Any) -> set[int]:
        """The dock (body, pad and base) and the park's movers, kept out of the clearance metric and the obstacle
        cull; the pad's concave surface must also stay out of the clearance's closest-point query, which can crash
        on concave meshes."""
        ep = env._solar
        if ep.park is None:
            return set()
        model = (ep.dock or {}).get("model")
        dock_bodies = {int(ep.dock_uid)} | ({model.pad_uid, model.base_uid} if model is not None else set())
        return set(park.moving_bodies(ep)) | dock_bodies

    def build_info(self, env: Any) -> dict[str, Any]:
        """Per-step fields the evaluator reads: the schema version and the patrol outcome so far."""
        return {
            "schema_version": SCHEMA_VERSION,
            "task_version": str(getattr(env.task, "version", "")),
            "solar_outcome": asdict(env._solar.outcome),
        }

    # ------------------------------------------------------------------ #
    # observation
    # ------------------------------------------------------------------ #
    def observation_part(self, env: Any, key: str) -> np.ndarray:
        """One key of what the model sees this decision, the whole view built once and shared by every key: the one
        built at the link's snapshot, or this step's own when no snapshot was taken since the last decision."""
        ep = env._solar
        if ep.view_step < ep.step:
            ep.view, ep.view_step = self._build_view(env, ep, self._clean_view(env, ep)), ep.step
            sensor_noise.show(ep)
        return ep.view[key]

    def _build_view(self, env: Any, ep: SolarEpisode, snapshot: dict[str, Any]) -> dict[str, np.ndarray]:
        """A clean snapshot with the sensor errors on top."""
        view = sensor_noise.observe(env, ep, snapshot)
        site_map = new_site_map()
        if ep.step == 0:
            drone_state.site_map(env, ep, site_map)
            flight_limit.site_map(env, ep, site_map)
        return dict(view, site_map=site_map)

    def _clean_view(self, env: Any, ep: SolarEpisode) -> dict[str, Any]:
        """Every part writes its own fields; the images come with the views they were taken from."""
        state = new_state()
        drone_state.observe(env, ep, state)
        wind.observe(env, ep, state)
        laser.observe(env, ep, state)
        ground_distance.observe(env, ep, state)
        flight_limit.observe(env, ep, state)
        dock.observe(env, ep, state)
        camera.observe(env, ep, state)
        zoom.observe(env, ep, state)
        return {"step": ep.step, "state": state, "rgb": ep.frames.rgb, "thermal": ep.frames.thermal,
                "zoom": ep.frames.zoom, "feed_view": camera.view(ep), "zoom_view": zoom.view(ep)}

    # ------------------------------------------------------------------ #
    # scoring
    # ------------------------------------------------------------------ #
    def stalled_rollout_metrics(self, task: Any, info: dict) -> dict:
        """The patrol outcome so far, so a model that stalls still answers for the threats it left unreported."""
        _ = task
        return dict((info or {}).get("solar_outcome") or asdict(Outcome()))

    def evaluate_rollout(self, *, task: Any, success: bool, t: float, horizon: float,
                         min_clearance: Optional[float], collision: bool, failure_reason: str,
                         info: Optional[dict] = None) -> ChallengeFamilyEvaluation:
        """Score a patrol from the outcome its last step carried; no outcome means nothing was earned."""
        outcome = (info or {}).get("solar_outcome") or asdict(Outcome())
        metrics = dict(outcome, time_sec=float(t), horizon_sec=float(horizon), success=bool(success))
        metrics["environment_type"] = CHALLENGE_TYPE_TO_ENVIRONMENT_TYPE.get(CHALLENGE_TYPE, "unknown")
        metrics["failure_reason"] = str(failure_reason)
        normalized = score.score(task, metrics)
        return ChallengeFamilyEvaluation(
            family_id=self.family_id,
            success=bool(success),
            score=float(normalized["final_score"]),
            failure_reason=str(failure_reason),
            metrics=metrics,
            normalized_metrics=normalized,
        )
