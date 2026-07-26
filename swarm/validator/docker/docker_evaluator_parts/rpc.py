import asyncio
import mmap
import os
import statistics
import threading
import time
from typing import Callable, Optional

import bittensor as bt
import capnp
import numpy as np

from swarm.challenge_families import evaluate_rollout, runtime_family_for_task
from swarm.constants import (
    CALIBRATION_MARGIN_SEC,
    CALIBRATION_RECAL_INTERVAL,
    FIRST_STEP_BUDGET_REF_SEC,
    FIRST_STEP_HARD_CAP_REF_SEC,
    HARD_CAP_MARGIN_SEC,
    HARD_CAP_REF_SEC,
    HARD_CAP_STRIKES_PER_SEED,
    MINER_COMPUTE_BUDGET_SEC,
    RPC_CONNECT_MAX_WAIT_SEC,
    RPC_FIRST_STEP_TIMEOUT_SEC,
    RPC_MAX_STRIKES_PER_SEED,
    RPC_PING_TIMEOUT_SEC,
    RPC_RESET_TIMEOUT_SEC,
    RPC_STEP_TIMEOUT_SEC,
    SIM_DT,
)
from swarm.protocol import FailureReason, ValidationResult
from swarm.utils.env_factory import make_env_with_initial_obs
from swarm.validator.calibration import act_hard_cap_sec, judge_act
from swarm.core.action import canonicalize_action

from ._shared import (
    _cleanup_env_quietly,
    _docker_evaluator_facade,
    _runtime_profile_from_payload,
    _submission_template_dir,
)
from .submission import _serialize_observation_shm
from swarm.config import RpcTraceSettings


def _strike_zero_action(n_drones: int, act_dim: int) -> np.ndarray:
    """Neutral substitute action in the exact family contract shape."""
    if n_drones > 1:
        return np.zeros((n_drones, act_dim), dtype=np.float32)
    return np.zeros(act_dim, dtype=np.float32)


def _run_multi_seed_rpc_sync(
    self,
    tasks: list,
    uid: int,
    rpc_port: int,
    on_seed_complete: Optional[Callable[..., None]] = None,
    rollout_observer: Optional[Callable[[dict], None]] = None,
    stop_event: Optional[threading.Event] = None,
    progress_state: Optional[dict] = None,
    task_offset: int = 0,
    task_total: Optional[int] = None,
    runtime_profile_payload: Optional[dict] = None,
    speed_factor: Optional[float] = None,
) -> list:
    """Run multiple seeds through the same RPC connection.

    This reuses the container for all seeds, calling agent.reset() between each.
    Much faster than creating a new container per seed.

    When ``speed_factor`` is provided and reference calibration is enabled, each
    act() is judged in baseline-equivalent time (hardware-fair scoring); otherwise
    the legacy per-step calibrated timeout is used.
    """
    use_ref = speed_factor is not None
    schema_file = _submission_template_dir() / "agent.capnp"
    agent_capnp = capnp.load(str(schema_file))

    shm_file = None
    shm_buf = None
    shm_path = f"/dev/shm/swarm_obs_{rpc_port}.bin"
    if os.path.exists(shm_path):
        try:
            shm_file = open(shm_path, "r+b")
            shm_buf = mmap.mmap(shm_file.fileno(), 0)
        except OSError:
            shm_buf = None

    def _build_observation(obs):
        if shm_buf is not None:
            try:
                return _serialize_observation_shm(agent_capnp, obs, shm_buf)
            except BufferError:
                pass
        return self._serialize_observation(agent_capnp, obs)
    trace_settings = RpcTraceSettings.from_env()
    trace_rpc = trace_settings.enabled
    trace_every = trace_settings.trace_every
    trace_heartbeat_sec = trace_settings.heartbeat_sec
    runtime_profile = _runtime_profile_from_payload(runtime_profile_payload, tasks)
    ping_timeout_sec = float(
        runtime_profile.rpc_ping_timeout_sec
        if runtime_profile.rpc_ping_timeout_sec is not None
        else RPC_PING_TIMEOUT_SEC
    )
    reset_timeout_sec = float(
        runtime_profile.rpc_reset_timeout_sec
        if runtime_profile.rpc_reset_timeout_sec is not None
        else RPC_RESET_TIMEOUT_SEC
    )
    first_step_timeout_sec = float(
        runtime_profile.rpc_first_step_timeout_sec
        if runtime_profile.rpc_first_step_timeout_sec is not None
        else RPC_FIRST_STEP_TIMEOUT_SEC
    )
    default_step_timeout_sec = float(
        runtime_profile.rpc_step_timeout_sec
        if runtime_profile.rpc_step_timeout_sec is not None
        else RPC_STEP_TIMEOUT_SEC
    )

    def _emit_seed_complete(
        task_obj=None,
        *,
        status: str = "done",
        success: bool = False,
        sim_t: float = 0.0,
        seed_wall_sec: float = 0.0,
        step_idx: int = 0,
        error: str = "",
        calibration_overhead_sec: Optional[float] = None,
        calibration_cpu_factor: Optional[float] = None,
        calibrated_timeout_sec: Optional[float] = None,
    ) -> None:
        if on_seed_complete is None:
            return

        payload = None
        if task_obj is not None:
            payload = {
                "uid": int(uid),
                "map_seed": int(getattr(task_obj, "map_seed", -1)),
                "challenge_type": int(getattr(task_obj, "challenge_type", -1)),
                "horizon_sec": float(getattr(task_obj, "horizon", 0.0)),
                "status": status,
                "success": bool(success),
                "sim_time_sec": float(sim_t),
                "seed_wall_sec": max(0.0, float(seed_wall_sec)),
                "step_idx": int(step_idx),
                "error": error,
                "calibration_overhead_sec": (
                    None
                    if calibration_overhead_sec is None
                    else float(calibration_overhead_sec)
                ),
                "calibration_cpu_factor": (
                    None
                    if calibration_cpu_factor is None
                    else float(calibration_cpu_factor)
                ),
                "calibrated_timeout_sec": (
                    None
                    if calibrated_timeout_sec is None
                    else float(calibrated_timeout_sec)
                ),
            }
        try:
            on_seed_complete(payload)
        except TypeError:
            try:
                on_seed_complete()
            except Exception:
                pass
        except Exception:
            pass

    def _trace(msg: str) -> None:
        if trace_rpc:
            line = f"[{time.strftime('%H:%M:%S')}] [RPC TRACE][UID {uid}][port {rpc_port}] {msg}"
            print(line, flush=True)
            bt.logging.info(line)

    def _emit_rollout_event(event: str, **payload: object) -> None:
        if rollout_observer is None:
            return
        try:
            rollout_observer({"event": event, **payload})
        except Exception:
            pass

    def _task_type_label(task_obj) -> str:
        raw_type = int(getattr(task_obj, "challenge_type", -1))
        # With schema v2 challenge_type is already explicit:
        # 1=city, 2=open, 3=mountain, 4=village, 5=warehouse.
        # Keep this hook to avoid changing trace call sites.
        return str(raw_type)

    phase_lock = threading.Lock()
    phase_state: dict[str, object] = {
        "phase": "init",
        "task": "n/a",
        "step": 0,
        "sim_t": 0.0,
        "updated_at": time.time(),
    }
    watchdog_stop = threading.Event()

    def _set_phase(
        phase: str, task: str = "n/a", step: int = 0, sim_t: float = 0.0
    ) -> None:
        with phase_lock:
            phase_state["phase"] = phase
            phase_state["task"] = task
            phase_state["step"] = int(step)
            phase_state["sim_t"] = float(sim_t)
            phase_state["updated_at"] = time.time()
        if progress_state is not None:
            progress_state["phase"] = phase
            progress_state["task"] = task
            progress_state["step_idx"] = int(step)
            progress_state["sim_t"] = float(sim_t)
            progress_state["ts"] = time.time()

    def _watchdog_loop() -> None:
        if not trace_rpc or trace_heartbeat_sec <= 0:
            return
        while not watchdog_stop.wait(timeout=trace_heartbeat_sec):
            now = time.time()
            with phase_lock:
                phase = str(phase_state.get("phase", "unknown"))
                task = str(phase_state.get("task", "n/a"))
                step = int(phase_state.get("step", 0))
                sim_t = float(phase_state.get("sim_t", 0.0))
                updated_at = float(phase_state.get("updated_at", now))
            idle_for = max(0.0, now - updated_at)
            _trace(
                f"heartbeat phase={phase} task={task} step={step} "
                f"sim_t={sim_t:.2f}s idle={idle_for:.1f}s"
            )

    async def run_all_seeds():
        results = []
        async with capnp.kj_loop():
            stream = None
            agent = None
            # Retries sized to the connect budget (each attempt ~ ping_timeout + 2s backoff).
            max_ping_attempts = max(6, int(RPC_CONNECT_MAX_WAIT_SEC / (ping_timeout_sec + 2.0)))

            for attempt in range(1, max_ping_attempts + 1):
                if stop_event is not None and stop_event.is_set():
                    _trace("stop requested during rpc connect; aborting batch")
                    for failed_task in tasks:
                        _emit_seed_complete(
                            failed_task,
                            status="stopped_during_connect",
                            success=False,
                            sim_t=0.0,
                        )
                    return [
                        ValidationResult(
                            uid, False, 0.0, 0.0, failure_reason=FailureReason.INFRA.value
                        )
                        for _ in tasks
                    ]
                try:
                    _set_phase("rpc_connect", task="n/a", step=attempt, sim_t=0.0)
                    _trace(f"connect attempt {attempt}/{max_ping_attempts}")
                    stream = await capnp.AsyncIoStream.create_connection(
                        host="localhost", port=rpc_port
                    )
                    client = capnp.TwoPartyClient(stream)
                    agent = client.bootstrap().cast_as(agent_capnp.Agent)

                    _set_phase("rpc_ping", task="n/a", step=attempt, sim_t=0.0)
                    ping_response = await asyncio.wait_for(
                        agent.ping("test"), timeout=ping_timeout_sec
                    )
                    if ping_response.response != "pong":
                        raise RuntimeError(
                            f"Unexpected ping response (attempt {attempt}/{max_ping_attempts})"
                        )

                    _trace(
                        f"ping ok (attempt {attempt}) response={ping_response.response}"
                    )
                    break
                except asyncio.TimeoutError:
                    _trace(
                        f"ping timeout on attempt {attempt}/{max_ping_attempts} "
                        f"({ping_timeout_sec}s)"
                    )
                    if attempt >= max_ping_attempts:
                        bt.logging.warning(
                            f"UID {uid}: RPC ping timeout after {max_ping_attempts} attempts "
                            f"({ping_timeout_sec}s each)"
                        )
                        for failed_task in tasks:
                            _emit_seed_complete(
                                failed_task,
                                status="rpc_ping_timeout",
                                success=False,
                                sim_t=0.0,
                            )
                        return [
                            ValidationResult(
                                uid, False, 0.0, 0.0,
                                failure_reason=FailureReason.INFRA.value,
                            )
                            for _ in tasks
                        ]
                    await asyncio.sleep(2)
                except Exception as e:
                    _trace(
                        f"connect/ping error on attempt {attempt}/{max_ping_attempts}: {type(e).__name__}: {e}"
                    )
                    if attempt >= max_ping_attempts:
                        bt.logging.warning(
                            f"Cap'n Proto connection/ping failed for UID {uid} on port {rpc_port} "
                            f"after {max_ping_attempts} attempts: {e}"
                        )
                        for failed_task in tasks:
                            _emit_seed_complete(
                                failed_task,
                                status="rpc_connect_failed",
                                success=False,
                                sim_t=0.0,
                                error=f"{type(e).__name__}: {e}",
                            )
                        return [
                            ValidationResult(
                                uid, False, 0.0, 0.0,
                                failure_reason=FailureReason.INFRA.value,
                            )
                            for _ in tasks
                        ]
                    await asyncio.sleep(2)

            if agent is None:
                for failed_task in tasks:
                    _emit_seed_complete(
                        failed_task,
                        status="rpc_agent_unavailable",
                        success=False,
                        sim_t=0.0,
                    )
                return [
                    ValidationResult(
                        uid, False, 0.0, 0.0, failure_reason=FailureReason.INFRA.value
                    )
                    for _ in tasks
                ]

            calibrated_timeout = default_step_timeout_sec
            rpc_overhead_sec = max(
                default_step_timeout_sec - MINER_COMPUTE_BUDGET_SEC, 0.010
            )
            cpu_factor = 1.0
            calibrated = False

            def _ref_hard_caps(overhead_sec: float) -> tuple[float, float]:
                return (
                    act_hard_cap_sec(
                        speed_factor, overhead_sec,
                        ref_sec=HARD_CAP_REF_SEC, margin_sec=HARD_CAP_MARGIN_SEC,
                    ),
                    act_hard_cap_sec(
                        speed_factor, overhead_sec,
                        ref_sec=FIRST_STEP_HARD_CAP_REF_SEC, margin_sec=HARD_CAP_MARGIN_SEC,
                    ),
                )

            if use_ref:
                act_hard_cap, first_hard_cap = _ref_hard_caps(rpc_overhead_sec)
            else:
                act_hard_cap, first_hard_cap = calibrated_timeout, first_step_timeout_sec
            task_idx = 0
            retry_signature: Optional[str] = None
            while task_idx < len(tasks):
                task = tasks[task_idx]
                if stop_event is not None and stop_event.is_set():
                    remaining = len(tasks) - task_idx
                    if remaining > 0:
                        _trace(
                            f"stop requested; aborting {remaining} remaining seed(s)"
                        )
                        results.extend(
                            [
                                ValidationResult(
                                    uid, False, 0.0, 0.0,
                                    failure_reason=FailureReason.INFRA.value,
                                )
                                for _ in range(remaining)
                            ]
                        )
                        for failed_task in tasks[task_idx:]:
                            _emit_seed_complete(
                                failed_task,
                                status="stopped_before_seed",
                                success=False,
                                sim_t=0.0,
                            )
                    break
                display_idx = task_offset + task_idx + 1
                display_total = task_total if task_total is not None else len(tasks)
                task_label = (
                    f"seed {display_idx}/{display_total} "
                    f"map_seed={getattr(task, 'map_seed', 'n/a')} "
                    f"type={_task_type_label(task)}"
                )
                seed_wall_start = time.time()
                try:
                    _set_phase("seed_start", task=task_label, step=0, sim_t=0.0)
                    _trace(
                        f"{task_label} start horizon={getattr(task, 'horizon', 0.0):.1f}s"
                    )
                    t_env_start = time.time()
                    _set_phase("env_build", task=task_label, step=0, sim_t=0.0)
                    _trace(f"{task_label} building env")
                    env, obs = make_env_with_initial_obs(task, gui=False)
                    _trace(
                        f"{task_label} env built in {(time.time() - t_env_start):.2f}s"
                    )

                    try:
                        t_reset_start = time.time()
                        try:
                            _set_phase(
                                "agent_reset", task=task_label, step=0, sim_t=0.0
                            )
                            await asyncio.wait_for(
                                agent.reset(), timeout=reset_timeout_sec
                            )
                        except Exception as e:
                            _trace(
                                f"{task_label} reset failed: {type(e).__name__}: {e}"
                            )
                            raise
                        reset_ms = (time.time() - t_reset_start) * 1000.0
                        _trace(f"{task_label} reset ok in {reset_ms:.1f}ms")

                        should_calibrate = not calibrated or (
                            CALIBRATION_RECAL_INTERVAL > 0
                            and task_idx > 0
                            and task_idx % CALIBRATION_RECAL_INTERVAL == 0
                        )
                        if should_calibrate:
                            phase_label = "rpc_recalibration" if calibrated else "rpc_calibration"
                            _set_phase(
                                phase_label,
                                task=task_label,
                                step=0,
                                sim_t=0.0,
                            )
                            if use_ref:
                                rpc_overhead_sec = await _measure_rpc_overhead_via_ping(
                                    agent, uid, ping_timeout_sec
                                )
                                cpu_factor = 1.0
                            else:
                                (
                                    rpc_overhead_sec,
                                    cpu_factor,
                                ) = await self._calibrate_rpc_overhead_async(
                                    agent, agent_capnp, obs, uid
                                )
                            calibrated_timeout = (
                                (MINER_COMPUTE_BUDGET_SEC * cpu_factor)
                                + rpc_overhead_sec
                                + CALIBRATION_MARGIN_SEC
                            )
                            calibrated = True
                            if use_ref:
                                act_hard_cap, first_hard_cap = _ref_hard_caps(rpc_overhead_sec)
                                cpu_factor = speed_factor
                                calibrated_timeout = act_hard_cap
                                _trace(
                                    f"{phase_label} speed_factor={speed_factor:.2f}x "
                                    f"overhead={rpc_overhead_sec*1000:.1f}ms "
                                    f"hard_cap={act_hard_cap*1000:.0f}ms budget={MINER_COMPUTE_BUDGET_SEC*1000:.0f}ms"
                                )
                            else:
                                _trace(
                                    f"{phase_label} step timeout={calibrated_timeout*1000:.1f}ms "
                                    f"(overhead={rpc_overhead_sec*1000:.1f}ms cpu_factor={cpu_factor:.2f}x)"
                                )

                        _emit_rollout_event(
                            "seed_ready",
                            task=task,
                            env=env,
                            uid=int(uid),
                            task_label=task_label,
                            step_idx=0,
                            sim_time_sec=0.0,
                        )

                        t_sim = 0.0
                        success = False
                        info = {}
                        strikes = 0
                        hard_cap_hits = 0
                        is_first_step = True
                        step_idx = 0
                        rpc_disconnected = False

                        n_drones = int(getattr(env, "NUM_DRONES", 1))
                        act_dim = int(env.action_space.shape[-1])
                        if n_drones > 1:
                            lo, hi = env.action_space.low, env.action_space.high
                        else:
                            lo, hi = (
                                env.action_space.low.flatten(),
                                env.action_space.high.flatten(),
                            )

                        while t_sim < task.horizon and not (
                            stop_event is not None and stop_event.is_set()
                        ):
                            step_idx += 1
                            act_ms = 0.0
                            if use_ref:
                                step_timeout = (
                                    first_hard_cap if is_first_step else act_hard_cap
                                )
                            else:
                                step_timeout = (
                                    first_step_timeout_sec
                                    if is_first_step
                                    else calibrated_timeout
                                )

                            observation = _build_observation(obs)
                            _set_phase(
                                "rpc_act",
                                task=task_label,
                                step=step_idx,
                                sim_t=t_sim,
                            )

                            action = None
                            step_striked = False
                            for act_attempt in (0, 1):
                                try:
                                    t_act_start = time.perf_counter()
                                    action_response = await asyncio.wait_for(
                                        agent.act(observation), timeout=step_timeout
                                    )
                                    act_ms = (time.perf_counter() - t_act_start) * 1000.0
                                    candidate = np.frombuffer(
                                        action_response.action.data,
                                        dtype=np.dtype(action_response.action.dtype),
                                    ).reshape(tuple(action_response.action.shape))
                                    if use_ref:
                                        budget = (
                                            FIRST_STEP_BUDGET_REF_SEC
                                            if is_first_step
                                            else MINER_COMPUTE_BUDGET_SEC
                                        )
                                        if judge_act(
                                            act_ms / 1000.0,
                                            overhead_sec=rpc_overhead_sec,
                                            speed_factor=speed_factor,
                                            budget_sec=budget,
                                            hard_cap_sec=step_timeout,
                                        ).strike:
                                            if not step_striked:
                                                step_striked = True
                                                strikes += 1
                                            _trace(
                                                f"{task_label} step={step_idx} act_slow {act_ms:.1f}ms "
                                                f"(>{budget*1000:.0f}ms@{speed_factor:.2f}x) "
                                                f"attempt {act_attempt + 1}/2 "
                                                f"strike {strikes}/{RPC_MAX_STRIKES_PER_SEED}"
                                            )
                                            if strikes >= RPC_MAX_STRIKES_PER_SEED:
                                                bt.logging.warning(
                                                    f"UID {uid} seed {task_idx}: {strikes} slow-act strikes, failing seed"
                                                )
                                                break
                                            if act_attempt == 0:
                                                continue
                                            break
                                    action = candidate
                                    if trace_rpc and (
                                        step_idx == 1 or step_idx % trace_every == 0
                                    ):
                                        _trace(
                                            f"{task_label} step={step_idx} t_sim={t_sim:.2f}s "
                                            f"act_ok={act_ms:.1f}ms timeout={step_timeout*1000:.0f}ms"
                                        )
                                    break
                                except asyncio.TimeoutError:
                                    act_ms = (time.perf_counter() - t_act_start) * 1000
                                    if not step_striked:
                                        step_striked = True
                                        strikes += 1
                                    if use_ref:
                                        hard_cap_hits += 1
                                    _trace(
                                        f"{task_label} step={step_idx} act timeout {act_ms:.1f}ms "
                                        f"attempt {act_attempt + 1}/2 "
                                        f"strike {strikes}/{RPC_MAX_STRIKES_PER_SEED}"
                                    )
                                    if is_first_step:
                                        bt.logging.warning(
                                            f"UID {uid}: first-step act() timeout ({act_ms:.0f}ms > {step_timeout*1000:.0f}ms), "
                                            f"strike {strikes}/{RPC_MAX_STRIKES_PER_SEED}"
                                        )
                                    else:
                                        bt.logging.warning(
                                            f"UID {uid}: act() timeout ({act_ms:.0f}ms > {step_timeout*1000:.0f}ms "
                                            f"[budget={MINER_COMPUTE_BUDGET_SEC*1000:.0f}x{cpu_factor:.2f}+overhead={rpc_overhead_sec*1000:.1f}]), "
                                            f"strike {strikes}/{RPC_MAX_STRIKES_PER_SEED}"
                                        )
                                    if use_ref and hard_cap_hits >= HARD_CAP_STRIKES_PER_SEED:
                                        rpc_disconnected = True
                                        bt.logging.warning(
                                            f"UID {uid} seed {task_idx}: {hard_cap_hits} hard-cap timeouts, "
                                            f"aborting seed and recycling container"
                                        )
                                        break
                                    if strikes >= RPC_MAX_STRIKES_PER_SEED:
                                        bt.logging.warning(
                                            f"UID {uid} seed {task_idx}: {strikes} RPC timeouts, failing seed"
                                        )
                                        break
                                    if act_attempt == 0:
                                        continue
                                except Exception as e:
                                    err_txt = f"{type(e).__name__}: {e}"
                                    if not step_striked:
                                        step_striked = True
                                        strikes += 1
                                    _trace(
                                        f"{task_label} step={step_idx} act error: {err_txt} "
                                        f"strike {strikes}/{RPC_MAX_STRIKES_PER_SEED}"
                                    )
                                    lowered = err_txt.lower()
                                    if (
                                        "broken pipe" in lowered
                                        or "disconnected" in lowered
                                        or "connection reset" in lowered
                                    ):
                                        rpc_disconnected = True
                                        _trace(
                                            f"{task_label} rpc disconnected; aborting seed"
                                        )
                                        break
                                    if strikes >= RPC_MAX_STRIKES_PER_SEED:
                                        bt.logging.warning(
                                            f"UID {uid} seed {task_idx}: {strikes} RPC errors, failing seed"
                                        )
                                    break

                            if action is None:
                                action = _strike_zero_action(n_drones, act_dim)
                            if rpc_disconnected or strikes >= RPC_MAX_STRIKES_PER_SEED:
                                break

                            is_first_step = False

                            act = canonicalize_action(
                                action,
                                lo,
                                hi,
                                n_drones=n_drones if n_drones > 1 else None,
                                act_dim=act_dim,
                            )
                            _set_phase(
                                "env_step", task=task_label, step=step_idx, sim_t=t_sim
                            )
                            obs, _r, terminated, truncated, info = env.step(
                                act if n_drones > 1 else act[None, :]
                            )

                            t_sim += SIM_DT
                            _emit_rollout_event(
                                "step",
                                task=task,
                                env=env,
                                uid=int(uid),
                                task_label=task_label,
                                step_idx=step_idx,
                                sim_time_sec=float(t_sim),
                                terminated=bool(terminated),
                                truncated=bool(truncated),
                                info=dict(info),
                                action=act.tolist(),
                                act_ms=float(act_ms),
                            )
                            if terminated or truncated:
                                success = info.get("success", False)
                                _trace(
                                    f"{task_label} terminated={terminated} truncated={truncated} "
                                    f"success={success} t_sim={t_sim:.2f}s strikes={strikes}"
                                )
                                break

                        seed_cancelled = (
                            stop_event is not None and stop_event.is_set()
                        )
                        if seed_cancelled:
                            _set_phase(
                                "seed_cancelled",
                                task=task_label,
                                step=step_idx,
                                sim_t=t_sim,
                            )
                            _trace(
                                f"{task_label} cancelled due to stop request at t_sim={t_sim:.2f}s"
                            )
                            results.append(
                                ValidationResult(
                                    uid, False, t_sim, 0.0,
                                    failure_reason=FailureReason.INFRA.value,
                                )
                            )
                            _emit_seed_complete(
                                task,
                                status="seed_cancelled",
                                success=False,
                                sim_t=t_sim,
                                seed_wall_sec=time.time() - seed_wall_start,
                                step_idx=step_idx,
                                calibration_overhead_sec=rpc_overhead_sec,
                                calibration_cpu_factor=cpu_factor,
                                calibrated_timeout_sec=calibrated_timeout,
                            )
                        elif rpc_disconnected:
                            _set_phase(
                                "seed_failed_rpc_disconnect",
                                task=task_label,
                                step=step_idx,
                                sim_t=t_sim,
                            )
                            _trace(f"{task_label} failed due to rpc disconnect")
                            results.append(
                                ValidationResult(
                                    uid, False, t_sim, 0.0,
                                    failure_reason=FailureReason.INFRA.value,
                                )
                            )
                            _emit_seed_complete(
                                task,
                                status="seed_rpc_disconnected",
                                success=False,
                                sim_t=t_sim,
                                seed_wall_sec=time.time() - seed_wall_start,
                                step_idx=step_idx,
                                calibration_overhead_sec=rpc_overhead_sec,
                                calibration_cpu_factor=cpu_factor,
                                calibrated_timeout_sec=calibrated_timeout,
                            )
                        elif strikes >= RPC_MAX_STRIKES_PER_SEED:
                            _set_phase(
                                "seed_failed_timeout_strikes",
                                task=task_label,
                                step=step_idx,
                                sim_t=t_sim,
                            )
                            _trace(
                                f"{task_label} failed due to strike limit; returning zero result"
                            )
                            results.append(
                                ValidationResult(
                                    uid, False, t_sim, 0.0,
                                    failure_reason=FailureReason.SLOW_ACT_STRIKES.value,
                                )
                            )
                            _emit_seed_complete(
                                task,
                                status="seed_timeout_strikes",
                                success=False,
                                sim_t=t_sim,
                                seed_wall_sec=time.time() - seed_wall_start,
                                step_idx=step_idx,
                                calibration_overhead_sec=rpc_overhead_sec,
                                calibration_cpu_factor=cpu_factor,
                                calibrated_timeout_sec=calibrated_timeout,
                            )
                        else:
                            if n_drones > 1:
                                family = runtime_family_for_task(task)
                                swarm = family.score_swarm(task, info)
                                score = float(swarm["final_score"])
                                per_succ = info.get("per_drone_success", [])
                                success = bool(swarm.get("success", bool(per_succ) and all(per_succ)))
                                per_fr = info.get("per_drone_failure_reason", [])
                                failure_reason = str(
                                    swarm.get("failure_reason")
                                    or ("NONE" if success else next(
                                        (r for r in per_fr if r != "NONE"), "NONE"
                                    ))
                                )
                                result_metrics = {
                                    "per_drone_final_score": swarm["per_drone_final_score"],
                                    "per_drone_success": list(per_succ),
                                    "per_drone_failure_reason": list(per_fr),
                                }
                            else:
                                min_clearance = info.get("min_clearance", None)
                                collision = info.get("collision", False)
                                failure_reason = info.get("failure_reason", "NONE")
                                evaluation = evaluate_rollout(
                                    task=task,
                                    success=success,
                                    t=t_sim,
                                    horizon=task.horizon,
                                    min_clearance=min_clearance,
                                    collision=collision,
                                    failure_reason=failure_reason,
                                )
                                score = evaluation.score
                                result_metrics = dict(evaluation.metrics)
                            _trace(
                                f"{task_label} result success={success} "
                                f"score={score:.4f} t_sim={t_sim:.2f}s"
                            )
                            _set_phase(
                                "seed_done",
                                task=task_label,
                                step=step_idx,
                                sim_t=t_sim,
                            )
                            _emit_rollout_event(
                                "seed_result",
                                task=task,
                                env=env,
                                uid=int(uid),
                                task_label=task_label,
                                step_idx=step_idx,
                                sim_time_sec=float(t_sim),
                                success=bool(success),
                                score=float(score),
                                info=dict(info),
                            )
                            results.append(
                                ValidationResult(
                                    uid, success, t_sim, score,
                                    failure_reason=failure_reason,
                                    metrics=result_metrics,
                                )
                            )
                            _emit_seed_complete(
                                task,
                                status="seed_done",
                                success=success,
                                sim_t=t_sim,
                                seed_wall_sec=time.time() - seed_wall_start,
                                step_idx=step_idx,
                                calibration_overhead_sec=rpc_overhead_sec,
                                calibration_cpu_factor=cpu_factor,
                                calibrated_timeout_sec=calibrated_timeout,
                            )

                    finally:
                        _cleanup_env_quietly(env)

                except Exception as e:
                    try:
                        exc_t_sim = t_sim
                    except NameError:
                        exc_t_sim = 0.0
                    signature = f"{type(e).__name__}: {e}"
                    if retry_signature is None:
                        retry_signature = signature
                        bt.logging.warning(
                            f"UID {uid} {task_label} failed: {signature}; "
                            f"rebuilding environment for one retry"
                        )
                        _set_phase(
                            "seed_env_retry", task=task_label, step=0, sim_t=exc_t_sim
                        )
                        continue
                    deterministic = signature == retry_signature
                    failure_reason = (
                        FailureReason.ENV_FAILURE
                        if deterministic
                        else FailureReason.INFRA
                    )
                    status = "seed_env_failure" if deterministic else "seed_exception"
                    bt.logging.warning(
                        f"UID {uid} {task_label} failed twice "
                        f"({'identical' if deterministic else 'different'} error): "
                        f"{signature}"
                    )
                    _set_phase(status, task=task_label, step=0, sim_t=exc_t_sim)
                    _trace(f"{task_label} failed with exception: {signature}")
                    results.append(
                        ValidationResult(
                            uid, False, exc_t_sim, 0.0,
                            failure_reason=failure_reason.value,
                        )
                    )
                    _emit_seed_complete(
                        task,
                        status=status,
                        success=False,
                        sim_t=exc_t_sim,
                        seed_wall_sec=time.time() - seed_wall_start,
                        step_idx=0,
                        error=signature,
                        calibration_overhead_sec=locals().get("rpc_overhead_sec"),
                        calibration_cpu_factor=locals().get("cpu_factor"),
                        calibrated_timeout_sec=locals().get("calibrated_timeout"),
                    )

                retry_signature = None
                task_idx += 1

        return results

    loop = asyncio.new_event_loop()
    watchdog_thread = None
    if trace_rpc and trace_heartbeat_sec > 0:
        _trace(f"rpc phase heartbeat enabled every {trace_heartbeat_sec:.1f}s")
        watchdog_thread = threading.Thread(
            target=_watchdog_loop,
            name=f"rpc_trace_watchdog_uid{uid}_{rpc_port}",
            daemon=True,
        )
        watchdog_thread.start()
    try:
        return loop.run_until_complete(run_all_seeds())
    finally:
        watchdog_stop.set()
        if watchdog_thread is not None:
            watchdog_thread.join(timeout=2.0)
        loop.close()
        if shm_buf is not None:
            shm_buf.close()
        if shm_file is not None:
            shm_file.close()

async def _measure_rpc_overhead_via_ping(agent, uid: int, ping_timeout_sec: float) -> float:
    """Pure RPC round-trip overhead from no-op pings (no miner-side compute)."""
    facade = _docker_evaluator_facade()
    samples = []
    for _ in range(facade.CALIBRATION_ROUNDS):
        try:
            t0 = time.perf_counter()
            await asyncio.wait_for(agent.ping("cal"), timeout=ping_timeout_sec)
            samples.append(time.perf_counter() - t0)
        except Exception:
            continue
    if len(samples) < 3:
        return max(facade.RPC_STEP_TIMEOUT_SEC - facade.MINER_COMPUTE_BUDGET_SEC, 0.010)
    samples.sort()
    trimmed = samples[1:-1] if len(samples) > 4 else samples
    return min(statistics.median(trimmed), facade.CALIBRATION_OVERHEAD_CAP_SEC)


async def _calibrate_rpc_overhead_async(self, agent, agent_capnp, obs, uid: int):
    """Measure trusted transport overhead; graph artifacts cannot calibrate hosts."""
    overhead = await _measure_rpc_overhead_via_ping(
        agent, uid, _docker_evaluator_facade().RPC_PING_TIMEOUT_SEC
    )
    return overhead, 1.0
