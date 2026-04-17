import asyncio
import statistics
import threading
import time
from typing import Callable, Optional

import bittensor as bt
import capnp
import numpy as np
from gym_pybullet_drones.utils.enums import ActionType

from swarm.constants import (
    CALIBRATION_MARGIN_SEC,
    CALIBRATION_RECAL_INTERVAL,
    MINER_COMPUTE_BUDGET_SEC,
    RPC_FIRST_STEP_TIMEOUT_SEC,
    RPC_MAX_STRIKES_PER_SEED,
    RPC_PING_TIMEOUT_SEC,
    RPC_RESET_TIMEOUT_SEC,
    RPC_STEP_TIMEOUT_SEC,
    SIM_DT,
    SPEED_LIMIT,
)
from swarm.protocol import ValidationResult
from swarm.utils.env_factory import make_env
from swarm.validator.reward import flight_reward

from ._shared import (
    _cleanup_env_quietly,
    _docker_evaluator_facade,
    _submission_template_dir,
)
from swarm.config import RpcTraceSettings


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
) -> list:
    """Run multiple seeds through the same RPC connection.

    This reuses the container for all seeds, calling agent.reset() between each.
    Much faster than creating a new container per seed.
    """
    schema_file = _submission_template_dir() / "agent.capnp"
    agent_capnp = capnp.load(str(schema_file))
    trace_settings = RpcTraceSettings.from_env()
    trace_rpc = trace_settings.enabled
    trace_every = trace_settings.trace_every
    trace_heartbeat_sec = trace_settings.heartbeat_sec

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
        timing_breakdown: Optional[dict] = None,
        rollout_breakdown: Optional[dict] = None,
        latency_stats: Optional[dict] = None,
        step_metrics: Optional[dict] = None,
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
                "moving_platform": bool(
                    getattr(task_obj, "moving_platform", False)
                ),
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
                "timing_breakdown": (
                    dict(timing_breakdown) if isinstance(timing_breakdown, dict) else {}
                ),
                "rollout_breakdown": (
                    dict(rollout_breakdown) if isinstance(rollout_breakdown, dict) else {}
                ),
                "latency_stats": (
                    dict(latency_stats) if isinstance(latency_stats, dict) else {}
                ),
                "step_metrics": (
                    dict(step_metrics) if isinstance(step_metrics, dict) else {}
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

    def _percentile_ms(samples_sec: list[float], q: float) -> float:
        if not samples_sec:
            return 0.0
        values = sorted(float(v) for v in samples_sec)
        if len(values) == 1:
            return values[0] * 1000.0
        q_clamped = min(1.0, max(0.0, float(q)))
        pos = q_clamped * (len(values) - 1)
        lower = int(pos)
        upper = min(lower + 1, len(values) - 1)
        frac = pos - lower
        sec = values[lower] + ((values[upper] - values[lower]) * frac)
        return sec * 1000.0

    def _latency_stats(samples_sec: list[float]) -> dict:
        total_sec = float(sum(samples_sec))
        count = int(len(samples_sec))
        avg_ms = (total_sec / count * 1000.0) if count > 0 else 0.0
        return {
            "count": count,
            "total_sec": total_sec,
            "avg_ms": avg_ms,
            "p50_ms": _percentile_ms(samples_sec, 0.50),
            "p95_ms": _percentile_ms(samples_sec, 0.95),
            "max_ms": (max(samples_sec) * 1000.0) if samples_sec else 0.0,
        }

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

    def _set_active_seed(task_obj=None) -> None:
        if progress_state is None:
            return
        if task_obj is None:
            progress_state["map_seed"] = -1
            progress_state["challenge_type"] = -1
            progress_state["seed_active"] = False
            progress_state["ts"] = time.time()
            return
        progress_state["map_seed"] = int(getattr(task_obj, "map_seed", -1))
        progress_state["challenge_type"] = int(getattr(task_obj, "challenge_type", -1))
        progress_state["seed_active"] = True
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
            max_ping_attempts = 6

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
                    return [ValidationResult(uid, False, 0.0, 0.0) for _ in tasks]
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
                        agent.ping("test"), timeout=RPC_PING_TIMEOUT_SEC
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
                        f"({RPC_PING_TIMEOUT_SEC}s)"
                    )
                    if attempt >= max_ping_attempts:
                        bt.logging.warning(
                            f"UID {uid}: RPC ping timeout after {max_ping_attempts} attempts "
                            f"({RPC_PING_TIMEOUT_SEC}s each)"
                        )
                        for failed_task in tasks:
                            _emit_seed_complete(
                                failed_task,
                                status="rpc_ping_timeout",
                                success=False,
                                sim_t=0.0,
                            )
                        return [
                            ValidationResult(uid, False, 0.0, 0.0) for _ in tasks
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
                            ValidationResult(uid, False, 0.0, 0.0) for _ in tasks
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
                return [ValidationResult(uid, False, 0.0, 0.0) for _ in tasks]

            calibrated_timeout = RPC_STEP_TIMEOUT_SEC
            rpc_overhead_sec = max(
                RPC_STEP_TIMEOUT_SEC - MINER_COMPUTE_BUDGET_SEC, 0.010
            )
            cpu_factor = 1.0
            calibrated = False
            for task_idx, task in enumerate(tasks):
                if stop_event is not None and stop_event.is_set():
                    remaining = len(tasks) - task_idx
                    if remaining > 0:
                        _trace(
                            f"stop requested; aborting {remaining} remaining seed(s)"
                        )
                        results.extend(
                            [
                                ValidationResult(uid, False, 0.0, 0.0)
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
                seed_wall_start = time.perf_counter()
                env = None
                t_sim = 0.0
                success = False
                info = {}
                strikes = 0
                is_first_step = True
                step_idx = 0
                rpc_disconnected = False
                seed_phase_sec = {
                    "env_build_sec": 0.0,
                    "env_reset_sec": 0.0,
                    "agent_reset_sec": 0.0,
                    "calibration_sec": 0.0,
                    "rollout_total_sec": 0.0,
                    "env_cleanup_sec": 0.0,
                }
                rollout_phase_sec = {
                    "observation_serialize_sec": 0.0,
                    "agent_act_sec": 0.0,
                    "action_decode_sec": 0.0,
                    "action_postprocess_sec": 0.0,
                    "env_step_sec": 0.0,
                }
                act_latencies_sec: list[float] = []
                env_step_latencies_sec: list[float] = []
                step_total_latencies_sec: list[float] = []
                act_timeout_count = 0
                act_error_count = 0
                act_budget_overrun_count = 0
                calibration_count = 0
                first_act_ms: Optional[float] = None
                last_act_ms = 0.0

                def _build_seed_report() -> tuple[float, dict, dict, dict, dict]:
                    seed_total_sec = max(0.0, time.perf_counter() - seed_wall_start)
                    rollout_total_sec = max(
                        0.0, float(seed_phase_sec.get("rollout_total_sec", 0.0))
                    )
                    rollout_known_sec = (
                        float(rollout_phase_sec.get("observation_serialize_sec", 0.0))
                        + float(rollout_phase_sec.get("agent_act_sec", 0.0))
                        + float(rollout_phase_sec.get("action_decode_sec", 0.0))
                        + float(rollout_phase_sec.get("action_postprocess_sec", 0.0))
                        + float(rollout_phase_sec.get("env_step_sec", 0.0))
                    )
                    rollout_breakdown_payload = {
                        "rollout_total_sec": rollout_total_sec,
                        **{
                            key: float(value)
                            for key, value in rollout_phase_sec.items()
                        },
                        "rollout_other_sec": max(
                            0.0, rollout_total_sec - rollout_known_sec
                        ),
                    }
                    known_seed_sec = (
                        float(seed_phase_sec.get("env_build_sec", 0.0))
                        + float(seed_phase_sec.get("env_reset_sec", 0.0))
                        + float(seed_phase_sec.get("agent_reset_sec", 0.0))
                        + float(seed_phase_sec.get("calibration_sec", 0.0))
                        + rollout_total_sec
                        + float(seed_phase_sec.get("env_cleanup_sec", 0.0))
                    )
                    timing_breakdown = {
                        "seed_total_sec": seed_total_sec,
                        **{key: float(value) for key, value in seed_phase_sec.items()},
                        "seed_unaccounted_sec": max(
                            0.0, seed_total_sec - known_seed_sec
                        ),
                    }
                    latency_payload = {
                        "act": _latency_stats(act_latencies_sec),
                        "env_step": _latency_stats(env_step_latencies_sec),
                        "step_total": _latency_stats(step_total_latencies_sec),
                    }
                    step_metrics = {
                        "step_count": int(step_idx),
                        "act_timeout_count": int(act_timeout_count),
                        "act_error_count": int(act_error_count),
                        "act_budget_overrun_count": int(act_budget_overrun_count),
                        "calibration_count": int(calibration_count),
                        "first_act_ms": (
                            None if first_act_ms is None else float(first_act_ms)
                        ),
                        "last_act_ms": float(last_act_ms),
                        "rpc_disconnected": bool(rpc_disconnected),
                    }
                    return (
                        seed_total_sec,
                        timing_breakdown,
                        rollout_breakdown_payload,
                        latency_payload,
                        step_metrics,
                    )

                try:
                    _set_active_seed(task)
                    _set_phase("seed_start", task=task_label, step=0, sim_t=0.0)
                    _trace(
                        f"{task_label} start horizon={getattr(task, 'horizon', 0.0):.1f}s"
                    )
                    t_env_start = time.perf_counter()
                    _set_phase("env_build", task=task_label, step=0, sim_t=0.0)
                    _trace(f"{task_label} building env")
                    env = make_env(task, gui=False)
                    seed_phase_sec["env_build_sec"] = max(
                        0.0, time.perf_counter() - t_env_start
                    )
                    _trace(
                        f"{task_label} env built in {seed_phase_sec['env_build_sec']:.2f}s"
                    )

                    try:
                        t_reset_env_start = time.perf_counter()
                        _set_phase("env_reset", task=task_label, step=0, sim_t=0.0)
                        _trace(f"{task_label} env.reset() start")
                        obs, _ = env.reset()
                        seed_phase_sec["env_reset_sec"] = max(
                            0.0, time.perf_counter() - t_reset_env_start
                        )
                        _trace(
                            f"{task_label} env.reset() done in {seed_phase_sec['env_reset_sec']:.2f}s"
                        )
                        t_reset_start = time.perf_counter()
                        try:
                            _set_phase(
                                "agent_reset", task=task_label, step=0, sim_t=0.0
                            )
                            await asyncio.wait_for(
                                agent.reset(), timeout=RPC_RESET_TIMEOUT_SEC
                            )
                        except Exception as e:
                            _trace(
                                f"{task_label} reset failed: {type(e).__name__}: {e}"
                            )
                            raise
                        seed_phase_sec["agent_reset_sec"] = max(
                            0.0, time.perf_counter() - t_reset_start
                        )
                        reset_ms = seed_phase_sec["agent_reset_sec"] * 1000.0
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
                            calibration_start = time.perf_counter()
                            (
                                rpc_overhead_sec,
                                cpu_factor,
                            ) = await self._calibrate_rpc_overhead_async(
                                agent, agent_capnp, obs, uid
                            )
                            seed_phase_sec["calibration_sec"] += max(
                                0.0, time.perf_counter() - calibration_start
                            )
                            calibration_count += 1
                            calibrated_timeout = (
                                (MINER_COMPUTE_BUDGET_SEC * cpu_factor)
                                + rpc_overhead_sec
                                + CALIBRATION_MARGIN_SEC
                            )
                            calibrated = True
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

                        lo, hi = (
                            env.action_space.low.flatten(),
                            env.action_space.high.flatten(),
                        )

                        rollout_start = time.perf_counter()
                        emit_status = "seed_exception"
                        emit_success = False
                        emit_error = ""
                        emit_score = 0.0
                        try:
                            while t_sim < task.horizon and not (
                                stop_event is not None and stop_event.is_set()
                            ):
                                step_idx += 1
                                step_timeout = (
                                    RPC_FIRST_STEP_TIMEOUT_SEC
                                    if is_first_step
                                    else calibrated_timeout
                                )
                                step_wall_start = time.perf_counter()

                                serialize_start = time.perf_counter()
                                observation = self._serialize_observation(
                                    agent_capnp, obs
                                )
                                rollout_phase_sec["observation_serialize_sec"] += max(
                                    0.0, time.perf_counter() - serialize_start
                                )
                                _set_phase(
                                    "rpc_act",
                                    task=task_label,
                                    step=step_idx,
                                    sim_t=t_sim,
                                )

                                decode_sec = 0.0
                                act_start = time.perf_counter()
                                try:
                                    action_response = await asyncio.wait_for(
                                        agent.act(observation), timeout=step_timeout
                                    )
                                    act_sec = max(
                                        0.0, time.perf_counter() - act_start
                                    )
                                    decode_start = time.perf_counter()
                                    action = np.frombuffer(
                                        action_response.action.data,
                                        dtype=np.dtype(action_response.action.dtype),
                                    ).reshape(tuple(action_response.action.shape))
                                    decode_sec = max(
                                        0.0, time.perf_counter() - decode_start
                                    )
                                    if trace_rpc and (
                                        step_idx == 1 or step_idx % trace_every == 0
                                    ):
                                        _trace(
                                            f"{task_label} step={step_idx} t_sim={t_sim:.2f}s "
                                            f"act_ok={act_sec*1000.0:.1f}ms timeout={step_timeout*1000:.0f}ms"
                                        )
                                except asyncio.TimeoutError:
                                    act_sec = max(
                                        0.0, time.perf_counter() - act_start
                                    )
                                    act_timeout_count += 1
                                    strikes += 1
                                    action = np.zeros(5, dtype=np.float32)
                                    _trace(
                                        f"{task_label} step={step_idx} act timeout {act_sec*1000.0:.1f}ms "
                                        f"strike {strikes}/{RPC_MAX_STRIKES_PER_SEED}"
                                    )
                                    if is_first_step:
                                        bt.logging.warning(
                                            f"UID {uid}: first-step act() timeout ({act_sec*1000.0:.0f}ms > {step_timeout*1000:.0f}ms), "
                                            f"strike {strikes}/{RPC_MAX_STRIKES_PER_SEED}"
                                        )
                                    else:
                                        bt.logging.warning(
                                            f"UID {uid}: act() timeout ({act_sec*1000.0:.0f}ms > {step_timeout*1000:.0f}ms "
                                            f"[budget={MINER_COMPUTE_BUDGET_SEC*1000:.0f}x{cpu_factor:.2f}+overhead={rpc_overhead_sec*1000:.1f}]), "
                                            f"strike {strikes}/{RPC_MAX_STRIKES_PER_SEED}"
                                        )
                                    if strikes >= RPC_MAX_STRIKES_PER_SEED:
                                        bt.logging.warning(
                                            f"UID {uid} seed {task_idx}: {strikes} RPC timeouts, failing seed"
                                        )
                                except Exception as e:
                                    act_sec = max(
                                        0.0, time.perf_counter() - act_start
                                    )
                                    act_error_count += 1
                                    action = np.zeros(5, dtype=np.float32)
                                    err_txt = f"{type(e).__name__}: {e}"
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
                                    elif strikes >= RPC_MAX_STRIKES_PER_SEED:
                                        bt.logging.warning(
                                            f"UID {uid} seed {task_idx}: {strikes} RPC errors, failing seed"
                                        )

                                rollout_phase_sec["agent_act_sec"] += act_sec
                                rollout_phase_sec["action_decode_sec"] += decode_sec
                                act_latencies_sec.append(act_sec)
                                if act_sec > MINER_COMPUTE_BUDGET_SEC:
                                    act_budget_overrun_count += 1
                                if first_act_ms is None:
                                    first_act_ms = act_sec * 1000.0
                                last_act_ms = act_sec * 1000.0
                                is_first_step = False

                                if rpc_disconnected or strikes >= RPC_MAX_STRIKES_PER_SEED:
                                    step_total_latencies_sec.append(
                                        max(0.0, time.perf_counter() - step_wall_start)
                                    )
                                    break

                                postprocess_start = time.perf_counter()
                                raw_act = np.nan_to_num(
                                    np.asarray(action, dtype=np.float32).reshape(-1),
                                    nan=0.0,
                                    posinf=0.0,
                                    neginf=0.0,
                                )
                                if raw_act.size != 5:
                                    raw_act = np.zeros(5, dtype=np.float32)
                                act = np.clip(raw_act, lo, hi)

                                if hasattr(env, "ACT_TYPE") and hasattr(
                                    env, "SPEED_LIMIT"
                                ):
                                    if (
                                        env.ACT_TYPE == ActionType.VEL
                                        and env.SPEED_LIMIT
                                    ):
                                        n = max(np.linalg.norm(act[:3]), 1e-6)
                                        scale = min(1.0, SPEED_LIMIT / n)
                                        act[:3] *= scale
                                        act = np.clip(act, lo, hi)
                                rollout_phase_sec["action_postprocess_sec"] += max(
                                    0.0, time.perf_counter() - postprocess_start
                                )

                                _set_phase(
                                    "env_step",
                                    task=task_label,
                                    step=step_idx,
                                    sim_t=t_sim,
                                )
                                env_step_start = time.perf_counter()
                                obs, _r, terminated, truncated, info = env.step(
                                    act[None, :]
                                )
                                env_step_sec = max(
                                    0.0, time.perf_counter() - env_step_start
                                )
                                rollout_phase_sec["env_step_sec"] += env_step_sec
                                env_step_latencies_sec.append(env_step_sec)
                                step_total_latencies_sec.append(
                                    max(0.0, time.perf_counter() - step_wall_start)
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
                                )
                                if terminated or truncated:
                                    success = info.get("success", False)
                                    _trace(
                                        f"{task_label} terminated={terminated} truncated={truncated} "
                                        f"success={success} t_sim={t_sim:.2f}s strikes={strikes}"
                                    )
                                    break
                        finally:
                            seed_phase_sec["rollout_total_sec"] += max(
                                0.0, time.perf_counter() - rollout_start
                            )

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
                            results.append(ValidationResult(uid, False, t_sim, 0.0))
                            emit_status = "seed_cancelled"
                        elif rpc_disconnected:
                            _set_phase(
                                "seed_failed_rpc_disconnect",
                                task=task_label,
                                step=step_idx,
                                sim_t=t_sim,
                            )
                            _trace(f"{task_label} failed due to rpc disconnect")
                            results.append(ValidationResult(uid, False, t_sim, 0.0))
                            emit_status = "seed_rpc_disconnected"
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
                            results.append(ValidationResult(uid, False, t_sim, 0.0))
                            emit_status = "seed_timeout_strikes"
                        else:
                            min_clearance = info.get("min_clearance", None)
                            collision = info.get("collision", False)
                            emit_score = flight_reward(
                                success=success,
                                t=t_sim,
                                horizon=task.horizon,
                                task=task,
                                min_clearance=min_clearance,
                                collision=collision,
                                legitimate_model=True,
                            )
                            _trace(
                                f"{task_label} result success={success} "
                                f"score={emit_score:.4f} t_sim={t_sim:.2f}s"
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
                                score=float(emit_score),
                                info=dict(info),
                            )
                            results.append(
                                ValidationResult(uid, success, t_sim, emit_score)
                            )
                            emit_status = "seed_done"
                            emit_success = bool(success)

                    finally:
                        cleanup_start = time.perf_counter()
                        _cleanup_env_quietly(env)
                        seed_phase_sec["env_cleanup_sec"] += max(
                            0.0, time.perf_counter() - cleanup_start
                        )

                    (
                        seed_total_sec,
                        timing_breakdown,
                        rollout_breakdown_payload,
                        latency_payload,
                        step_metrics,
                    ) = _build_seed_report()
                    _emit_seed_complete(
                        task,
                        status=emit_status,
                        success=emit_success,
                        sim_t=t_sim,
                        seed_wall_sec=seed_total_sec,
                        step_idx=step_idx,
                        error=emit_error,
                        calibration_overhead_sec=rpc_overhead_sec,
                        calibration_cpu_factor=cpu_factor,
                        calibrated_timeout_sec=calibrated_timeout,
                        timing_breakdown=timing_breakdown,
                        rollout_breakdown=rollout_breakdown_payload,
                        latency_stats=latency_payload,
                        step_metrics=step_metrics,
                    )
                    _set_active_seed(None)

                except Exception as e:
                    exc_t_sim = t_sim
                    bt.logging.warning(
                        f"UID {uid} {task_label} failed: {type(e).__name__}: {e}"
                    )
                    _set_phase(
                        "seed_exception", task=task_label, step=0, sim_t=exc_t_sim
                    )
                    _trace(
                        f"{task_label} failed with exception: {type(e).__name__}: {e}"
                    )
                    results.append(ValidationResult(uid, False, exc_t_sim, 0.0))
                    (
                        seed_total_sec,
                        timing_breakdown,
                        rollout_breakdown_payload,
                        latency_payload,
                        step_metrics,
                    ) = _build_seed_report()
                    _emit_seed_complete(
                        task,
                        status="seed_exception",
                        success=False,
                        sim_t=exc_t_sim,
                        seed_wall_sec=seed_total_sec,
                        step_idx=step_idx,
                        error=f"{type(e).__name__}: {e}",
                        calibration_overhead_sec=locals().get("rpc_overhead_sec"),
                        calibration_cpu_factor=locals().get("cpu_factor"),
                        calibrated_timeout_sec=locals().get("calibrated_timeout"),
                        timing_breakdown=timing_breakdown,
                        rollout_breakdown=rollout_breakdown_payload,
                        latency_stats=latency_payload,
                        step_metrics=step_metrics,
                    )
                    _set_active_seed(None)

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

async def _calibrate_rpc_overhead_async(self, agent, agent_capnp, obs, uid: int):
    """Measure RPC pipeline overhead and CPU speed factor.

    The round-trip time includes both network overhead and benchmark compute.
    We subtract the benchmark compute to isolate pure network/serialization cost,
    so the timeout formula doesn't double-count compute via both cpu_factor and overhead.
    """
    docker_evaluator_mod = _docker_evaluator_facade()
    pure_overheads = []
    benchmark_times_ns = []

    for r in range(docker_evaluator_mod.CALIBRATION_ROUNDS):
        cal_obs = self._serialize_observation(agent_capnp, obs)

        try:
            t0 = time.time()
            cal_response = await asyncio.wait_for(
                agent.calibrate(cal_obs),
                timeout=docker_evaluator_mod.CALIBRATION_TIMEOUT_SEC,
            )
            dt = time.time() - t0
            bench_ns = cal_response.benchmarkNs
            bench_sec = bench_ns / 1e9 if bench_ns > 0 else 0.0
            pure_overheads.append(max(0.001, dt - bench_sec))
            if bench_ns > 0:
                benchmark_times_ns.append(bench_ns)
        except (asyncio.TimeoutError, Exception) as e:
            bt.logging.warning(
                f"UID {uid}: calibration round {r+1}/{docker_evaluator_mod.CALIBRATION_ROUNDS} failed: {e}"
            )

    if len(pure_overheads) < 3:
        fallback = max(
            docker_evaluator_mod.RPC_STEP_TIMEOUT_SEC
            - docker_evaluator_mod.MINER_COMPUTE_BUDGET_SEC,
            0.010,
        )
        bt.logging.warning(
            f"UID {uid}: calibration mostly failed ({len(pure_overheads)}/{docker_evaluator_mod.CALIBRATION_ROUNDS} ok), "
            f"using fallback overhead={fallback*1000:.0f}ms, cpu_factor=1.0"
        )
        return fallback, 1.0

    pure_overheads.sort()
    trimmed = pure_overheads[1:-1] if len(pure_overheads) > 4 else pure_overheads
    median_overhead = statistics.median(trimmed)

    if median_overhead > docker_evaluator_mod.CALIBRATION_OVERHEAD_CAP_SEC:
        bt.logging.warning(
            f"UID {uid}: measured RPC overhead {median_overhead*1000:.1f}ms exceeds cap "
            f"{docker_evaluator_mod.CALIBRATION_OVERHEAD_CAP_SEC*1000:.0f}ms — capping."
        )
        median_overhead = docker_evaluator_mod.CALIBRATION_OVERHEAD_CAP_SEC

    cpu_factor = 1.0
    if len(benchmark_times_ns) >= 3:
        benchmark_times_ns.sort()
        trimmed_bench = (
            benchmark_times_ns[1:-1]
            if len(benchmark_times_ns) > 4
            else benchmark_times_ns
        )
        median_bench_ns = statistics.median(trimmed_bench)
        cpu_factor = (
            median_bench_ns / docker_evaluator_mod.CALIBRATION_BENCHMARK_REF_NS
        )
        cpu_factor = max(
            1.0,
            min(cpu_factor, docker_evaluator_mod.CALIBRATION_CPU_FACTOR_CAP),
        )

    bench_median_ms = (
        statistics.median(benchmark_times_ns) / 1e6 if benchmark_times_ns else 0.0
    )
    overhead_ms = median_overhead * 1000
    if overhead_ms > docker_evaluator_mod.CALIBRATION_WARN_OVERHEAD_MS or cpu_factor > docker_evaluator_mod.CALIBRATION_WARN_CPU_FACTOR:
        bt.logging.warning(
            f"UID {uid}: abnormal calibration — "
            f"overhead={overhead_ms:.1f}ms, cpu_factor={cpu_factor:.2f}x, "
            f"benchmark={bench_median_ms:.1f}ms"
        )

    return median_overhead, cpu_factor
