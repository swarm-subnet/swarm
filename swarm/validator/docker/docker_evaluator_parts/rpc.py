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
                seed_wall_start = time.time()
                try:
                    _set_phase("seed_start", task=task_label, step=0, sim_t=0.0)
                    _trace(
                        f"{task_label} start horizon={getattr(task, 'horizon', 0.0):.1f}s"
                    )
                    t_env_start = time.time()
                    _set_phase("env_build", task=task_label, step=0, sim_t=0.0)
                    _trace(f"{task_label} building env")
                    env = make_env(task, gui=False)
                    _trace(
                        f"{task_label} env built in {(time.time() - t_env_start):.2f}s"
                    )

                    try:
                        t_reset_env_start = time.time()
                        _set_phase("env_reset", task=task_label, step=0, sim_t=0.0)
                        _trace(f"{task_label} env.reset() start")
                        obs, _ = env.reset()
                        _trace(
                            f"{task_label} env.reset() done in {(time.time() - t_reset_env_start):.2f}s"
                        )
                        t_reset_start = time.time()
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
                        reset_ms = (time.time() - t_reset_start) * 1000.0
                        _trace(f"{task_label} reset ok in {reset_ms:.1f}ms")

                        if not calibrated:
                            _set_phase(
                                "rpc_calibration",
                                task=task_label,
                                step=0,
                                sim_t=0.0,
                            )
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
                            bt.logging.info(
                                f"UID {uid}: calibrated timeout = {calibrated_timeout*1000:.1f}ms "
                                f"(budget={MINER_COMPUTE_BUDGET_SEC*1000:.0f}ms x {cpu_factor:.2f} + overhead={rpc_overhead_sec*1000:.1f}ms + margin={CALIBRATION_MARGIN_SEC*1000:.0f}ms)"
                            )
                            _trace(
                                f"calibrated step timeout={calibrated_timeout*1000:.1f}ms "
                                f"(overhead={rpc_overhead_sec*1000:.1f}ms cpu_factor={cpu_factor:.2f}x)"
                            )

                        t_sim = 0.0
                        success = False
                        info = {}
                        strikes = 0
                        is_first_step = True
                        step_idx = 0
                        rpc_disconnected = False

                        lo, hi = (
                            env.action_space.low.flatten(),
                            env.action_space.high.flatten(),
                        )

                        while t_sim < task.horizon and not (
                            stop_event is not None and stop_event.is_set()
                        ):
                            step_idx += 1
                            step_timeout = (
                                RPC_FIRST_STEP_TIMEOUT_SEC
                                if is_first_step
                                else calibrated_timeout
                            )

                            observation = self._serialize_observation(
                                agent_capnp, obs
                            )
                            _set_phase(
                                "rpc_act",
                                task=task_label,
                                step=step_idx,
                                sim_t=t_sim,
                            )

                            try:
                                t_act_start = time.time()
                                action_response = await asyncio.wait_for(
                                    agent.act(observation), timeout=step_timeout
                                )
                                act_ms = (time.time() - t_act_start) * 1000.0
                                action = np.frombuffer(
                                    action_response.action.data,
                                    dtype=np.dtype(action_response.action.dtype),
                                ).reshape(tuple(action_response.action.shape))
                                if trace_rpc and (
                                    step_idx == 1 or step_idx % trace_every == 0
                                ):
                                    _trace(
                                        f"{task_label} step={step_idx} t_sim={t_sim:.2f}s "
                                        f"act_ok={act_ms:.1f}ms timeout={step_timeout*1000:.0f}ms"
                                    )
                            except asyncio.TimeoutError:
                                act_ms = (time.time() - t_act_start) * 1000
                                strikes += 1
                                action = np.zeros(5, dtype=np.float32)
                                _trace(
                                    f"{task_label} step={step_idx} act timeout {act_ms:.1f}ms "
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
                                if strikes >= RPC_MAX_STRIKES_PER_SEED:
                                    bt.logging.warning(
                                        f"UID {uid} seed {task_idx}: {strikes} RPC timeouts, failing seed"
                                    )
                                    break
                            except Exception as e:
                                action = np.zeros(5, dtype=np.float32)
                                err_txt = f"{type(e).__name__}: {e}"
                                _trace(
                                    f"{task_label} step={step_idx} act error: {err_txt}"
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

                            is_first_step = False

                            act = np.clip(
                                np.asarray(action, dtype=np.float32).reshape(-1),
                                lo,
                                hi,
                            )

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

                            _set_phase(
                                "env_step",
                                task=task_label,
                                step=step_idx,
                                sim_t=t_sim,
                            )
                            obs, _r, terminated, truncated, info = env.step(
                                act[None, :]
                            )

                            t_sim += SIM_DT
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
                            results.append(ValidationResult(uid, False, t_sim, 0.0))
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
                            results.append(ValidationResult(uid, False, t_sim, 0.0))
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
                            results.append(ValidationResult(uid, False, t_sim, 0.0))
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
                            min_clearance = info.get("min_clearance", None)
                            collision = info.get("collision", False)
                            score = flight_reward(
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
                                f"score={score:.4f} t_sim={t_sim:.2f}s"
                            )
                            _set_phase(
                                "seed_done",
                                task=task_label,
                                step=step_idx,
                                sim_t=t_sim,
                            )
                            results.append(
                                ValidationResult(uid, success, t_sim, score)
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
                    _emit_seed_complete(
                        task,
                        status="seed_exception",
                        success=False,
                        sim_t=exc_t_sim,
                        seed_wall_sec=time.time() - seed_wall_start,
                        step_idx=0,
                        error=f"{type(e).__name__}: {e}",
                        calibration_overhead_sec=locals().get("rpc_overhead_sec"),
                        calibration_cpu_factor=locals().get("cpu_factor"),
                        calibrated_timeout_sec=locals().get("calibrated_timeout"),
                    )

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
    bt.logging.info(
        f"UID {uid}: calibration results — "
        f"overhead={median_overhead*1000:.1f}ms (pure network), "
        f"cpu_factor={cpu_factor:.2f}x, "
        f"benchmark_median={bench_median_ms:.1f}ms, "
        f"rtt=[{', '.join(f'{t*1000:.1f}' for t in pure_overheads)}]ms"
    )

    return median_overhead, cpu_factor

async def _run_multi_seed_rpc_host(
    self,
    tasks: list,
    uid: int,
    rpc_port: int,
    on_seed_complete: Optional[Callable[..., None]] = None,
    stop_event: Optional[threading.Event] = None,
    progress_state: Optional[dict] = None,
    task_offset: int = 0,
    task_total: Optional[int] = None,
) -> list:
    """Async wrapper that runs multi-seed RPC evaluation in thread pool."""
    try:
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: self._run_multi_seed_rpc_sync(
                tasks,
                uid,
                rpc_port,
                on_seed_complete,
                stop_event,
                progress_state,
                task_offset,
                task_total,
            ),
        )
        return results
    except Exception as e:
        bt.logging.warning(f"Multi-seed RPC evaluation failed for UID {uid}: {e}")
        for _ in tasks:
            if on_seed_complete:
                try:
                    on_seed_complete()
                except Exception:
                    pass
        return [ValidationResult(uid, False, 0.0, 0.0) for _ in tasks]
