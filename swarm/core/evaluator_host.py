#!/usr/bin/env python3
import asyncio
import time
import socket
import sys
from pathlib import Path
import numpy as _np

swarm_path = str(Path(__file__).resolve().parent.parent.parent)
if swarm_path not in sys.path:
    sys.path.insert(0, swarm_path)

from swarm.protocol import MapTask, ValidationResult
from swarm.utils.env_factory import make_env
from swarm.constants import SIM_DT, SPEED_LIMIT
from gym_pybullet_drones.utils.enums import ActionType


async def evaluate_with_rpc_client(
    task: MapTask,
    uid: int,
    container_ip: str,
    auth_token: str = None
) -> ValidationResult:
    
    max_wait = 30
    wait_interval = 0.5
    
    for _ in range(int(max_wait / wait_interval)):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.5)
            result = sock.connect_ex((container_ip, 9000))
            sock.close()
            if result == 0:
                break
        except Exception:
            pass
        time.sleep(wait_interval)
    else:
        return ValidationResult(uid, False, 0.0, 0.0)
    
    try:
        import capnp
        schema_file = Path(__file__).parent.parent / "submission_template" / "agent.capnp"
        agent_capnp = capnp.load(str(schema_file))
        
        async def run_evaluation():
            async with capnp.kj_loop():
                client = capnp.TwoPartyClient(f"{container_ip}:9000")
                agent = client.bootstrap().cast_as(agent_capnp.Agent)
                
                if auth_token:
                    ping_result = await agent.ping(auth_token)
                    if ping_result != "pong":
                        raise RuntimeError("RPC authentication failed")
                else:
                    ping_result = await agent.ping("test")
                    if ping_result != "pong":
                        raise RuntimeError("RPC ping failed")
                
                env = make_env(task, gui=False)
                
                try:
                    obs, _ = env.reset()
                    
                    pos0 = _np.asarray(task.start, dtype=float)
                    t_sim = 0.0
                    success = False
                    
                    lo, hi = env.action_space.low.flatten(), env.action_space.high.flatten()
                    last_pos = pos0
                    
                    while t_sim < task.horizon:
                        try:
                            obs_tensor = agent_capnp.Tensor.new_message()
                            obs_tensor.data = obs.tobytes()
                            obs_tensor.shape = list(obs.shape)
                            obs_tensor.dtype = str(obs.dtype)
                            
                            observation = agent_capnp.Observation.new_message()
                            entry = observation.init("entries", 1)[0]
                            entry.key = "__value__"
                            entry.tensor = obs_tensor
                            
                            action_tensor = await agent.act(observation)
                            action = _np.frombuffer(
                                action_tensor.data,
                                dtype=_np.dtype(action_tensor.dtype)
                            ).reshape(tuple(action_tensor.shape))
                        except Exception:
                            action = _np.zeros(4, dtype=_np.float32)
                        
                        act = _np.clip(_np.asarray(action, dtype=_np.float32).reshape(-1), lo, hi)
                        
                        if hasattr(env, 'ACT_TYPE') and hasattr(env, 'SPEED_LIMIT'):
                            if env.ACT_TYPE == ActionType.VEL and env.SPEED_LIMIT:
                                n = max(_np.linalg.norm(act[:3]), 1e-6)
                                scale = min(1.0, SPEED_LIMIT / n)
                                act[:3] *= scale
                                act = _np.clip(act, lo, hi)
                        
                        prev = last_pos
                        obs, _r, terminated, truncated, info = env.step(act[None, :])
                        last_pos = env._getDroneStateVector(0)[0:3]
                        
                        t_sim += SIM_DT
                        if terminated or truncated:
                            success = info.get("success", False)
                            break
                    
                    from swarm.validator.reward import flight_reward
                    score = flight_reward(
                        success=success,
                        t=t_sim,
                        horizon=task.horizon,
                        task=task,
                    )
                    
                    return ValidationResult(uid, success, t_sim, score)
                finally:
                    try:
                        env.close()
                    except Exception:
                        pass
        
        result = await run_evaluation()
    except Exception:
        result = ValidationResult(uid, False, 0.0, 0.0)
    
    return result

