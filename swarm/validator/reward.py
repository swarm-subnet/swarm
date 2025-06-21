# swarm/validator/reward.py
def flight_reward(success: bool, t: float, e: float, horizon: float,
                  w_t=0.9, w_e=0.1) -> float:
    if not success:
        return 0.0
    t_norm = min(1.0, t / horizon)
    e_norm = e / (e + 50)
    score  = 1.0 - w_t*t_norm - w_e*e_norm
    return max(0.0, score)
