# swarm/utils/weight_utils.py
from __future__ import annotations
from typing import Dict
import numpy as np

def update_ema_weights(old: Dict[int, float]|None, new: Dict[int, float],
                       alpha: float=0.2) -> Dict[int,int]:
    if old is None: old = {}
    ema = {uid: (1-alpha)*old.get(uid,0.0)+alpha*score for uid,score in new.items()}
    for uid,val in old.items():
        if uid not in ema:
            ema[uid] = (1-alpha)*val
    vals = np.array(list(ema.values()), float)
    if vals.sum() == 0: return {u:0 for u in ema}
    vals = vals/vals.sum()*(2**64-1)
    return {uid:int(v) for uid,v in zip(ema.keys(), vals)}
