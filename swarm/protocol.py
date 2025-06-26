# swarm/protocol.py
# -----------------------------------------------------------------------------
#  Swarm – Bittensor subnet “Swarm” 
# -----------------------------------------------------------------------------
from __future__ import annotations

import hashlib
import msgpack         # still used locally for hashing
from dataclasses import asdict, dataclass
from typing import List, Tuple

import bittensor as bt

# --------------------------------------------------------------------------- #
# 1.  Pure‑Python dataclasses                                  #
# --------------------------------------------------------------------------- #

@dataclass(slots=True)
class MapTask:
    map_seed: int
    start:  Tuple[float, float, float]
    goal:   Tuple[float, float, float]
    sim_dt: float
    horizon: float
    version: str = "1"

    # msgpack helpers remain for local persistence / hashing if you want them
    def pack(self) -> bytes:
        return msgpack.packb(asdict(self), use_bin_type=True)

    @staticmethod
    def unpack(blob: bytes) -> "MapTask":
        return MapTask(**msgpack.unpackb(blob, raw=False))


@dataclass(slots=True)
class RPMCmd:
    t: float
    rpm: Tuple[float, float, float, float]


@dataclass(slots=True)
class FlightPlan:
    commands: List[RPMCmd]
    sha256: str | None = None

    def __post_init__(self):
        if self.sha256 is None:
            packed = msgpack.packb(
                [(c.t, c.rpm) for c in self.commands], use_bin_type=True
            )
            self.sha256 = hashlib.sha256(packed).hexdigest()

    def pack(self) -> bytes:
        return msgpack.packb(
            {
                "commands": [(c.t, c.rpm) for c in self.commands],
                "sha256": self.sha256,
            },
            use_bin_type=True,
        )

    @staticmethod
    def unpack(blob: bytes) -> "FlightPlan":
        obj = msgpack.unpackb(blob, raw=False)
        cmds = [RPMCmd(t, tuple(rpm)) for t, rpm in obj["commands"]]
        return FlightPlan(commands=cmds, sha256=obj["sha256"])


@dataclass(slots=True)
class ValidationResult:
    uid: int
    success: bool
    time_sec: float
    energy: float
    score: float


# --------------------------------------------------------------------------- #
# 2.  Synapse wrappers                          #
# --------------------------------------------------------------------------- #

def _tuple_to_list3(t: Tuple[float, float, float]) -> List[float]:
    """Helper so tuples don’t get rejected by the JSON encoder."""
    return [float(x) for x in t]


class MapTaskSynapse(bt.Synapse):
    """
    Validator ➜ Miner  (pure JSON payload)
    """
    # --- payload fields (all JSON native types) --------------------------- #
    map_seed: int
    start: List[float]      # len == 3
    goal:  List[float]
    sim_dt: float
    horizon: float
    version: str = "1"

    # --- bittensor hook --------------------------------------------------- #
    def deserialize(self) -> "MapTaskSynapse":      # noqa: D401
        # Nothing to do – attributes are already native types
        return self

    # --- convenience accessors ------------------------------------------- #
    @property
    def task(self) -> MapTask:
        return MapTask(
            map_seed=self.map_seed,
            start=tuple(self.start),
            goal=tuple(self.goal),
            sim_dt=self.sim_dt,
            horizon=self.horizon,
            version=self.version,
        )

    # --- builders --------------------------------------------------------- #
    @staticmethod
    def from_task(task: MapTask) -> "MapTaskSynapse":
        return MapTaskSynapse(
            map_seed=task.map_seed,
            start=_tuple_to_list3(task.start),
            goal=_tuple_to_list3(task.goal),
            sim_dt=float(task.sim_dt),
            horizon=float(task.horizon),
            version=task.version,
        )


class FlightPlanSynapse(bt.Synapse):
    """
    Miner ➜ Validator  (pure JSON payload)
    """
    # commands is a **list of dicts**: {"t": float, "rpm": [f,f,f,f]}
    commands: List[dict]
    sha256: str
    version: str = "1"

    def deserialize(self) -> "FlightPlanSynapse":    # noqa: D401
        return self

    @property
    def plan(self) -> FlightPlan:
        cmds = [RPMCmd(c["t"], tuple(c["rpm"])) for c in self.commands]
        return FlightPlan(commands=cmds, sha256=self.sha256)

    @staticmethod
    def from_plan(plan: FlightPlan) -> "FlightPlanSynapse":
        return FlightPlanSynapse(
            commands=[{"t": c.t, "rpm": list(c.rpm)} for c in plan.commands],
            sha256=plan.sha256,
            version="1",
        )

__all__ = [
    "MapTask",
    "RPMCmd",
    "FlightPlan",
    "ValidationResult",
    "MapTaskSynapse",
    "FlightPlanSynapse",
]
