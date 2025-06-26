# swarm/protocol.py
# -----------------------------------------------------------------------------
#  Swarm – Bittensor subnet “Swarm”
# -----------------------------------------------------------------------------
"""Unified protocol definitions for the Swarm subnet.

This revision *merges the former ``MapTaskSynapse`` into ``FlightPlanSynapse``* so
only **one** synapse class remains.  The new ``FlightPlanSynapse`` is bidirectional:

* **Validator ➜ Miner** – carries *only* the planning task fields.
* **Miner ➜ Validator** – carries the flight‑plan fields and can optionally echo
  the originating task back for stateless validation.

All attributes are therefore declared **optional**; the producer of the message
simply omits the fields that are not relevant in that direction.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import List, Tuple, Optional

import msgpack  # still used locally for hashing / persistence
import bittensor as bt

# --------------------------------------------------------------------------- #
# 1.  Pure‑Python dataclasses                                                  #
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class MapTask:
    map_seed: int
    start: Tuple[float, float, float]
    goal: Tuple[float, float, float]
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
# 2.  Synapse helpers                                                          #
# --------------------------------------------------------------------------- #


def _tuple_to_list3(t: Tuple[float, float, float]) -> List[float]:
    """Helper so tuples don’t get rejected by the JSON encoder."""
    return [float(x) for x in t]


# --------------------------------------------------------------------------- #
# 3.  Unified Synapse                                                          #
# --------------------------------------------------------------------------- #


Synapse = bt.Synapse  # type alias for brevity; comes from bittensor


class FlightPlanSynapse(Synapse):
    """Bidirectional synapse used *both* for planning tasks and plan results.

    **Direction 1 – Validator ➜ Miner**
        ``map_seed`` … ``horizon`` fields *must* be present.

    **Direction 2 – Miner ➜ Validator**
        ``commands`` & ``sha256`` fields *must* be present and the task fields
        *may* be echoed back as a convenience.
    """

    # --- MapTask fields --------------------------------------------------- #
    map_seed: Optional[int] = None
    start: Optional[List[float]] = None     # len == 3
    goal: Optional[List[float]] = None      # len == 3
    sim_dt: Optional[float] = None
    horizon: Optional[float] = None

    # --- FlightPlan fields ------------------------------------------------ #
    commands: Optional[List[dict]] = None   # {"t": float, "rpm": [f,f,f,f]}
    sha256: Optional[str] = None

    # --- protocol meta ---------------------------------------------------- #
    version: str = "1"

    # --- bittensor hook --------------------------------------------------- #
    def deserialize(self) -> "FlightPlanSynapse":  # noqa: D401
        # Attributes are already JSON‑native; simply return self.
        return self

    # --- convenience accessors ------------------------------------------- #
    @property
    def task(self) -> Optional[MapTask]:
        """Convert to :class:`MapTask` if task fields are present."""
        if None in (
            self.map_seed,
            self.start,
            self.goal,
            self.sim_dt,
            self.horizon,
        ):
            return None
        return MapTask(
            map_seed=self.map_seed,  # type: ignore[arg-type]
            start=tuple(self.start),  # type: ignore[arg-type]
            goal=tuple(self.goal),    # type: ignore[arg-type]
            sim_dt=self.sim_dt,
            horizon=self.horizon,
            version=self.version,
        )

    @property
    def plan(self) -> Optional[FlightPlan]:
        """Convert to :class:`FlightPlan` if plan fields are present."""
        if self.commands is None or self.sha256 is None:
            return None
        cmds = [RPMCmd(c["t"], tuple(c["rpm"])) for c in self.commands]
        return FlightPlan(commands=cmds, sha256=self.sha256)

    # --- builders --------------------------------------------------------- #
    @staticmethod
    def from_task(task: MapTask) -> "FlightPlanSynapse":
        """Factory for the *Validator ➜ Miner* direction."""
        return FlightPlanSynapse(
            map_seed=task.map_seed,
            start=_tuple_to_list3(task.start),
            goal=_tuple_to_list3(task.goal),
            sim_dt=float(task.sim_dt),
            horizon=float(task.horizon),
            version=task.version,
        )

    @staticmethod
    def from_plan(
        plan: FlightPlan,
        *,
        task: Optional[MapTask] = None,
    ) -> "FlightPlanSynapse":
        """Factory for the *Miner ➜ Validator* direction.

        The originating :class:`MapTask` can be attached so the validator can
        match the plan to its task without additional state.
        """
        payload: dict = {
            "commands": [{"t": c.t, "rpm": list(c.rpm)} for c in plan.commands],
            "sha256": plan.sha256,
            "version": "1",
        }
        if task is not None:
            payload.update(
                {
                    "map_seed": task.map_seed,
                    "start": _tuple_to_list3(task.start),
                    "goal": _tuple_to_list3(task.goal),
                    "sim_dt": float(task.sim_dt),
                    "horizon": float(task.horizon),
                }
            )
        return FlightPlanSynapse(**payload)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# 4.  Export list                                                             #
# --------------------------------------------------------------------------- #

__all__ = [
    "MapTask",
    "RPMCmd",
    "FlightPlan",
    "ValidationResult",
    "FlightPlanSynapse",
]
