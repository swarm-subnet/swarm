# swarm/protocol.py
# -----------------------------------------------------------------------------
#  Swarm – Bittensor subnet “Drone-Nav”
#
#  All **over-the-wire** payloads (validator⇆miner) live here.  The file gives:
#    1.  Dataclasses that the business logic works with locally.
#    2.  Thin Synapse wrappers that Bittensor uses for transmission.
#
#  Why both?
#  ----------
#  •  Dataclasses → fast, type-safe, IDE-friendly, no bittensor import cost
#  •  Synapses    → exactly what Bittensor expects on the wire (Pydantic model)
#     They simply embed / extract the dataclass binary blob.
# -----------------------------------------------------------------------------
from __future__ import annotations

import hashlib
import msgpack
from dataclasses import asdict, dataclass
from typing import List, Tuple

import bittensor as bt
from bittensor import Synapse


# --------------------------------------------------------------------------- #
# 1.  Pure-Python dataclasses                                                  #
# --------------------------------------------------------------------------- #
#
# NOTE:  We serialise with **msgpack** and *never* with pickle / JSON:
#        • msgpack is deterministic, cross-language, binary-compact
#        • pickle breaks determinism & is a security hazard
#

@dataclass(slots=True)
class MapTask:
    """
    Single flight task issued by the *validator*.

    Parameters
    ----------
    map_seed
        Global seed — both miner & validator call ``build_world(seed)``.
    start, goal
        Cartesian (x, y, z) in **world metres**.
    sim_dt
        Control step used by the miner in seconds (e.g. 0.02 = 50 Hz).
    horizon
        Max episode length in seconds; validator truncates beyond this.
    version
        Increment whenever a breaking protocol change happens.
    """
    map_seed: int
    start: Tuple[float, float, float]
    goal:  Tuple[float, float, float]
    sim_dt: float
    horizon: float
    version: str = "1"   # bump → miners running old code auto-fail

    # --------------------------------------------------------------------- #
    #  MsgPack helpers                                                      #
    # --------------------------------------------------------------------- #
    def pack(self) -> bytes:
        """→ wire bytes (deterministic)."""
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
    """
    Deterministic control trace returned by the *miner*.

    • The SHA-256 is re-calculated by the validator to catch tampering
      or floating-point divergence.
    """
    commands: List[RPMCmd]
    sha256: str | None = None              # filled in __post_init__

    # energy/time stats are computed by the validator — keep payload minimal

    # --------------------------------------------------------------------- #
    #  Construction / serialisation                                         #
    # --------------------------------------------------------------------- #
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
    """
    Outcome **internal to validator** → converted to bittensor weights.
    """
    uid: int
    success: bool
    time_sec: float
    energy: float
    score: float


# --------------------------------------------------------------------------- #
# 2.  Synapse wrappers                                                        #
# --------------------------------------------------------------------------- #
#
# They do *zero* logical work – merely embed the blob created above so the
# bittensor network can ship it around.  The `.deserialize()` override is
# mandatory so Synapse returns itself untouched on the miner side.
#

class MapTaskSynapse(Synapse):
    """
    Validator ➜ Miner

    Payload = msgpack-encoded :class:`MapTask`.
    """

    version: str = ""
    blob: bytes  # raw msgpack

    # ------ bittensor hook ------------------------------------------------ #
    def deserialize(self) -> "MapTaskSynapse":  # noqa: D401  (bittensor API)
        return self

    # ------ helpers ------------------------------------------------------- #
    @property
    def task(self) -> MapTask:
        return MapTask.unpack(self.blob)

    @staticmethod
    def from_task(task: MapTask) -> "MapTaskSynapse":
        return MapTaskSynapse(version=task.version, blob=task.pack())


class FlightPlanSynapse(Synapse):
    """
    Miner ➜ Validator

    Payload = msgpack-encoded :class:`FlightPlan`.
    """

    version: str = ""
    blob: bytes

    def deserialize(self) -> "FlightPlanSynapse":
        return self

    @property
    def plan(self) -> FlightPlan:
        return FlightPlan.unpack(self.blob)

    @staticmethod
    def from_plan(plan: FlightPlan) -> "FlightPlanSynapse":
        return FlightPlanSynapse(version="1", blob=plan.pack())


# --------------------------------------------------------------------------- #
#  Convenience re-exports                                                    #
# --------------------------------------------------------------------------- #
__all__ = [
    # dataclasses
    "MapTask",
    "RPMCmd",
    "FlightPlan",
    "ValidationResult",
    # synapses
    "MapTaskSynapse",
    "FlightPlanSynapse",
]
