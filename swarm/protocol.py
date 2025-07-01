# swarm/protocol.py
# --------------------------------------------------------------------------- #
#  Swarm – unified protocol definitions (SDK v2.1)
# --------------------------------------------------------------------------- #
"""
Adds an **update‑poll handshake**:

    Validator            Miner
    ────────── ask_ref? ───►
                       [no_update]   (nothing changed)
                       [ref]         (new SHA → validator may set need_blob)

*If* the validator already has the model cached and only wants to confirm
freshness it sends ``ask_ref=True`` without a task.  The miner replies
``no_update=True`` or a full ``ref``.  The payload is therefore limited to a
few hundred bytes when nothing changed.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import List, Tuple, Optional, Dict, Any

import msgpack
import bittensor as bt


# --------------------------------------------------------------------------- #
# 1.  Core dataclasses (unchanged)                                            #
# --------------------------------------------------------------------------- #

@dataclass(slots=True)
class MapTask:
    map_seed: int
    start: Tuple[float, float, float]
    goal: Tuple[float, float, float]
    sim_dt: float
    horizon: float
    version: str = "1"

    def pack(self) -> bytes:
        return msgpack.packb(asdict(self), use_bin_type=True)

    @staticmethod
    def unpack(blob: bytes) -> "MapTask":
        return MapTask(**msgpack.unpackb(blob, raw=False))


@dataclass(slots=True)
class RPMCmd:
    t: float
    rpm: Tuple[float, float, float, float]


# Legacy – kept one cycle
@dataclass(slots=True)
class FlightPlan:
    commands: List[RPMCmd]
    sha256: str | None = None

    def __post_init__(self):
        if self.sha256 is None:
            payload = [(c.t, c.rpm) for c in self.commands]
            self.sha256 = hashlib.sha256(
                msgpack.packb(payload, use_bin_type=True)
            ).hexdigest()

    def pack(self) -> bytes:
        return msgpack.packb(
            {"commands": [(c.t, c.rpm) for c in self.commands],
             "sha256": self.sha256},
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
# 2.  Model‑streaming helpers                                                 #
# --------------------------------------------------------------------------- #

@dataclass(slots=True)
class PolicyRef:
    sha256: str
    entrypoint: str
    framework: str
    size_bytes: int
    version: str = "1"

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PolicyChunk:
    sha256: str
    data: bytes

    def as_dict(self) -> Dict[str, Any]:
        return {"sha256": self.sha256, "data": self.data}


# --------------------------------------------------------------------------- #
# 3.  Synapse utilities                                                       #
# --------------------------------------------------------------------------- #

def _tuple_to_list(t: Tuple[float, float, float]) -> List[float]:
    return [float(x) for x in t]


# --------------------------------------------------------------------------- #
# 4.  Bidirectional Synapse                                                   #
# --------------------------------------------------------------------------- #

Synapse = bt.Synapse      # alias

class PolicySynapse(Synapse):
    """
    Fields                               Direction
    ------                               ---------
    ask_ref      : bool                V → M   “send PolicyRef if updated”
    need_blob    : bool                V → M   “I miss <sha>; stream it”
    task*        : MapTask fields      V → M   evaluation request

    ref          : PolicyRef dict      M → V   manifest of new model
    no_update    : bool                M → V   model unchanged
    chunk        : PolicyChunk dict    M → V   wheel / zip fragments
    result       : ValidationResult    V → M   score returned to miner
    """
    # MapTask (validator ➜ miner)
    map_seed: Optional[int] = None
    start: Optional[List[float]] = None
    goal: Optional[List[float]] = None
    sim_dt: Optional[float] = None
    horizon: Optional[float] = None

    # Handshake
    ask_ref: Optional[bool] = None      # validator ➜ miner
    need_blob: Optional[bool] = None    # validator ➜ miner

    ref: Optional[Dict[str, Any]] = None      # miner ➜ validator
    no_update: Optional[bool] = None          # miner ➜ validator
    chunk: Optional[Dict[str, Any]] = None    # miner ➜ validator

    # Evaluation result
    result: Optional[Dict[str, Any]] = None   # validator ➜ miner

    version: str = "1"

    # Bittensor hook
    def deserialize(self) -> "PolicySynapse":
        return self

    # -------- convenience accessors -------------------------------------

    @property
    def task(self) -> Optional[MapTask]:
        if None in (self.map_seed, self.start, self.goal,
                    self.sim_dt, self.horizon):
            return None
        return MapTask(
            map_seed=self.map_seed,           # type: ignore[arg-type]
            start=tuple(self.start),          # type: ignore[arg-type]
            goal=tuple(self.goal),            # type: ignore[arg-type]
            sim_dt=self.sim_dt,
            horizon=self.horizon,
            version=self.version,
        )

    @property
    def policy_ref(self) -> Optional[PolicyRef]:
        return PolicyRef(**self.ref) if self.ref else None   # type: ignore[arg-type]

    @property
    def policy_chunk(self) -> Optional[PolicyChunk]:
        return PolicyChunk(**self.chunk) if self.chunk else None  # type: ignore[arg-type]

    @property
    def validation_result(self) -> Optional[ValidationResult]:
        return ValidationResult(**self.result) if self.result else None  # type: ignore[arg-type]

    # -------- static builders -------------------------------------------

    # 1) validator asks “have anything new?”
    @staticmethod
    def query_update() -> "PolicySynapse":
        return PolicySynapse(ask_ref=True)

    # 2) validator requests missing blob after seeing ref
    @staticmethod
    def request_blob(task: Optional[MapTask] = None) -> "PolicySynapse":
        payload: Dict[str, Any] = {"need_blob": True}
        if task:
            payload.update(
                dict(
                    map_seed=task.map_seed,
                    start=_tuple_to_list(task.start),
                    goal=_tuple_to_list(task.goal),
                    sim_dt=float(task.sim_dt),
                    horizon=float(task.horizon),
                )
            )
        return PolicySynapse(**payload)       # type: ignore[arg-type]

    # 3) miner says “model unchanged”
    @staticmethod
    def no_update_msg() -> "PolicySynapse":
        return PolicySynapse(no_update=True)

    # 4) miner sends manifest
    @staticmethod
    def from_ref(ref: PolicyRef) -> "PolicySynapse":
        return PolicySynapse(ref=ref.as_dict())

    # 5) miner streams chunk
    @staticmethod
    def from_chunk(chunk: PolicyChunk) -> "PolicySynapse":
        return PolicySynapse(chunk=chunk.as_dict())

    # 6) validator sends MapTask for evaluation (no blob needed)
    @staticmethod
    def task_request(task: MapTask) -> "PolicySynapse":
        return PolicySynapse(
            map_seed=task.map_seed,
            start=_tuple_to_list(task.start),
            goal=_tuple_to_list(task.goal),
            sim_dt=float(task.sim_dt),
            horizon=float(task.horizon),
            ask_ref=False,
            need_blob=False,
            version=task.version,
        )

    # 7) validator returns score
    @staticmethod
    def from_result(res: ValidationResult) -> "PolicySynapse":
        return PolicySynapse(result=asdict(res))


# --------------------------------------------------------------------------- #
# 5.  Export list                                                             #
# --------------------------------------------------------------------------- #

__all__ = [
    # core mission objects
    "MapTask",
    "RPMCmd",
    "ValidationResult",
    # legacy
    "FlightPlan",
    # model‑streaming
    "PolicyRef",
    "PolicyChunk",
    "PolicySynapse",
]
