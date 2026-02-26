# --------------------------------------------------------------------------- #
#  Swarm – unified protocol definitions (SDK v2.2, simplified handshake)
# --------------------------------------------------------------------------- #
"""
Handshake (always two messages max):

    Validator            Miner
    ────────── empty ─────►      (request PolicyRef)
                   ref   ◄──────
    ─── need_blob=True ──►      (only if SHA mismatch)
               chunks   ◄──────  (streamed until EOF)

No MapTask data is exchanged; miners never know the evaluation map.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple, Optional, Any

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
    challenge_type: int
    search_radius: float = 10.0
    moving_platform: bool = False
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


@dataclass(slots=True)
class ValidationResult:
    uid: int
    success: bool
    time_sec: float
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
    data: str

    def as_dict(self) -> Dict[str, Any]:
        return {"sha256": self.sha256, "data": self.data}


# --------------------------------------------------------------------------- #
# 3.  Bidirectional Synapse                                                   #
# --------------------------------------------------------------------------- #
Synapse = bt.Synapse     # alias


class PolicySynapse(Synapse):
    """
    Fields                               Direction
    ------                               ---------
    need_blob    : bool                V → M   request model payload

    ref          : PolicyRef dict      M → V   model manifest
    chunk        : PolicyChunk dict    M → V   streamed binary data
    result       : ValidationResult    V → M   evaluation score
    """
    need_blob: Optional[bool] = None           # validator ➜ miner

    ref:   Optional[Dict[str, Any]] = None     # miner ➜ validator
    chunk: Optional[Dict[str, Any]] = None     # miner ➜ validator

    result: Optional[Dict[str, Any]] = None    # validator ➜ miner

    version: str = "1"
    timeout: float = 5.0                      # custom timeout in seconds

    # Bittensor hook
    def deserialize(self) -> "PolicySynapse":
        return self

    # -------- convenience accessors ---------------------------------
    @property
    def policy_ref(self) -> Optional[PolicyRef]:
        return PolicyRef(**self.ref) if self.ref else None   # type: ignore[arg-type]

    @property
    def policy_chunk(self) -> Optional[PolicyChunk]:
        return PolicyChunk(**self.chunk) if self.chunk else None  # type: ignore[arg-type]

    @property
    def validation_result(self) -> Optional[ValidationResult]:
        return ValidationResult(**self.result) if self.result else None  # type: ignore[arg-type]

    # -------- static builders ---------------------------------------
    @staticmethod
    def request_ref() -> "PolicySynapse":
        """Validator → Miner: “send me your current PolicyRef”"""
        return PolicySynapse()

    @staticmethod
    def request_blob() -> "PolicySynapse":
        """Validator → Miner: “stream me the binary”"""
        return PolicySynapse(need_blob=True)

    @staticmethod
    def from_ref(ref: PolicyRef) -> "PolicySynapse":
        return PolicySynapse(ref=ref.as_dict())

    @staticmethod
    def from_chunk(chunk: PolicyChunk) -> "PolicySynapse":
        return PolicySynapse(chunk=chunk.as_dict())

    @staticmethod
    def from_result(res: ValidationResult) -> "PolicySynapse":
        return PolicySynapse(result=asdict(res))


# --------------------------------------------------------------------------- #
# 4.  Export list                                                             #
# --------------------------------------------------------------------------- #
__all__ = [
    "MapTask",
    "RPMCmd",
    "ValidationResult",
    "PolicyRef",
    "PolicyChunk",
    "PolicySynapse",
]
