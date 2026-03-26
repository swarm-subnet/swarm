# --------------------------------------------------------------------------- #
#  Swarm – unified protocol definitions (SDK v2.3, GitHub-hosted models)
# --------------------------------------------------------------------------- #
"""
Handshake (single message):

    Validator            Miner
    ────────── empty ─────►      (request PolicyRef)
                   ref   ◄──────  (includes github_url for model download)

Validators download submission.zip directly from the miner's public GitHub
repository.  No binary streaming over the wire.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import msgpack
try:
    import bittensor as bt
except ImportError:
    bt = None


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
class ValidationResult:
    uid: int
    success: bool
    time_sec: float
    score: float


# --------------------------------------------------------------------------- #
# 2.  Model reference                                                         #
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class PolicyRef:
    sha256: str
    entrypoint: str
    framework: str
    size_bytes: int
    github_url: str
    version: str = "1"

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# 3.  Bidirectional Synapse                                                   #
# --------------------------------------------------------------------------- #
Synapse = bt.Synapse if bt is not None else None  # alias


class PolicySynapse(Synapse):
    """
    Fields                               Direction
    ------                               ---------
    ref          : PolicyRef dict      M → V   model manifest + github_url
    result       : ValidationResult    V → M   evaluation score
    """
    ref: Optional[Dict[str, Any]] = None  # miner → validator
    result: Optional[Dict[str, Any]] = None  # validator → miner

    version: str = "1"
    timeout: float = 5.0

    def deserialize(self) -> "PolicySynapse":
        return self

    # -------- convenience accessors ---------------------------------
    @property
    def policy_ref(self) -> Optional[PolicyRef]:
        return PolicyRef(**self.ref) if self.ref else None  # type: ignore[arg-type]

    @property
    def validation_result(self) -> Optional[ValidationResult]:
        return ValidationResult(**self.result) if self.result else None  # type: ignore[arg-type]

    # -------- static builders ---------------------------------------
    @staticmethod
    def request_ref() -> "PolicySynapse":
        """Validator -> Miner: send me your current PolicyRef."""
        return PolicySynapse()

    @staticmethod
    def from_ref(ref: PolicyRef) -> "PolicySynapse":
        return PolicySynapse(ref=ref.as_dict())

    @staticmethod
    def from_result(res: ValidationResult) -> "PolicySynapse":
        return PolicySynapse(result=asdict(res))


# --------------------------------------------------------------------------- #
# 4.  Export list                                                             #
# --------------------------------------------------------------------------- #
__all__ = [
    "MapTask",
    "ValidationResult",
    "PolicyRef",
    "PolicySynapse",
]
