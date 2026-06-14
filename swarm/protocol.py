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

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import msgpack
try:
    import bittensor as bt
except ImportError:
    bt = None


SCHEMA_VERSION = "5.0.0"


class FailureReason(str, Enum):
    NONE = "NONE"
    OBSTACLE_COLLISION = "OBSTACLE_COLLISION"
    NO_TOUCH_SPHERE = "NO_TOUCH_SPHERE"
    TILT = "TILT"
    TIMEOUT = "TIMEOUT"
    INFEASIBLE = "INFEASIBLE"
    SPAWN_FAILURE = "SPAWN_FAILURE"
    EVAL_ERROR = "EVAL_ERROR"
    INFRA = "INFRA"


_SUPPORTED_SCHEMA_VERSIONS: set[str] = {SCHEMA_VERSION}


def normalize_version(s) -> str:
    if not isinstance(s, str):
        s = str(s) if s is not None else ""
    if not s:
        return s
    return s[1:] if s[0] in ("V", "v") else s


def is_supported_schema(version) -> bool:
    return normalize_version(version) in _SUPPORTED_SCHEMA_VERSIONS


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
    family_id: str = "cf_search_and_rescue"
    version: str = SCHEMA_VERSION
    search_centre: Tuple[float, float] = (0.0, 0.0)

    def pack(self) -> bytes:
        return msgpack.packb(asdict(self), use_bin_type=True)

    @staticmethod
    def unpack(blob: bytes) -> "MapTask":
        data = msgpack.unpackb(blob, raw=False)
        data.pop("moving_platform", None)
        data.pop("search_radius", None)
        if not data.get("family_id"):
            version = normalize_version(data.get("version", ""))
            data["family_id"] = (
                "cf_search_and_rescue" if version.startswith("5.") else "cf_autopilot"
            )
        sc = data.get("search_centre")
        if isinstance(sc, (list, tuple)):
            data["search_centre"] = tuple(sc)
        return MapTask(**data)


@dataclass(slots=True)
class ValidationResult:
    uid: int
    success: bool
    time_sec: float
    score: float
    failure_reason: str = field(default="NONE", kw_only=True)
    metrics: Dict[str, Any] = field(default_factory=dict, kw_only=True)


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
Synapse = bt.Synapse if bt is not None else type("Synapse", (), {})  # alias


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
