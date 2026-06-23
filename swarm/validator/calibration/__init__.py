"""Reference-time normalization for hardware-fair miner scoring."""

from .judge import StepVerdict, act_hard_cap_sec, judge_act
from .speed_factor import (
    CALIBRATION_STATE,
    CalibrationEntry,
    CalibrationState,
    SpeedFactor,
    baseline_model_available,
    baseline_model_path,
    load_baseline_manifest,
    normalize_speed_factor,
    percentile,
)

__all__ = [
    "CALIBRATION_STATE",
    "CalibrationEntry",
    "CalibrationState",
    "SpeedFactor",
    "StepVerdict",
    "act_hard_cap_sec",
    "baseline_model_available",
    "baseline_model_path",
    "judge_act",
    "load_baseline_manifest",
    "normalize_speed_factor",
    "percentile",
]
