"""Shared training utilities for the Swarm training pipeline."""

from .common import EpisodeSummary, ensure_dir, load_json, rollout_episode, save_json, seed_everything
from .dataset import MODE_TO_ID, list_episode_files, make_perception_labels, save_episode_dataset
from .experts import PrivilegedExpertConfig, PrivilegedExpertPolicy, load_expert_config, save_expert_config

__all__ = [
    "EpisodeSummary",
    "MODE_TO_ID",
    "PrivilegedExpertConfig",
    "PrivilegedExpertPolicy",
    "ensure_dir",
    "list_episode_files",
    "load_expert_config",
    "load_json",
    "make_perception_labels",
    "rollout_episode",
    "save_episode_dataset",
    "save_expert_config",
    "save_json",
    "seed_everything",
]

try:
    from .models import (
        AsymmetricCritic,
        StudentInferencePolicy,
        StudentModelConfig,
        StudentPolicy,
        load_checkpoint,
        load_model_config,
        save_checkpoint,
        save_model_config,
    )

    __all__.extend(
        [
            "AsymmetricCritic",
            "StudentInferencePolicy",
            "StudentModelConfig",
            "StudentPolicy",
            "load_checkpoint",
            "load_model_config",
            "save_checkpoint",
            "save_model_config",
        ]
    )
except ImportError:
    pass
