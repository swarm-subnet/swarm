import subprocess
import time

from swarm.constants import (
    CALIBRATION_BENCHMARK_REF_NS,
    CALIBRATION_CPU_FACTOR_CAP,
    CALIBRATION_OVERHEAD_CAP_SEC,
    CALIBRATION_ROUNDS,
    CALIBRATION_TIMEOUT_SEC,
    MINER_COMPUTE_BUDGET_SEC,
    RPC_STEP_TIMEOUT_SEC,
)

from .docker_evaluator_parts import (
    batch,
    lifecycle,
    networking,
    parallel,
    rpc,
    submission,
)
from .docker_evaluator_parts._shared import (
    _HEAVY_CHALLENGE_TYPES,
    _cleanup_env_quietly,
    _heavy_aware_chunk,
    _run_multi_seed_rpc_sync_isolated_payload,
    _submission_template_dir,
)


class DockerSecureEvaluator:
    """Docker-based secure model evaluation."""

    _instance = None
    _base_ready = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


DockerSecureEvaluator.__init__ = lifecycle.__init__
DockerSecureEvaluator._check_docker_available = lifecycle._check_docker_available
DockerSecureEvaluator._env_truthy = staticmethod(lifecycle._env_truthy)
DockerSecureEvaluator._docker_env_overrides = classmethod(
    lifecycle._docker_env_overrides
)
DockerSecureEvaluator._split_worker_cpuset_map = staticmethod(
    lifecycle._split_worker_cpuset_map
)
DockerSecureEvaluator._resolve_worker_limits = classmethod(
    lifecycle._resolve_worker_limits
)
DockerSecureEvaluator._calculate_docker_hash = lifecycle._calculate_docker_hash
DockerSecureEvaluator._get_image_hash_label = lifecycle._get_image_hash_label
DockerSecureEvaluator._should_rebuild_base_image = lifecycle._should_rebuild_base_image
DockerSecureEvaluator._setup_base_container = lifecycle._setup_base_container

DockerSecureEvaluator._find_free_port = networking._find_free_port
DockerSecureEvaluator._check_rpc_ready = networking._check_rpc_ready
DockerSecureEvaluator._get_docker_host_ip = networking._get_docker_host_ip
DockerSecureEvaluator._get_container_pid = networking._get_container_pid
DockerSecureEvaluator._apply_network_lockdown = networking._apply_network_lockdown

DockerSecureEvaluator._normalize_package_name = staticmethod(
    submission._normalize_package_name
)
DockerSecureEvaluator._validate_requirements = submission._validate_requirements
DockerSecureEvaluator._serialize_observation = staticmethod(
    submission._serialize_observation
)

DockerSecureEvaluator._run_multi_seed_rpc_sync = rpc._run_multi_seed_rpc_sync
DockerSecureEvaluator._calibrate_rpc_overhead_async = rpc._calibrate_rpc_overhead_async
DockerSecureEvaluator.evaluate_seeds_batch = batch.evaluate_seeds_batch
DockerSecureEvaluator.evaluate_seeds_parallel = parallel.evaluate_seeds_parallel
DockerSecureEvaluator.cleanup = batch.cleanup

__all__ = [
    "DockerSecureEvaluator",
    "CALIBRATION_BENCHMARK_REF_NS",
    "CALIBRATION_CPU_FACTOR_CAP",
    "CALIBRATION_OVERHEAD_CAP_SEC",
    "CALIBRATION_ROUNDS",
    "CALIBRATION_TIMEOUT_SEC",
    "MINER_COMPUTE_BUDGET_SEC",
    "RPC_STEP_TIMEOUT_SEC",
    "_HEAVY_CHALLENGE_TYPES",
    "_cleanup_env_quietly",
    "_heavy_aware_chunk",
    "_run_multi_seed_rpc_sync_isolated_payload",
    "_submission_template_dir",
    "batch",
    "lifecycle",
    "networking",
    "parallel",
    "rpc",
    "submission",
    "subprocess",
    "time",
]
