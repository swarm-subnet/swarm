#!/usr/bin/env python3
"""Public benchmark engine facade.

The implementation is split across smaller modules under
``swarm.benchmark.engine_parts`` while preserving the historical
``swarm.benchmark.engine`` symbol surface used by the CLI and tests.
"""

from .engine_parts._shared import (
    BENCH_GROUP_ORDER,
    BENCH_GROUP_TO_TYPE,
    _BatchStat,
    _ProcessBatchRequest,
    _ProcessBatchResult,
    _ProcessSeedEvent,
    _ProcessWorkerHeartbeat,
)
from .engine_parts.config import (
    _RunOptions,
    _Tee,
    _active_runtime_overrides,
    _apply_relaxed_overrides,
    _build_progress_bar,
    _debug_profile_options,
    _parse_args,
    _resolve_run_options,
    _temporary_env,
    _ts,
)
from .engine_parts.dispatch import (
    _AdaptiveBackoffController,
    _PARENT_WORKER_HEARTBEAT_SEC,
    _PARENT_WORKER_STALL_TIMEOUT_SEC,
    _PRESSURE_POLL_INTERVAL_SEC,
    _build_worker_stall_seed_meta,
    _resource_class_for_group,
    _resource_cost_dict_for_group,
    _resource_model_rows,
    _is_backoff_infra_status,
    _is_backoff_timeout_status,
    _is_clean_execution_status,
    _max_heavy_active,
    _select_next_batch_index,
    _worker_cap_levels,
)
from .engine_parts.entry import main
from .engine_parts.reporting import _print_results
from .engine_parts.seeds import (
    _batch_indices,
    _find_seeds,
    _infer_bench_group,
    _infer_uid_from_model_path,
    _load_type_seeds,
    _normalize_type_seeds,
    _save_type_seeds,
)
from .engine_parts.workers import (
    _benchmark_mp_context,
    _benchmark_worker_main,
    _create_prepared_benchmark_evaluator,
    _run_benchmark,
    _run_benchmark_process_mode,
)

__all__ = [
    "BENCH_GROUP_ORDER",
    "BENCH_GROUP_TO_TYPE",
    "_AdaptiveBackoffController",
    "_BatchStat",
    "_PARENT_WORKER_HEARTBEAT_SEC",
    "_PARENT_WORKER_STALL_TIMEOUT_SEC",
    "_PRESSURE_POLL_INTERVAL_SEC",
    "_ProcessBatchRequest",
    "_ProcessBatchResult",
    "_ProcessSeedEvent",
    "_ProcessWorkerHeartbeat",
    "_RunOptions",
    "_Tee",
    "_active_runtime_overrides",
    "_apply_relaxed_overrides",
    "_batch_indices",
    "_benchmark_mp_context",
    "_benchmark_worker_main",
    "_build_progress_bar",
    "_build_worker_stall_seed_meta",
    "_create_prepared_benchmark_evaluator",
    "_debug_profile_options",
    "_find_seeds",
    "_resource_class_for_group",
    "_resource_cost_dict_for_group",
    "_resource_model_rows",
    "_is_backoff_infra_status",
    "_is_backoff_timeout_status",
    "_infer_bench_group",
    "_infer_uid_from_model_path",
    "_is_clean_execution_status",
    "_load_type_seeds",
    "_max_heavy_active",
    "_normalize_type_seeds",
    "_parse_args",
    "_print_results",
    "_resolve_run_options",
    "_run_benchmark",
    "_run_benchmark_process_mode",
    "_save_type_seeds",
    "_select_next_batch_index",
    "_temporary_env",
    "_ts",
    "_worker_cap_levels",
    "main",
]


if __name__ == "__main__":
    main()
