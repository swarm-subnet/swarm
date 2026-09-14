# The MIT License (MIT)
# Copyright © 2026 Swarm

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""CLI arguments, run options, log teeing and progress bars for the full Docker evaluation."""

from __future__ import annotations

from swarm.challenge_families import DEFAULT_RUNTIME_FAMILY_ID
from swarm.constants import N_DOCKER_WORKERS

from ._shared import (
    Any,
    Dict,
    Optional,
    Path,
    _RunOptions,
    _tqdm,
    argparse,
    contextmanager,
    io,
    os,
    sys,
    threading,
    time,
)


class _Tee(io.TextIOBase):
    """Write to multiple file objects simultaneously."""

    def __init__(self, *files):
        """Keep the given streams, treat the first as primary, and guard writes with a lock."""
        self.files = files
        self._primary = files[0] if files else sys.__stdout__
        self._lock = threading.Lock()

    def write(self, data):
        """Send data to every stream still open, skipping any that raise, and report its length."""
        with self._lock:
            for f in self.files:
                try:
                    if getattr(f, "closed", False):
                        continue
                    f.write(data)
                    f.flush()
                except Exception:
                    continue
        return len(data)

    def flush(self):
        """Push every stream still open, skipping any that raise."""
        with self._lock:
            for f in self.files:
                try:
                    if getattr(f, "closed", False):
                        continue
                    f.flush()
                except Exception:
                    continue

    @property
    def buffer(self):
        """Return the primary stream's binary view, None when it does not expose one."""
        return getattr(self._primary, "buffer", None)

    @property
    def encoding(self):
        """Return the primary stream's text codec, utf-8 when it declares none."""
        return getattr(self._primary, "encoding", "utf-8")

    @property
    def errors(self):
        """Return the primary stream's decode error policy, strict when it declares none."""
        return getattr(self._primary, "errors", "strict")

    def reconfigure(self, *args, **kwargs):
        """Pass the call on to every stream that supports one, skipping any that raise."""
        for f in self.files:
            try:
                reconfigure = getattr(f, "reconfigure", None)
                if callable(reconfigure):
                    reconfigure(*args, **kwargs)
            except Exception:
                continue

    def fileno(self):
        """Return the primary stream's OS descriptor number."""
        return self._primary.fileno()

    def isatty(self):
        """Report whether the primary stream is a terminal."""
        return self._primary.isatty()

    def writable(self):
        """Return True; a tee always accepts output."""
        return True

    def __getattr__(self, name: str):
        """Delegate every unknown attribute to the primary stream."""
        return getattr(self._primary, name)


def _ts() -> str:
    """Return the local clock as HH:MM:SS for the prefix on every log line."""
    return time.strftime("%H:%M:%S")


class _NoopProgressBar:
    """Stands in for tqdm when it is not installed, swallowing every call the run makes."""

    def update(self, _n: int) -> None:
        """Ignore the step count; there is no bar to advance."""
        return None

    def set_postfix_str(self, _text: str, refresh: bool = True) -> None:
        """Ignore the trailing text; there is no bar to show it on."""
        _ = refresh
        return None

    def close(self) -> None:
        """Do nothing: there is no bar to tear down."""
        return None


def _build_progress_bar(total_seeds: int):
    """Return a tqdm bar counting seeds, or the silent stand-in when tqdm is unavailable."""
    if _tqdm is None:
        return _NoopProgressBar()
    return _tqdm(
        total=total_seeds,
        desc="Seed progress",
        unit="seed",
        dynamic_ncols=True,
        mininterval=0.5,
        leave=True,
    )


def _debug_profile_options() -> _RunOptions:
    """Return the _RunOptions of the debug profile: verbose RPC tracing, a 300 s batch cap and a 1800 s per-seed cap."""
    return _RunOptions(
        heartbeat_sec=15.0,
        rpc_trace=True,
        rpc_trace_every=250,
        rpc_heartbeat_sec=120.0,
        max_batch_timeout_sec=300.0,
        timeout_multiplier=1.0,
        extend_timeout_on_progress=True,
        timeout_extend_sec=30.0,
        timeout_progress_stale_sec=3.0,
        timeout_progress_min_sim_advance=0.02,
        max_seed_walltime_sec=1800.0,
        default_log_out=str(
            Path("/tmp") / f"bench_full_eval_{os.getuid()}_{os.getpid()}.log"
        ),
    )


@contextmanager
def _temporary_env(overrides: Dict[str, Optional[str]]):
    """Apply the given environment variables for the block, then put the earlier values back; a None value unsets."""
    previous = {k: os.environ.get(k) for k in overrides}
    try:
        for key, value in overrides.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Return the command line namespace for a full Docker evaluation: model, UID, family, seeds and output paths."""
    parser = argparse.ArgumentParser(
        description="Full Docker evaluation benchmark across all challenge types.",
    )
    parser.add_argument(
        "--model", type=Path, required=True,
        help="Path to miner submission zip (e.g. model/UID_178.zip).",
    )
    parser.add_argument(
        "--uid", type=int, default=None,
        help="Miner UID (default: inferred from --model filename, fallback 0).",
    )
    parser.add_argument(
        "--family-id",
        type=str,
        default=DEFAULT_RUNTIME_FAMILY_ID,
        help=(
            "Challenge family to benchmark "
            f"(default: {DEFAULT_RUNTIME_FAMILY_ID})."
        ),
    )
    parser.add_argument(
        "--seeds-per-group", type=int, default=3,
        help="Number of seeds per benchmark map group (default: 3).",
    )
    parser.add_argument(
        "--workers", type=int, default=N_DOCKER_WORKERS,
        help="Parallel Docker workers (default: one per CPU group; optionally capped by SWARM_MAX_DOCKER_WORKERS).",
    )
    parser.add_argument(
        "--log-out", type=Path, default=None,
        help="Path to write log file (default: a per-run /tmp/bench_full_eval_<uid>_<pid>.log file).",
    )
    parser.add_argument(
        "--relax-timeouts", action="store_true", default=False,
        help="Override timing constants for slow machines (longer timeouts, more strikes).",
    )
    parser.add_argument(
        "--rpc-verbosity",
        choices=["low", "mid", "high"],
        default="mid",
        help="RPC log verbosity level (default: mid).",
    )
    parser.add_argument(
        "--seed-file",
        type=Path,
        default=None,
        help="JSON file with exact benchmark seeds to reuse across runs.",
    )
    parser.add_argument(
        "--save-seed-file",
        type=Path,
        default=None,
        help="Write the resolved benchmark seed map to JSON.",
    )
    parser.add_argument(
        "--seed-search-rng",
        type=int,
        default=None,
        help="Seed used for reproducible seed discovery when --seed-file is not provided.",
    )
    parser.add_argument(
        "--summary-json-out",
        type=Path,
        default=None,
        help="Write benchmark summary JSON to this path.",
    )
    return parser.parse_args(argv)


def _resolve_run_options(args: argparse.Namespace) -> _RunOptions:
    """Return the debug profile with its RPC tracing tuned to the chosen verbosity level."""
    opts = _debug_profile_options()
    if args.rpc_verbosity == "low":
        opts.rpc_trace = False
        opts.rpc_trace_every = 1000
        opts.rpc_heartbeat_sec = 0.0
    elif args.rpc_verbosity == "mid":
        opts.rpc_trace = True
        opts.rpc_trace_every = 250
        opts.rpc_heartbeat_sec = 120.0
    else:
        opts.rpc_trace = True
        opts.rpc_trace_every = 25
        opts.rpc_heartbeat_sec = 30.0
    return opts


def _apply_relaxed_overrides() -> Dict[str, Any]:
    """Loosen the timing and worker constants in swarm.constants in place for slow machines, and return what was set."""
    import swarm.constants as _c

    overrides = {
        "MINER_COMPUTE_BUDGET_SEC": 3.0,
        "CALIBRATION_OVERHEAD_CAP_SEC": 1.0,
        "RPC_MAX_STRIKES_PER_SEED": 15,
        "RPC_STEP_TIMEOUT_SEC": 4.0,
        "DOCKER_WORKER_CPUS": "4",
        "GLOBAL_EVAL_BASE_SEC": 7200.0,
        "GLOBAL_EVAL_PER_SEED_SEC": 600.0,
        "GLOBAL_EVAL_CAP_SEC": 7200.0,
    }
    for attr, val in overrides.items():
        if hasattr(_c, attr):
            setattr(_c, attr, val)
    return overrides


def _active_runtime_overrides() -> Dict[str, str]:
    """Return the worker thread, CPU, memory and cpuset environment variables that currently carry a value."""
    keys = [
        "SWARM_DOCKER_THREAD_CAPS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
        "SWARM_TORCH_NUM_THREADS",
        "SWARM_TORCH_INTEROP_THREADS",
        "SWARM_DOCKER_WORKER_CPUS_OVERRIDE",
        "SWARM_DOCKER_WORKER_MEMORY_OVERRIDE",
        "SWARM_DOCKER_WORKER_CPUSETS",
        "SWARM_HOST_WORKER_MEMORY_MB",
        "SWARM_HOST_WORKER_CPUSETS",
    ]
    active: Dict[str, str] = {}
    for key in keys:
        value = os.getenv(key)
        if value not in (None, ""):
            active[key] = value
    for key, value in os.environ.items():
        if key.startswith("SWARM_DOCKER_WORKER_CPUSET_CPUS_") and value not in ("",):
            active[key] = value
        if key.startswith("SWARM_HOST_WORKER_CPUSET_CPUS_") and value not in ("",):
            active[key] = value
    return active
