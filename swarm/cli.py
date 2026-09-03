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

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from swarm.constants import N_DOCKER_WORKERS
from swarm.domain_model import (
    CHALLENGE_FAMILY_IDS,
    get_challenge_family_definition,
)
from swarm.core.submission_policy import validate_submission_zip
from swarm.policy_interface import (
    POLICY_CONTRACT_FILENAME,
    PolicyInterfaceError,
    render_artifact_policy_contract,
    resolve_policy_interface_version,
    smoke_test_policy_package,
    verify_policy_package_contract,
)

# The submission module pulls in bittensor, which parses sys.argv at import unless told not to;
# the swarm CLI owns its arguments, so the flag is set for that import and then put back.
_BT_PARSE_FLAG = os.environ.get("BT_NO_PARSE_CLI_ARGS")
os.environ["BT_NO_PARSE_CLI_ARGS"] = "true"

from miner.src.miner import DEFAULT_BACKEND_URL, submit_private  # noqa: E402

if _BT_PARSE_FLAG is None:
    del os.environ["BT_NO_PARSE_CLI_ARGS"]
else:
    os.environ["BT_NO_PARSE_CLI_ARGS"] = _BT_PARSE_FLAG

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BENCH_LOG = Path("/tmp/bench_full_eval.log")
DEFAULT_MODEL_ZIP = REPO_ROOT / "Submission" / "submission.zip"

MODEL_EXTENSIONS = {
    ".bin",
    ".ckpt",
    ".h5",
    ".json",
    ".npy",
    ".npz",
    ".onnx",
    ".pb",
    ".pkl",
    ".pt",
    ".pth",
    ".safetensors",
    ".tflite",
    ".weights",
    ".zip",
}

REQUIRED_TEMPLATE_FILES = {
    "main.py",
    "agent.capnp",
    "agent_server.py",
    "runtime_caps.py",
}

REPORT_FIELD_PATTERNS = {
    "seeds_evaluated": re.compile(r"Seeds evaluated:\s+(\d+)"),
    "success_rate_pct": re.compile(r"Success rate:\s+\d+/\d+\s+\(([\d.]+)%\)"),
    "clean_execution_rate_pct": re.compile(
        r"Clean execution rate:\s+\d+/\d+\s+\(([\d.]+)%\)"
    ),
    "total_wall_clock_sec": re.compile(r"Total wall-clock:\s+([\d.]+)s"),
    "avg_wall_per_seed_sec": re.compile(r"Avg wall / seed:\s+([\d.]+)s"),
    "median_wall_per_seed_sec": re.compile(r"Median wall / seed:\s+([\d.]+)s"),
    "p90_wall_per_seed_sec": re.compile(r"P90 wall / seed:\s+([\d.]+)s"),
    "avg_sim_time_per_seed_sec": re.compile(r"Avg sim time / seed:\s+([\d.]+)s"),
    "total_seed_worker_time_sec": re.compile(r"Total seed-worker time:\s+([\d.]+)s"),
    "throughput_seeds_per_min": re.compile(r"Throughput:\s+([\d.]+)\s+seeds/min"),
    "throughput_per_worker": re.compile(
        r"Throughput per worker:\s+([\d.]+)\s+seeds/min/worker"
    ),
    "effective_parallelism": re.compile(r"Effective parallelism:\s+([\d.]+)x"),
    "worker_utilization_pct": re.compile(r"utilization\s+([\d.]+)%\s+of"),
    "batches_run": re.compile(r"Batches run:\s+(\d+)"),
    "avg_seeds_per_container": re.compile(r"Avg seeds / container:\s+([\d.]+)"),
    "total_startup_overhead_sec": re.compile(r"Total startup overhead:\s+([\d.]+)s"),
    "avg_startup_per_container_sec": re.compile(r"Avg startup / container:\s+([\d.]+)s"),
    "workers_used": re.compile(r"Workers used:\s+(\d+)"),
    "estimated_wall_clock_sec_1000": re.compile(r"Estimated wall-clock:\s+([\d.]+)s"),
    "estimated_avg_wall_per_seed_sec_1000": re.compile(
        r"Estimated avg wall / seed:\s+([\d.]+)s"
    ),
    "estimated_throughput_1000": re.compile(
        r"Estimated throughput:\s+([\d.]+)\s+seeds/min"
    ),
}

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
BENCH_GROUP_ORDER = [
    "type1_city",
    "type2_open",
    "type3_mountain",
    "type4_village",
    "type5_warehouse",
    "type6_forest",
    "type7_office",
]
BENCH_GROUP_TO_TYPE = {
    "type1_city": 1,
    "type2_open": 2,
    "type3_mountain": 3,
    "type4_village": 4,
    "type5_warehouse": 5,
    "type6_forest": 6,
    "type7_office": 7,
}
TYPE_LABELS = {
    1: "city",
    2: "open",
    3: "mountain",
    4: "village",
    5: "warehouse",
    6: "forest",
    7: "office",
}


@dataclass
class DoctorCheck:
    name: str
    ok: bool
    detail: str
    required: bool = True


@dataclass(frozen=True)
class VisualizeTarget:
    challenge_type: int
    seed: Optional[int] = None
    family_id: str = "cf_autopilot"
    note: Optional[str] = None


@dataclass(frozen=True)
class PackagedModelArtifact:
    family_id: str
    interface_version: str
    output_zip: Path
    sha256: str
    packaged_files_count: int


def _check_module_available(module_name: str) -> DoctorCheck:
    spec = importlib.util.find_spec(module_name)
    return DoctorCheck(
        name=f"module:{module_name}",
        ok=spec is not None,
        detail="available" if spec is not None else "missing",
        required=True,
    )


def _check_python_version() -> DoctorCheck:
    ok = sys.version_info >= (3, 11)
    return DoctorCheck(
        name="python",
        ok=ok,
        detail=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        required=True,
    )


def _check_docker_binary() -> DoctorCheck:
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return DoctorCheck("docker_binary", True, result.stdout.strip(), True)
        return DoctorCheck("docker_binary", False, result.stderr.strip() or "not found", True)
    except FileNotFoundError:
        return DoctorCheck("docker_binary", False, "docker command not found", True)


def _check_binary_available(binary_name: str, *, required: bool = True) -> DoctorCheck:
    path = shutil.which(binary_name)
    return DoctorCheck(
        name=f"binary:{binary_name}",
        ok=path is not None,
        detail=path if path is not None else "not found on PATH",
        required=required,
    )


def _binary_capabilities(path: str) -> set[str]:
    getcap = shutil.which("getcap")
    if getcap is None:
        return set()
    try:
        result = subprocess.run(
            [getcap, path],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return set()
    if result.returncode != 0 or not result.stdout.strip():
        return set()
    _, _, caps_blob = result.stdout.partition(" ")
    caps: set[str] = set()
    for token in caps_blob.replace("=", ",").split(","):
        token = token.strip()
        if token.startswith("cap_"):
            caps.add(token)
    return caps


def _check_sandbox_lockdown_permissions() -> DoctorCheck:
    nsenter_path = shutil.which("nsenter")
    iptables_path = shutil.which("iptables")
    if nsenter_path is None or iptables_path is None:
        missing = []
        if nsenter_path is None:
            missing.append("nsenter")
        if iptables_path is None:
            missing.append("iptables")
        return DoctorCheck(
            "sandbox_lockdown_permissions",
            False,
            f"cannot assess without required binaries: {', '.join(missing)}",
            False,
        )

    if hasattr(os, "geteuid") and os.geteuid() == 0:
        return DoctorCheck(
            "sandbox_lockdown_permissions",
            True,
            "running as root; network lockdown should be permitted",
            False,
        )

    resolved_iptables = os.path.realpath(iptables_path)
    nsenter_caps = _binary_capabilities(nsenter_path)
    iptables_caps = _binary_capabilities(resolved_iptables)
    if "cap_sys_admin" in nsenter_caps and "cap_net_admin" in iptables_caps:
        return DoctorCheck(
            "sandbox_lockdown_permissions",
            True,
            (
                "binary capabilities detected "
                f"(nsenter={nsenter_path}, iptables={resolved_iptables})"
            ),
            False,
        )

    detail = (
        "current user is not root and sandbox network lockdown may fail; "
        f"run with sudo -E or grant cap_sys_admin to {nsenter_path} and "
        f"cap_net_admin to {resolved_iptables}"
    )
    if shutil.which("getcap") is None:
        detail += " (getcap unavailable, binary capabilities could not be inspected)"
    return DoctorCheck("sandbox_lockdown_permissions", False, detail, False)


def _check_docker_daemon() -> DoctorCheck:
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if result.returncode == 0:
            return DoctorCheck("docker_daemon", True, "reachable", True)
        return DoctorCheck(
            "docker_daemon",
            False,
            result.stderr.strip() or result.stdout.strip() or "unreachable",
            True,
        )
    except FileNotFoundError:
        return DoctorCheck("docker_daemon", False, "docker command not found", True)
    except subprocess.TimeoutExpired:
        return DoctorCheck("docker_daemon", False, "timeout while contacting daemon", True)


def _check_writable_dir(path: Path, name: str) -> DoctorCheck:
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path, delete=True):
            pass
        return DoctorCheck(name, True, str(path), True)
    except Exception as exc:  # pragma: no cover - depends on host FS perms.
        return DoctorCheck(name, False, f"{path}: {exc}", True)


def _check_submission_template() -> DoctorCheck:
    template_dir = REPO_ROOT / "swarm" / "submission_template"
    missing = [f for f in sorted(REQUIRED_TEMPLATE_FILES) if not (template_dir / f).exists()]
    if missing:
        return DoctorCheck(
            "submission_template",
            False,
            f"missing files: {', '.join(missing)}",
            True,
        )
    return DoctorCheck("submission_template", True, str(template_dir), True)


def _check_benchmark_engine() -> DoctorCheck:
    spec = importlib.util.find_spec("swarm.benchmark.engine")
    if spec is not None:
        return DoctorCheck("benchmark_engine", True, "swarm.benchmark.engine", True)
    return DoctorCheck("benchmark_engine", False, "swarm.benchmark.engine not found", True)


def _check_env_var(name: str, required: bool = False) -> DoctorCheck:
    value = os.getenv(name)
    if value:
        return DoctorCheck(name, True, "set", required)
    return DoctorCheck(name, False, "not set", required)


def _runtime_state_dir() -> Path:
    return REPO_ROOT / "swarm" / "state"


def _run_doctor_checks() -> list[DoctorCheck]:
    from swarm.constants import MODEL_DIR

    return [
        _check_python_version(),
        _check_docker_binary(),
        _check_docker_daemon(),
        _check_binary_available("nsenter"),
        _check_binary_available("iptables"),
        _check_sandbox_lockdown_permissions(),
        _check_module_available("capnp"),
        _check_module_available("pybullet"),
        _check_module_available("gym_pybullet_drones"),
        _check_writable_dir(_runtime_state_dir(), "state_dir"),
        _check_writable_dir(Path(MODEL_DIR), "model_dir"),
        _check_submission_template(),
        _check_benchmark_engine(),
    ]


def _print_doctor_text(checks: list[DoctorCheck]) -> None:
    print("Swarm Doctor")
    for check in checks:
        status = "OK" if check.ok else "FAIL"
        req = "required" if check.required else "optional"
        print(f"- {status:4} [{req}] {check.name}: {check.detail}")


def _cmd_doctor(args: argparse.Namespace) -> int:
    checks = _run_doctor_checks()
    _print_doctor_text(checks)
    failed_required = any((not c.ok) and c.required for c in checks)
    return 1 if failed_required else 0


def _build_benchmark_argv(args: argparse.Namespace) -> list[str]:
    argv = ["--model", str(args.model)]
    if args.uid is not None:
        argv.extend(["--uid", str(args.uid)])
    if args.family_id is not None:
        argv.extend(["--family-id", str(args.family_id)])
    argv.extend(["--seeds-per-group", str(args.seeds_per_group)])
    argv.extend(["--workers", str(args.workers)])
    if args.log_out is not None:
        argv.extend(["--log-out", str(args.log_out)])
    if args.seed_file is not None:
        argv.extend(["--seed-file", str(args.seed_file)])
    if args.save_seed_file is not None:
        argv.extend(["--save-seed-file", str(args.save_seed_file)])
    if args.seed_search_rng is not None:
        argv.extend(["--seed-search-rng", str(args.seed_search_rng)])
    if args.summary_json_out is not None:
        argv.extend(["--summary-json-out", str(args.summary_json_out)])
    if args.relax_timeouts:
        argv.append("--relax-timeouts")
    argv.extend(["--rpc-verbosity", str(args.rpc_verbosity)])
    return argv


def _champion_zip_name(uid: int, family_id: Optional[str]) -> str:
    if family_id:
        return f"champion_{family_id}_UID_{uid}.zip"
    return f"champion_UID_{uid}.zip"


def _download_champion_model(family_id: Optional[str] = None) -> Optional[Path]:
    import httpx

    base_url = os.environ.get("SWARM_BACKEND_API_URL", "https://api.swarm124.com").rstrip("/")
    params = {"family_id": family_id} if family_id else {}
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(f"{base_url}/champion", params=params)
            if resp.status_code != 200:
                print("No champion model available to download.", file=sys.stderr)
                return None
            champ = resp.json()
            if not champ.get("is_released"):
                print(f"Champion UID {champ['uid']} is not released for download yet.", file=sys.stderr)
                return None

            uid = champ["uid"]
            expected_hash = champ.get("model_hash")
            output = Path(_champion_zip_name(uid, family_id))

            if output.exists() and expected_hash:
                existing_hash = hashlib.sha256(output.read_bytes()).hexdigest()
                if existing_hash == expected_hash:
                    print(f"Using cached champion: {output}")
                    return output

            print(f"Downloading champion UID {uid} (score: {champ.get('benchmark_score', 0):.4f})...")
            dl = client.get(f"{base_url}/models/{uid}/download", params=params)
            if dl.status_code != 200:
                print(f"Download failed: HTTP {dl.status_code}", file=sys.stderr)
                return None

            if expected_hash:
                dl_hash = hashlib.sha256(dl.content).hexdigest()
                if dl_hash != expected_hash:
                    print("Download integrity check failed.", file=sys.stderr)
                    return None

            output.write_bytes(dl.content)
            print(f"Saved: {output} ({len(dl.content) / (1024*1024):.1f} MB)")
            return output
    except Exception as exc:
        print(f"Failed to download champion: {exc}", file=sys.stderr)
        return None


def _cmd_benchmark(args: argparse.Namespace) -> int:
    if args.model is None:
        downloaded = _download_champion_model(args.family_id)
        if downloaded is None:
            print("No --model specified and champion download failed.", file=sys.stderr)
            return 1
        args.model = downloaded

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}", file=sys.stderr)
        return 1

    from swarm.benchmark.engine import main as benchmark_main

    argv = _build_benchmark_argv(args)
    try:
        benchmark_main(argv)
        return 0
    except (SystemExit, KeyboardInterrupt):
        return 1
    except Exception as exc:
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        return 1


def _group_label(group_name: str) -> str:
    challenge_type = BENCH_GROUP_TO_TYPE.get(str(group_name))
    if challenge_type is None:
        return str(group_name)
    return TYPE_LABELS.get(int(challenge_type), str(group_name))


def _load_visualize_summary_groups(summary_json: Path) -> dict[str, list[dict[str, Any]]]:
    payload = json.loads(Path(summary_json).read_text())
    raw_groups = payload.get("group_results")
    if not isinstance(raw_groups, dict):
        raise ValueError("Summary JSON missing group_results.")

    normalized: dict[str, list[dict[str, Any]]] = {}
    for group_name in BENCH_GROUP_ORDER:
        rows = raw_groups.get(group_name, [])
        if not isinstance(rows, list):
            raise ValueError(f"Summary JSON group_results[{group_name}] must be a list.")
        normalized[group_name] = []
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"Summary JSON row for {group_name} is not an object.")
            normalized[group_name].append(dict(row))
    return normalized


def _load_visualize_seed_groups(
    seed_file: Path, family_id: str = "cf_autopilot"
) -> dict[str, list[int]]:
    """Read a seed file written by ``swarm benchmark --save-seed-file``.

    Uses the benchmark's own reader so the envelope it writes, the family the seeds
    belong to, and the groups that family actually runs all stay in one definition.
    """
    from swarm.benchmark.engine_parts.seeds import _load_type_seeds

    return dict(_load_type_seeds(Path(seed_file), family_id=family_id))


def _lookup_seed_type_in_summary(summary_json: Path, seed: int) -> int:
    groups = _load_visualize_summary_groups(summary_json)
    matches = [
        BENCH_GROUP_TO_TYPE[group_name]
        for group_name in BENCH_GROUP_ORDER
        for row in groups[group_name]
        if int(row.get("seed", -1)) == int(seed)
    ]
    unique = sorted(set(int(match) for match in matches))
    if not unique:
        raise ValueError(f"Seed {seed} was not found in summary JSON: {summary_json}")
    if len(unique) > 1:
        raise ValueError(
            f"Seed {seed} appeared under multiple challenge types in summary JSON: {summary_json}"
        )
    return unique[0]


def _lookup_seed_type_in_seed_file(
    seed_file: Path, seed: int, family_id: str = "cf_autopilot"
) -> int:
    groups = _load_visualize_seed_groups(seed_file, family_id=family_id)
    matches = [
        BENCH_GROUP_TO_TYPE[group_name]
        for group_name in groups
        if int(seed) in groups[group_name]
    ]
    unique = sorted(set(int(match) for match in matches))
    if not unique:
        raise ValueError(f"Seed {seed} was not found in seed file: {seed_file}")
    if len(unique) > 1:
        raise ValueError(
            f"Seed {seed} appeared under multiple challenge types in seed file: {seed_file}"
        )
    return unique[0]


def _infer_benchmark_type_from_seed(seed: int, family_id: str = "cf_autopilot") -> int:
    from swarm.constants import SIM_DT
    from swarm.validator.task_gen import random_task

    task = random_task(sim_dt=SIM_DT, seed=int(seed), family_id=family_id)
    return int(task.challenge_type)


def _load_failed_visualize_rows(summary_json: Path) -> list[dict[str, Any]]:
    groups = _load_visualize_summary_groups(summary_json)
    failed_rows: list[dict[str, Any]] = []
    for group_name in BENCH_GROUP_ORDER:
        challenge_type = BENCH_GROUP_TO_TYPE[group_name]
        for row in groups[group_name]:
            if bool(row.get("success", False)):
                continue
            failed_rows.append(
                {
                    "group": group_name,
                    "challenge_type": int(challenge_type),
                    "seed": int(row["seed"]),
                    "score": float(row.get("score", 0.0)),
                    "sim_time": float(row.get("sim_time", 0.0)),
                    "execution_status": str(row.get("execution_status", "unknown")),
                }
            )
    return failed_rows


def _print_failed_visualize_rows(summary_json: Path, failed_rows: Sequence[dict[str, Any]]) -> None:
    print(f"Failed benchmark seeds from {summary_json}:")
    if not failed_rows:
        print("  none")
        return

    for idx, row in enumerate(failed_rows, start=1):
        label = _group_label(str(row["group"]))
        print(
            f"  {idx:>2}. seed {int(row['seed'])}  "
            f"type {int(row['challenge_type'])} ({label})  "
            f"score={float(row['score']):.4f}  sim={float(row['sim_time']):.2f}s  "
            f"status={row['execution_status']}"
        )
    print()
    print("Re-run with `swarm visualize --summary-json <path> --failed-index N` to inspect one.")


def _resolve_visualize_target(args: argparse.Namespace) -> Optional[VisualizeTarget]:
    failed_mode = bool(args.failed or args.failed_index is not None)

    if failed_mode:
        if args.summary_json is None:
            raise ValueError("`--failed` and `--failed-index` require `--summary-json`.")
        if args.seed is not None or args.type is not None or args.seed_file is not None:
            raise ValueError(
                "`--failed` and `--failed-index` cannot be combined with `--seed`, `--type`, or `--seed-file`."
            )
        failed_rows = _load_failed_visualize_rows(Path(args.summary_json))
        if args.failed_index is None:
            _print_failed_visualize_rows(Path(args.summary_json), failed_rows)
            return None

        failed_index = int(args.failed_index)
        if failed_index <= 0:
            raise ValueError("`--failed-index` must be a positive 1-based index.")
        if failed_index > len(failed_rows):
            raise ValueError(
                f"`--failed-index` {failed_index} is out of range for {len(failed_rows)} failed seeds."
            )
        row = failed_rows[failed_index - 1]
        return VisualizeTarget(
            challenge_type=int(row["challenge_type"]),
            seed=int(row["seed"]),
            family_id=args.family_id,
            note=(
                f"Reviewing failed seed {int(row['seed'])} as "
                f"type {int(row['challenge_type'])} ({_group_label(str(row['group']))}) "
                f"from {args.summary_json}."
            ),
        )

    if args.seed is None:
        if args.type is None:
            raise ValueError(
                "Provide `--type`, or provide `--seed` so the type can be inferred, "
                "or use `--summary-json --failed`."
            )
        return VisualizeTarget(challenge_type=int(args.type), seed=None, family_id=args.family_id)

    inferred_type: int | None = None
    inferred_note: str | None = None
    if args.summary_json is not None:
        inferred_type = _lookup_seed_type_in_summary(Path(args.summary_json), int(args.seed))
        inferred_note = (
            f"Resolved seed {int(args.seed)} to type {inferred_type} "
            f"({TYPE_LABELS.get(inferred_type, 'unknown')}) from {args.summary_json}."
        )
    elif args.seed_file is not None:
        inferred_type = _lookup_seed_type_in_seed_file(
            Path(args.seed_file), int(args.seed), family_id=args.family_id
        )
        inferred_note = (
            f"Resolved seed {int(args.seed)} to type {inferred_type} "
            f"({TYPE_LABELS.get(inferred_type, 'unknown')}) from {args.seed_file}."
        )
    elif args.type is None:
        inferred_type = _infer_benchmark_type_from_seed(int(args.seed), family_id=args.family_id)
        inferred_note = (
            f"Resolved seed {int(args.seed)} to benchmark type {inferred_type} "
            f"({TYPE_LABELS.get(inferred_type, 'unknown')})."
        )

    if args.type is not None:
        if inferred_type is not None and int(args.type) != int(inferred_type):
            raise ValueError(
                f"Explicit `--type {int(args.type)}` does not match the inferred type "
                f"{int(inferred_type)} for seed {int(args.seed)}."
            )
        return VisualizeTarget(challenge_type=int(args.type), seed=int(args.seed), family_id=args.family_id)

    if inferred_type is None:
        raise ValueError("Could not resolve a challenge type for visualization.")

    return VisualizeTarget(
        challenge_type=int(inferred_type),
        seed=int(args.seed),
        family_id=args.family_id,
        note=inferred_note,
    )


def _build_visualize_argv(args: argparse.Namespace, target: VisualizeTarget) -> list[str]:
    argv = ["--type", str(target.challenge_type), "--family-id", str(target.family_id)]
    if target.seed is not None:
        argv.extend(["--seed", str(target.seed)])
    argv.extend(["--speed", str(args.speed)])
    argv.extend(["--boost", str(args.boost)])
    argv.extend(["--camera", str(args.camera)])
    argv.extend(["--width", str(args.width)])
    argv.extend(["--height", str(args.height)])
    if args.render_scale is not None:
        argv.extend(["--render-scale", str(args.render_scale)])
    if args.render_distance is not None:
        argv.extend(["--render-distance", str(args.render_distance)])
    if args.render_fps is not None:
        argv.extend(["--render-fps", str(args.render_fps)])
    if args.sim_fps is not None:
        argv.extend(["--sim-fps", str(args.sim_fps)])
    if args.gpu:
        argv.append("--gpu")
    return argv


def _cmd_visualize(args: argparse.Namespace) -> int:
    try:
        target = _resolve_visualize_target(args)
        if target is None:
            return 0
        if target.note:
            print(target.note)
        from validator.scripts.visualize_map import main as visualize_main

        visualize_main(_build_visualize_argv(args, target))
        return 0
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except (SystemExit, KeyboardInterrupt):
        return 1
    except Exception as exc:
        print(f"Visualizer failed: {exc}", file=sys.stderr)
        return 1


def _build_video_argv(args: argparse.Namespace) -> list[str]:
    argv = ["--model", str(args.model), "--family-id", str(args.family_id)]
    if args.seed_file is not None:
        argv.extend(["--seed-file", str(args.seed_file)])
    else:
        argv.extend(["--seed", str(args.seed)])
        argv.extend(["--type", str(args.type)])
    argv.extend(["--mode", str(args.mode)])
    argv.extend(["--backend", str(args.backend)])
    argv.extend(["--width", str(args.width)])
    argv.extend(["--height", str(args.height)])
    argv.extend(["--fps", str(args.fps)])
    if args.out is not None:
        argv.extend(["--out", str(args.out)])
    if args.summary_json is not None:
        argv.extend(["--summary-json", str(args.summary_json)])
    if args.skip_existing:
        argv.append("--skip-existing")
    if args.progress_file is not None:
        argv.extend(["--progress-file", str(args.progress_file)])
    argv.extend(["--chase-back", str(args.chase_back)])
    argv.extend(["--chase-up", str(args.chase_up)])
    argv.extend(["--chase-fov", str(args.chase_fov)])
    argv.extend(["--fpv-fov", str(args.fpv_fov)])
    argv.extend(["--overview-fov", str(args.overview_fov)])
    if getattr(args, "save_actions", None) is not None:
        argv.extend(["--save-actions", str(args.save_actions)])
    if getattr(args, "replay_actions", None) is not None:
        argv.extend(["--replay-actions", str(args.replay_actions)])
    return argv


def _cmd_video(args: argparse.Namespace) -> int:
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}", file=sys.stderr)
        return 1
    if args.seed_file is None and (args.seed is None or args.type is None):
        print("Provide either --seed-file, or both --seed and --type.", file=sys.stderr)
        return 1
    if args.seed_file is not None and (args.seed is not None or args.type is not None):
        print(
            "--seed-file renders every seed in the file; drop --seed/--type, or drop "
            "--seed-file to render just one.",
            file=sys.stderr,
        )
        return 1
    try:
        from validator.scripts.generate_video import main as video_main

        video_main(_build_video_argv(args))
        return 0
    except (SystemExit, KeyboardInterrupt) as exc:
        return int(exc.code) if isinstance(exc, SystemExit) and isinstance(exc.code, int) else 1
    except Exception as exc:
        print(f"Video generation failed: {exc}", file=sys.stderr)
        return 1


def _collect_packable_files(source_dir: Path) -> list[Path]:
    allowed_names = {"drone_agent.py", "requirements.txt"}
    files: list[Path] = []
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts:
            continue
        if path.name in allowed_names or path.suffix.lower() in MODEL_EXTENSIONS:
            files.append(path)
    return files


def _sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _package_model_artifact(
    *,
    source_dir: Path,
    output_zip: Path,
    family_id: str,
    interface_version: str | None,
    overwrite: bool,
) -> PackagedModelArtifact:
    if not source_dir.is_dir():
        raise ValueError(f"Source directory not found: {source_dir}")
    drone_agent = source_dir / "drone_agent.py"
    if not drone_agent.is_file():
        raise ValueError("Source must contain drone_agent.py")
    if output_zip.exists() and not overwrite:
        raise ValueError(
            f"Output already exists: {output_zip} (use --overwrite to replace)"
        )

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    files_to_pack = _collect_packable_files(source_dir)
    if drone_agent not in files_to_pack:
        files_to_pack.append(drone_agent)
        files_to_pack.sort()
    try:
        interface_version = resolve_policy_interface_version(
            family_id,
            interface_version,
        )
        policy_contract_json = render_artifact_policy_contract(
            family_id,
            interface_version,
        )
    except PolicyInterfaceError as exc:
        raise ValueError(str(exc)) from exc

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in files_to_pack:
            zf.write(file_path, arcname=str(file_path.relative_to(source_dir)))
        zf.writestr(POLICY_CONTRACT_FILENAME, policy_contract_json)

    accepted, detail = validate_submission_zip(output_zip)
    if not accepted:
        output_zip.unlink(missing_ok=True)
        raise ValueError(detail)

    return PackagedModelArtifact(
        family_id=family_id,
        interface_version=interface_version,
        output_zip=output_zip,
        sha256=_sha256sum(output_zip),
        packaged_files_count=len(files_to_pack) + 1,
    )


def _family_display_name(family_id: str) -> str:
    try:
        return str(get_challenge_family_definition(family_id)["name"])
    except Exception:
        return family_id


def _prompt_family_id() -> Optional[str]:
    """Ask the miner which challenge family the artifact targets. Returns the
    chosen family id, or None if the prompt was cancelled."""
    families = sorted(CHALLENGE_FAMILY_IDS)
    names = [_family_display_name(fid) for fid in families]
    width = max(len(name) for name in names)

    print("\nWhich challenge family did you train for?\n")
    for i, (fid, name) in enumerate(zip(families, names), start=1):
        print(f"  {i}) {name.ljust(width)}  ({fid})")
    print()

    while True:
        try:
            choice = input(f"Select [1-{len(families)}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.", file=sys.stderr)
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(families):
            return families[int(choice) - 1]
        print(f"Please enter a number between 1 and {len(families)}.", file=sys.stderr)


def _resolve_family_id(args: argparse.Namespace) -> Optional[str]:
    """The family from the flag, or from a menu when there is a terminal to ask."""
    if args.family_id is not None:
        return str(args.family_id)
    if sys.stdin.isatty() and sys.stdout.isatty():
        return _prompt_family_id()
    print(
        "--family-id is required when not run interactively (no terminal to prompt).",
        file=sys.stderr,
    )
    return None


def _cmd_model_package(args: argparse.Namespace) -> int:
    family_id = _resolve_family_id(args)
    if family_id is None:
        return 1

    source_dir = Path(args.source)
    output_zip = Path(args.output)
    print(f"\nPackaging for {_family_display_name(family_id)} ({family_id})...")
    try:
        packaged = _package_model_artifact(
            source_dir=source_dir,
            output_zip=output_zip,
            family_id=str(family_id),
            interface_version=args.interface_version,
            overwrite=bool(args.overwrite),
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Created package: {packaged.output_zip}")
    print(f"Files included: {packaged.packaged_files_count}")
    print(f"Policy family: {packaged.family_id}")
    print(f"Policy interface: {packaged.interface_version}")
    return 0


def _verify_model_zip(model_path: Path, max_uncompressed_mb: float) -> dict[str, Any]:
    """Every local check a submission must pass, as one report; ``compliant`` sums it up."""
    from swarm.constants import MAX_MODEL_BYTES
    from swarm.core.model_verify import (
        classify_model_validity,
        inspect_model_structure,
        zip_is_safe,
    )

    size_bytes = model_path.stat().st_size
    max_uncompressed = int(max_uncompressed_mb * 1024 * 1024)
    size_ok = size_bytes <= MAX_MODEL_BYTES
    zip_safe = zip_is_safe(model_path, max_uncompressed=max_uncompressed)
    inspection = inspect_model_structure(model_path)
    status, reason = classify_model_validity(inspection)
    contract_ok, contract_reason, contract = verify_policy_package_contract(model_path)
    smoke_ok = False
    smoke_reason = "skipped_due_to_contract_failure"
    if contract_ok:
        smoke_ok, smoke_reason = smoke_test_policy_package(model_path)
    compliant = bool(
        size_ok
        and zip_safe
        and status == "legitimate"
        and contract_ok
        and smoke_ok
    )

    return {
        "model": str(model_path),
        "compliant": compliant,
        "size_bytes": size_bytes,
        "size_limit_bytes": MAX_MODEL_BYTES,
        "size_ok": size_ok,
        "zip_safe": zip_safe,
        "status": status,
        "reason": reason,
        "policy_contract_ok": contract_ok,
        "policy_contract_reason": contract_reason,
        "runtime_smoke_ok": smoke_ok,
        "runtime_smoke_reason": smoke_reason,
        "policy_contract": contract,
        "inspection": inspection,
    }


def _print_model_report(payload: dict[str, Any]) -> None:
    print(f"Model: {payload['model']}")
    print(f"Compliant: {payload['compliant']}")
    print(f"Status: {payload['status']}")
    print(f"Reason: {payload['reason']}")
    print(f"Size: {payload['size_bytes']} bytes (limit {payload['size_limit_bytes']})")
    print(f"Policy contract: {payload['policy_contract_ok']} ({payload['policy_contract_reason']})")
    if payload["policy_contract"] is not None:
        print(f"Policy family: {payload['policy_contract']['family_id']}")
        print(f"Policy interface: {payload['policy_contract']['interface_version']}")
    print(f"Runtime smoke: {payload['runtime_smoke_ok']} ({payload['runtime_smoke_reason']})")


def _cmd_model_verify(args: argparse.Namespace) -> int:
    model_path = Path(args.model)
    if not model_path.is_file():
        print(f"Model zip not found: {model_path}", file=sys.stderr)
        return 1
    payload = _verify_model_zip(model_path, args.max_uncompressed_mb)
    _print_model_report(payload)
    return 0 if payload["compliant"] else 1


def _cmd_model_submit(args: argparse.Namespace) -> int:
    """Package (or take) an artifact, verify it locally, then commit and upload it privately."""
    family_id = _resolve_family_id(args)
    if family_id is None:
        return 1

    if args.artifact is not None:
        artifact = Path(args.artifact)
        if not artifact.is_file():
            print(f"Artifact not found: {artifact}", file=sys.stderr)
            return 1
    else:
        print(f"\nPackaging for {_family_display_name(family_id)} ({family_id})...")
        try:
            packaged = _package_model_artifact(
                source_dir=Path(args.source),
                output_zip=Path(args.output),
                family_id=family_id,
                interface_version=args.interface_version,
                overwrite=True,
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        artifact = packaged.output_zip
        print(f"Created package: {artifact} ({packaged.packaged_files_count} files)")

    if not args.upload_only:
        payload = _verify_model_zip(artifact, max_uncompressed_mb=50.0)
        _print_model_report(payload)
        if not payload["compliant"]:
            print("\nThe archive did not pass the local checks; nothing was committed on-chain.", file=sys.stderr)
            return 1
        print()

    return submit_private(
        family_id=family_id,
        artifact=str(artifact),
        backend_url=args.backend_url,
        wallet_name=args.wallet_name,
        wallet_hotkey=args.wallet_hotkey,
        netuid=args.netuid,
        network=args.network,
        upload_only=bool(args.upload_only),
    )


def _cmd_model_test(args: argparse.Namespace) -> int:
    source_dir = Path(args.source)
    if not source_dir.is_dir():
        print(f"Source directory not found: {source_dir}", file=sys.stderr)
        return 1

    checks: list[DoctorCheck] = []
    with tempfile.TemporaryDirectory(prefix="swarm_graph_test_") as tmp:
        output = Path(tmp) / "submission.zip"
        try:
            packaged = _package_model_artifact(
                source_dir=source_dir,
                output_zip=output,
                family_id=str(args.family_id),
                interface_version=args.interface_version,
                overwrite=True,
            )
            ok, reason = smoke_test_policy_package(packaged.output_zip)
            checks.append(DoctorCheck("submission_admission", True, "ok", True))
            checks.append(DoctorCheck("runtime_probe", ok, reason, True))
        except ValueError as exc:
            checks.append(DoctorCheck("submission_admission", False, str(exc), True))

    print("Model Test")
    for check in checks:
        status = "OK" if check.ok else "FAIL"
        req = "required" if check.required else "optional"
        print(f"- {status:4} [{req}] {check.name}: {check.detail}")

    failed_required = any((not c.ok) and c.required for c in checks)
    return 1 if failed_required else 0


def sanitize_benchmark_log_text(text: str) -> str:
    text = ANSI_ESCAPE_RE.sub("", text)
    text = text.replace("\r", "")
    return text


def extract_benchmark_results_block(text: str) -> str | None:
    clean_text = sanitize_benchmark_log_text(text)
    start = clean_text.rfind("=== RESULTS ===")
    if start < 0:
        return None

    tail = clean_text[start:]
    for marker in ("=== BENCHMARK COMPLETE ===", "=== BENCHMARK FAILED ==="):
        marker_index = tail.find(marker)
        if marker_index >= 0:
            line_end = tail.find("\n", marker_index)
            if line_end < 0:
                return tail.strip()
            return tail[:line_end].strip()
    return tail.strip()


def parse_benchmark_report_text(text: str) -> dict[str, Any]:
    text = sanitize_benchmark_log_text(text)
    output: dict[str, Any] = {}
    for field, pattern in REPORT_FIELD_PATTERNS.items():
        match = pattern.search(text)
        if not match:
            continue
        token = match.group(1)
        if field in {"seeds_evaluated", "workers_used", "batches_run"}:
            output[field] = int(token)
        else:
            output[field] = float(token)

    required_fields = {"seeds_evaluated", "total_wall_clock_sec", "workers_used"}
    missing = required_fields - output.keys()
    if missing:
        raise ValueError(f"Could not parse benchmark summary fields: {sorted(missing)}")
    return output


def _latest_bench_log() -> Optional[Path]:
    """Newest benchmark log this user wrote; the engine stamps uid and pid into the name."""
    logs = sorted(
        DEFAULT_BENCH_LOG.parent.glob(f"{DEFAULT_BENCH_LOG.stem}_{os.getuid()}_*.log"),
        key=lambda p: p.stat().st_mtime,
    )
    return logs[-1] if logs else None


def _cmd_report(args: argparse.Namespace) -> int:
    input_path = Path(args.input) if args.input is not None else (
        _latest_bench_log() or DEFAULT_BENCH_LOG
    )
    if not input_path.is_file():
        print(f"Report input file not found: {input_path}", file=sys.stderr)
        if args.input is None:
            print(
                f"No benchmark log found in {DEFAULT_BENCH_LOG.parent}. Run `swarm benchmark` "
                "first, or pass --input with the path it printed as `Log file:`.",
                file=sys.stderr,
            )
        return 1
    text = input_path.read_text()
    results_block = extract_benchmark_results_block(text)
    try:
        summary = parse_benchmark_report_text(results_block or text)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Report source: {input_path}")
    if results_block:
        print()
        print(results_block)
    else:
        print(f"Seeds evaluated: {summary['seeds_evaluated']}")
        print(f"Workers used: {summary['workers_used']}")
        print(f"Total wall-clock: {summary['total_wall_clock_sec']:.1f}s")
        if "throughput_seeds_per_min" in summary:
            print(f"Throughput: {summary['throughput_seeds_per_min']:.2f} seeds/min")
        if "estimated_wall_clock_sec_1000" in summary:
            print(
                "Estimated wall-clock for 1000 seeds: "
                f"{summary['estimated_wall_clock_sec_1000']:.1f}s"
            )

    return 0


def _cmd_monitor(args: argparse.Namespace) -> int:
    try:
        from swarm.validator.runtime_dashboard import run_runtime_dashboard

        return run_runtime_dashboard(
            snapshot_path=args.snapshot,
            events_path=args.events,
            refresh_sec=args.refresh_sec,
            once=args.once,
            no_clear=args.no_clear,
            max_events=args.max_events,
        )
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"Monitor failed: {exc}", file=sys.stderr)
        return 1


def _cmd_champion(args: argparse.Namespace) -> int:
    import httpx

    base_url = args.backend_url
    if not base_url:
        print("Backend URL required. Set --backend-url or SWARM_BACKEND_API_URL.", file=sys.stderr)
        return 1
    base_url = base_url.rstrip("/")
    params = {"family_id": args.family_id} if args.family_id else {}

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(f"{base_url}/champion", params=params)
            if resp.status_code == 404:
                print("No champion model yet.", file=sys.stderr)
                return 1
            if resp.status_code != 200:
                print(f"Failed to fetch champion: HTTP {resp.status_code}", file=sys.stderr)
                return 1

            champ = resp.json()
            uid = champ["uid"]
            score = champ.get("benchmark_score", 0)
            released = champ.get("is_released", False)
            per_type = champ.get("per_type_scores") or {}
            expected_hash = champ.get("model_hash")

            if not released:
                print(f"Champion: UID {uid}  Score: {score:.4f}")
                print("Model is not published for download yet; a fresh crown is published within minutes.")
                return 2

            output = args.output or Path(_champion_zip_name(uid, args.family_id))
            print(f"Champion: UID {uid}  Score: {score:.4f}")
            if per_type:
                parts = [f"{k}: {v:.3f}" for k, v in sorted(per_type.items()) if v]
                if parts:
                    print(f"Per-map:  {', '.join(parts)}")
            print(f"Downloading to {output} ...")

            dl = client.get(f"{base_url}/models/{uid}/download", params=params)
            if dl.status_code == 403:
                print("Model not released for public download.", file=sys.stderr)
                return 1
            if dl.status_code != 200:
                print(f"Download failed: HTTP {dl.status_code}", file=sys.stderr)
                return 1

            if expected_hash:
                dl_hash = hashlib.sha256(dl.content).hexdigest()
                if dl_hash != expected_hash:
                    print(f"Download integrity check failed (expected {expected_hash[:16]}...)", file=sys.stderr)
                    return 1

            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(dl.content)

            size_mb = len(dl.content) / (1024 * 1024)
            print(f"Saved: {output} ({size_mb:.1f} MB)")
            return 0

    except httpx.ConnectError:
        print(f"Cannot connect to backend at {base_url}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="swarm", description="Swarm CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser(
        "doctor", help="Check local environment readiness for Swarm benchmarking."
    )
    doctor_parser.set_defaults(func=_cmd_doctor)

    monitor_parser = subparsers.add_parser(
        "monitor",
        help="Live validator runtime dashboard.",
    )
    monitor_parser.add_argument(
        "--snapshot",
        type=Path,
        default=None,
        help="Path to validator_runtime.json snapshot file.",
    )
    monitor_parser.add_argument(
        "--events",
        type=Path,
        default=None,
        help="Path to validator_events.jsonl events file.",
    )
    monitor_parser.add_argument(
        "--refresh-sec",
        type=float,
        default=1.0,
        help="Refresh interval for the live dashboard.",
    )
    monitor_parser.add_argument(
        "--max-events",
        type=int,
        default=8,
        help="How many recent events to display.",
    )
    monitor_parser.add_argument(
        "--once",
        action="store_true",
        help="Render one frame and exit.",
    )
    monitor_parser.add_argument(
        "--no-clear",
        action="store_true",
        help="Do not clear the terminal between frames.",
    )
    monitor_parser.set_defaults(func=_cmd_monitor)

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Run benchmark workflows.",
    )
    benchmark_parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to submission zip. If omitted, auto-downloads the current champion.",
    )
    benchmark_parser.add_argument(
        "--uid",
        type=int,
        default=None,
        help="Miner UID. If omitted, benchmark script infers from model name.",
    )
    benchmark_parser.add_argument(
        "--family-id",
        choices=sorted(CHALLENGE_FAMILY_IDS),
        default=None,
        help="Challenge family to benchmark (default: the engine's default family).",
    )
    benchmark_parser.add_argument(
        "--seeds-per-group",
        type=int,
        default=3,
        help="Seeds per map group.",
    )
    benchmark_parser.add_argument(
        "--workers",
        type=int,
        default=N_DOCKER_WORKERS,
        help="Parallel workers for benchmark (default: one per CPU group; optionally capped by SWARM_MAX_DOCKER_WORKERS).",
    )
    benchmark_parser.add_argument(
        "--log-out",
        type=Path,
        default=None,
        help="Output benchmark log path (default: a per-run /tmp/bench_full_eval_<uid>_<pid>.log).",
    )
    benchmark_parser.add_argument(
        "--seed-file",
        type=Path,
        default=None,
        help="Reuse an exact benchmark seed JSON instead of discovering seeds.",
    )
    benchmark_parser.add_argument(
        "--save-seed-file",
        type=Path,
        default=None,
        help="Write the resolved benchmark seeds to JSON for later replay.",
    )
    benchmark_parser.add_argument(
        "--seed-search-rng",
        type=int,
        default=None,
        help="Random seed used for reproducible benchmark seed discovery.",
    )
    benchmark_parser.add_argument(
        "--summary-json-out",
        type=Path,
        default=None,
        help="Write benchmark summary JSON to this path.",
    )
    benchmark_parser.add_argument(
        "--relax-timeouts",
        action="store_true",
        help="Enable slow-machine timeout overrides.",
    )
    benchmark_parser.add_argument(
        "--rpc-verbosity",
        choices=["low", "mid", "high"],
        default="mid",
        help="RPC tracing verbosity.",
    )
    benchmark_parser.set_defaults(func=_cmd_benchmark)

    model_parser = subparsers.add_parser("model", help="Model packaging and validation.")
    model_subparsers = model_parser.add_subparsers(dest="model_command", required=True)

    model_admit_parser = model_subparsers.add_parser(
        "verify",
        help="Verify submission zip compliance.",
    )
    model_admit_parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to submission zip.",
    )
    model_admit_parser.add_argument(
        "--max-uncompressed-mb",
        type=float,
        default=300.0,
        help="Maximum allowed uncompressed ZIP size in MB for safety checks.",
    )
    model_admit_parser.set_defaults(func=_cmd_model_verify)

    model_package_parser = model_subparsers.add_parser(
        "package",
        help="Build submission.zip from a source folder.",
    )
    model_package_parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Source directory containing drone_agent.py and any model weights.",
    )
    model_package_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_MODEL_ZIP,
        help=f"Output submission zip path (default: {DEFAULT_MODEL_ZIP}).",
    )
    model_package_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output zip if it exists.",
    )
    model_package_parser.add_argument(
        "--family-id",
        choices=sorted(CHALLENGE_FAMILY_IDS),
        default=None,
        help=(
            "Challenge family implemented by this artifact. Omit it in a "
            "terminal to pick from a menu; required for non-interactive runs."
        ),
    )
    model_package_parser.add_argument(
        "--interface-version",
        default=None,
        help=(
            "Explicit policy interface version. Defaults to the first supported "
            "version for the selected family."
        ),
    )
    model_package_parser.set_defaults(func=_cmd_model_package)

    model_test_parser = model_subparsers.add_parser(
        "test",
        help="Test source folder formatting and packaging readiness.",
    )
    model_test_parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Source directory containing drone_agent.py and any model weights.",
    )
    model_test_parser.add_argument("--family-id", choices=sorted(CHALLENGE_FAMILY_IDS), required=True)
    model_test_parser.add_argument("--interface-version", default=None)
    model_test_parser.set_defaults(func=_cmd_model_test)

    model_submit_parser = model_subparsers.add_parser(
        "submit",
        help="Package, verify, commit on-chain and upload a model privately.",
    )
    submit_input = model_submit_parser.add_mutually_exclusive_group(required=True)
    submit_input.add_argument(
        "--source",
        type=Path,
        help="Source directory to package first (drone_agent.py and model weights).",
    )
    submit_input.add_argument(
        "--artifact",
        type=Path,
        help="An already packaged submission zip to submit as is.",
    )
    model_submit_parser.add_argument(
        "--family-id",
        choices=sorted(CHALLENGE_FAMILY_IDS),
        default=None,
        help="Challenge family this model competes in (prompted in a terminal when omitted).",
    )
    model_submit_parser.add_argument(
        "--interface-version",
        default=None,
        help="Explicit policy interface version when packaging from --source.",
    )
    model_submit_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_MODEL_ZIP,
        help=f"Where --source is packaged to (default: {DEFAULT_MODEL_ZIP}).",
    )
    model_submit_parser.add_argument(
        "--backend-url",
        type=str,
        default=os.environ.get("SWARM_BACKEND_API_URL", DEFAULT_BACKEND_URL),
        help=f"Backend API URL (default: {DEFAULT_BACKEND_URL}).",
    )
    model_submit_parser.add_argument("--wallet.name", dest="wallet_name", default="default")
    model_submit_parser.add_argument("--wallet.hotkey", dest="wallet_hotkey", default="default")
    model_submit_parser.add_argument("--netuid", type=int, default=124)
    model_submit_parser.add_argument("--subtensor.network", dest="network", default="finney")
    model_submit_parser.add_argument(
        "--upload-only",
        action="store_true",
        help="Skip the chain commit and only (re)upload the artifact committed earlier.",
    )
    model_submit_parser.set_defaults(func=_cmd_model_submit)

    report_parser = subparsers.add_parser(
        "report",
        help="Summarize benchmark logs.",
    )
    report_parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Benchmark log input path (default: the newest log the last benchmark wrote).",
    )
    report_parser.set_defaults(func=_cmd_report)

    champion_parser = subparsers.add_parser(
        "champion",
        help="Download the current champion model.",
    )
    champion_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path. Defaults to champion_UID_{uid}.zip in current directory.",
    )
    champion_parser.add_argument(
        "--family-id",
        choices=sorted(CHALLENGE_FAMILY_IDS),
        default=None,
        help="Challenge family to download the champion for (default: the best champion across families).",
    )
    champion_parser.add_argument(
        "--backend-url",
        type=str,
        default=os.environ.get("SWARM_BACKEND_API_URL", "https://api.swarm124.com"),
        help="Backend API URL (default: https://api.swarm124.com).",
    )
    champion_parser.set_defaults(func=_cmd_champion)

    visualize_parser = subparsers.add_parser(
        "visualize",
        help="Open an interactive visualizer for a specific benchmark seed or map type.",
    )
    visualize_parser.add_argument(
        "--type",
        type=int,
        default=None,
        choices=[1, 2, 3, 4, 5, 6, 7],
        help="Challenge type (1=City 2=Open 3=Mountain 4=Village 5=Warehouse 6=Forest 7=Office).",
    )
    visualize_parser.add_argument(
        "--family-id",
        type=str,
        default="cf_autopilot",
        choices=sorted(CHALLENGE_FAMILY_IDS),
        help="Challenge family id (default: cf_autopilot).",
    )
    visualize_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Map seed. If `--type` is omitted, infer the type from `--summary-json`, "
            "`--seed-file`, or the seed's deterministic benchmark assignment."
        ),
    )
    visualize_parser.add_argument(
        "--seed-file",
        type=Path,
        default=None,
        help="Saved benchmark seed JSON used for seed->type lookup.",
    )
    visualize_parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Benchmark summary JSON used for failed-seed review and seed->type lookup.",
    )
    visualize_parser.add_argument(
        "--failed",
        action="store_true",
        help="List failed seeds from `--summary-json` for review.",
    )
    visualize_parser.add_argument(
        "--failed-index",
        type=int,
        default=None,
        help="1-based failed-seed index from `--summary-json` to open directly.",
    )
    visualize_parser.add_argument(
        "--speed",
        type=float,
        default=4.0,
        help="Base flight speed in metres per second.",
    )
    visualize_parser.add_argument(
        "--boost",
        type=float,
        default=2.0,
        help="Multiplier for shifted movement.",
    )
    visualize_parser.add_argument(
        "--camera",
        choices=["follow", "fixed"],
        default="follow",
        help="Viewer camera mode.",
    )
    visualize_parser.add_argument(
        "--width",
        type=int,
        default=960,
        help="Window width.",
    )
    visualize_parser.add_argument(
        "--height",
        type=int,
        default=540,
        help="Window height.",
    )
    visualize_parser.add_argument(
        "--render-scale",
        type=float,
        default=None,
        help="Internal render scale. Defaults depend on map type.",
    )
    visualize_parser.add_argument(
        "--render-distance",
        type=float,
        default=None,
        help="Maximum camera/render distance in metres. Defaults depend on map type.",
    )
    visualize_parser.add_argument(
        "--render-fps",
        type=float,
        default=None,
        help="Maximum render FPS. Defaults depend on map type.",
    )
    visualize_parser.add_argument(
        "--sim-fps",
        type=float,
        default=None,
        help="Maximum world simulation FPS for the viewer. Defaults depend on map type.",
    )
    visualize_parser.add_argument(
        "--gpu",
        action="store_true",
        help="Enable Bullet EGL hardware rendering for the visualizer if available.",
    )
    visualize_parser.set_defaults(func=_cmd_visualize)

    video_parser = subparsers.add_parser(
        "video",
        help="Render mp4 flight videos for one seed or a saved benchmark seed file.",
    )
    video_parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to submission zip (e.g., model/UID_178.zip).",
    )
    video_parser.add_argument(
        "--family-id",
        type=str,
        default="cf_autopilot",
        choices=sorted(CHALLENGE_FAMILY_IDS),
        help="Challenge family id (default: cf_autopilot).",
    )
    video_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Single seed to replay.",
    )
    video_parser.add_argument(
        "--type",
        type=int,
        default=None,
        choices=[1, 2, 3, 4, 5, 6, 7],
        help="Challenge type (1=City 2=Open 3=Mountain 4=Village 5=Warehouse 6=Forest 7=Office).",
    )
    video_parser.add_argument(
        "--seed-file",
        type=Path,
        default=None,
        help="Benchmark seed JSON generated by swarm benchmark --save-seed-file.",
    )
    video_parser.add_argument(
        "--mode",
        type=str,
        default="chase",
        help="Camera mode(s): depth, fpv, chase, overview, or all.",
    )
    video_parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Directory where mp4 files will be written.",
    )
    video_parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Frame width.",
    )
    video_parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Frame height.",
    )
    video_parser.add_argument(
        "--fps",
        type=int,
        default=25,
        help="Video frames per second.",
    )
    video_parser.add_argument(
        "--backend",
        choices=["local", "benchmark"],
        default="benchmark",
        help="Replay backend: local fast replay, or exact benchmark Docker/RPC replay.",
    )
    video_parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Benchmark summary JSON from swarm benchmark --summary-json-out; replay must match when provided.",
    )
    video_parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a seed if all requested mp4 outputs already exist.",
    )
    video_parser.add_argument(
        "--progress-file",
        type=Path,
        default=None,
        help="Optional JSON progress path for single-seed video generation.",
    )
    video_parser.add_argument(
        "--chase-back",
        type=float,
        default=2.5,
        help="Chase camera distance behind the drone in metres.",
    )
    video_parser.add_argument(
        "--chase-up",
        type=float,
        default=1.0,
        help="Chase camera height above the drone in metres.",
    )
    video_parser.add_argument(
        "--chase-fov",
        type=float,
        default=65.0,
        help="Chase camera field of view in degrees.",
    )
    video_parser.add_argument(
        "--fpv-fov",
        type=float,
        default=90.0,
        help="FPV camera field of view in degrees.",
    )
    video_parser.add_argument(
        "--overview-fov",
        type=float,
        default=60.0,
        help="Overview camera field of view in degrees.",
    )
    video_parser.add_argument(
        "--save-actions",
        type=Path,
        default=None,
        help="Save recorded actions per seed for deterministic replay.",
    )
    video_parser.add_argument(
        "--replay-actions",
        type=Path,
        default=None,
        help="Replay pre-recorded actions instead of running the policy.",
    )
    video_parser.set_defaults(func=_cmd_video)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
