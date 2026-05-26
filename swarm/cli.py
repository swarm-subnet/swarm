from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import py_compile
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
    BENCHMARK_GROUP_TO_CHALLENGE_TYPE,
    CHALLENGE_FAMILY_IDS,
    CHALLENGE_TYPE_TO_ENVIRONMENT_TYPE,
)
from swarm.policy_interface import (
    POLICY_CONTRACT_FILENAME,
    PolicyInterfaceError,
    render_artifact_policy_contract,
    resolve_policy_interface_version,
    smoke_test_policy_package,
    verify_policy_package_contract,
)
from swarm.submission_manifest import (
    LEGACY_FALLBACK,
    REPO_LAYOUT_RULES,
    SUBMISSION_MANIFEST_FILENAME,
    SubmissionArtifact,
    SubmissionManifestError,
    load_submission_manifest,
    validate_submission_repo,
    write_submission_manifest,
)

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
    "drone_agent.py",
}

REQUIREMENTS_DIRECT_REF_RE = re.compile(r"\s@\s")
REQUIREMENTS_URL_RE = re.compile(r"^(?:https?://|git\+|file:|/|\.\.?/)")

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
BENCH_GROUP_ORDER = list(BENCHMARK_GROUP_TO_CHALLENGE_TYPE)
BENCH_GROUP_TO_TYPE = dict(BENCHMARK_GROUP_TO_CHALLENGE_TYPE)
TYPE_LABELS = dict(CHALLENGE_TYPE_TO_ENVIRONMENT_TYPE)


@dataclass
class DoctorCheck:
    name: str
    ok: bool
    detail: str
    required: bool = True


@dataclass(frozen=True)
class PackagedModelArtifact:
    family_id: str
    interface_version: str
    output_zip: Path
    sha256: str
    packaged_files_count: int


@dataclass(frozen=True)
class RepoPackageSource:
    family_id: str
    source_dir: Path
    interface_version: str | None = None



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


def _download_champion_model() -> Optional[Path]:
    import httpx

    base_url = os.environ.get("SWARM_BACKEND_API_URL", "https://api.swarm124.com").rstrip("/")
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(f"{base_url}/champion")
            if resp.status_code != 200:
                print("No champion model available to download.", file=sys.stderr)
                return None
            champ = resp.json()
            if not champ.get("is_released"):
                print(f"Champion UID {champ['uid']} is not released for download yet.", file=sys.stderr)
                return None

            uid = champ["uid"]
            expected_hash = champ.get("model_hash")
            output = Path(f"champion_UID_{uid}.zip")

            if output.exists() and expected_hash:
                existing_hash = hashlib.sha256(output.read_bytes()).hexdigest()
                if existing_hash == expected_hash:
                    print(f"Using cached champion: {output}")
                    return output

            print(f"Downloading champion UID {uid} (score: {champ.get('benchmark_score', 0):.4f})...")
            dl = client.get(f"{base_url}/models/{uid}/download")
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
        downloaded = _download_champion_model()
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




def _validate_requirements_file(requirements_path: Path) -> list[str]:
    issues: list[str] = []
    for idx, raw_line in enumerate(requirements_path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-"):
            issues.append(f"line {idx}: pip option not allowed ({line})")
            continue
        if REQUIREMENTS_DIRECT_REF_RE.search(line):
            issues.append(f"line {idx}: direct reference not allowed ({line})")
            continue
        if REQUIREMENTS_URL_RE.search(line):
            issues.append(f"line {idx}: direct URL/path not allowed ({line})")
    return issues


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
    if not drone_agent.exists():
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

    return PackagedModelArtifact(
        family_id=family_id,
        interface_version=interface_version,
        output_zip=output_zip,
        sha256=_sha256sum(output_zip),
        packaged_files_count=len(files_to_pack) + 1,
    )


def _default_repo_artifact_relpath(family_id: str) -> str:
    artifact_basename = Path(str(LEGACY_FALLBACK["artifact_path"])).name
    return (
        Path(REPO_LAYOUT_RULES["artifacts_dir"]) / family_id / artifact_basename
    ).as_posix()


def _parse_repo_family_source(raw_value: str) -> RepoPackageSource:
    family_and_version, separator, source_token = raw_value.partition("=")
    if not separator or not source_token.strip():
        raise ValueError(
            "invalid_family_source:expected FAMILY_ID=PATH or FAMILY_ID@INTERFACE_VERSION=PATH"
        )
    family_id, version_separator, interface_version = family_and_version.partition("@")
    family_id = family_id.strip()
    if family_id not in CHALLENGE_FAMILY_IDS:
        raise ValueError(f"unsupported_family_id:{family_id}")
    normalized_version = interface_version.strip() if version_separator else None
    return RepoPackageSource(
        family_id=family_id,
        source_dir=Path(source_token.strip()),
        interface_version=normalized_version or None,
    )


def _resolve_repo_package_sources(args: argparse.Namespace) -> list[RepoPackageSource]:
    specs: list[RepoPackageSource] = []
    for raw_value in args.family_source or ():
        specs.append(_parse_repo_family_source(raw_value))

    if args.source is not None or args.family_id is not None:
        if args.source is None or args.family_id is None:
            raise ValueError("repo_package_requires_source_and_family_id_together")
        specs.append(
            RepoPackageSource(
                family_id=str(args.family_id),
                source_dir=Path(args.source),
                interface_version=args.interface_version,
            )
        )

    if not specs:
        raise ValueError("no_family_sources_specified")

    seen_family_ids: set[str] = set()
    for spec in specs:
        if spec.family_id in seen_family_ids:
            raise ValueError(f"duplicate_family_id:{spec.family_id}")
        seen_family_ids.add(spec.family_id)
    return specs


def _inspect_submission_repo(
    repo_root: Path,
    *,
    allow_legacy_fallback: bool = True,
) -> tuple[bool, str, list[dict[str, Any]]]:
    ok, reason, manifest = validate_submission_repo(
        repo_root,
        allow_legacy_fallback=allow_legacy_fallback,
    )
    if not ok or manifest is None:
        return False, reason, []

    artifact_reports: list[dict[str, Any]] = []
    first_failure = "ok"
    repo_ok = True
    for artifact in manifest.artifacts:
        artifact_file = repo_root / artifact.artifact_path
        contract_ok, contract_reason, contract = verify_policy_package_contract(artifact_file)
        smoke_ok = False
        smoke_reason = "skipped_due_to_contract_failure"
        if contract_ok:
            smoke_ok, smoke_reason = smoke_test_policy_package(artifact_file)
        compliant = bool(contract_ok and smoke_ok)
        artifact_reports.append(
            {
                "family_id": artifact.family_id,
                "artifact_path": artifact.artifact_path,
                "sha256": artifact.sha256,
                "interface_version": artifact.interface_version,
                "metadata": dict(artifact.metadata),
                "policy_contract_ok": contract_ok,
                "policy_contract_reason": contract_reason,
                "runtime_smoke_ok": smoke_ok,
                "runtime_smoke_reason": smoke_reason,
                "legacy_fallback_used": manifest.legacy_fallback_used,
                "compliant": compliant,
                "policy_contract": contract,
            }
        )
        if not compliant and repo_ok:
            repo_ok = False
            first_failure = (
                f"policy_contract:{artifact.family_id}:{contract_reason}"
                if not contract_ok
                else f"runtime_smoke:{artifact.family_id}:{smoke_reason}"
            )

    return repo_ok, first_failure, artifact_reports


def _cmd_model_package(args: argparse.Namespace) -> int:
    source_dir = Path(args.source)
    output_zip = Path(args.output)
    try:
        packaged = _package_model_artifact(
            source_dir=source_dir,
            output_zip=output_zip,
            family_id=str(args.family_id),
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


def _cmd_model_verify(args: argparse.Namespace) -> int:
    from swarm.constants import MAX_MODEL_BYTES
    from swarm.core.model_verify import (
        classify_model_validity,
        inspect_model_structure,
        zip_is_safe,
    )

    model_path = Path(args.model)
    if not model_path.is_file():
        print(f"Model zip not found: {model_path}", file=sys.stderr)
        return 1

    size_bytes = model_path.stat().st_size
    max_uncompressed = int(args.max_uncompressed_mb * 1024 * 1024)
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

    payload = {
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

    return 0 if compliant else 1


def _cmd_repo_package(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root)
    repo_root.mkdir(parents=True, exist_ok=True)

    try:
        package_sources = _resolve_repo_package_sources(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    existing_artifacts: dict[str, SubmissionArtifact] = {}
    manifest_path = repo_root / SUBMISSION_MANIFEST_FILENAME
    if manifest_path.exists():
        try:
            existing_manifest = load_submission_manifest(manifest_path)
        except SubmissionManifestError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        existing_artifacts = {
            artifact.family_id: artifact for artifact in existing_manifest.artifacts
        }

    packaged_artifacts: list[PackagedModelArtifact] = []
    for spec in package_sources:
        artifact_relpath = _default_repo_artifact_relpath(spec.family_id)
        artifact_output = repo_root / artifact_relpath
        try:
            packaged = _package_model_artifact(
                source_dir=spec.source_dir,
                output_zip=artifact_output,
                family_id=spec.family_id,
                interface_version=spec.interface_version,
                overwrite=bool(args.overwrite),
            )
        except ValueError as exc:
            print(f"{spec.family_id}: {exc}", file=sys.stderr)
            return 1

        existing_artifacts[spec.family_id] = SubmissionArtifact(
            family_id=packaged.family_id,
            interface_version=packaged.interface_version,
            artifact_path=artifact_relpath,
            sha256=packaged.sha256,
            metadata={
                "packaged_with": "swarm_cli",
                "source_dir_name": spec.source_dir.name,
            },
        )
        packaged_artifacts.append(packaged)

    write_submission_manifest(repo_root, tuple(existing_artifacts.values()))

    print(f"Repo root: {repo_root}")
    print(f"Manifest: {repo_root / SUBMISSION_MANIFEST_FILENAME}")
    print(f"Artifacts updated: {len(packaged_artifacts)}")
    for packaged in packaged_artifacts:
        print(
            f"- {packaged.family_id}: {packaged.output_zip} "
            f"({packaged.interface_version}, sha256={packaged.sha256[:12]}...)"
        )
    print("Run `swarm repo verify --repo-root <path>` before publishing.")
    return 0


def _cmd_repo_verify(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root)
    if not repo_root.is_dir():
        print(f"Repo root not found: {repo_root}", file=sys.stderr)
        return 1

    repo_ok, reason, artifact_reports = _inspect_submission_repo(
        repo_root,
        allow_legacy_fallback=not args.strict_manifest,
    )

    manifest_path = repo_root / SUBMISSION_MANIFEST_FILENAME
    print(f"Repo: {repo_root}")
    print(f"Manifest path: {manifest_path}")
    print(f"Compliant: {repo_ok}")
    if not artifact_reports:
        print(f"Reason: {reason}")
        return 0 if repo_ok else 1

    for report in artifact_reports:
        print(f"Family: {report['family_id']}")
        print(f"Artifact: {report['artifact_path']}")
        print(
            "Policy contract: "
            f"{report['policy_contract_ok']} ({report['policy_contract_reason']})"
        )
        print(
            "Runtime smoke: "
            f"{report['runtime_smoke_ok']} ({report['runtime_smoke_reason']})"
        )
    if not repo_ok:
        print(f"Reason: {reason}")
    return 0 if repo_ok else 1


def _cmd_model_test(args: argparse.Namespace) -> int:
    from swarm.constants import MAX_MODEL_BYTES

    source_dir = Path(args.source)
    if not source_dir.is_dir():
        print(f"Source directory not found: {source_dir}", file=sys.stderr)
        return 1

    checks: list[DoctorCheck] = []
    drone_agent = source_dir / "drone_agent.py"
    checks.append(
        DoctorCheck(
            name="drone_agent.py",
            ok=drone_agent.exists(),
            detail="present" if drone_agent.exists() else "missing",
            required=True,
        )
    )

    if drone_agent.exists():
        try:
            py_compile.compile(str(drone_agent), doraise=True)
            checks.append(DoctorCheck("drone_agent_syntax", True, "valid python", True))
        except py_compile.PyCompileError as exc:
            checks.append(DoctorCheck("drone_agent_syntax", False, str(exc), True))

    requirements_path = source_dir / "requirements.txt"
    if requirements_path.exists():
        req_issues = _validate_requirements_file(requirements_path)
        checks.append(
            DoctorCheck(
                "requirements.txt",
                ok=not req_issues,
                detail="ok" if not req_issues else "; ".join(req_issues),
                required=True,
            )
        )
    else:
        checks.append(
            DoctorCheck(
                "requirements.txt",
                ok=True,
                detail="not present (optional)",
                required=False,
            )
        )

    files_to_pack = _collect_packable_files(source_dir)
    total_size = sum(f.stat().st_size for f in files_to_pack)
    checks.append(
        DoctorCheck(
            "estimated_package_size",
            ok=total_size <= MAX_MODEL_BYTES,
            detail=f"{total_size} bytes (limit {MAX_MODEL_BYTES})",
            required=True,
        )
    )

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


def _cmd_report(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"Report input file not found: {input_path}", file=sys.stderr)
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

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(f"{base_url}/champion")
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
                print("Model is not released for download yet.")
                return 0

            output = args.output or Path(f"champion_UID_{uid}.zip")
            print(f"Champion: UID {uid}  Score: {score:.4f}")
            if per_type:
                parts = [f"{k}: {v:.3f}" for k, v in sorted(per_type.items()) if v]
                if parts:
                    print(f"Per-map:  {', '.join(parts)}")
            print(f"Downloading to {output} ...")

            dl = client.get(f"{base_url}/models/{uid}/download")
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
        "--seeds-per-group",
        type=int,
        default=3,
        help="Seeds per map group.",
    )
    benchmark_parser.add_argument(
        "--workers",
        type=int,
        default=N_DOCKER_WORKERS,
        help="Parallel workers for benchmark (default: available vCPUs, capped at 12).",
    )
    benchmark_parser.add_argument(
        "--log-out",
        type=Path,
        default=None,
        help=f"Output benchmark log path (default in script: {DEFAULT_BENCH_LOG}).",
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

    model_verify_parser = model_subparsers.add_parser(
        "verify",
        help="Verify submission zip compliance.",
    )
    model_verify_parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to submission zip.",
    )
    model_verify_parser.add_argument(
        "--max-uncompressed-mb",
        type=float,
        default=300.0,
        help="Maximum allowed uncompressed ZIP size in MB for safety checks.",
    )
    model_verify_parser.set_defaults(func=_cmd_model_verify)

    model_package_parser = model_subparsers.add_parser(
        "package",
        help="Build submission.zip from a source folder.",
    )
    model_package_parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Source directory containing drone_agent.py and model files.",
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
        default="cf_search_and_rescue",
        help="Challenge family implemented by this artifact.",
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
        help="Source directory containing drone_agent.py.",
    )
    model_test_parser.set_defaults(func=_cmd_model_test)

    repo_parser = subparsers.add_parser(
        "repo",
        help="Repository-level multi-family packaging and verification.",
    )
    repo_subparsers = repo_parser.add_subparsers(dest="repo_command", required=True)

    repo_package_parser = repo_subparsers.add_parser(
        "package",
        help="Build or update a multi-family submission repo manifest and artifacts.",
    )
    repo_package_parser.add_argument(
        "--repo-root",
        type=Path,
        required=True,
        help="Repo root where submission_manifest.json and artifacts/ live.",
    )
    repo_package_parser.add_argument(
        "--family-source",
        action="append",
        default=[],
        help=(
            "Family source mapping in the form FAMILY_ID=PATH or "
            "FAMILY_ID@INTERFACE_VERSION=PATH. May be repeated."
        ),
    )
    repo_package_parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Single-family source directory shortcut.",
    )
    repo_package_parser.add_argument(
        "--family-id",
        choices=sorted(CHALLENGE_FAMILY_IDS),
        default=None,
        help="Single-family shortcut for --source.",
    )
    repo_package_parser.add_argument(
        "--interface-version",
        default=None,
        help="Optional interface version for the single-family shortcut.",
    )
    repo_package_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite targeted artifact ZIPs if they already exist.",
    )
    repo_package_parser.set_defaults(func=_cmd_repo_package)

    repo_verify_parser = repo_subparsers.add_parser(
        "verify",
        help="Verify a repo-root submission manifest and all published artifacts.",
    )
    repo_verify_parser.add_argument(
        "--repo-root",
        type=Path,
        required=True,
        help="Repo root containing submission_manifest.json.",
    )
    repo_verify_parser.add_argument(
        "--strict-manifest",
        action="store_true",
        help="Require submission_manifest.json and disable legacy submission.zip fallback.",
    )
    repo_verify_parser.set_defaults(func=_cmd_repo_verify)

    report_parser = subparsers.add_parser(
        "report",
        help="Summarize benchmark logs.",
    )
    report_parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_BENCH_LOG,
        help=f"Benchmark log input path (default: {DEFAULT_BENCH_LOG}).",
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
        "--backend-url",
        type=str,
        default=os.environ.get("SWARM_BACKEND_API_URL", "https://api.swarm124.com"),
        help="Backend API URL (default: https://api.swarm124.com).",
    )
    champion_parser.set_defaults(func=_cmd_champion)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
