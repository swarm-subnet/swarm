from __future__ import annotations

import argparse
import importlib.util
import json
import os
import py_compile
import re
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BENCH_SCRIPT = REPO_ROOT / "debugging" / "bench_full_eval.py"
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
    "total_wall_clock_sec": re.compile(r"Total wall-clock:\s+([\d.]+)s"),
    "avg_wall_per_seed_sec": re.compile(r"Avg wall / seed:\s+([\d.]+)s"),
    "median_wall_per_seed_sec": re.compile(r"Median wall / seed:\s+([\d.]+)s"),
    "p90_wall_per_seed_sec": re.compile(r"P90 wall / seed:\s+([\d.]+)s"),
    "throughput_seeds_per_min": re.compile(r"Throughput:\s+([\d.]+)\s+seeds/min"),
    "throughput_per_worker": re.compile(
        r"Throughput per worker:\s+([\d.]+)\s+seeds/min/worker"
    ),
    "workers_used": re.compile(r"Workers used:\s+(\d+)"),
    "estimated_wall_clock_sec_1000": re.compile(r"Estimated wall-clock:\s+([\d.]+)s"),
    "estimated_throughput_1000": re.compile(
        r"Estimated throughput:\s+([\d.]+)\s+seeds/min"
    ),
}


@dataclass
class DoctorCheck:
    name: str
    ok: bool
    detail: str
    required: bool = True


def _check_module_available(module_name: str) -> DoctorCheck:
    spec = importlib.util.find_spec(module_name)
    return DoctorCheck(
        name=f"module:{module_name}",
        ok=spec is not None,
        detail="available" if spec is not None else "missing",
        required=True,
    )


def _check_python_version() -> DoctorCheck:
    ok = sys.version_info >= (3, 10)
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


def _check_benchmark_script() -> DoctorCheck:
    if DEFAULT_BENCH_SCRIPT.exists():
        return DoctorCheck("benchmark_script", True, str(DEFAULT_BENCH_SCRIPT), True)
    return DoctorCheck("benchmark_script", False, f"missing: {DEFAULT_BENCH_SCRIPT}", True)


def _check_env_var(name: str, required: bool = False) -> DoctorCheck:
    value = os.getenv(name)
    if value:
        return DoctorCheck(name, True, "set", required)
    return DoctorCheck(name, False, "not set", required)


def _run_doctor_checks() -> list[DoctorCheck]:
    from swarm.constants import MODEL_DIR

    return [
        _check_python_version(),
        _check_docker_binary(),
        _check_docker_daemon(),
        _check_module_available("capnp"),
        _check_module_available("pybullet"),
        _check_module_available("gym_pybullet_drones"),
        _check_writable_dir(Path("state"), "state_dir"),
        _check_writable_dir(Path(MODEL_DIR), "model_dir"),
        _check_submission_template(),
        _check_benchmark_script(),
        _check_env_var("SWARM_PRIVATE_BENCHMARK_SECRET", required=False),
    ]


def _print_doctor_text(checks: list[DoctorCheck]) -> None:
    print("Swarm Doctor")
    for check in checks:
        status = "OK" if check.ok else "FAIL"
        req = "required" if check.required else "optional"
        print(f"- {status:4} [{req}] {check.name}: {check.detail}")


def _cmd_doctor(args: argparse.Namespace) -> int:
    checks = _run_doctor_checks()
    if args.json:
        payload = [asdict(c) for c in checks]
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_doctor_text(checks)
    failed_required = any((not c.ok) and c.required for c in checks)
    return 1 if failed_required else 0


def _build_benchmark_command(args: argparse.Namespace) -> list[str]:
    cmd = [sys.executable, "-u", str(DEFAULT_BENCH_SCRIPT), "--model", str(args.model)]
    if args.uid is not None:
        cmd.extend(["--uid", str(args.uid)])
    cmd.extend(["--seeds-per-group", str(args.seeds_per_group)])
    cmd.extend(["--workers", str(args.workers)])
    if args.log_out is not None:
        cmd.extend(["--log-out", str(args.log_out)])
    if args.relax_timeouts:
        cmd.append("--relax-timeouts")
    cmd.extend(["--rpc-verbosity", str(args.rpc_verbosity)])
    return cmd


def _cmd_benchmark(args: argparse.Namespace) -> int:
    if not DEFAULT_BENCH_SCRIPT.exists():
        print(f"Benchmark script not found: {DEFAULT_BENCH_SCRIPT}", file=sys.stderr)
        return 1
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}", file=sys.stderr)
        return 1
    command = _build_benchmark_command(args)
    return subprocess.run(command, check=False).returncode


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


def _cmd_model_package(args: argparse.Namespace) -> int:
    source_dir = Path(args.source)
    output_zip = Path(args.output)

    if not source_dir.is_dir():
        print(f"Source directory not found: {source_dir}", file=sys.stderr)
        return 1
    drone_agent = source_dir / "drone_agent.py"
    if not drone_agent.exists():
        print("Source must contain drone_agent.py", file=sys.stderr)
        return 1
    if output_zip.exists() and not args.overwrite:
        print(
            f"Output already exists: {output_zip} (use --overwrite to replace)",
            file=sys.stderr,
        )
        return 1

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    files_to_pack = _collect_packable_files(source_dir)
    if drone_agent not in files_to_pack:
        files_to_pack.append(drone_agent)
        files_to_pack.sort()

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in files_to_pack:
            zf.write(file_path, arcname=str(file_path.relative_to(source_dir)))

    print(f"Created package: {output_zip}")
    print(f"Files included: {len(files_to_pack)}")
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
    compliant = bool(size_ok and zip_safe and status == "legitimate")

    payload = {
        "model": str(model_path),
        "compliant": compliant,
        "size_bytes": size_bytes,
        "size_limit_bytes": MAX_MODEL_BYTES,
        "size_ok": size_ok,
        "zip_safe": zip_safe,
        "status": status,
        "reason": reason,
        "inspection": inspection,
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Model: {payload['model']}")
        print(f"Compliant: {payload['compliant']}")
        print(f"Status: {payload['status']}")
        print(f"Reason: {payload['reason']}")
        print(f"Size: {payload['size_bytes']} bytes (limit {payload['size_limit_bytes']})")

    return 0 if compliant else 1


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

    if args.json:
        print(json.dumps([asdict(c) for c in checks], indent=2, sort_keys=True))
    else:
        print("Model Test")
        for check in checks:
            status = "OK" if check.ok else "FAIL"
            req = "required" if check.required else "optional"
            print(f"- {status:4} [{req}] {check.name}: {check.detail}")

    failed_required = any((not c.ok) and c.required for c in checks)
    return 1 if failed_required else 0


def parse_benchmark_report_text(text: str) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for field, pattern in REPORT_FIELD_PATTERNS.items():
        match = pattern.search(text)
        if not match:
            continue
        token = match.group(1)
        if field in {"seeds_evaluated", "workers_used"}:
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
    try:
        summary = parse_benchmark_report_text(text)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"Report source: {input_path}")
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="swarm", description="Swarm CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser(
        "doctor", help="Check local environment readiness for Swarm benchmarking."
    )
    doctor_parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    doctor_parser.set_defaults(func=_cmd_doctor)

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Run benchmark workflows.",
    )
    benchmark_parser.add_argument(
        "--full",
        action="store_true",
        help="Compatibility flag; full benchmark is the default mode.",
    )
    benchmark_parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to submission zip (e.g., model/UID_178.zip).",
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
        default=2,
        help="Parallel workers for benchmark.",
    )
    benchmark_parser.add_argument(
        "--log-out",
        type=Path,
        default=None,
        help=f"Output benchmark log path (default in script: {DEFAULT_BENCH_LOG}).",
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
    model_verify_parser.add_argument("--json", action="store_true", help="Emit JSON output.")
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
    model_test_parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    model_test_parser.set_defaults(func=_cmd_model_test)

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
    report_parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    report_parser.set_defaults(func=_cmd_report)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
