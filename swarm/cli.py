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
    if args.json:
        payload = [asdict(c) for c in checks]
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
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


def _cmd_benchmark(args: argparse.Namespace) -> int:
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


def _build_visualize_argv(args: argparse.Namespace) -> list[str]:
    argv = ["--type", str(args.type)]
    if args.seed is not None:
        argv.extend(["--seed", str(args.seed)])
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
        from scripts.visualize_map import main as visualize_main

        visualize_main(_build_visualize_argv(args))
        return 0
    except (SystemExit, KeyboardInterrupt):
        return 1
    except Exception as exc:
        print(f"Visualizer failed: {exc}", file=sys.stderr)
        return 1


def _build_video_argv(args: argparse.Namespace) -> list[str]:
    argv = ["--model", str(args.model)]
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
    try:
        from scripts.generate_video import main as video_main

        video_main(_build_video_argv(args))
        return 0
    except (SystemExit, KeyboardInterrupt) as exc:
        return int(exc.code) if isinstance(exc, SystemExit) and isinstance(exc.code, int) else 1
    except Exception as exc:
        print(f"Video generation failed: {exc}", file=sys.stderr)
        return 1


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

    if args.json:
        payload = dict(summary)
        if results_block:
            payload["results_block"] = results_block
        payload["report_source"] = str(input_path)
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
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
            resp = client.get(f"{base_url}/leaderboard?page_size=1")
            if resp.status_code != 200:
                print(f"Failed to fetch leaderboard: HTTP {resp.status_code}", file=sys.stderr)
                return 1

            data = resp.json()
            entries = data.get("entries", [])
            if not entries:
                print("No evaluated models on the leaderboard yet.", file=sys.stderr)
                return 1

            champ = None
            for entry in entries:
                if entry.get("is_champion"):
                    champ = entry
                    break
            if champ is None:
                champ = entries[0]

            uid = champ["uid"]
            score = champ.get("benchmark_score", 0)
            released = champ.get("is_released", False)
            per_type = champ.get("per_type_scores") or {}

            if args.json:
                print(json.dumps(champ, indent=2))
                if not released:
                    return 0

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
    doctor_parser.add_argument("--json", action="store_true", help="Emit JSON output.")
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

    visualize_parser = subparsers.add_parser(
        "visualize",
        help="Open an interactive visualizer for a specific map type and seed.",
    )
    visualize_parser.add_argument(
        "--type",
        type=int,
        required=True,
        choices=[1, 2, 3, 4, 5, 6],
        help="Challenge type (1=City 2=Open 3=Mountain 4=Village 5=Warehouse 6=Forest).",
    )
    visualize_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Map seed. If omitted, a random valid seed is chosen.",
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
        "--seed",
        type=int,
        default=None,
        help="Single seed to replay.",
    )
    video_parser.add_argument(
        "--type",
        type=int,
        default=None,
        choices=[1, 2, 3, 4, 5, 6],
        help="Challenge type for single-seed replay.",
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
        default=os.environ.get("SWARM_BACKEND_API_URL", ""),
        help="Backend API URL (or set SWARM_BACKEND_API_URL env var).",
    )
    champion_parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    champion_parser.set_defaults(func=_cmd_champion)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
