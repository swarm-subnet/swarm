#!/usr/bin/env python3
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

"""Does one model on one seed score the same on two different machines?

The crown moves on a margin near +0.008. If the same flight scores differently on two
validators, the board is ordered partly by which host drew the seed, and no amount of
seed sharing fixes that. This answers the question with numbers instead of opinion.

Run it on every machine, then compare the reports:

    python validator/scripts/cross_machine_check.py run --out mybox.json
    python validator/scripts/cross_machine_check.py compare mybox.json ownervali.json ...

It reruns the tracked default model over a frozen seed list, so every machine flies
exactly the same missions. Nothing is downloaded, no wallet or backend is needed.

Read the comparison this way:

* identical everywhere: the simulation is reproducible across hosts, and a shared image
  would lock it in.
* tiny differences, far below 0.001: floating point noise, harmless against the margin.
* differences at the second decimal, or success flipping to failure: the score depends on
  the host. A shared Docker image pins the libraries but not the CPU, so it would only
  fix this if the divergence comes from the libraries rather than the processor. The
  environment block in each report is what tells the two apart.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = REPO_ROOT / "validator" / "tests" / "default_model" / "default_model.zip"
DEFAULT_SEED_FILE = (
    REPO_ROOT / "validator" / "tests" / "fixtures"
    / "benchmark_default_model_fixed_100_seeds_v1.json"
)
# Any difference at or above this is far past floating-point noise and moves a crown.
MEANINGFUL_DIFFERENCE = 0.001


def _cpu_model() -> str:
    """The processor's marketing name, which is what actually differs between hosts."""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _cpu_flags() -> List[str]:
    """The vector instruction sets present, because they decide how float maths is dispatched."""
    interesting = ("avx", "avx2", "avx512f", "fma", "sse4_2")
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("flags"):
                have = set(line.split(":", 1)[1].split())
                return [flag for flag in interesting if flag in have]
    except OSError:
        pass
    return []


def _package_version(name: str) -> str:
    """An installed package's version, or a marker when it is absent."""
    try:
        module = __import__(name)
    except Exception:
        return "absent"
    return str(getattr(module, "__version__", "unknown"))


RUNNER_IMAGE = "swarm_evaluator_base:latest"


def _docker(*args: str, timeout: int = 900) -> subprocess.CompletedProcess:
    """Run a docker command and hand back the finished process."""
    return subprocess.run(
        ["docker", *args], capture_output=True, text=True, timeout=timeout,
    )


def _runner_image_id() -> str:
    """The runner image's full content id.

    Untruncated on purpose: this is the evidence that two machines ran the same image,
    which is the entire question. A short id is not proof.
    """
    try:
        result = _docker("images", "--no-trunc", "-q", RUNNER_IMAGE, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return "docker unavailable"
    return result.stdout.strip() or "not built"


def _prepare_image(image_ref: Optional[str], image_tar: Optional[str]) -> None:
    """Put one agreed image on this machine, so every host flies inside the same one.

    Without this each machine builds its own image from the same Dockerfile, and that
    Dockerfile is not reproducible: the base tag moves, apt is unpinned, and numpy is a
    range. Comparing those builds answers a different question from the one being asked.

    The validator skips its rebuild when the image's swarm.code_hash label matches the
    hash of the checkout, and that hash is the same on every machine, so a correctly
    labelled image is simply adopted.
    """
    if image_tar:
        print(f"Loading the agreed image from {image_tar}")
        loaded = _docker("load", "-i", image_tar)
        if loaded.returncode != 0:
            raise RuntimeError(f"docker load failed: {loaded.stderr.strip()}")
        ref = image_ref
        if not ref:
            match = [w for w in loaded.stdout.split() if ":" in w and "/" in w or ":" in w]
            ref = match[-1] if match else None
        if not ref:
            raise RuntimeError("could not tell which image the tarball loaded")
    elif image_ref:
        print(f"Pulling the agreed image {image_ref}")
        pulled = _docker("pull", image_ref)
        if pulled.returncode != 0:
            raise RuntimeError(f"docker pull failed: {pulled.stderr.strip()}")
        ref = image_ref
    else:
        return

    tagged = _docker("tag", ref, RUNNER_IMAGE, timeout=60)
    if tagged.returncode != 0:
        raise RuntimeError(f"docker tag failed: {tagged.stderr.strip()}")
    print(f"Runner image pinned to {ref} ({_runner_image_id()[:24]}...)")


def _environment() -> Dict[str, Any]:
    """Everything about this host that could plausibly move a score."""
    return {
        "hostname": platform.node(),
        "cpu": _cpu_model(),
        "cpu_flags": _cpu_flags(),
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": _package_version("numpy"),
        "pybullet": _package_version("pybullet"),
        "torch": _package_version("torch"),
        "runner_image_id": _runner_image_id(),
        "thread_env": {
            name: os.environ.get(name, "unset")
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }


def _seed_rows(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every flown seed as one row, ordered so two machines produce comparable lists."""
    rows: List[Dict[str, Any]] = []
    for group, entries in (summary.get("group_results") or {}).items():
        for entry in entries:
            rows.append({
                "group": group,
                "seed": int(entry["seed"]),
                "challenge_type": int(entry["challenge_type"]),
                "score": float(entry["score"]),
                "success": bool(entry.get("success")),
                "execution_status": entry.get("execution_status"),
            })
    rows.sort(key=lambda row: (row["group"], row["seed"]))
    return rows


def _digest(rows: List[Dict[str, Any]], places: Optional[int]) -> str:
    """A fingerprint of every score, exact when ``places`` is None and rounded otherwise.

    Two fingerprints are reported because they answer different questions: the exact one
    says whether the flights were bit-identical, the rounded one says whether the
    differences are large enough to matter.
    """
    hasher = hashlib.sha256()
    for row in rows:
        score = row["score"] if places is None else round(row["score"], places)
        hasher.update(f"{row['group']}|{row['seed']}|{score!r}\n".encode())
    return hasher.hexdigest()[:16]


def _as_model_archive(model: Path) -> Path:
    """The model as a zip the benchmark accepts, packing a folder when one is given.

    A published champion lands on disk as a folder of source and weights, and that is the
    model worth testing: its crown is the one the margin decides.
    """
    if model.is_file():
        return model
    if not model.is_dir():
        raise FileNotFoundError(model)
    archive = Path(tempfile.mkdtemp()) / "submission.zip"
    root = model / "model" if (model / "model").is_dir() else model
    # zipfile rather than the zip command: the machines running this are validator hosts
    # and laptops, and a missing zip binary should not be what stops the check.
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(root).as_posix())
    return archive


def _run_benchmark(
    model: Path, seed_file: Path, workers: int, limit: Optional[int], family_id: str,
) -> Dict[str, Any]:
    """Fly the model over the seed file for one family and return the benchmark's summary."""
    from swarm.benchmark.engine import main as benchmark_main

    model = _as_model_archive(model)
    if limit is not None:
        seed_file = _trimmed_seed_file(seed_file, limit)

    with tempfile.TemporaryDirectory() as tmp:
        summary_path = Path(tmp) / "summary.json"
        benchmark_main([
            "--model", str(model),
            "--family-id", family_id,
            "--workers", str(workers),
            "--seed-file", str(seed_file),
            "--summary-json-out", str(summary_path),
            "--relax-timeouts",
        ])
        return json.loads(summary_path.read_text())


def _trimmed_seed_file(seed_file: Path, limit: int) -> Path:
    """A shorter copy of the seed file, taking seeds evenly from every map group.

    Every machine must fly the same subset, so the trim is by position rather than by
    anything host-specific.
    """
    groups = json.loads(seed_file.read_text())
    per_group = max(1, limit // max(1, len(groups)))
    trimmed = {name: list(seeds)[:per_group] for name, seeds in groups.items()}
    out = Path(tempfile.mkdtemp()) / f"seeds_{limit}.json"
    out.write_text(json.dumps(trimmed, indent=2))
    return out


def _report(rows: List[Dict[str, Any]], label: str, family_id: str, model: str) -> Dict[str, Any]:
    """The full record for one machine: what it flew, what it scored, and on what."""
    scores = [row["score"] for row in rows]
    return {
        "label": label,
        "family_id": family_id,
        "model": model,
        "environment": _environment(),
        "seed_count": len(rows),
        "mean_score": sum(scores) / len(scores) if scores else 0.0,
        "success_count": sum(1 for row in rows if row["success"]),
        "digest_exact": _digest(rows, None),
        "digest_3dp": _digest(rows, 3),
        "seeds": rows,
    }


def _command_run(args: argparse.Namespace) -> int:
    """Fly the frozen seeds on this machine and write its report."""
    model, seed_file = Path(args.model), Path(args.seed_file)
    if not model.exists():
        print(f"missing model: {model}", file=sys.stderr)
        return 2
    if not seed_file.is_file():
        print(f"missing seed file: {seed_file}", file=sys.stderr)
        return 2

    try:
        _prepare_image(args.image, args.image_tar)
    except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
        print(f"could not pin the runner image: {exc}", file=sys.stderr)
        return 2

    print(f"Flying {args.limit or 'all'} {args.family_id} seeds on {platform.node()} ({_cpu_model()})")
    summary = _run_benchmark(model, seed_file, args.workers, args.limit, args.family_id)
    rows = _seed_rows(summary)
    if not rows:
        print("the benchmark returned no seeds", file=sys.stderr)
        return 2

    report = _report(rows, args.label or platform.node(), args.family_id, Path(args.model).name)
    Path(args.out).write_text(json.dumps(report, indent=2))

    print(f"\nseeds        {report['seed_count']}")
    print(f"mean score   {report['mean_score']:.6f}")
    print(f"successes    {report['success_count']}")
    print(f"digest       {report['digest_exact']}  (exact)")
    print(f"digest       {report['digest_3dp']}  (3 decimal places)")
    print(f"\nwritten to {args.out}. Run compare with the reports from the other machines.")
    return 0


def _command_compare(args: argparse.Namespace) -> int:
    """Put two or more machines' reports side by side and say whether they agree."""
    reports = [json.loads(Path(p).read_text()) for p in args.reports]
    if len(reports) < 2:
        print("give at least two reports", file=sys.stderr)
        return 2
    families = {r.get("family_id") for r in reports}
    if len(families) > 1:
        print(f"these reports are not comparable, they fly different families: {families}",
              file=sys.stderr)
        return 2

    print("MACHINES\n")
    for report in reports:
        env = report["environment"]
        print(f"  {report['label']}")
        print(f"    cpu          {env['cpu']}")
        print(f"    flags        {','.join(env['cpu_flags']) or 'unknown'}")
        print(f"    numpy        {env['numpy']}   pybullet {env['pybullet']}")
        print(f"    runner image {env['runner_image_id']}")
        print(f"    family       {report.get('family_id', '?')}   model {report.get('model', '?')}")
        print(f"    mean         {report['mean_score']:.6f}   successes {report['success_count']}")
        print(f"    digest       {report['digest_exact']} exact / {report['digest_3dp']} 3dp\n")

    images = {r["environment"].get("runner_image_id", "?") for r in reports}
    same_image = len(images) == 1 and "not built" not in images

    if not same_image:
        print("THESE MACHINES DID NOT RUN THE SAME IMAGE\n")
        for report in reports:
            print(f"  {report['label']:<16} {report['environment'].get('runner_image_id', '?')[:32]}")
        print()
        print("So this run cannot answer the question that was asked. It compares the images")
        print("each machine happened to build, which is what already happens in production.")
        print("Rerun every machine with the same --image or --image-tar, then compare again.")
        print("Whatever the scores below say, they do not separate the image from the CPU.\n")
    else:
        print(f"ALL MACHINES RAN THE SAME IMAGE: {images.pop()[:32]}...\n")

    if len({r["digest_exact"] for r in reports}) == 1:
        print("VERDICT: every machine produced byte-identical scores.")
        if same_image:
            print("One image gives one score on every host. Shipping a central image fixes this.")
        else:
            print("The simulation reproduces across these hosts even before pinning the image.")
        return 0

    base, *others = reports
    base_seeds = {(row["group"], row["seed"]): row for row in base["seeds"]}
    worst_overall = 0.0
    print("DIFFERENCES against", base["label"], "\n")

    for report in others:
        differing = 0
        flips = 0
        meaningful = 0
        worst = 0.0
        compared = 0
        for row in report["seeds"]:
            other = base_seeds.get((row["group"], row["seed"]))
            if other is None:
                continue
            compared += 1
            gap = abs(row["score"] - other["score"])
            if gap > 0:
                differing += 1
            if gap >= MEANINGFUL_DIFFERENCE:
                meaningful += 1
            if row["success"] != other["success"]:
                flips += 1
            worst = max(worst, gap)
        worst_overall = max(worst_overall, worst)
        share = 100.0 * differing / compared if compared else 0.0
        print(f"  {report['label']}: {compared} seeds compared")
        print(f"    differ at all          {differing} ({share:.0f}%)")
        print(f"    differ by >= {MEANINGFUL_DIFFERENCE}     {meaningful}")
        print(f"    success/failure flips  {flips}")
        print(f"    largest difference     {worst:.6f}")
        print(f"    mean gap               {abs(report['mean_score'] - base['mean_score']):.6f}\n")

    print("VERDICT:", end=" ")
    if worst_overall < MEANINGFUL_DIFFERENCE:
        print("the scores differ, but only as floating-point noise.")
        print("Nothing here is large enough to move a crown.")
    elif same_image:
        print("one image, different scores. The image is not the cause.")
        print("Every machine ran identical libraries inside an identical image and still")
        print("disagreed, so what is left is the processor. Shipping a central image will")
        print("not fix this on its own; the fix has to tolerate the difference instead, by")
        print("comparing a challenger against the champion on the seeds they actually shared.")
    else:
        print("the scores differ, but the machines ran different images.")
        print("Pin the image with --image or --image-tar and rerun before concluding")
        print("anything: this result cannot tell the image apart from the processor.")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Parse the command line and dispatch to run or compare."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="fly the frozen seeds on this machine")
    run.add_argument("--out", default="cross_machine_report.json", help="where to write the report")
    run.add_argument("--label", default=None, help="a name for this machine (default: hostname)")
    run.add_argument("--limit", type=int, default=30, help="seeds to fly, 0 for all (default: 30)")
    run.add_argument("--workers", type=int, default=4, help="parallel workers (default: 4)")
    run.add_argument(
        "--family-id", default="cf_search_and_rescue",
        help="challenge family to fly (default: cf_search_and_rescue)",
    )
    run.add_argument(
        "--image", default=None,
        help="pull this image and run inside it, so every machine uses the same one",
    )
    run.add_argument(
        "--image-tar", default=None,
        help="load the agreed image from a docker save tarball instead of pulling it",
    )
    run.add_argument(
        "--model", default=str(DEFAULT_MODEL),
        help="model archive, or a published champion folder which is zipped for you",
    )
    run.add_argument("--seed-file", default=str(DEFAULT_SEED_FILE), help="frozen seed list")
    run.set_defaults(func=_command_run)

    compare = sub.add_parser("compare", help="compare two or more machines' reports")
    compare.add_argument("reports", nargs="+", help="report files written by run")
    compare.set_defaults(func=_command_compare)

    args = parser.parse_args(argv)
    if getattr(args, "limit", None) == 0:
        args.limit = None
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
