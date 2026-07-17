"""Generate the deterministic graph-format hardware calibration artifact."""

from __future__ import annotations

import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

from swarm.model_graph import EXECUTION_PROFILE_ID, RUNNER_ABI, profile_digest
from swarm.model_graph_template.generate_template import generate


def generate_baseline(output_dir: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="swarm_baseline_") as tmp:
        source = Path(tmp)
        generate(source)
        artifact = output_dir / "baseline_model.zip"
        with zipfile.ZipFile(artifact, "w", zipfile.ZIP_STORED) as archive:
            for relative in ("manifest.json", "models/policy.onnx"):
                info = zipfile.ZipInfo(relative, date_time=(2026, 7, 10, 0, 0, 0))
                info.external_attr = 0o100644 << 16
                archive.writestr(info, (source / relative).read_bytes())

    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = {
        "calibration_version": "swarm-model-graph-ref-v2",
        "scope": "all_challenge_families",
        "interface_version": "model_graph.v1",
        "execution_profile": EXECUTION_PROFILE_ID,
        "execution_profile_digest": profile_digest(),
        "runner_abi": RUNNER_ABI,
        "baseline_model": {
            "artifact": artifact.name,
            "sha256": digest,
            "run_as_family_id": "cf_autopilot",
            "run_as_challenge_type": 2,
        },
        "owner_compute_p90_ms": 17.6,
        "owner_overhead_ms": 2.7,
        "measurement": {
            "sample_seeds": [1001],
            "sample_horizon_sec": 30.0,
            "sample_count": 1499,
            "warmup_steps": 1,
            "docker_worker_cpus": "2",
        },
    }
    (output_dir / "baseline_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    generate_baseline(Path(__file__).resolve().parent)
