"""Full-loop integration: artifact -> admission -> container -> RPC -> sim -> score.

Runs the committed calibration baseline through the real evaluator: the graph
runner image is built or verified, the sandboxed container is launched, and a
complete episode is simulated and scored over live Cap'n Proto RPC.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess

import pytest

from swarm.constants import SIM_DT

pytestmark = pytest.mark.integration


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        probe = subprocess.run(["docker", "info"], capture_output=True, timeout=10)
    except Exception:
        return False
    return probe.returncode == 0


@pytest.mark.skipif(not _docker_available(), reason="docker daemon unavailable")
def test_full_evaluation_loop_scores_the_baseline_artifact():
    from swarm.validator.calibration import baseline_model_path
    from swarm.validator.docker.docker_evaluator import DockerSecureEvaluator
    from swarm.validator.docker.docker_evaluator_parts.batch import evaluate_seeds_batch
    from swarm.validator.task_gen import task_for_seed_and_type

    evaluator = DockerSecureEvaluator()
    assert DockerSecureEvaluator._base_ready

    task = task_for_seed_and_type(
        SIM_DT, seed=1001, challenge_type=2, family_id="cf_autopilot"
    )
    seed_meta = []
    results = asyncio.run(
        evaluate_seeds_batch(
            evaluator,
            [task],
            uid=0,
            model_path=baseline_model_path(),
            worker_id=0,
            on_seed_complete=lambda meta=None: seed_meta.append(meta),
            is_calibration_run=True,
        )
    )

    assert len(results) == 1
    result = results[0]
    assert seed_meta and seed_meta[-1]["status"] == "seed_done"
    assert seed_meta[-1]["step_idx"] > 0
    assert 0.0 <= float(result.score) <= 1.0
    assert result.failure_reason in ("TIMEOUT", "NONE")
