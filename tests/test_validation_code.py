import json
from pathlib import Path
from typing import List, Optional

import pytest

pytest.importorskip("pybullet")
pytest.importorskip("gym_pybullet_drones")

from RL.test_secure_RL import secure_ppo_load_weights_only, _run_episode_speed_limit
from swarm.validator.task_gen import random_task
from swarm.constants import SIM_DT, HORIZON_SEC
from swarm.utils.env_factory import make_env


MODEL_PATH = Path("/root/swarm/miner_models_v2/UID_117.zip")
SNAPSHOT_PATH = Path(__file__).with_name("test_validation_code_results.json")
SEEDS = tuple(range(1, 101))
REFERENCE_UID = 117


def _load_reference_model():
    if not MODEL_PATH.exists():
        pytest.skip(f"Reference model not found at {MODEL_PATH}")

    baseline_task = random_task(sim_dt=SIM_DT, horizon=HORIZON_SEC, seed=0)
    env = make_env(baseline_task, gui=False)
    try:
        try:
            model = secure_ppo_load_weights_only(MODEL_PATH, env=env, device="cpu")
        except RuntimeError as exc:
            pytest.skip(f"Unable to load reference model: {exc}")
    finally:
        try:
            env.close()
        except Exception:
            pass

    policy = getattr(model, "policy", None)
    if policy is not None and hasattr(policy, "set_training_mode"):
        policy.set_training_mode(False)
    return model


def _run_reference_suite(model) -> List[dict]:
    results = []
    for seed in SEEDS:
        task = random_task(sim_dt=SIM_DT, horizon=HORIZON_SEC, seed=seed)
        validation, _ = _run_episode_speed_limit(
            task=task,
            uid=REFERENCE_UID,
            model=model,
            gui=False,
        )
        results.append(
            {
                "seed": seed,
                "success": bool(validation.success),
                "time_sec": round(float(validation.time_sec), 6),
                "score": round(float(validation.score), 6),
            }
        )
    return results


def _read_snapshot() -> Optional[List[dict]]:
    if not SNAPSHOT_PATH.exists():
        return None
    return json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))


def _write_snapshot(data: List[dict]) -> None:
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_PATH.write_text(
        json.dumps(data, indent=2, sort_keys=True), encoding="utf-8"
    )


def _format_diff(expected: List[dict], actual: List[dict]) -> str:
    if len(expected) != len(actual):
        return (
            f"Snapshot length mismatch: expected {len(expected)} entries "
            f"but computed {len(actual)}"
        )
    for idx, (exp, act) in enumerate(zip(expected, actual)):
        if exp != act:
            seed = exp.get("seed", idx)
            return (
                f"Snapshot mismatch at index {idx} (seed {seed}): "
                f"expected {exp}, got {act}"
            )
    return "Snapshot mismatch with no detailed diff available."


@pytest.mark.slow
@pytest.mark.integration
def test_validation_code_snapshot():
    model = _load_reference_model()
    current_results = _run_reference_suite(model)

    snapshot = _read_snapshot()
    if snapshot is None:
        _write_snapshot(current_results)
        snapshot = current_results

    assert snapshot == current_results, _format_diff(snapshot, current_results)
