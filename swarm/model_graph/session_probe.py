"""Startup probe: build every session and confirm deterministic inference.

Runs once when a runner boots, before the RPC server reports ready. Catches
models that admit statically but fail to construct a session or produce
non-repeatable output, and warms sessions so the first scored step pays no
extra cost.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import resource
import sys
from pathlib import Path

import numpy as np

from .action import action_digest
from .constants import (
    FAMILY_GRAPH_CONTRACTS,
    SESSION_PROBE_ADDRESS_BYTES,
    SESSION_PROBE_WALL_SEC,
    SWARM_NUM_DRONES_RANGE,
    resolve_shape,
)
from .errors import ModelGraphError, ReasonCode
from .manifest import GraphManifest
from .runner import GraphRunner


def _zero_observation(manifest: GraphManifest, num_drones: int | None) -> dict[str, np.ndarray]:
    contract = FAMILY_GRAPH_CONTRACTS[manifest.family_id]
    observation = {}
    for key, shape in contract["observations"].items():
        observation[key] = np.zeros(resolve_shape(shape, num_drones), dtype=np.float32)
    return observation


def _drone_counts(family_id: str) -> tuple[int | None, ...]:
    from .constants import family_has_swarm_axis

    if family_has_swarm_axis(family_id):
        return (SWARM_NUM_DRONES_RANGE[0], SWARM_NUM_DRONES_RANGE[1])
    return (None,)


def probe_runner(runner: GraphRunner) -> None:
    """Warm every family shape and assert two identical warm-up ticks agree."""
    for num_drones in _drone_counts(runner.manifest.family_id):
        observation = _zero_observation(runner.manifest, num_drones)
        runner.reset(num_drones=num_drones)
        try:
            first = runner.act(observation, num_drones=num_drones)
            runner.reset(num_drones=num_drones)
            second = runner.act(observation, num_drones=num_drones)
        except ModelGraphError:
            raise
        except Exception as exc:
            raise ModelGraphError(ReasonCode.LOAD_FAILED, f"probe inference failed: {exc}") from exc
        if action_digest(first) != action_digest(second):
            raise ModelGraphError(
                ReasonCode.LOAD_FAILED,
                f"non-deterministic warm-up at num_drones={num_drones}",
            )
    runner.reset(num_drones=_drone_counts(runner.manifest.family_id)[0])


def _probe_child(artifact: str, connection) -> None:
    try:
        if os.name == "posix":
            resource.setrlimit(
                resource.RLIMIT_AS,
                (SESSION_PROBE_ADDRESS_BYTES, SESSION_PROBE_ADDRESS_BYTES),
            )
        runner = GraphRunner.from_artifact(Path(artifact))
        probe_runner(runner)
        connection.send({"ok": True})
    except BaseException as exc:
        if isinstance(exc, ModelGraphError):
            record = exc.to_record()
        else:
            record = ModelGraphError(ReasonCode.LOAD_FAILED, str(exc)).to_record()
        connection.send({"ok": False, **record})
    finally:
        connection.close()


def probe_artifact_subprocess(
    artifact: Path, timeout: float = SESSION_PROBE_WALL_SEC
) -> None:
    """Probe an artifact under an address-space and wall-time boundary."""
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_probe_child, args=(str(Path(artifact)), child))
    process.start()
    child.close()
    try:
        if not parent.poll(timeout):
            raise ModelGraphError(
                ReasonCode.LOAD_FAILED,
                f"session probe exceeded {timeout:.1f}s",
            )
        try:
            record = parent.recv()
        except (EOFError, OSError) as exc:
            raise ModelGraphError(
                ReasonCode.LOAD_FAILED,
                "session probe terminated without a result",
            ) from exc
    finally:
        parent.close()
        process.join(2)
        if process.is_alive():
            process.kill()
            process.join()
    if not record.get("ok"):
        reason = ReasonCode(record.get("reason_code", ReasonCode.LOAD_FAILED.value))
        raise ModelGraphError(
            reason,
            record.get("detail", "session probe failed"),
            node_id=record.get("node_id"),
            model_id=record.get("model_id"),
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe an admitted model-graph artifact")
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args(argv)
    try:
        runner = GraphRunner.from_artifact(args.artifact)
        probe_runner(runner)
    except ModelGraphError as exc:
        print(json.dumps(exc.to_record(), sort_keys=True), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
