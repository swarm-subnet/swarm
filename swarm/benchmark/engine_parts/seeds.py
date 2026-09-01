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

from swarm.challenge_families import DEFAULT_RUNTIME_FAMILY_ID, build_random_task
from swarm.domain_model import (
    BENCHMARK_GROUP_TO_ENVIRONMENT_TYPE,
    CHALLENGE_FAMILY_TO_ENVIRONMENT_TYPES,
    CHALLENGE_TYPE_TO_BENCHMARK_GROUP,
)

from ._shared import (
    _UID_RE,
    BENCH_GROUP_ORDER,
    Any,
    Dict,
    List,
    Optional,
    Path,
    json,
    random,
)


def family_bench_groups(family_id: str) -> List[str]:
    environment_types = set(CHALLENGE_FAMILY_TO_ENVIRONMENT_TYPES[family_id])
    return [
        group
        for group in BENCH_GROUP_ORDER
        if BENCHMARK_GROUP_TO_ENVIRONMENT_TYPE[group] in environment_types
    ]


def _infer_uid_from_model_path(model_path: Path) -> Optional[int]:
    for candidate in (model_path.stem, model_path.name):
        match = _UID_RE.search(candidate)
        if match:
            try:
                return int(match.group(1))
            except Exception:
                continue
    return None


def _normalize_type_seeds(raw: Any, *, family_id: str) -> Dict[str, List[int]]:
    if not isinstance(raw, dict):
        raise ValueError(
            "Seed file must contain a JSON object mapping benchmark groups to seed lists."
        )

    if "type_seeds" in raw:
        envelope_family_id = str(raw.get("family_id") or DEFAULT_RUNTIME_FAMILY_ID)
        if envelope_family_id != family_id:
            raise ValueError(
                f"Seed file family_id mismatch: expected {family_id}, got {envelope_family_id}"
            )
        raw = raw.get("type_seeds")
        if not isinstance(raw, dict):
            raise ValueError("Seed file type_seeds must be a JSON object.")

    family_groups = family_bench_groups(family_id)
    normalized: Dict[str, List[int]] = {}
    missing = [group for group in family_groups if group not in raw]
    if missing:
        raise ValueError(f"Seed file missing groups: {', '.join(missing)}")

    for group in family_groups:
        values = raw.get(group)
        if not isinstance(values, list) or not values:
            raise ValueError(f"Seed group {group} must be a non-empty list.")
        normalized[group] = [int(seed) for seed in values]

    return normalized


def _load_type_seeds(
    seed_file: Path,
    *,
    family_id: str = DEFAULT_RUNTIME_FAMILY_ID,
) -> Dict[str, List[int]]:
    return _normalize_type_seeds(json.loads(seed_file.read_text()), family_id=family_id)


def _save_type_seeds(
    seed_file: Path,
    type_seeds: Dict[str, List[int]],
    *,
    family_id: str = DEFAULT_RUNTIME_FAMILY_ID,
) -> None:
    seed_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "challenge_family_seed_file.v1",
        "family_id": family_id,
        "type_seeds": type_seeds,
    }
    seed_file.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _infer_bench_group(challenge_type: int, seed: int) -> Optional[str]:
    _ = seed
    return CHALLENGE_TYPE_TO_BENCHMARK_GROUP.get(challenge_type)


def _find_seeds(
    seeds_per_group: int,
    *,
    family_id: str = DEFAULT_RUNTIME_FAMILY_ID,
) -> Dict[str, List[int]]:
    from swarm.constants import SIM_DT

    groups: Dict[str, List[int]] = {g: [] for g in family_bench_groups(family_id)}

    seed = random.randint(100000, 900000)
    max_search = seed + 500000
    while seed < max_search:
        task = build_random_task(
            sim_dt=SIM_DT,
            seed=seed,
            family_id=family_id,
        )
        group = _infer_bench_group(int(task.challenge_type), seed)
        if group is not None and group in groups and len(groups[group]) < seeds_per_group:
            groups[group].append(seed)

        if all(len(v) >= seeds_per_group for v in groups.values()):
            break
        seed += 1

    missing = [g for g, seeds in groups.items() if len(seeds) < seeds_per_group]
    if missing:
        raise RuntimeError(
            "Could not find enough seeds for groups: "
            + ", ".join(f"{g} ({len(groups[g])}/{seeds_per_group})" for g in missing)
        )

    return groups


def _batch_indices(total_tasks: int) -> List[List[int]]:
    if total_tasks <= 0:
        return []
    return [[index] for index in range(total_tasks)]
