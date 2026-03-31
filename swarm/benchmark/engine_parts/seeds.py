from __future__ import annotations

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


def _infer_uid_from_model_path(model_path: Path) -> Optional[int]:
    for candidate in (model_path.stem, model_path.name):
        match = _UID_RE.search(candidate)
        if match:
            try:
                return int(match.group(1))
            except Exception:
                continue
    return None


def _normalize_type_seeds(raw: Any) -> Dict[str, List[int]]:
    if not isinstance(raw, dict):
        raise ValueError("Seed file must contain a JSON object mapping benchmark groups to seed lists.")

    normalized: Dict[str, List[int]] = {}
    missing = [group for group in BENCH_GROUP_ORDER if group not in raw]
    if missing:
        raise ValueError(f"Seed file missing groups: {', '.join(missing)}")

    for group in BENCH_GROUP_ORDER:
        values = raw.get(group)
        if not isinstance(values, list) or not values:
            raise ValueError(f"Seed group {group} must be a non-empty list.")
        normalized[group] = [int(seed) for seed in values]

    return normalized


def _load_type_seeds(seed_file: Path) -> Dict[str, List[int]]:
    return _normalize_type_seeds(json.loads(seed_file.read_text()))


def _save_type_seeds(seed_file: Path, type_seeds: Dict[str, List[int]]) -> None:
    seed_file.parent.mkdir(parents=True, exist_ok=True)
    seed_file.write_text(json.dumps(type_seeds, indent=2, sort_keys=True))


def _infer_bench_group(challenge_type: int, seed: int) -> Optional[str]:
    _ = seed
    if challenge_type == 1:
        return "type1_city"
    if challenge_type == 2:
        return "type2_open"
    if challenge_type == 3:
        return "type3_mountain"
    if challenge_type == 4:
        return "type4_village"
    if challenge_type == 5:
        return "type5_warehouse"
    if challenge_type == 6:
        return "type6_forest"
    return None


def _find_seeds(seeds_per_group: int) -> Dict[str, List[int]]:
    from swarm.constants import SIM_DT
    from swarm.validator.task_gen import random_task

    groups: Dict[str, List[int]] = {g: [] for g in BENCH_GROUP_ORDER}

    seed = random.randint(100000, 900000)
    max_search = seed + 500000
    while seed < max_search:
        task = random_task(sim_dt=SIM_DT, seed=seed)
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
