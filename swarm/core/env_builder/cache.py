from __future__ import annotations

from . import _shared as shared

STATIC_WORLD_CACHE_VERSION = 2


def set_map_cache_epoch(epoch: int) -> None:
    shared.current_epoch_number = epoch


def cleanup_old_epoch_cache(keep_epoch: int) -> None:
    base = shared.MAP_CACHE_DIR / shared.BENCHMARK_VERSION
    if not base.exists():
        return
    for child in base.iterdir():
        if child.is_dir() and child.name.startswith("epoch_"):
            try:
                ep = int(child.name.split("_", 1)[1])
                if ep < keep_epoch:
                    shared.shutil.rmtree(child, ignore_errors=True)
            except (ValueError, IndexError):
                continue


def _normalize_xy(
    point: shared.Optional[shared.Tuple[float, float, float]]
) -> shared.Optional[shared.Tuple[float, float]]:
    if point is None:
        return None
    return (round(float(point[0]), 6), round(float(point[1]), 6))


def _static_world_cache_file(
    seed: int,
    challenge_type: int,
    start: shared.Optional[shared.Tuple[float, float, float]],
    goal: shared.Optional[shared.Tuple[float, float, float]],
) -> shared.Path:
    payload = {
        "static_world_cache_version": STATIC_WORLD_CACHE_VERSION,
        "benchmark_version": shared.BENCHMARK_VERSION,
        "seed": int(seed),
        "challenge_type": int(challenge_type),
        "start_xy": _normalize_xy(start),
        "goal_xy": _normalize_xy(goal),
    }
    key_json = shared.json.dumps(payload, sort_keys=True, separators=(",", ":"))
    key_hash = shared.hashlib.sha256(key_json.encode("utf-8")).hexdigest()
    epoch_dir = (
        f"epoch_{shared.current_epoch_number}"
        if shared.current_epoch_number is not None
        else "no_epoch"
    )
    return (
        shared.MAP_CACHE_DIR
        / shared.BENCHMARK_VERSION
        / epoch_dir
        / f"type{challenge_type}"
        / f"{key_hash}.bullet"
    )


def _static_world_cache_meta_file(
    seed: int,
    challenge_type: int,
    start: shared.Optional[shared.Tuple[float, float, float]],
    goal: shared.Optional[shared.Tuple[float, float, float]],
) -> shared.Path:
    return _static_world_cache_file(seed, challenge_type, start, goal).with_suffix(
        ".json"
    )


def _build_static_world_cache_meta(
    cli: int,
    *,
    start: shared.Optional[shared.Tuple[float, float, float]],
    goal: shared.Optional[shared.Tuple[float, float, float]],
    challenge_type: int,
    base_body_count: int = 0,
) -> dict:
    from .generation import _raycast_surface_z

    total_bodies = int(shared.p.getNumBodies(physicsClientId=cli))
    map_body_count = max(0, total_bodies - int(base_body_count))
    meta = {
        "body_count": total_bodies,
        "map_body_count": map_body_count,
    }

    if challenge_type in (3, 4):
        if start is not None:
            meta["start_surface_z"] = float(
                _raycast_surface_z(cli, float(start[0]), float(start[1]))
            )
        if goal is not None:
            meta["goal_surface_z"] = float(
                _raycast_surface_z(cli, float(goal[0]), float(goal[1]))
            )

    return meta


def _write_static_world_cache_meta(meta_file: shared.Path, meta: dict) -> None:
    tmp_meta_file = meta_file.with_suffix(".json.tmp")
    tmp_meta_file.unlink(missing_ok=True)
    tmp_meta_file.write_text(
        shared.json.dumps(meta, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    tmp_meta_file.replace(meta_file)


def _read_static_world_cache_meta(meta_file: shared.Path) -> shared.Optional[dict]:
    if not meta_file.exists():
        return None
    try:
        return shared.json.loads(meta_file.read_text(encoding="utf-8"))
    except Exception:
        return None


def _invalidate_static_world_cache(cache_file: shared.Path, meta_file: shared.Path) -> None:
    cache_file.unlink(missing_ok=True)
    meta_file.unlink(missing_ok=True)


def prebuild_static_world_cache(
    seed: int,
    challenge_type: int,
    *,
    start: shared.Optional[shared.Tuple[float, float, float]],
    goal: shared.Optional[shared.Tuple[float, float, float]],
) -> shared.Path:
    from .generation import _build_static_world

    cache_file = _static_world_cache_file(seed, challenge_type, start, goal)
    if challenge_type == 2:
        return cache_file
    meta_file = _static_world_cache_meta_file(seed, challenge_type, start, goal)
    if not shared.MAP_CACHE_ENABLED:
        return cache_file
    if cache_file.exists() and meta_file.exists():
        return cache_file
    if cache_file.exists() or meta_file.exists():
        _invalidate_static_world_cache(cache_file, meta_file)

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = cache_file.with_suffix(".tmp")
    tmp_file.unlink(missing_ok=True)

    cli = shared.p.connect(shared.p.DIRECT)
    try:
        base_body_count = shared.p.getNumBodies(physicsClientId=cli)
        _build_static_world(
            seed=seed,
            cli=cli,
            start=start,
            goal=goal,
            challenge_type=challenge_type,
        )
        shared.p.saveBullet(str(tmp_file), physicsClientId=cli)
        meta = _build_static_world_cache_meta(
            cli,
            start=start,
            goal=goal,
            challenge_type=challenge_type,
            base_body_count=base_body_count,
        )
    except Exception:
        tmp_file.unlink(missing_ok=True)
        _invalidate_static_world_cache(cache_file, meta_file)
        raise
    finally:
        shared.p.disconnect(cli)

    tmp_file.replace(cache_file)
    try:
        _write_static_world_cache_meta(meta_file, meta)
    except Exception:
        _invalidate_static_world_cache(cache_file, meta_file)
    return cache_file


def _save_static_world_cache_from_client(
    seed: int,
    cli: int,
    *,
    start: shared.Optional[shared.Tuple[float, float, float]],
    goal: shared.Optional[shared.Tuple[float, float, float]],
    challenge_type: int,
    base_body_count: int = 0,
) -> None:
    if challenge_type == 2:
        return

    cache_file = _static_world_cache_file(seed, challenge_type, start, goal)
    meta_file = _static_world_cache_meta_file(seed, challenge_type, start, goal)
    if cache_file.exists() and meta_file.exists():
        return
    if cache_file.exists() or meta_file.exists():
        _invalidate_static_world_cache(cache_file, meta_file)

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = cache_file.with_suffix(".tmp")
    tmp_file.unlink(missing_ok=True)

    try:
        shared.p.saveBullet(str(tmp_file), physicsClientId=cli)
        tmp_file.replace(cache_file)
        meta = _build_static_world_cache_meta(
            cli,
            start=start,
            goal=goal,
            challenge_type=challenge_type,
            base_body_count=base_body_count,
        )
        _write_static_world_cache_meta(meta_file, meta)
    except Exception:
        tmp_file.unlink(missing_ok=True)
        _invalidate_static_world_cache(cache_file, meta_file)


def _try_load_static_world_cache(
    seed: int,
    cli: int,
    *,
    start: shared.Optional[shared.Tuple[float, float, float]],
    goal: shared.Optional[shared.Tuple[float, float, float]],
    challenge_type: int,
) -> bool:
    cache_file = _static_world_cache_file(seed, challenge_type, start, goal)
    meta_file = _static_world_cache_meta_file(seed, challenge_type, start, goal)
    if challenge_type == 2:
        return False
    if not cache_file.exists() or not meta_file.exists():
        _invalidate_static_world_cache(cache_file, meta_file)
        return False

    expected_meta = _read_static_world_cache_meta(meta_file)
    if not isinstance(expected_meta, dict):
        _invalidate_static_world_cache(cache_file, meta_file)
        return False

    before_bodies = shared.p.getNumBodies(physicsClientId=cli)

    try:
        shared.p.loadBullet(str(cache_file), physicsClientId=cli)
        after_bodies = shared.p.getNumBodies(physicsClientId=cli)

        actual_meta = _build_static_world_cache_meta(
            cli,
            start=start,
            goal=goal,
            challenge_type=challenge_type,
            base_body_count=before_bodies,
        )

        loaded_map_bodies = max(0, int(after_bodies - before_bodies))
        expected_map_bodies = int(
            expected_meta.get("map_body_count", expected_meta.get("body_count", -2))
        )

        if challenge_type in (1, 3, 4, 6) and loaded_map_bodies <= 0:
            _invalidate_static_world_cache(cache_file, meta_file)
            return False

        if int(actual_meta.get("map_body_count", -1)) != expected_map_bodies:
            _invalidate_static_world_cache(cache_file, meta_file)
            return False

        if challenge_type in (3, 4):
            tolerance = 1.0
            expected_start_surface = expected_meta.get("start_surface_z")
            actual_start_surface = actual_meta.get("start_surface_z")
            if expected_start_surface is None or actual_start_surface is None:
                _invalidate_static_world_cache(cache_file, meta_file)
                return False
            if abs(float(actual_start_surface) - float(expected_start_surface)) > tolerance:
                _invalidate_static_world_cache(cache_file, meta_file)
                return False

            expected_goal_surface = expected_meta.get("goal_surface_z")
            actual_goal_surface = actual_meta.get("goal_surface_z")
            if expected_goal_surface is None or actual_goal_surface is None:
                _invalidate_static_world_cache(cache_file, meta_file)
                return False
            if abs(float(actual_goal_surface) - float(expected_goal_surface)) > tolerance:
                _invalidate_static_world_cache(cache_file, meta_file)
                return False

        return True
    except Exception:
        _invalidate_static_world_cache(cache_file, meta_file)
        return False
