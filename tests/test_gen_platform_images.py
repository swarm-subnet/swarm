from __future__ import annotations

from pathlib import Path

from scripts import gen_platform_images as image_mod


def test_challenge_names_cover_all_map_types() -> None:
    assert image_mod.CHALLENGE_NAMES == {
        1: "city",
        2: "open",
        3: "mountain",
        4: "village",
        5: "warehouse",
        6: "forest",
    }


def test_pick_seeds_classifies_explicit_seed_list() -> None:
    explicit = [323517, 323521, 323527, 323530, 323518, 431623]
    mapping = {
        323517: 1,
        323521: 2,
        323527: 3,
        323530: 4,
        323518: 5,
        431623: 6,
    }

    original = image_mod._choose_challenge_type
    image_mod._choose_challenge_type = lambda seed: mapping[seed]
    try:
        picked = image_mod._pick_seeds(
            per_type=1,
            max_scan=100,
            sim_dt=1 / 240.0,
            explicit_seeds=explicit,
        )
    finally:
        image_mod._choose_challenge_type = original

    assert picked[1] == [323517]
    assert picked[2] == [323521]
    assert picked[3] == [323527]
    assert picked[4] == [323530]
    assert picked[5] == [323518]
    assert picked[6] == [431623]


def test_local_ansible_temp_uses_writable_tmp(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "ansible_local"
    monkeypatch.setenv("ANSIBLE_LOCAL_TEMP", str(target))

    image_mod._ensure_local_ansible_temp()

    assert target.is_dir()


def test_pick_seeds_accepts_explicit_type_seed_specs() -> None:
    picked = image_mod._pick_seeds(
        per_type=1,
        max_scan=10,
        sim_dt=1 / 240.0,
        explicit_seed_specs=["1:323517", "6:431623"],
    )

    assert picked[1] == [323517]
    assert picked[6] == [431623]
    assert picked[2] == []
