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

"""World seeds: the wide secret numbers Swarm Sentinel and Swarm Test Field are flown on, so a model cannot find
the seed it is flying by trying every list seed."""
from __future__ import annotations

import json

import numpy as np
import pytest

from swarm.challenge_families import require_runtime_family
from swarm.challenge_families.solar_patrol import flight_limit, sensor_noise, theft, wind
from swarm.challenge_families.solar_patrol import seed_checks as solar_seed_checks
from swarm.challenge_families.test_field import people
from swarm.challenge_families.test_field import seed_checks as field_seed_checks
from swarm.core.maps.solar import builder
from swarm.validator.seed_scheme import derive_seeds, derive_world_seed, derive_world_seeds, key_fingerprint

KEY = "11" * 32
OTHER_KEY = "22" * 32
EPOCH = 22
SENTINEL = "cf_solar_patrol"
TEST_FIELD = "cf_test_field"
PUBLIC_FAMILY = "cf_autopilot"
# Below this validator's own version, so the manager treats itself as on the shared scheme.
LIVE_FROM = "5.0.0"

# Every validator must build the same worlds from one key, so the encoding is pinned like the list's own.
KNOWN_WORLD_SEEDS = {
    SENTINEL: [195400129957381213844017116229595907684, 42541674157329983098622528642135832512],
    TEST_FIELD: [184103304775973437905105808848360527071, 139582012338641331408293332398441288338],
}
# Two world seeds that differ only above their lowest 32 bits: a draw that cut the seed would treat them as one.
LOW_BITS = 0x9E3779B9
WIDE_A = (0x1234_5678_9ABC_DEF0_1122_3344 << 32) | LOW_BITS
WIDE_B = (0x0FED_CBA9_8765_4321_5566_7788 << 32) | LOW_BITS


@pytest.fixture
def manager(reload_module, monkeypatch, tmp_path):
    """A seed manager rooted in tmp_path, on the shared scheme, holding the test epoch's key."""
    module = reload_module("swarm.validator.seed_manager")
    state_dir = tmp_path / "state"
    monkeypatch.setattr(module, "STATE_DIR", state_dir)
    monkeypatch.setattr(module, "EPOCH_SEEDS_DIR", state_dir / "epoch_seeds")
    monkeypatch.setattr(module, "PREEVAL_SEEDS_DIR", state_dir / "preeval_seeds")
    monkeypatch.setattr(module, "DEFAULT_SEED_SCHEME_MIN_VERSION", "99.0.0")
    built = module.BenchmarkSeedManager()
    built.epoch_number = EPOCH
    built.apply_backend_scheme(LIVE_FROM, {str(EPOCH): {"key": KEY, "commitment": key_fingerprint(KEY)}})
    built.module = module
    return built


@pytest.mark.parametrize("family_id", [SENTINEL, TEST_FIELD])
def test_world_seeds_match_the_known_answer_vector(family_id):
    """Proves the pinned encoding still gives the worlds every validator builds from this key."""
    assert derive_world_seeds(KEY, EPOCH, family_id, 2) == KNOWN_WORLD_SEEDS[family_id]


def test_a_world_seed_is_far_wider_than_a_list_seed_and_needs_the_key():
    """Proves a world seed cannot be found by trying every list seed: it fills 128 bits, never repeats a list seed,
    and another key, epoch, family or index gives another one."""
    seeds = derive_world_seeds(KEY, EPOCH, SENTINEL, 256)
    assert len(set(seeds)) == 256 and all(2**96 < seed < 2**128 for seed in seeds)
    assert not set(seeds) & set(derive_seeds(KEY, EPOCH, SENTINEL, 256))
    base = derive_world_seed(KEY, EPOCH, SENTINEL, 0)
    assert len({base, derive_world_seed(OTHER_KEY, EPOCH, SENTINEL, 0), derive_world_seed(KEY, EPOCH + 1, SENTINEL, 0),
                derive_world_seed(KEY, EPOCH, TEST_FIELD, 0), derive_world_seed(KEY, EPOCH, SENTINEL, 1)}) == 5


@pytest.mark.parametrize("family_id", [SENTINEL, TEST_FIELD])
def test_the_validator_flies_world_seeds_and_publishes_the_list(manager, family_id):
    """Proves the patrols are built from world seeds while the stored and published list stays the 32-bit one, so
    the backend and the seed-set identity see no change."""
    public = derive_seeds(KEY, EPOCH, family_id, manager.module.epoch_seed_count(family_id))
    flown = manager.get_all_seeds(family_id)
    assert require_runtime_family(family_id).secret_worlds
    assert flown == derive_world_seeds(KEY, EPOCH, family_id, len(public))
    assert manager.get_screening_seeds(family_id) + manager.get_benchmark_seeds(family_id) == flown
    assert manager.get_all_seeds(family_id, EPOCH) == flown
    suffix = f"__{family_id}"
    stored = json.loads((manager.module.EPOCH_SEEDS_DIR / f"epoch_{EPOCH}{suffix}.json").read_text())
    assert stored["seeds"] == public and max(stored["seeds"]) < 2**32


def test_a_family_without_secret_worlds_flies_its_list(manager):
    """Proves the change reaches only the families that ask for it."""
    assert not require_runtime_family(PUBLIC_FAMILY).secret_worlds
    assert manager.get_all_seeds(PUBLIC_FAMILY) == derive_seeds(
        KEY, EPOCH, PUBLIC_FAMILY, manager.module.epoch_seed_count(PUBLIC_FAMILY))


def test_a_family_with_no_runtime_flies_its_list(manager, monkeypatch):
    """Proves a family the simulator has no runtime for falls back to its list instead of failing the lookup."""
    monkeypatch.setattr(manager.module, "require_runtime_family", require_runtime_family)
    monkeypatch.setattr(manager, "_seeds_for", lambda family_id, epoch: [1, 2, 3])
    monkeypatch.setattr(manager, "uses_derived_seeds", lambda family_id: True)
    assert manager.get_all_seeds("cf_not_a_family") == [1, 2, 3]


def test_a_new_key_gives_new_world_seeds(manager):
    """Proves the kept world seeds follow the key they were built from."""
    before = manager.get_all_seeds(SENTINEL)
    manager.apply_backend_scheme(LIVE_FROM, {str(EPOCH): {"key": OTHER_KEY, "commitment": key_fingerprint(OTHER_KEY)}})
    after = manager.get_all_seeds(SENTINEL)
    assert after != before and after == derive_world_seeds(OTHER_KEY, EPOCH, SENTINEL, len(after))


def test_the_pad_reading_names_a_list_seed_but_not_the_world_flown_at_its_index():
    """Reproduces the exploit and its end: the position error on the pad is a public function of the seed, so a
    table of every list seed finds the seed from that one reading; the world flown at the same index is drawn from
    the world seed, and no list seed gives its reading."""
    public = derive_seeds(KEY, EPOCH, SENTINEL, 1)[0]
    table = {tuple(np.round(sensor_noise.rtk_drift(seed, 1)[0], 9)): seed for seed in range(public - 200, public + 200)}
    assert table[tuple(np.round(sensor_noise.rtk_drift(public, 1)[0], 9))] == public
    flown = derive_world_seed(KEY, EPOCH, SENTINEL, 0)
    assert tuple(np.round(sensor_noise.rtk_drift(flown, 1)[0], 9)) not in table
    assert tuple(np.round(sensor_noise.rtk_drift(flown & 0xFFFFFFFF, 1)[0], 9)) != tuple(
        np.round(sensor_noise.rtk_drift(flown, 1)[0], 9))


def test_no_draw_of_the_park_cuts_a_world_seed_to_its_low_bits():
    """Proves every per-seed draw reads the whole world seed: two seeds that share their lowest 32 bits get a
    different position error, theft, wind, flight limit, vegetation, movers and people."""
    assert WIDE_A & 0xFFFFFFFF == WIDE_B & 0xFFFFFFFF
    assert not np.allclose(sensor_noise.rtk_drift(WIDE_A, 4), sensor_noise.rtk_drift(WIDE_B, 4))
    rolls = [np.random.default_rng([theft.THEFT_SEED_STREAM, seed]).random() for seed in (WIDE_A, WIDE_B)]
    assert rolls[0] != rolls[1]
    assert theft.has_theft(WIDE_A) == (rolls[0] < theft.THEFT_SHARE)
    low = np.zeros(4)
    assert not np.allclose(flight_limit.pushes(WIDE_A, low), flight_limit.pushes(WIDE_B, low))
    assert builder.solar_densities(WIDE_A) != builder.solar_densities(WIDE_B)
    assert builder.solar_mover_rules(WIDE_A) != builder.solar_mover_rules(WIDE_B)
    assert [people.count(WIDE_A + n) for n in range(40)] != [people.count(WIDE_B + n) for n in range(40)]
    assert [wind.strength(WIDE_A + n, night=False) for n in range(40)] != [
        wind.strength(WIDE_B + n, night=False) for n in range(40)]


def test_a_list_seed_builds_the_park_it_always_did():
    """Proves a 32-bit seed is drawn exactly as before the cuts were removed, so a local benchmark flies the same
    parks: the values are the ones the old masked streams gave."""
    seed = 3594827619
    assert builder.solar_densities(seed) == builder.solar_densities(seed & 0xFFFFFFFF)
    assert (seed ^ builder.CONFIG["seed_offset"]) < 2**32 and (seed ^ builder.CONFIG["shift_seed_offset"]) < 2**32


@pytest.mark.parametrize("checks", [solar_seed_checks, field_seed_checks])
def test_a_replaced_world_seed_stays_as_wide_as_the_one_it_stands_in_for(checks):
    """Proves a seed that fails its check is not swapped for one a table could find: a world seed's replacements
    are wide, and a list seed's are the 32-bit ones they always were."""
    wide = list(checks.candidates(WIDE_A))
    assert wide[0] == WIDE_A and all(seed > 2**96 for seed in wide[1:]) and len(set(wide)) == len(wide)
    narrow = list(checks.candidates(12345))
    assert narrow[0] == 12345 and all(seed < 2**32 for seed in narrow)
