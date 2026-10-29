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

"""The seed manager under the shared scheme: derived lists, readiness, and the stale file trap."""
from __future__ import annotations

import json

import pytest

from swarm.constants import BENCHMARK_SCREENING_SEED_COUNT, BENCHMARK_TOTAL_SEED_COUNT
from swarm.validator.seed_scheme import derive_seed, derive_seeds, key_fingerprint

KEY = "11" * 32
OTHER_KEY = "22" * 32
EPOCH = 22
FAMILY = "cf_autopilot"
SENTINEL = "cf_solar_patrol"
SENTINEL_SEED_COUNT = 2000
OTHER_FAMILY = "cf_search_and_rescue"
# Below this validator's own version, so the manager treats itself as on the shared scheme.
LIVE_FROM = "5.0.0"


@pytest.fixture
def seed_manager_module(reload_module):
    """A freshly imported seed_manager, so patched paths never leak between tests."""
    return reload_module("swarm.validator.seed_manager")


@pytest.fixture
def manager(seed_manager_module, monkeypatch, tmp_path):
    """A manager rooted in tmp_path, already aligned to the test epoch."""
    module = seed_manager_module
    state_dir = tmp_path / "state"
    monkeypatch.setattr(module, "STATE_DIR", state_dir)
    monkeypatch.setattr(module, "EPOCH_SEEDS_DIR", state_dir / "epoch_seeds")
    monkeypatch.setattr(module, "PREEVAL_SEEDS_DIR", state_dir / "preeval_seeds")
    # Starts where a validator starts, on its own seeds, until the backend moves it.
    # Pinned rather than inherited from swarm.__version__ so these tests keep describing
    # the transition itself once the release has activated the shared scheme.
    monkeypatch.setattr(module, "DEFAULT_SEED_SCHEME_MIN_VERSION", "99.0.0")
    built = module.BenchmarkSeedManager()
    built.epoch_number = EPOCH
    return built


def _go_live(manager, key: str = KEY, epoch: int = EPOCH) -> None:
    """Put the manager on the shared scheme holding one epoch's key."""
    manager.apply_backend_scheme(
        LIVE_FROM, {str(epoch): {"key": key, "commitment": key_fingerprint(key)}}
    )


def _stored_file(module, epoch: int = EPOCH, family_id: str = FAMILY):
    """The seed file the manager writes for one epoch and family."""
    suffix = "" if family_id == "cf_autopilot" else f"__{family_id}"
    return module.EPOCH_SEEDS_DIR / f"epoch_{epoch}{suffix}.json"


def test_seeds_come_from_the_key_not_the_dice(manager):
    """Proves the list a validator flies is exactly what the epoch key derives."""
    _go_live(manager)

    assert manager.get_all_seeds(FAMILY) == derive_seeds(
        KEY, EPOCH, FAMILY, BENCHMARK_TOTAL_SEED_COUNT
    )


def test_without_a_key_there_are_no_seeds_and_no_dice(manager, seed_manager_module):
    """Proves a missing key stops the validator instead of quietly restoring per-validator seeds."""
    manager.apply_backend_scheme(LIVE_FROM, {})

    assert not manager.seeds_ready()
    with pytest.raises(seed_manager_module.SeedsNotReady):
        manager.get_all_seeds(FAMILY)


def test_readiness_is_per_epoch(manager):
    """Proves holding this epoch's key says nothing about the next one's."""
    _go_live(manager)

    assert manager.seeds_ready(EPOCH)
    assert not manager.seeds_ready(EPOCH + 1)


def test_a_key_that_breaks_its_commitment_is_refused(manager):
    """Proves a key the network did not commit to cannot silently replace the real one."""
    manager.apply_backend_scheme(
        LIVE_FROM, {str(EPOCH): {"key": KEY, "commitment": key_fingerprint(OTHER_KEY)}}
    )

    assert not manager.seeds_ready()


def test_seeds_left_on_disk_from_before_the_switch_are_thrown_away(
    manager, seed_manager_module
):
    """Proves the update trap: state/ survives an upgrade, so an unstamped file must not be trusted.

    Without the stamp check a validator that upgraded mid-epoch would keep flying the seeds it
    rolled itself, matching nobody, and nothing in the logs would say so.
    """
    module = seed_manager_module
    path = _stored_file(module)
    path.parent.mkdir(parents=True, exist_ok=True)
    stale = list(range(BENCHMARK_TOTAL_SEED_COUNT))
    path.write_text(json.dumps({
        "epoch_number": EPOCH,
        "family_id": FAMILY,
        "seeds": stale,
        "published": False,
    }))

    _go_live(manager)
    seeds = manager.get_all_seeds(FAMILY)

    assert seeds != stale
    assert seeds == derive_seeds(KEY, EPOCH, FAMILY, BENCHMARK_TOTAL_SEED_COUNT)


def test_a_file_from_another_key_is_thrown_away(manager, seed_manager_module):
    """Proves a list built under a different key is never reused for this one."""
    module = seed_manager_module
    _go_live(manager, OTHER_KEY)
    manager.get_all_seeds(FAMILY)
    assert json.loads(_stored_file(module).read_text())["key_fingerprint"] == (
        key_fingerprint(OTHER_KEY)
    )

    manager._family_seeds = {}
    _go_live(manager, KEY)

    assert manager.get_all_seeds(FAMILY) == derive_seeds(
        KEY, EPOCH, FAMILY, BENCHMARK_TOTAL_SEED_COUNT
    )


def test_a_new_key_drops_the_list_already_in_memory(manager):
    """Proves the cache follows the key, not just the file, so a stale list cannot stay live."""
    _go_live(manager, OTHER_KEY)
    first = manager.get_all_seeds(FAMILY)

    _go_live(manager, KEY)

    assert manager.get_all_seeds(FAMILY) != first


def test_stored_seeds_are_reused_when_the_key_still_matches(manager, seed_manager_module):
    """Proves the stamp check costs nothing on a restart that changed nothing."""
    module = seed_manager_module
    _go_live(manager)
    first = manager.get_all_seeds(FAMILY)
    written = json.loads(_stored_file(module).read_text())["seeds"]

    manager._family_seeds = {}

    assert manager.get_all_seeds(FAMILY) == first == written


def test_the_old_scheme_still_reads_its_own_unstamped_files(manager, seed_manager_module):
    """Proves a validator left on random seeds keeps its history, which no key can rebuild."""
    module = seed_manager_module
    path = _stored_file(module)
    path.parent.mkdir(parents=True, exist_ok=True)
    kept = list(range(BENCHMARK_TOTAL_SEED_COUNT))
    path.write_text(json.dumps({
        "epoch_number": EPOCH,
        "family_id": FAMILY,
        "seeds": kept,
        "published": False,
    }))

    assert not manager.uses_derived_seeds()
    assert manager.get_all_seeds(FAMILY) == kept


def test_the_seed_set_identity_only_exists_under_the_shared_scheme(manager):
    """Proves scores carry a list identity when there is one, and nothing when there is not."""
    assert manager.seed_set_id_for(EPOCH, FAMILY) is None

    _go_live(manager)

    assert manager.seed_set_id_for(EPOCH, FAMILY) is not None
    assert manager.seed_set_id_for(EPOCH, FAMILY) != manager.seed_set_id_for(
        EPOCH, "cf_search_and_rescue"
    )


def test_a_keyless_manager_holds_no_seeds_at_all(manager, seed_manager_module):
    """Proves "no key means no seeds" holds in memory too, not only on the next build.

    An old random file on disk must not be loaded into a derived manager that has just said
    it cannot build this epoch, or the invariant is true in the logs and false in the object.
    """
    module = seed_manager_module
    path = _stored_file(module)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "epoch_number": EPOCH,
        "family_id": FAMILY,
        "seeds": list(range(BENCHMARK_TOTAL_SEED_COUNT)),
        "published": False,
    }))

    manager.apply_backend_scheme(LIVE_FROM, {})

    assert not manager.seeds_ready()
    with pytest.raises(seed_manager_module.SeedsNotReady):
        manager.get_all_seeds(FAMILY)


def test_changing_the_scheme_drops_what_the_other_one_cached(manager):
    """Proves a list built under one scheme is never served once the other becomes live."""
    random_seeds = manager.get_all_seeds(FAMILY)
    assert not manager.uses_derived_seeds()

    _go_live(manager)

    assert manager.uses_derived_seeds()
    assert manager.get_all_seeds(FAMILY) != random_seeds


def test_sentinel_flies_a_list_of_two_thousand_seeds(manager):
    """Proves Swarm Sentinel's epoch list is 2,000 long and its indexes 1000 to 1999 resolve to the derived seeds."""
    _go_live(manager)

    seeds = manager.get_all_seeds(SENTINEL)

    assert seeds == derive_seeds(KEY, EPOCH, SENTINEL, SENTINEL_SEED_COUNT)
    assert seeds[1000:] == [derive_seed(KEY, EPOCH, SENTINEL, index) for index in range(1000, SENTINEL_SEED_COUNT)]
    assert len(manager.get_screening_seeds(SENTINEL)) == BENCHMARK_SCREENING_SEED_COUNT
    assert manager.get_benchmark_seeds(SENTINEL) == seeds[BENCHMARK_SCREENING_SEED_COUNT:]


def test_other_families_keep_one_thousand_seeds_and_their_values(manager):
    """Proves a family with no seed count of its own still gets 1,000 seeds, split 300 and 700, unchanged."""
    _go_live(manager)

    seeds = manager.get_all_seeds(OTHER_FAMILY)

    assert seeds == derive_seeds(KEY, EPOCH, OTHER_FAMILY, BENCHMARK_TOTAL_SEED_COUNT)
    assert len(manager.get_screening_seeds(OTHER_FAMILY)) == BENCHMARK_SCREENING_SEED_COUNT
    assert len(manager.get_benchmark_seeds(OTHER_FAMILY)) == BENCHMARK_TOTAL_SEED_COUNT - BENCHMARK_SCREENING_SEED_COUNT


@pytest.mark.parametrize("family_id, expected", [(FAMILY, 1000), (OTHER_FAMILY, 1000), (SENTINEL, 2000)])
def test_a_stored_file_of_the_families_length_is_reused(manager, seed_manager_module, family_id, expected):
    """Proves a cached epoch file with the family's own length is accepted, not rebuilt."""
    module = seed_manager_module
    _go_live(manager)
    first = manager.get_all_seeds(family_id)
    assert len(first) == expected
    manager._family_seeds = {}
    manager._build_seeds = lambda *_: pytest.fail("a file of the right length must not be rebuilt")

    assert manager.get_all_seeds(family_id) == first == json.loads(_stored_file(module, family_id=family_id).read_text())["seeds"]


def test_sentinel_file_from_before_the_growth_is_rebuilt_with_the_same_first_seeds(manager, seed_manager_module):
    """Proves a 1,000 seed Sentinel file is replaced by the 2,000 list, which starts with the same seeds."""
    module = seed_manager_module
    _go_live(manager)
    old = derive_seeds(KEY, EPOCH, SENTINEL, BENCHMARK_TOTAL_SEED_COUNT)
    path = _stored_file(module, family_id=SENTINEL)
    manager._save_epoch_file(EPOCH, SENTINEL, old, published=False)
    assert path.exists()

    rebuilt = manager.get_all_seeds(SENTINEL)

    assert len(rebuilt) == SENTINEL_SEED_COUNT
    assert rebuilt[:BENCHMARK_TOTAL_SEED_COUNT] == old


def test_without_the_shared_scheme_sentinel_still_rolls_two_thousand_seeds(manager):
    """Proves a validator on its own random seeds also gets a 2,000 seed list for Sentinel and 1,000 for the rest."""
    assert len(manager.get_all_seeds(SENTINEL)) == SENTINEL_SEED_COUNT
    assert len(manager.get_all_seeds(OTHER_FAMILY)) == BENCHMARK_TOTAL_SEED_COUNT


def test_epoch_seed_count_is_the_larger_of_the_shared_total_and_the_family_count(seed_manager_module):
    """Proves Sentinel's list holds its own 2,000 seeds while other and unknown families hold the 1,000 total."""
    count = seed_manager_module.epoch_seed_count

    assert count(SENTINEL) == SENTINEL_SEED_COUNT
    assert count(OTHER_FAMILY) == BENCHMARK_TOTAL_SEED_COUNT
    assert count("cf_not_a_family") == BENCHMARK_TOTAL_SEED_COUNT


def test_family_bump_keeps_seed_bytes_and_publication_provenance(manager, seed_manager_module, monkeypatch):
    """Another family's bump preserves seed files; even this family's bump retains stored seed identity and provenance."""
    module = seed_manager_module
    _go_live(manager)
    versions = {FAMILY: '5.1.6', OTHER_FAMILY: '5.1.6', SENTINEL: '5.1.6'}
    monkeypatch.setattr(module, 'get_family_benchmark_version', versions.__getitem__)
    first = manager.get_all_seeds(OTHER_FAMILY)
    path = _stored_file(module, family_id=OTHER_FAMILY)
    original = path.read_bytes()
    versions[SENTINEL] = '6.0.0'
    manager._family_seeds.clear()
    assert manager.get_all_seeds(OTHER_FAMILY) == first
    assert path.read_bytes() == original
    solar = manager.get_all_seeds(SENTINEL)
    assert json.loads(_stored_file(module, family_id=SENTINEL).read_text())['benchmark_version'] == '6.0.0'
    versions[SENTINEL] = '6.0.1'
    manager._family_seeds.clear()
    assert manager.get_all_seeds(SENTINEL) == solar
    assert json.loads(_stored_file(module, family_id=SENTINEL).read_text())['benchmark_version'] == '6.0.0'
    manager._publish_unpublished_epochs()
    manager.epoch_number = EPOCH + 1
    manager._publish_unpublished_epochs()
    publication = next(item for item in manager.get_pending_publications() if item['family_id'] == SENTINEL)
    assert publication['benchmark_version'] == '6.0.0'
    assert publication['seeds'] == solar


def test_seed_scheme_uses_the_named_familys_version(manager, seed_manager_module, monkeypatch):
    """A global release cannot opt unchanged families into a different seed scheme."""
    versions = {FAMILY: '5.1.6', OTHER_FAMILY: '5.1.6', SENTINEL: '6.0.0'}
    monkeypatch.setattr(seed_manager_module, 'get_family_benchmark_version', versions.__getitem__)
    manager.apply_backend_scheme('6.0.0', {})
    assert not manager.uses_derived_seeds(OTHER_FAMILY)
    assert manager.uses_derived_seeds(SENTINEL)
    assert manager.seeds_ready(EPOCH, OTHER_FAMILY)
    assert not manager.seeds_ready(EPOCH, SENTINEL)
