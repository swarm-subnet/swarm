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

from swarm.constants import BENCHMARK_TOTAL_SEED_COUNT
from swarm.validator.seed_scheme import derive_seeds, key_fingerprint


KEY = "11" * 32
OTHER_KEY = "22" * 32
EPOCH = 22
FAMILY = "cf_autopilot"
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
