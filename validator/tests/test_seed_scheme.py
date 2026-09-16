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

"""The seed derivation every validator shares: its pinned encoding and its version boundary."""

import hashlib

from swarm.validator.seed_scheme import (
    derive_seed,
    derive_seeds,
    key_fingerprint,
    seed_set_id,
    uses_derived_seeds,
)


KEY = "11" * 32
EPOCH = 22
FAMILY = "cf_autopilot"

# The contract with the backend. These exact numbers are what the network flies, so a change
# to the encoding has to fail here rather than quietly move every map in the subnet.
KNOWN_SEEDS = [3594827619, 151986188, 2070247283, 1556279107]


def test_derivation_matches_the_known_answer_vector():
    """Proves the pinned encoding still produces the seeds the backend expects."""
    assert derive_seeds(KEY, EPOCH, FAMILY, len(KNOWN_SEEDS)) == KNOWN_SEEDS


def test_two_validators_with_one_key_build_the_same_list():
    """Proves the whole point: the same key gives the same 1,100 maps everywhere."""
    assert derive_seeds(KEY, EPOCH, FAMILY, 128) == derive_seeds(KEY, EPOCH, FAMILY, 128)


def test_each_family_and_epoch_gets_its_own_list():
    """Proves one key still separates families and epochs instead of repeating maps."""
    base = derive_seeds(KEY, EPOCH, FAMILY, 16)
    assert derive_seeds(KEY, EPOCH, "cf_search_and_rescue", 16) != base
    assert derive_seeds(KEY, EPOCH + 1, FAMILY, 16) != base


def test_seeds_fit_the_thirty_two_bit_range():
    """Proves every derived seed is a value the simulator accepts."""
    assert all(0 <= seed <= 2**32 - 1 for seed in derive_seeds(KEY, EPOCH, FAMILY, 256))


def test_fingerprint_is_the_hash_of_the_raw_key():
    """Proves the validator checks a key against the same commitment the backend published."""
    assert key_fingerprint(KEY) == hashlib.sha256(bytes.fromhex(KEY)).hexdigest()


def test_seed_set_id_is_stable_and_family_scoped():
    """Proves the identity sent with scores names one list and nothing else."""
    identity = seed_set_id(key_fingerprint(KEY), EPOCH, FAMILY)
    assert identity == seed_set_id(key_fingerprint(KEY), EPOCH, FAMILY)
    assert identity != seed_set_id(key_fingerprint(KEY), EPOCH, "cf_search_and_rescue")


def test_the_scheme_starts_exactly_at_its_version():
    """Proves derived seeds begin at the activating version and not one release early."""
    assert not uses_derived_seeds("5.1.5", "5.1.6")
    assert not uses_derived_seeds(None, "5.1.6")
    assert uses_derived_seeds("5.1.6", "5.1.6")
    assert uses_derived_seeds("5.2.0", "5.1.6")


def test_one_index_is_one_mission():
    """Proves a single index resolves to the same seed as the full list's entry."""
    assert derive_seed(KEY, EPOCH, FAMILY, 7) == derive_seeds(KEY, EPOCH, FAMILY, 8)[7]
