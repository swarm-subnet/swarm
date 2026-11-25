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

"""The mix of a seed: six independent switches at their decided shares, each seed's the same every time."""

from __future__ import annotations

import numpy as np

from swarm.challenge_families.test_field_v2 import mix


def test_each_switch_comes_up_at_its_share_and_on_its_own():
    """Over many seeds each switch is on at its share, and knowing one says nothing about another."""
    drawn = [mix.present(seed) for seed in range(20000)]
    people = np.array([m.people for m in drawn])
    animals = np.array([m.animals for m in drawn])
    outsiders = np.array([m.outsiders for m in drawn])
    birds = np.array([m.birds for m in drawn])
    assert abs(people.mean() - mix.PEOPLE_SHARE) < 0.01
    assert abs(animals.mean() - mix.ANIMALS_SHARE) < 0.015
    assert abs(outsiders.mean() - mix.OUTSIDERS_SHARE) < 0.015
    assert abs(birds.mean() - mix.BIRDS_SHARE) < 0.015
    assert abs(animals[outsiders].mean() - animals.mean()) < 0.02
    assert abs(outsiders[~people].mean() - outsiders.mean()) < 0.05
    assert abs(birds[animals].mean() - birds.mean()) < 0.02 and abs(birds[outsiders].mean() - birds.mean()) < 0.025


def test_the_birds_switch_moves_none_of_the_three_before_it():
    """Each of the first three switches is still the draw it was, the first three of the mix's stream."""
    for seed in (0, 1, 7, 12345, 2 ** 100 + 3):
        first = [float(np.random.default_rng([mix.MIX_SEED_STREAM, seed, n]).random()) for n in range(3)]
        found = mix.present(seed)
        assert (found.people, found.animals, found.outsiders) == (first[0] < mix.PEOPLE_SHARE,
                                                                  first[1] < mix.ANIMALS_SHARE,
                                                                  first[2] < mix.OUTSIDERS_SHARE)


def test_boars_and_rodents_are_rare_by_day_come_to_one_night_in_five_and_move_no_switch_before_them():
    """Each of the two comes at its day share by day and its night share at night, on a draw of its own: neither
    says anything about the other or about the animals, and the four switches before them are what they were."""
    day = [mix.present(seed) for seed in range(20000)]
    night = [mix.present(seed, night=True) for seed in range(20000)]
    for name, shares in (("boars", mix.BOARS_SHARE), ("rodents", mix.RODENTS_SHARE)):
        assert abs(np.mean([getattr(m, name) for m in day]) - shares[0]) < 0.006
        assert abs(np.mean([getattr(m, name) for m in night]) - shares[1]) < 0.01
        assert all(getattr(n, name) for d, n in zip(day, night) if getattr(d, name))
    boars, rodents = np.array([m.boars for m in night]), np.array([m.rodents for m in night])
    animals = np.array([m.animals for m in night])
    assert abs(rodents[boars].mean() - rodents.mean()) < 0.03 and abs(boars[animals].mean() - boars.mean()) < 0.02
    assert [(m.people, m.animals, m.outsiders, m.birds) for m in day] == [(m.people, m.animals, m.outsiders, m.birds) for m in night]
    for seed in (0, 1, 7, 12345, 2 ** 100 + 3):
        first = [float(np.random.default_rng([mix.MIX_SEED_STREAM, seed, n]).random()) for n in range(4)]
        found = mix.present(seed)
        assert (found.people, found.animals, found.outsiders, found.birds) == (
            first[0] < mix.PEOPLE_SHARE, first[1] < mix.ANIMALS_SHARE, first[2] < mix.OUTSIDERS_SHARE, first[3] < mix.BIRDS_SHARE)
    assert (mix.BOARS_SHARE, mix.RODENTS_SHARE) == ((0.05, 0.20), (0.06, 0.20))
    for seed in list(range(400)) + [2 ** 100 + 3, 2 ** 127 + 11]:
        fifth, sixth = (float(np.random.default_rng([mix.MIX_SEED_STREAM, seed, n]).random()) for n in (4, 5))
        assert (mix.present(seed).boars, mix.present(seed).rodents) == (fifth < 0.05, sixth < 0.06)
        assert (mix.present(seed, True).boars, mix.present(seed, True).rodents) == (fifth < 0.20, sixth < 0.20)


def test_a_seed_keeps_its_mix_and_a_wide_seed_uses_all_its_bits():
    """The same seed gives the same mix, and seeds that differ only above 32 bits do not all share one."""
    assert mix.present(12345) == mix.present(12345)
    wide = {mix.present((top << 96) + 7) for top in range(1, 200)}
    assert len(wide) > 1
