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

"""Family declarations fence local work and preserve unaffected benchmark identity."""
from __future__ import annotations

import asyncio
import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from swarm.domain_model import get_family_benchmark_version, load_challenge_family_registry, validate_family_task_version
from swarm.validator.utils_parts import evaluation, heartbeat, run_task, state
from validator.scripts.check_family_versions import declared_reset, release_tuple, validate_release, version_tuple
from validator.scripts.sync_family_registry import _validate_registry


@pytest.fixture
def registry():
    """Return an isolated bootstrap registry, every family at 5.1.6 with no reset declared, for release edits."""
    snapshot = copy.deepcopy(load_challenge_family_registry())
    for entry in snapshot['challenge_families'].values():
        entry['benchmark_version'] = '5.1.6'
        entry.pop('benchmark_version_reset', None)
    return snapshot


def test_bootstrap_hotfix_accepts_legacy_registry(registry):
    """Adding declarations at the existing production label is a valid bootstrap."""
    base = copy.deepcopy(registry)
    for entry in base['challenge_families'].values():
        del entry['benchmark_version']
    validate_release(registry, '5.1.6.6', base, '5.1.6.5')
    _validate_registry(registry)


def test_selective_sentinel_release_is_valid(registry):
    """A later Sentinel release keeps every other family's original label."""
    changed = copy.deepcopy(registry)
    changed['challenge_families']['cf_solar_patrol']['benchmark_version'] = '6.0.0'
    validate_release(changed, '6.0.0', registry, '5.1.6.6')
    validate_release(changed, '6.0.0.1', changed, '6.0.0')


def test_new_family_in_a_hotfix_is_not_a_change(registry):
    """A family the base never had restarts nothing, so a hotfix may introduce it."""
    base = copy.deepcopy(registry)
    del base['challenge_families']['cf_solar_patrol']
    validate_release(registry, '5.1.6.6', base, '5.1.6.6')


def test_new_family_at_the_new_prefix_declares_the_bump(registry):
    """A family introduced at the new global prefix is the change that release declares."""
    base = copy.deepcopy(registry)
    del base['challenge_families']['cf_solar_patrol']
    changed = copy.deepcopy(registry)
    changed['challenge_families']['cf_solar_patrol']['benchmark_version'] = '6.0.0'
    validate_release(changed, '6.0.0', base, '5.1.6.6')


@pytest.fixture
def live(registry):
    """Return the registry as production runs it before the reset: Sentinel 6.0.0, Test Field 6.0.1, the rest 5.1.6."""
    registry['challenge_families']['cf_solar_patrol']['benchmark_version'] = '6.0.0'
    registry['challenge_families']['cf_test_field']['benchmark_version'] = '6.0.1'
    return registry


def _reset(live, **overrides):
    """Return the live registry with every family restarted at 1.0.0 from its live label."""
    reset = copy.deepcopy(live)
    for family, entry in reset['challenge_families'].items():
        old = entry['benchmark_version']
        entry['benchmark_version'] = overrides.get(family, '1.0.0')
        entry['benchmark_version_reset'] = {'from': old, 'to': '1.0.0'}
    return reset


def test_bundled_registry_is_the_reset(live):
    """The shipped registry restarts the eight families it had at 1.0.0 from their live labels; Swarm Test Field v2,
    added since, was never live under an old label and has none to restart."""
    del live['challenge_families']['cf_test_field_v2']
    validate_release(load_challenge_family_registry(), '6.1.0', live, '6.0.1.1')
    _validate_registry(load_challenge_family_registry())


def test_bundled_registry_adds_swarm_test_field_v2_without_touching_a_label():
    """Against the registry of release 6.1.0, the shipped one adds one family at 1.0.0 with no reset and changes no
    other family's label, so no model of another family is evaluated again."""
    shipped = load_challenge_family_registry()
    before = copy.deepcopy(shipped)
    del before['challenge_families']['cf_test_field_v2']
    before['entity_types']['challenge_family']['ids'].remove('cf_test_field_v2')
    validate_release(shipped, '6.2.0', before, '6.1.0')
    validate_release(shipped, '6.1.0.1', before, '6.1.0')
    added = shipped['challenge_families']['cf_test_field_v2']
    assert added['benchmark_version'] == '1.0.0' and 'benchmark_version_reset' not in added
    assert all(entry['benchmark_version'] == before['challenge_families'][family]['benchmark_version']
               for family, entry in shipped['challenge_families'].items() if family in before['challenge_families'])


def test_declared_reset_lowers_labels_in_a_release(live):
    """A three-part release may restart every family, and may raise one past 1.0.0 in the same step."""
    validate_release(_reset(live), '6.1.0', live, '6.0.0')
    validate_release(_reset(live, cf_solar_patrol='1.1.0'), '7.0.0', live, '6.0.0')


@pytest.mark.parametrize('release,mutate,message', [
    ('6.1.0', lambda r: r['challenge_families']['cf_solar_patrol'].pop('benchmark_version_reset'), 'without a declared reset'),
    ('6.1.0', lambda r: r['challenge_families']['cf_solar_patrol']['benchmark_version_reset'].update({'from': '5.1.6'}), 'without a declared reset'),
    ('6.0.0.1', lambda r: None, 'without a three-part release bump'),
    ('6.1.0', lambda r: r['challenge_families']['cf_swarm_sar'].update({'benchmark_version': '0.9.0'}), 'reset label is above'),
    ('6.1.0', lambda r: r['challenge_families']['cf_swarm_sar']['benchmark_version_reset'].update({'extra': 1}), 'exactly'),
    ('6.1.0', lambda r: r['challenge_families']['cf_swarm_sar'].update({'benchmark_version': '6.0.0', 'benchmark_version_reset': {'from': '5.1.6', 'to': '6.0.0'}}), 'must lower'),
    ('6.1.0', lambda r: r['challenge_families']['cf_swarm_sar'].update({'benchmark_version': '5.1.6', 'benchmark_version_reset': {'from': '5.1.6', 'to': '5.1.6'}}), 'must lower'),
])
def test_bad_resets_are_refused(live, release, mutate, message):
    """An undeclared or mismatched reset, a reset in a hotfix, a malformed declaration and a reset that does not lower the label all fail."""
    reset = _reset(live)
    mutate(reset)
    with pytest.raises(ValueError, match=message):
        validate_release(reset, release, live, '6.0.0')


def test_family_labels_move_on_their_own_after_the_reset(live):
    """After the reset a family rises without matching the release number, and releases need no family change."""
    after = _reset(live)
    scoring = _reset(live, cf_solar_patrol='1.1.0')
    validate_release(scoring, '6.2.0', after, '6.1.0')
    validate_release(after, '6.1.1', after, '6.1.0')
    validate_release(after, '6.1.0.1', after, '6.1.0')
    with pytest.raises(ValueError, match='without a declared reset'):
        validate_release(after, '6.3.0', scoring, '6.2.0')


@pytest.mark.parametrize('value', [None, 516, '5.1', '5.1.6.1', 'v5.1.6', '5.a.6', '-1.2.3', '5.1.6\n'])
def test_malformed_versions_fail(value):
    """Only three numeric components form a family version."""
    with pytest.raises(ValueError):
        version_tuple(value)


@pytest.mark.parametrize('release,updates', [
    ('5.1.6.7', {'cf_search_and_rescue': '5.1.5'}),
    ('5.1.6.7', {'cf_solar_patrol': '6.0.0'}),
    ('6.0.0', {'cf_solar_patrol': '6.0.0', 'cf_swarm_sar': '5.1.5'}),
    ('5.1.5', {}),
])
def test_invalid_release_declarations_fail(registry, release, updates):
    """Decreases, hotfix family edits and a lower release are refused."""
    changed = copy.deepcopy(registry)
    for family, value in updates.items():
        changed['challenge_families'][family]['benchmark_version'] = value
    with pytest.raises(ValueError):
        validate_release(changed, release, registry, '5.1.6.6')


def test_hotfix_cannot_advance_a_nonmaximum_family(registry):
    """Keeping the maximum unchanged cannot hide another family's hotfix edit."""
    base = copy.deepcopy(registry)
    base['challenge_families']['cf_swarm_sar']['benchmark_version'] = '5.1.5'
    with pytest.raises(ValueError, match='without a three-part release bump'):
        validate_release(registry, '5.1.6.7', base, '5.1.6.6')


def test_accessors_refuse_unknown_and_missing_versions(registry):
    """A missing declaration never falls back to the global version."""
    with pytest.raises(KeyError):
        get_family_benchmark_version('unknown', registry=registry)
    del registry['challenge_families']['cf_swarm_sar']['benchmark_version']
    with pytest.raises(ValueError):
        get_family_benchmark_version('cf_swarm_sar', registry=registry)
    with pytest.raises(SystemExit):
        _validate_registry(registry)


@pytest.mark.parametrize('version', [None, '5.1.5', '6.0.0'])
def test_task_mismatch_refused_before_fetch(monkeypatch, version):
    """Incompatible work cannot fetch a model, execute or submit a result."""
    fetch = AsyncMock()
    monkeypatch.setattr(run_task, '_ensure_models_from_backend', fetch)
    validator = SimpleNamespace()
    asyncio.run(run_task.run_task(validator, {
        'task_id': 1, 'uid': 42, 'phase': 'BENCHMARK',
        'family_id': 'cf_search_and_rescue', 'benchmark_version': version,
    }, cancel_flag=asyncio.Event(), wake_flag=asyncio.Event()))
    fetch.assert_not_called()


def test_family_version_agreement():
    """The accepted version is always the bundled family declaration."""
    assert validate_family_task_version('cf_swarm_sar', '1.0.0') == '1.0.0'


def test_cache_survives_another_familys_bump(monkeypatch):
    """A Sentinel release preserves SAR cache keys and changes only Sentinel keys."""
    versions = {'cf_search_and_rescue': '5.1.6', 'cf_solar_patrol': '5.1.6'}
    monkeypatch.setattr(state, 'get_family_benchmark_version', versions.__getitem__)
    before = state._score_cache_key('hash', 10, 'cf_search_and_rescue')
    solar = state._score_cache_key('hash', 10, 'cf_solar_patrol')
    versions['cf_solar_patrol'] = '6.0.0'
    assert state._score_cache_key('hash', 10, 'cf_search_and_rescue') == before
    assert state._score_cache_key('hash', 10, 'cf_solar_patrol') != solar
    cache = {}
    monkeypatch.setattr(state, 'load_benchmark_cache', lambda: cache)
    monkeypatch.setattr(state, 'save_benchmark_cache', cache.update)
    state.set_cached_score('hash', 10, {'score': 0.5}, 'cf_search_and_rescue')
    assert state.has_cached_score('hash', 10, 'cf_search_and_rescue')
    assert state.get_cached_score('hash', 10, 'cf_search_and_rescue')['benchmark_version'] == '5.1.6'



@pytest.mark.asyncio
async def test_heartbeat_fallback_keeps_family_identity():
    """A queue-derived heartbeat retains both the family's identity and its evaluation version."""
    api = SimpleNamespace(post_heartbeat=AsyncMock(return_value={}))
    manager = heartbeat.HeartbeatManager(api, asyncio.get_running_loop())
    try:
        manager.start('evaluating_benchmark', 42, 1, queue=[{
            'uid': 42, 'phase': 'BENCHMARK', 'assignment_id': 9,
            'family_id': 'cf_swarm_sar', 'benchmark_version': '5.1.6', 'epoch_number': 10,
        }])
        await asyncio.sleep(0.01)
        active = api.post_heartbeat.call_args.kwargs['active_task']
        assert active['family_id'] == 'cf_swarm_sar'
        assert active['benchmark_version'] == '5.1.6'
    finally:
        manager.finish()
        await asyncio.sleep(0.01)


@pytest.mark.parametrize('phase', ['SCREENING', 'BENCHMARK', 'REEVAL'])
@pytest.mark.asyncio
async def test_evaluation_heartbeat_uses_local_family_version(monkeypatch, tmp_path, phase):
    """Screening, benchmark and defence heartbeats identify the family's evaluation scope."""
    hb = MagicMock()
    monkeypatch.setattr(evaluation, 'get_family_benchmark_version', lambda _family: '5.1.5')
    monkeypatch.setattr(evaluation, 'HeartbeatManager', lambda *_args: hb)
    monkeypatch.setattr(evaluation, 'build_screening_tasks', lambda **_kwargs: [])
    monkeypatch.setattr(evaluation, '_run_streaming_phase', AsyncMock(return_value=([0.5], {'city': [0.5]}, [], None)))
    validator = SimpleNamespace(backend_api=object(), seed_manager=SimpleNamespace(
        epoch_number=10, get_screening_seeds=lambda **_kwargs: [11],
    ))
    if phase == 'SCREENING':
        await evaluation._run_screening(validator, 42, tmp_path / 'model.zip', task_id=9, family_id='cf_swarm_sar')
    else:
        await evaluation._run_full_benchmark(validator, 42, tmp_path / 'model.zip', seeds=[11], reeval=phase == 'REEVAL',
                                              task_id=9, family_id='cf_swarm_sar')
    active = hb.start.call_args.kwargs['active_task']
    assert active['phase'] == phase
    assert active['family_id'] == 'cf_swarm_sar'
    assert active['benchmark_version'] == '5.1.5'
    hb.finish.assert_called_once()


@pytest.mark.parametrize('value,expected', [('6.1.0', (6, 1, 0, 0)), ('5.1.6.6', (5, 1, 6, 6))])
def test_release_tuple_pads_three_part_releases(value, expected):
    """Three- and four-part releases compare on four parts."""
    assert release_tuple(value) == expected


@pytest.mark.parametrize('value', [None, '6.1', '6.1.0.0.1', 'v6.1.0'])
def test_release_tuple_refuses_malformed_releases(value):
    """Only numeric three- or four-part releases are ordered."""
    with pytest.raises(ValueError):
        release_tuple(value)


def test_declared_reset_reads_the_pair():
    """A family without a declaration has no reset; one with it yields its old and new labels."""
    assert declared_reset('cf_solar_patrol', {'benchmark_version': '1.0.0'}) is None
    entry = {'benchmark_version': '1.0.0', 'benchmark_version_reset': {'from': '6.0.0', 'to': '1.0.0'}}
    assert declared_reset('cf_solar_patrol', entry) == ('6.0.0', '1.0.0')
