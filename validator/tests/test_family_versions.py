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
from validator.scripts.check_family_versions import validate_release, version_tuple
from validator.scripts.sync_family_registry import _validate_registry


@pytest.fixture
def registry():
    """Return an isolated bootstrap registry, every family at 5.1.6, for release edits."""
    snapshot = copy.deepcopy(load_challenge_family_registry())
    for entry in snapshot['challenge_families'].values():
        entry['benchmark_version'] = '5.1.6'
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


@pytest.mark.parametrize('value', [None, 516, '5.1', '5.1.6.1', 'v5.1.6', '5.a.6', '-1.2.3', '5.1.6\n'])
def test_malformed_versions_fail(value):
    """Only three numeric components form a family version."""
    with pytest.raises(ValueError):
        version_tuple(value)


@pytest.mark.parametrize('release,updates', [
    ('6.0.0', {}),
    ('5.1.6.7', {'cf_search_and_rescue': '5.1.5'}),
    ('5.1.6.7', {'cf_solar_patrol': '6.0.0'}),
    ('6.0.0', {'cf_solar_patrol': '6.0.0', 'cf_swarm_sar': '5.1.7'}),
    ('6.0.0', {'cf_solar_patrol': '6.0.0', 'cf_swarm_sar': '5.1.5'}),
    ('5.1.5', {}),
])
def test_invalid_release_declarations_fail(registry, release, updates):
    """Global bumps, hotfixes and family edits must obey the same release declaration."""
    changed = copy.deepcopy(registry)
    for family, value in updates.items():
        changed['challenge_families'][family]['benchmark_version'] = value
    with pytest.raises(ValueError):
        validate_release(changed, release, registry, '5.1.6.6')


def test_hotfix_cannot_advance_a_nonmaximum_family(registry):
    """Keeping the maximum unchanged cannot hide another family's hotfix edit."""
    base = copy.deepcopy(registry)
    base['challenge_families']['cf_swarm_sar']['benchmark_version'] = '5.1.5'
    with pytest.raises(ValueError, match='without a global'):
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
    assert validate_family_task_version('cf_swarm_sar', '5.1.6') == '5.1.6'


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
