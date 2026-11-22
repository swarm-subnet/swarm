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
"""Shared observation buffers retain exactly the uncached payload and message."""

from pathlib import Path

import capnp
import numpy as np
import pytest

from swarm.core.observation.frozen import freeze, is_frozen
from swarm.validator.docker.docker_evaluator_parts import submission


def test_only_the_array_its_producer_froze_counts_as_frozen():
    """A view of a frozen frame, or a frame only flagged read-only, is never vouched for."""
    frame = freeze(np.ones(4, dtype=np.float32))
    flagged = np.ones(4, dtype=np.float32)
    flagged.flags.writeable = False
    assert is_frozen(frame)
    assert not frame.flags.writeable
    assert not is_frozen(frame.view())
    assert not is_frozen(flagged)


def test_cache_matches_uncached_sequence(monkeypatch):
    """Held, mutable, reordered and overlapping frames preserve every byte."""
    schema = capnp.load(str(Path('swarm/submission_template/agent.capnp').resolve()))
    buffers = [bytearray(16_000_000), bytearray(16_000_000)]
    cache = []
    frame = freeze(np.ones((480, 640, 3), dtype=np.float32))
    blank = freeze(np.zeros((512, 640, 1), dtype=np.float32))
    mutable = np.zeros_like(frame)
    view_base = np.zeros_like(frame)
    view = view_base.view()
    view.flags.writeable = False
    observations = [dict(rgb=mutable), dict(rgb=frame, thermal=blank), dict(rgb=frame, thermal=blank),
                    dict(thermal=frame, rgb=frame), dict(prefix=np.ones(16), rgb=frame),
                    dict(rgb=mutable), dict(rgb=view), dict(rgb=frame)]
    for obs in observations:
        for _ in range(2):
            old = submission._serialize_observation_shm(schema, obs, buffers[0])
            new = submission._serialize_observation_shm(schema, obs, buffers[1], cache)
            assert old.to_bytes() == new.to_bytes()
            assert buffers[0] == buffers[1]
            if obs.get("rgb") is mutable:
                mutable.flat[0] += np.float32(1)
            if obs.get("rgb") is view:
                view_base.flat[0] += np.float32(1)
        mutable[0, 0, 0] = 7
        view_base[0, 0, 0] = 9
    resized = np.array([0, 0, 1], dtype=np.float32)
    resized.flags.writeable = False
    for size in (3, 2):
        resized.resize(size, refcheck=False)
        old = submission._serialize_observation_shm(schema, resized, buffers[0])
        new = submission._serialize_observation_shm(schema, resized, buffers[1], cache)
        assert old.to_bytes() == new.to_bytes()
        assert buffers[0] == buffers[1]
    restrided = np.zeros((2, 2), dtype=np.float32)
    restrided[1, 1] = 1
    freeze(restrided)
    for strides in ((8, 4), (0, 4)):
        restrided.strides = strides
        old = submission._serialize_observation_shm(schema, restrided, buffers[0])
        new = submission._serialize_observation_shm(schema, restrided, buffers[1], cache)
        assert old.to_bytes() == new.to_bytes()
        assert buffers[0] == buffers[1]
    with pytest.raises(BufferError):
        submission._serialize_observation_shm(schema, {'rgb': frame}, bytearray(1), cache)
    assert cache == []
    with pytest.raises(ValueError):
        frame[0, 0, 0] = 0


def test_held_frame_skips_scan(monkeypatch):
    """A frozen held frame needs no second zero scan."""
    schema = capnp.load(str(Path('swarm/submission_template/agent.capnp').resolve()))
    frame = freeze(np.ones((480, 640, 3), dtype=np.float32))
    cache, buffer = [], bytearray(frame.nbytes)
    submission._serialize_observation_shm(schema, frame, buffer, cache)
    def fail_scan(arr):
        """Reject a repeated scan."""
        raise AssertionError('held frame scanned')
    monkeypatch.setattr(submission, '_all_zero_bits', fail_scan)
    monkeypatch.setattr(submission.np, 'frombuffer', fail_scan)
    submission._serialize_observation_shm(schema, frame, buffer, cache)


def test_a_frame_frozen_only_by_its_flag_is_copied_again():
    """A writable view taken before the flag was cleared can still change the bytes, so only producer-frozen frames
    are trusted."""
    schema = capnp.load(str(Path('swarm/submission_template/agent.capnp').resolve()))
    frame = np.ones(4, dtype=np.float32)
    writer = frame.view()
    frame.flags.writeable = False
    cache, buffer, fresh = [], bytearray(64), bytearray(64)
    submission._serialize_observation_shm(schema, frame, buffer, cache)
    writer[0] = 9
    cached = submission._serialize_observation_shm(schema, frame, buffer, cache)
    uncached = submission._serialize_observation_shm(schema, frame, fresh)
    assert cached.to_bytes() == uncached.to_bytes()
    assert buffer == fresh


def test_a_cache_entry_never_vouches_for_another_buffer():
    """A frame written into one buffer is written again in full when the same cache serves a new buffer."""
    schema = capnp.load(str(Path('swarm/submission_template/agent.capnp').resolve()))
    frame = freeze(np.ones(4, dtype=np.float32))
    cache, first, second, fresh = [], bytearray(64), bytearray(64), bytearray(64)
    submission._serialize_observation_shm(schema, frame, first, cache)
    cached = submission._serialize_observation_shm(schema, frame, second, cache)
    uncached = submission._serialize_observation_shm(schema, frame, fresh)
    assert cached.to_bytes() == uncached.to_bytes()
    assert second == fresh


@pytest.mark.parametrize("initial", [0, 1])
def test_a_producer_frozen_frame_made_writable_is_copied_again(initial):
    """Re-enabled writes invalidate both cached zero scans and cached frame bytes."""
    schema = capnp.load(str(Path("swarm/submission_template/agent.capnp").resolve()))
    frame = freeze(np.full(4, initial, dtype=np.float32))
    cache, buffer, fresh = [], bytearray(64), bytearray(64)
    submission._serialize_observation_shm(schema, frame, buffer, cache)
    frame.flags.writeable = True
    frame[0] = 9
    cached = submission._serialize_observation_shm(schema, frame, buffer, cache)
    uncached = submission._serialize_observation_shm(schema, frame, fresh)
    assert cached.to_bytes() == uncached.to_bytes()
    assert buffer == fresh
    assert cache == []
