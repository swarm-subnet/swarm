"""Camera byte conversion retains float32 division bits."""

from types import SimpleNamespace

import numpy as np
import pytest

from swarm.challenge_families.solar_patrol import camera


@pytest.mark.parametrize("shape", [(16, 16), (480, 640), (512, 640)])
def test_frame_conversion_bits(monkeypatch, shape):
    """Every byte value and real frame shapes match the original cast and divide."""
    height, width = shape
    rgba = np.resize(np.repeat(np.arange(256, dtype=np.uint8), 4), (height, width, 4))
    if height > 16:
        rgba = np.random.default_rng(4).integers(0, 256, rgba.shape, dtype=np.uint8)
    env = SimpleNamespace(CLIENT=0, _sun=None, _render_flags=0, _sky_flags=0,
                          _daylight_flags=0, _light_direction=[0, 0, 1])
    shot = SimpleNamespace(height=height, width=width, matrices=lambda: ([], []))
    monkeypatch.setattr(camera, "RAYCAST", 0)
    monkeypatch.setattr(camera.airframe, "pose", lambda *args: None)
    monkeypatch.setattr(camera.p, "getCameraImage", lambda *args, **kwargs:
                        (width, height, rgba, None, np.zeros(shape, dtype=np.int32)))
    frame, _ = camera.colour_frame(env, shot)
    expected = rgba[:, :, :3].astype(np.float32) / 255.0
    assert frame.tobytes() == expected.tobytes()
    assert frame.flags.c_contiguous
    assert np.true_divide(rgba[:, :, :1], np.float32(255), dtype=np.float32).tobytes() == (
        rgba[:, :, :1].astype(np.float32) / 255.0).tobytes()
