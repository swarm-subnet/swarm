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

"""Inference thread ceilings hold over miner-supplied options and late runtime imports."""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

from swarm.submission_template import runtime_caps


def test_onnxruntime_default_session_gets_worker_thread_options():
    """A session built with no options still receives the capped intra and inter op counts."""
    class Options:
        """Stub session options holding only the two thread-count attributes."""
        def __init__(self):
            """Leave both counts at 0, the value onnxruntime reads as unset."""
            self.intra_op_num_threads = 0
            self.inter_op_num_threads = 0

    class Session:
        """Stub InferenceSession that keeps the path and the options it was handed."""
        def __init__(self, path, options=None, *args, **kwargs):
            """Store the model path and the session options for later inspection."""
            self.path = path
            self.options = options

    module = SimpleNamespace(InferenceSession=Session, SessionOptions=Options)
    runtime_caps._patch_onnxruntime(module, runtime_caps.ThreadCaps(2, 1))

    session = module.InferenceSession("policy.onnx")

    assert session.options.intra_op_num_threads == 2
    assert session.options.inter_op_num_threads == 1


def test_onnxruntime_preserves_safe_explicit_value_and_clamps_high_value():
    """An explicit thread request below the ceiling is kept, one above it is pulled down."""
    class Options:
        """Stub session options that start both counts at the same requested number."""
        def __init__(self, intra):
            """Set the intra-op and inter-op counts to the caller's requested value."""
            self.intra_op_num_threads = intra
            self.inter_op_num_threads = intra

    class Session:
        """Stub InferenceSession that keeps only the options it was constructed with."""
        def __init__(self, path, options=None, *args, **kwargs):
            """Store the session options and drop the model path."""
            self.options = options

    module = SimpleNamespace(
        InferenceSession=Session,
        SessionOptions=lambda: Options(0),
    )
    runtime_caps._patch_onnxruntime(module, runtime_caps.ThreadCaps(2, 1))

    safe = module.InferenceSession("safe.onnx", Options(1))
    excessive = module.InferenceSession("excessive.onnx", Options(16))

    assert safe.options.intra_op_num_threads == 1
    assert safe.options.inter_op_num_threads == 1
    assert excessive.options.intra_op_num_threads == 2
    assert excessive.options.inter_op_num_threads == 1


def test_torch_defaults_and_future_public_setters_are_capped():
    """Patching torch applies both ceilings at once and clamps every later setter call."""
    intra_calls = []
    inter_calls = []

    def set_num_threads(value):
        """Record the intra-op count torch was asked to use."""
        intra_calls.append(value)

    def set_num_interop_threads(value):
        """Record the inter-op count torch was asked to use."""
        inter_calls.append(value)

    module = SimpleNamespace(
        set_num_threads=set_num_threads,
        set_num_interop_threads=set_num_interop_threads,
    )
    runtime_caps._patch_torch(module, runtime_caps.ThreadCaps(2, 1))

    module.set_num_threads(64)
    module.set_num_threads(1)
    module.set_num_interop_threads(64)

    assert intra_calls == [2, 2, 1]
    assert inter_calls == [1, 1]


def test_caps_are_disabled_only_by_absent_canonical_env(monkeypatch):
    """No thread variable in the environment means no ceilings; the canonical one with the per-runtime ones gives all four counts."""
    monkeypatch.delenv("SWARM_INFERENCE_THREADS", raising=False)
    monkeypatch.delenv("SWARM_TORCH_NUM_THREADS", raising=False)
    monkeypatch.delenv("SWARM_TORCH_THREADS", raising=False)
    assert runtime_caps._configured_caps() is None

    monkeypatch.setenv("SWARM_INFERENCE_THREADS", "2")
    monkeypatch.setenv("SWARM_ORT_INTRA_OP_THREADS", "2")
    monkeypatch.setenv("SWARM_ORT_INTER_OP_THREADS", "1")
    monkeypatch.setenv("SWARM_TORCH_NUM_THREADS", "2")
    monkeypatch.setenv("SWARM_TORCH_INTEROP_THREADS", "1")
    assert runtime_caps._configured_caps() == runtime_caps.ThreadCaps(2, 1, 2, 1)


def test_explicit_legacy_torch_cap_does_not_enable_onnxruntime(monkeypatch):
    """The legacy torch variables alone bound torch and leave ONNX Runtime unpatched."""
    monkeypatch.delenv("SWARM_INFERENCE_THREADS", raising=False)
    monkeypatch.setenv("SWARM_TORCH_THREADS", "1")
    monkeypatch.setenv("SWARM_TORCH_INTEROP_THREADS", "1")

    assert runtime_caps._configured_caps() == runtime_caps.ThreadCaps(
        1,
        1,
        1,
        1,
        onnxruntime_enabled=False,
    )


def test_post_import_finder_patches_module_after_execution(tmp_path, monkeypatch):
    """The finder runs its patcher once the module body has already run, and the change sticks."""
    module_name = "_swarm_runtime_caps_import_probe"
    (tmp_path / f"{module_name}.py").write_text("loaded = True\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    patched = []

    def patcher(module, caps):
        """Note the executed module's state and the ceiling, then mark the module capped."""
        patched.append((module.loaded, caps.intra_op))
        module.capped = True

    monkeypatch.setitem(runtime_caps._PATCHERS, module_name, patcher)
    finder = runtime_caps._ThreadCapFinder(runtime_caps.ThreadCaps(2, 1))
    sys.meta_path.insert(0, finder)
    try:
        module = importlib.import_module(module_name)
    finally:
        sys.meta_path.remove(finder)
        sys.modules.pop(module_name, None)

    assert module.capped is True
    assert patched == [(True, 2)]
