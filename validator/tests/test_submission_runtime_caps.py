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

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

from swarm.submission_template import runtime_caps


def test_onnxruntime_default_session_gets_worker_thread_options():
    class Options:
        def __init__(self):
            self.intra_op_num_threads = 0
            self.inter_op_num_threads = 0

    class Session:
        def __init__(self, path, options=None, *args, **kwargs):
            self.path = path
            self.options = options

    module = SimpleNamespace(InferenceSession=Session, SessionOptions=Options)
    runtime_caps._patch_onnxruntime(module, runtime_caps.ThreadCaps(2, 1))

    session = module.InferenceSession("policy.onnx")

    assert session.options.intra_op_num_threads == 2
    assert session.options.inter_op_num_threads == 1


def test_onnxruntime_preserves_safe_explicit_value_and_clamps_high_value():
    class Options:
        def __init__(self, intra):
            self.intra_op_num_threads = intra
            self.inter_op_num_threads = intra

    class Session:
        def __init__(self, path, options=None, *args, **kwargs):
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
    intra_calls = []
    inter_calls = []

    def set_num_threads(value):
        intra_calls.append(value)

    def set_num_interop_threads(value):
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
    module_name = "_swarm_runtime_caps_import_probe"
    (tmp_path / f"{module_name}.py").write_text("loaded = True\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    patched = []

    def patcher(module, caps):
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
