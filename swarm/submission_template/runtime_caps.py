"""Trusted inference thread ceilings installed before miner code is imported."""

from __future__ import annotations

import functools
import importlib.abc
import importlib.machinery
import os
import sys
from dataclasses import dataclass
from types import ModuleType
from typing import Callable, Optional


@dataclass(frozen=True)
class ThreadCaps:
    intra_op: int
    inter_op: int
    torch_intra_op: Optional[int] = None
    torch_inter_op: Optional[int] = None
    onnxruntime_enabled: bool = True

    @property
    def effective_torch_intra_op(self) -> int:
        return self.torch_intra_op or self.intra_op

    @property
    def effective_torch_inter_op(self) -> int:
        return self.torch_inter_op or self.inter_op


def _positive_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return max(1, int(default))


def _configured_caps() -> Optional[ThreadCaps]:
    raw = os.environ.get("SWARM_INFERENCE_THREADS")
    if raw in (None, ""):
        torch_raw = (
            os.environ.get("SWARM_TORCH_NUM_THREADS")
            or os.environ.get("SWARM_TORCH_THREADS")
        )
        if torch_raw in (None, ""):
            return None
        try:
            torch_intra_op = max(1, int(torch_raw))
        except (TypeError, ValueError):
            return None
        torch_inter_op = _positive_env("SWARM_TORCH_INTEROP_THREADS", 1)
        return ThreadCaps(
            intra_op=torch_intra_op,
            inter_op=torch_inter_op,
            torch_intra_op=torch_intra_op,
            torch_inter_op=torch_inter_op,
            onnxruntime_enabled=False,
        )
    try:
        intra_op = max(1, int(raw))
    except (TypeError, ValueError):
        return None
    return ThreadCaps(
        intra_op=min(
            intra_op,
            _positive_env("SWARM_ORT_INTRA_OP_THREADS", intra_op),
        ),
        inter_op=min(
            intra_op,
            _positive_env("SWARM_ORT_INTER_OP_THREADS", 1),
        ),
        torch_intra_op=min(
            intra_op,
            _positive_env("SWARM_TORCH_NUM_THREADS", intra_op),
        ),
        torch_inter_op=min(
            intra_op,
            _positive_env("SWARM_TORCH_INTEROP_THREADS", 1),
        ),
    )


def _bounded(current: object, ceiling: int) -> int:
    try:
        value = int(current)
    except (TypeError, ValueError):
        value = 0
    return ceiling if value <= 0 else min(value, ceiling)


def _patch_onnxruntime(module: ModuleType, caps: ThreadCaps) -> None:
    session_class = getattr(module, "InferenceSession", None)
    options_class = getattr(module, "SessionOptions", None)
    if session_class is None or options_class is None:
        return
    original = getattr(session_class, "__init__", None)
    if original is None or getattr(original, "__swarm_thread_capped__", False):
        return

    @functools.wraps(original)
    def capped_init(self, path_or_bytes, sess_options=None, *args, **kwargs):
        options = sess_options if sess_options is not None else options_class()
        try:
            options.intra_op_num_threads = _bounded(
                options.intra_op_num_threads, caps.intra_op
            )
        except Exception:
            pass
        try:
            options.inter_op_num_threads = _bounded(
                options.inter_op_num_threads, caps.inter_op
            )
        except Exception:
            pass
        return original(self, path_or_bytes, options, *args, **kwargs)

    capped_init.__swarm_thread_capped__ = True
    session_class.__init__ = capped_init


def _patch_torch(module: ModuleType, caps: ThreadCaps) -> None:
    intra_op = caps.effective_torch_intra_op
    inter_op = caps.effective_torch_inter_op
    original_set = getattr(module, "set_num_threads", None)
    if callable(original_set) and not getattr(
        original_set, "__swarm_thread_capped__", False
    ):
        original_set(intra_op)

        @functools.wraps(original_set)
        def capped_set_num_threads(value):
            return original_set(_bounded(value, intra_op))

        capped_set_num_threads.__swarm_thread_capped__ = True
        module.set_num_threads = capped_set_num_threads

    original_interop = getattr(module, "set_num_interop_threads", None)
    if callable(original_interop) and not getattr(
        original_interop, "__swarm_thread_capped__", False
    ):
        try:
            original_interop(inter_op)
        except RuntimeError:
            pass

        @functools.wraps(original_interop)
        def capped_set_num_interop_threads(value):
            try:
                return original_interop(_bounded(value, inter_op))
            except RuntimeError:
                return None

        capped_set_num_interop_threads.__swarm_thread_capped__ = True
        module.set_num_interop_threads = capped_set_num_interop_threads


_PATCHERS: dict[str, Callable[[ModuleType, ThreadCaps], None]] = {
    "onnxruntime": _patch_onnxruntime,
    "torch": _patch_torch,
}


class _PostImportLoader(importlib.abc.Loader):
    def __init__(
        self,
        loader: importlib.abc.Loader,
        callback: Callable[[ModuleType], None],
    ) -> None:
        self._loader = loader
        self._callback = callback

    def create_module(self, spec):
        create = getattr(self._loader, "create_module", None)
        return create(spec) if create is not None else None

    def exec_module(self, module: ModuleType) -> None:
        self._loader.exec_module(module)
        self._callback(module)


class _ThreadCapFinder(importlib.abc.MetaPathFinder):
    def __init__(self, caps: ThreadCaps) -> None:
        self._caps = caps

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "onnxruntime" and not self._caps.onnxruntime_enabled:
            return None
        patcher = _PATCHERS.get(fullname)
        if patcher is None:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.loader is None:
            return spec
        spec.loader = _PostImportLoader(
            spec.loader,
            lambda module: patcher(module, self._caps),
        )
        return spec


def install_runtime_thread_caps() -> None:
    """Cap supported inference runtimes without importing unused frameworks."""
    caps = _configured_caps()
    if caps is None:
        return

    for name, patcher in _PATCHERS.items():
        if name == "onnxruntime" and not caps.onnxruntime_enabled:
            continue
        module = sys.modules.get(name)
        if module is not None:
            patcher(module, caps)

    if not any(isinstance(finder, _ThreadCapFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _ThreadCapFinder(caps))
