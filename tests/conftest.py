from __future__ import annotations

import importlib
import multiprocessing as mp
import os
import queue
import sys
import types
from pathlib import Path
from typing import Any

import pytest


def _make_bittensor_stub() -> types.ModuleType:
    bt = types.ModuleType("bittensor")

    class _Logger:
        def info(self, *args: Any, **kwargs: Any) -> None:
            return None

        def warning(self, *args: Any, **kwargs: Any) -> None:
            return None

        def error(self, *args: Any, **kwargs: Any) -> None:
            return None

        def success(self, *args: Any, **kwargs: Any) -> None:
            return None

        def debug(self, *args: Any, **kwargs: Any) -> None:
            return None

        def check_config(self, config: Any) -> None:
            return None

        def register_primary_logger(self, name: str) -> None:
            return None

        def add_args(self, parser: Any) -> None:
            return None

    class _Synapse:
        def __init__(self, **kwargs: Any):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class _Wallet:
        @staticmethod
        def add_args(parser: Any) -> None:
            return None

    class _SubtensorClass:
        @staticmethod
        def add_args(parser: Any) -> None:
            return None

    class _Axon:
        @staticmethod
        def add_args(parser: Any) -> None:
            return None

    def _config(parser: Any) -> Any:
        return parser.parse_args([])

    def _subtensor(*args: Any, **kwargs: Any) -> Any:
        class _Metagraph:
            hotkeys = []

        class _Subtensor:
            def metagraph(self, netuid: int) -> _Metagraph:
                _ = netuid
                return _Metagraph()

        return _Subtensor()

    bt.logging = _Logger()
    bt.Synapse = _Synapse
    bt.Wallet = _Wallet
    bt.Subtensor = _SubtensorClass
    bt.Axon = _Axon
    bt.Config = _config
    bt.subtensor = _subtensor
    return bt


def _make_capnp_stub() -> types.ModuleType:
    capnp = types.ModuleType("capnp")

    class _KjLoop:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            _ = exc_type, exc, tb
            return False

    class _AsyncIoStream:
        @staticmethod
        async def create_connection(*args: Any, **kwargs: Any):
            _ = args, kwargs
            return object()

        @staticmethod
        async def create_server(*args: Any, **kwargs: Any):
            _ = args, kwargs
            class _Server:
                async def __aenter__(self):
                    return self

                async def __aexit__(self, exc_type, exc, tb):
                    _ = exc_type, exc, tb
                    return False

                async def serve_forever(self):
                    return None
            return _Server()

    class _TwoPartyClient:
        def __init__(self, stream: Any):
            _ = stream

        def bootstrap(self):
            class _Bootstrap:
                def cast_as(self, _agent):
                    return object()
            return _Bootstrap()

    class _TwoPartyServer:
        def __init__(self, stream: Any, bootstrap: Any):
            _ = stream, bootstrap

        async def on_disconnect(self):
            return None

    def _load(path: str):
        _ = path

        class _Observation:
            @staticmethod
            def new_message():
                class _Msg:
                    def init(self, field: str, n: int):
                        _ = field
                        entries = []
                        for _i in range(n):
                            entries.append(
                                types.SimpleNamespace(
                                    key="",
                                    tensor=types.SimpleNamespace(data=b"", shape=[], dtype=""),
                                )
                            )
                        self.entries = entries
                        return entries
                return _Msg()

        return types.SimpleNamespace(Observation=_Observation, Agent=object, Tensor=object)

    capnp.kj_loop = lambda: _KjLoop()
    capnp.AsyncIoStream = _AsyncIoStream
    capnp.TwoPartyClient = _TwoPartyClient
    capnp.TwoPartyServer = _TwoPartyServer
    capnp.load = _load
    return capnp


def _import_bittensor_with_fallback() -> types.ModuleType:
    try:
        import bittensor as real_bt  # type: ignore
        return real_bt
    except PermissionError:
        # Some sandboxed environments disallow SemLock creation used by bittensor logging.
        # Retry with an in-process queue so tests can still import the real package.
        original_queue = mp.Queue
        mp.Queue = lambda maxsize=-1: queue.Queue(maxsize=0 if maxsize < 0 else maxsize)  # type: ignore[assignment]
        try:
            sys.modules.pop("bittensor", None)
            import bittensor as real_bt  # type: ignore
            return real_bt
        finally:
            mp.Queue = original_queue  # type: ignore[assignment]


def pytest_sessionstart(session: pytest.Session) -> None:
    _ = session
    use_stub_bt = os.getenv("SWARM_TEST_USE_STUB_BITTENSOR", "0") == "1"
    use_stub_capnp = os.getenv("SWARM_TEST_USE_STUB_CAPNP", "0") == "1"

    ansible_tmp = Path("/tmp") / "swarm_ansible_tmp"
    ansible_tmp.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("ANSIBLE_LOCAL_TEMP", str(ansible_tmp))
    os.environ.setdefault("ANSIBLE_REMOTE_TEMP", str(ansible_tmp))

    if use_stub_bt:
        sys.modules["bittensor"] = _make_bittensor_stub()
    else:
        try:
            real_bt = _import_bittensor_with_fallback()
            sys.modules["bittensor"] = real_bt
        except Exception as exc:
            raise RuntimeError(
                "bittensor is required for the default test run. "
                "Install requirements or set SWARM_TEST_USE_STUB_BITTENSOR=1."
            ) from exc

    if use_stub_capnp:
        sys.modules["capnp"] = _make_capnp_stub()
    elif "capnp" not in sys.modules:
        try:
            import capnp as real_capnp  # type: ignore
            sys.modules["capnp"] = real_capnp
        except Exception as exc:
            raise RuntimeError(
                "pycapnp is required for the default test run. "
                "Install requirements or set SWARM_TEST_USE_STUB_CAPNP=1."
            ) from exc


@pytest.fixture
def bt_stub() -> types.ModuleType:
    return sys.modules["bittensor"]


@pytest.fixture
def reload_module():
    def _reload(module_name: str):
        if module_name in sys.modules:
            del sys.modules[module_name]
        return importlib.import_module(module_name)

    return _reload
