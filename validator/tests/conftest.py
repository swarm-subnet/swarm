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

"""Pytest session setup for the validator suite: bittensor and capnp are installed real or stubbed."""

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
    """Return a fake bittensor module whose logging, wallet, axon and subtensor all do nothing."""
    bt = types.ModuleType("bittensor")

    class _Logger:
        """A bt.logging replacement that swallows every line and every config hook."""
        def info(self, *args: Any, **kwargs: Any) -> None:
            """Accept an info log line and discard it."""
            return None

        def warning(self, *args: Any, **kwargs: Any) -> None:
            """Accept a warning log line and discard it."""
            return None

        def error(self, *args: Any, **kwargs: Any) -> None:
            """Accept an error log line and discard it."""
            return None

        def success(self, *args: Any, **kwargs: Any) -> None:
            """Accept a success log line and discard it."""
            return None

        def debug(self, *args: Any, **kwargs: Any) -> None:
            """Accept a debug log line and discard it."""
            return None

        def check_config(self, config: Any) -> None:
            """Accept a config object and raise nothing, whatever it holds."""
            return None

        def register_primary_logger(self, name: str) -> None:
            """Accept a logger registration and keep no record of the name."""
            return None

        def add_args(self, parser: Any) -> None:
            """Leave the parser untouched; the logging stub declares no flags."""
            return None

    class _Synapse:
        """A synapse whose constructor stores every keyword as an attribute."""
        def __init__(self, **kwargs: Any):
            """Set one attribute per keyword argument on the new synapse."""
            for key, value in kwargs.items():
                setattr(self, key, value)

    class _Wallet:
        """A wallet stub with no keys, only the parser-args hook."""
        @staticmethod
        def add_args(parser: Any) -> None:
            """Leave the parser untouched; the wallet stub declares no flags."""
            return None

    class _SubtensorClass:
        """A bt.Subtensor stand-in that contributes no command-line arguments."""
        @staticmethod
        def add_args(parser: Any) -> None:
            """Leave the parser untouched; the subtensor stub declares no flags."""
            return None

    class _Axon:
        """An axon stand-in whose only surface is the parser hook."""
        @staticmethod
        def add_args(parser: Any) -> None:
            """Leave the parser untouched; the axon stub declares no flags."""
            return None

    def _config(parser: Any) -> Any:
        """Return the parser defaults by parsing an empty argument list."""
        return parser.parse_args([])

    def _subtensor(*args: Any, **kwargs: Any) -> Any:
        """Return a subtensor object whose metagraph carries an empty hotkey list."""
        class _Metagraph:
            """A metagraph carrying an empty hotkey list."""
            hotkeys = []

        class _Subtensor:
            """A subtensor whose only call is metagraph()."""
            def metagraph(self, netuid: int) -> _Metagraph:
                """Return an empty metagraph and ignore the netuid asked for."""
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
    """Return a fake capnp module: kj loop, streams, two-party client and server, all inert."""
    capnp = types.ModuleType("capnp")

    class _KjLoop:
        """An async context manager standing in for capnp's kj event loop."""
        async def __aenter__(self):
            """Return the loop itself as the context value."""
            return self

        async def __aexit__(self, exc_type, exc, tb):
            """Return False so an exception raised inside the block propagates."""
            _ = exc_type, exc, tb
            return False

    class _AsyncIoStream:
        """Stream factory stub: a connection is a bare object, a server is a context manager that serves nothing."""
        @staticmethod
        async def create_connection(*args: Any, **kwargs: Any):
            """Return a bare object standing in for a connected capnp stream."""
            _ = args, kwargs
            return object()

        @staticmethod
        async def create_server(*args: Any, **kwargs: Any):
            """Return a server context manager that serves nothing."""
            _ = args, kwargs
            class _Server:
                """A server whose serve_forever returns at once."""
                async def __aenter__(self):
                    """Return the server itself as the context value."""
                    return self

                async def __aexit__(self, exc_type, exc, tb):
                    """Return False so an exception raised inside the block propagates."""
                    _ = exc_type, exc, tb
                    return False

                async def serve_forever(self):
                    """Return at once instead of blocking on a serve loop."""
                    return None
            return _Server()

    class _TwoPartyClient:
        """A capnp RPC client stub whose bootstrap hands back a bare object."""
        def __init__(self, stream: Any):
            """Accept and discard the stream the real client would own."""
            _ = stream

        def bootstrap(self):
            """Return an object whose cast_as hands back a bare capability."""
            class _Bootstrap:
                """A bootstrap capability that casts to a bare object."""
                def cast_as(self, _agent):
                    """Return a bare object for whatever interface is asked for."""
                    return object()
            return _Bootstrap()

    class _TwoPartyServer:
        """A capnp RPC server stub whose disconnect wait returns at once."""
        def __init__(self, stream: Any, bootstrap: Any):
            """Accept and discard the stream and the bootstrap capability."""
            _ = stream, bootstrap

        async def on_disconnect(self):
            """Return None at once instead of waiting for a peer to drop."""
            return None

    def _load(path: str):
        """Return a schema namespace with Observation, Agent and Tensor, ignoring the path."""
        _ = path

        class _Observation:
            """An Observation schema whose messages hold blank tensor entries."""
            @staticmethod
            def new_message():
                """Return a message object that can initialise a list of blank entries."""
                class _Msg:
                    """A message stub that keeps the entry list it initialises."""
                    def init(self, field: str, n: int):
                        """Attach n blank key/tensor entries to the message and return them."""
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
    """Import the real bittensor, retrying with a plain queue when SemLock is denied."""
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
    """Put bittensor and capnp into sys.modules, real or stubbed, and set the ansible temp dirs."""
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
    """The bittensor module the session installed, real or stubbed."""
    return sys.modules["bittensor"]


@pytest.fixture
def reload_module():
    """A callable that drops a module from sys.modules and imports it fresh."""
    def _reload(module_name: str):
        """Return the named module after forcing a fresh import of it."""
        if module_name in sys.modules:
            del sys.modules[module_name]
        return importlib.import_module(module_name)

    return _reload
