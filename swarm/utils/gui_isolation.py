"""
swarm.utils.gui_isolation
─────────────────────────
Transparent helper that executes a callable in a *dedicated subprocess*
iff `gui=True`.  It guarantees:

* The parent always receives a result **first**, before the child touches
  any flaky OpenGL/Bullet shutdown code.
* The child is terminated immediately after sending the result, so no
  destructor can crash it afterwards.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import traceback
from typing import Any, Callable


# ──────────────────────────────────────────────────────────────────────
# Internal worker
# ──────────────────────────────────────────────────────────────────────
def _worker(fn: Callable, args: tuple, kwargs: dict, pipe):
    """
    Run the target callable, send ("ok", result) or ("err", traceback)
    to the parent, **flush**, and hard‑exit the interpreter with
    `os._exit(0)` so that PyBullet never has a chance to crash on quit.
    """
    try:
        out = fn(*args, **kwargs)
        pipe.send(("ok", out))
    except BaseException:                           # noqa: BLE001
        pipe.send(("err", traceback.format_exc()))
    finally:
        pipe.close()
        # Hard exit – skip atexit handlers, C++ static destructors, …
        os._exit(0)                                 # noqa: PLE1102


# ──────────────────────────────────────────────────────────────────────
# Public façade
# ──────────────────────────────────────────────────────────────────────
def run_isolated(fn: Callable, *args: Any, **kwargs: Any) -> Any:
    """
    Execute *fn* in a subprocess **only if** ``kwargs.get("gui")`` is
    truthy.  Otherwise call it synchronously.

    The call is transparent for the caller – you receive exactly what
    *fn* would have returned or a `RuntimeError` wrapping the traceback
    if the child raised.
    """
    if not kwargs.get("gui", False):
        # Head‑less → direct call (fast path)
        return fn(*args, **kwargs)

    # GUI path → spawn an isolated interpreter
    try:
        mp.set_start_method("spawn", force=False)
    except RuntimeError:
        pass  # already set elsewhere

    parent_end, child_end = mp.Pipe(duplex=False)
    proc = mp.Process(target=_worker,
                      args=(fn, args, kwargs, child_end),
                      daemon=True)
    proc.start()
    child_end.close()               # parent never uses this end

    try:
        status, payload = parent_end.recv()  # blocks until child sends
    except EOFError:
        # Child died before sending anything (hard seg‑fault, SIGKILL…)
        proc.join()
        raise RuntimeError(
            f"GUI subprocess terminated abnormally "
            f"(exit code {proc.exitcode})."
        ) from None
    finally:
        proc.join()

    if status == "ok":
        return payload
    else:  # "err"
        raise RuntimeError(
            "GUI subprocess raised an exception:\n" + payload
        ) from None
