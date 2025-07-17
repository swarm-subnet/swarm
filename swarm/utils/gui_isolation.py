"""
swarm.utils.gui_isolation
─────────────────────────
Run a function in a dedicated *sub‑process* **iff** ``gui=True`` is passed.
The child sends its result (or traceback) over a unidirectional pipe and
terminates with `os._exit(0)` immediately afterwards, so that no C++ /
OpenGL / PyBullet destructors can crash the parent process.

Typical use
-----------

    from swarm.utils.gui_isolation import run_isolated

    res = run_isolated(my_callable, *args, gui=True, **kwargs)

If ``gui`` evaluates to *False* the call happens in‑process (fast path).
"""
from __future__ import annotations

import multiprocessing as mp
import os
import traceback
from typing import Any, Callable, Tuple


# ════════════════════════════════════════════════════════════════════════
# Internal worker
# ════════════════════════════════════════════════════════════════════════
def _worker(fn: Callable, a: Tuple, kw: dict, pipe) -> None:
    """
    Execute *fn*, send either ``("ok", result)`` or ``("err", tb_string)``
    through *pipe*, flush, and **hard‑exit** the interpreter so that
    no buggy library code can run afterwards.
    """
    try:
        out = fn(*a, **kw)
        pipe.send(("ok", out))
    except BaseException:                           # noqa: BLE001
        pipe.send(("err", traceback.format_exc()))
    finally:
        pipe.close()
        os._exit(0)                                 # noqa: PLE1102


# ════════════════════════════════════════════════════════════════════════
# Public façade
# ════════════════════════════════════════════════════════════════════════
def run_isolated(fn: Callable, *args: Any, **kwargs: Any) -> Any:
    """
    Transparent call wrapper.

    * If ``kwargs.get("gui")`` is falsy → call *fn* synchronously.
    * Otherwise execute *fn* in a *spawned* sub‑process as described above.
    """
    if not kwargs.get("gui", False):
        # Head‑less: simple direct call
        return fn(*args, **kwargs)

    # GUI path: run in an isolated interpreter ---------------------------
    try:
        mp.set_start_method("spawn", force=False)
    except RuntimeError:
        pass  # already chosen elsewhere

    parent, child = mp.Pipe(duplex=False)
    proc = mp.Process(target=_worker, args=(fn, args, kwargs, child), daemon=True)
    proc.start()
    child.close()          # the parent never uses this end

    try:
        status, payload = parent.recv()             # blocks
    except EOFError:
        proc.join()
        raise RuntimeError(
            f"GUI subprocess terminated abnormally "
            f"(exit code {proc.exitcode})."
        ) from None
    finally:
        proc.join()

    if status == "ok":
        return payload
    raise RuntimeError("GUI subprocess raised:\n" + payload)  # status == "err"
