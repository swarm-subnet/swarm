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

from swarm.challenge_families import DEFAULT_RUNTIME_FAMILY_ID

from ._shared import *

PUBLISH_BACKOFF_CAP_CYCLES = 12


def _seed_manager_method(seed_manager, method_name: str, *args, **kwargs):
    method = getattr(seed_manager, method_name)
    try:
        return method(*args, **kwargs)
    except TypeError:
        return method(*args)


def _publish_backoff_cycles(failures: int) -> int:
    """Return how many forward cycles to skip before retrying a failed publish.

    Sequence: 1 → 1, 2 → 2, 3 → 4, 4 → 8, 5+ → 12 (capped). Doubles per
    failure so a permanently broken backend stops spamming logs while
    transient outages still recover quickly.
    """
    if failures <= 0:
        return 0
    return min(PUBLISH_BACKOFF_CAP_CYCLES, 2 ** (failures - 1))


async def _publish_pending_epoch_seeds(self) -> None:
    """Publish unpublished past-epoch seed records to the backend.

    Marks an epoch as published only when the backend confirms acceptance,
    so rejected or failed calls remain pending. Repeatedly failing epochs
    back off exponentially (up to ``PUBLISH_BACKOFF_CAP_CYCLES``) so
    permanent rejections do not flood logs every cycle.
    """
    skip_until: Dict[tuple[int, str], int] = getattr(self, "_epoch_publish_skip_until", {})
    failures: Dict[tuple[int, str], int] = getattr(self, "_epoch_publish_failures", {})
    current_cycle = int(getattr(self, "forward_count", 0))

    for pub in _seed_manager_method(self.seed_manager, "get_pending_publications"):
        ep = pub.get("epoch_number")
        family_id = str(pub.get("family_id") or DEFAULT_RUNTIME_FAMILY_ID)
        if ep is None:
            continue
        scope = (int(ep), family_id)

        if current_cycle < skip_until.get(scope, 0):
            continue

        try:
            response = await self.backend_api.publish_epoch_seeds(
                epoch_number=ep,
                family_id=family_id,
                seeds=pub.get("seeds", []),
                started_at=pub.get("started_at", ""),
                ended_at=pub.get("ended_at", ""),
                benchmark_version=pub.get("benchmark_version"),
            )
        except Exception as e:
            bt.logging.warning(f"Failed to publish epoch {ep} ({family_id}) seeds: {e}")
            failures[scope] = failures.get(scope, 0) + 1
            skip_until[scope] = current_cycle + _publish_backoff_cycles(failures[scope])
            continue

        accepted = isinstance(response, dict) and (
            response.get("published") or response.get("accepted")
        )
        if accepted:
            _seed_manager_method(
                self.seed_manager,
                "mark_epoch_published",
                ep,
                family_id=family_id,
            )
            failures.pop(scope, None)
            skip_until.pop(scope, None)
            bt.logging.info(f"Published epoch {ep} ({family_id}) seeds to backend")
        else:
            failures[scope] = failures.get(scope, 0) + 1
            backoff = _publish_backoff_cycles(failures[scope])
            skip_until[scope] = current_cycle + backoff
            bt.logging.warning(
                f"Backend rejected epoch {ep} ({family_id}) publish "
                f"(attempt {failures[scope]}); "
                f"retry in {backoff} cycle(s): {response}"
            )

    self._epoch_publish_skip_until = skip_until
    self._epoch_publish_failures = failures


# ──────────────────────────────────────────────────────────────────────────
# Queue worker – processes a single normal-model pipeline item
# ──────────────────────────────────────────────────────────────────────────
