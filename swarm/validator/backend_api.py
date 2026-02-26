"""
Backend API Client for Swarm v4 Benchmark System.

Validators need to report scores to backend. Backend aggregates
scores from all validators (51% stake, median) and calculates final weights.
This creates HTTP client to talk to backend.

Endpoints (all under /validators prefix):
- POST /validators/models/new           - Tell backend "I found a new model"
- POST /validators/models/{uid}/screening - Submit screening result (200 private seeds)
- POST /validators/models/{uid}/score   - Submit full benchmark score (1200 seeds)
- GET  /validators/sync                 - Get current weights + re-eval queue

Freeze-last behavior:
- If backend is down → use last known weights (saved locally)
- Validator doesn't crash, keeps running with old weights

Rate Limiting:
- Each miner hotkey can only submit ONE model ever (lifetime limit)
- Backend tracks this and rejects duplicate submissions

Scoring Thresholds (calculated by backend):
- Screening pass: score >= 0.1 OR score >= 80% of current top model
- Full benchmark: 51% stake must report before score is finalized
- Champion: highest benchmark score after 51% threshold met
"""

import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Dict, Any, Optional

import httpx
import bittensor as bt

from swarm.constants import BENCHMARK_VERSION

STATE_DIR = Path(__file__).parent.parent.parent / "state"
RUNTIME_STATE_FILE = STATE_DIR / "runtime_state.json"


def _load_runtime_state() -> dict:
    """Load runtime state (last known weights, re-eval queue)."""
    try:
        if RUNTIME_STATE_FILE.exists():
            with open(RUNTIME_STATE_FILE, 'r') as f:
                return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        bt.logging.warning(f"Runtime state load failed: {e}")
    return {"last_weights": {}, "reeval_queue": [], "last_sync": 0}


def _save_runtime_state(state: dict) -> None:
    """Save runtime state atomically."""
    STATE_DIR.mkdir(exist_ok=True)
    temp_file = RUNTIME_STATE_FILE.with_suffix(".tmp")
    try:
        with open(temp_file, 'w') as f:
            json.dump(state, f)
        temp_file.replace(RUNTIME_STATE_FILE)
    except IOError as e:
        bt.logging.error(f"Runtime state save failed: {e}")
        temp_file.unlink(missing_ok=True)


class BackendApiClient:
    """HTTP client for backend API communication with signature authentication."""

    def __init__(
        self,
        wallet: "bt.wallet" = None,
        base_url: str = None,
        timeout: float = 30.0
    ):
        """Initialize backend API client.

        Args:
            wallet: Bittensor wallet for signing requests. If None, will try to sign
                    but requests may fail auth on backend.
            base_url: Backend URL. If None, reads from SWARM_BACKEND_API_URL env var.
            timeout: Request timeout in seconds.

        Raises:
            ValueError: If SWARM_BACKEND_API_URL is not set.
        """
        self.base_url = base_url or os.getenv("SWARM_BACKEND_API_URL")
        if not self.base_url:
            raise ValueError(
                "SWARM_BACKEND_API_URL env var required for v4 benchmark. "
                "Set it to your backend server URL."
            )

        self.base_url = self.base_url.rstrip("/")
        self.timeout = timeout
        self.wallet = wallet
        self.client = httpx.AsyncClient(timeout=timeout)

        self._runtime_state = _load_runtime_state()
        bt.logging.info(f"BackendApiClient initialized: {self.base_url}")

    def set_wallet(self, wallet: "bt.wallet") -> None:
        """Set wallet for signing requests (can be called after init)."""
        self.wallet = wallet

    async def close(self) -> None:
        """Close HTTP client."""
        await self.client.aclose()

    def _sign_request(self, method: str, endpoint: str, body: bytes) -> Dict[str, str]:
        """Create authentication headers with signed request."""
        if not self.wallet:
            bt.logging.warning("No wallet configured - requests will not be signed")
            return {}

        nonce = str(uuid.uuid4())
        timestamp = str(int(time.time()))
        body_hash = hashlib.sha256(body).hexdigest()
        path = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        message = f"{timestamp}:{nonce}:{method.upper()}:{path}:{body_hash}"

        signature = self.wallet.hotkey.sign(message.encode()).hex()

        return {
            "X-Validator-Hotkey": self.wallet.hotkey.ss58_address,
            "X-Validator-Signature": signature,
            "X-Validator-Nonce": nonce,
            "X-Validator-Timestamp": timestamp,
        }

    async def _post_signed(self, endpoint: str, data: dict) -> Dict[str, Any]:
        """Make a signed POST request to the backend."""
        body = json.dumps(data).encode()
        headers = self._sign_request("POST", endpoint, body)
        headers["Content-Type"] = "application/json"

        try:
            resp = await self.client.post(
                f"{self.base_url}{endpoint}",
                content=body,
                headers=headers
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            bt.logging.warning(f"Backend rejected {endpoint}: {e.response.status_code}")
            try:
                return e.response.json()
            except:
                return {"error": str(e), "status_code": e.response.status_code}
        except Exception as e:
            bt.logging.warning(f"Backend API error ({endpoint}): {e}")
            return {"error": str(e)}

    async def _get_signed(self, endpoint: str) -> Dict[str, Any]:
        """Make a signed GET request to the backend."""
        body = b""
        headers = self._sign_request("GET", endpoint, body)

        try:
            resp = await self.client.get(
                f"{self.base_url}{endpoint}",
                headers=headers
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            bt.logging.warning(f"Backend API error ({endpoint}): {e}")
            return {"error": str(e)}

    # ──────────────────────────────────────────────────────────────────────
    # POST /validators/models/new
    # ──────────────────────────────────────────────────────────────────────
    async def post_new_model(
        self,
        uid: int,
        model_hash: str,
        coldkey: str,
        validator_hotkey: str
    ) -> Dict[str, Any]:
        """Notify backend of new model.

        Args:
            uid: Miner UID
            model_hash: SHA256 hash of model
            coldkey: Miner coldkey
            validator_hotkey: This validator's hotkey (used to get miner hotkey)

        Returns:
            {"accepted": True, "model_id": 123} or {"accepted": False, "reason": "..."}
        """
        miner_hotkey = self._get_miner_hotkey(uid)
        if not miner_hotkey:
            bt.logging.warning(f"Cannot register UID {uid}: miner hotkey unavailable")
            return {"accepted": False, "reason": "miner hotkey unavailable"}

        result = await self._post_signed(
            "/validators/models/new",
            {
                "uid": uid,
                "model_hash": model_hash,
                "coldkey": coldkey,
                "hotkey": miner_hotkey
            }
        )

        # Map backend response to expected format
        if "model_id" in result:
            return {"accepted": True, "model_id": result["model_id"]}
        elif "error" in result or "detail" in result:
            return {"accepted": False, "reason": result.get("detail", result.get("error", "unknown"))}
        return result

    def _get_miner_hotkey(self, uid: int) -> str:
        """Get miner hotkey from metagraph by UID."""
        try:
            subtensor = bt.subtensor(network="finney")
            metagraph = subtensor.metagraph(netuid=124)
            if 0 <= uid < len(metagraph.hotkeys):
                return metagraph.hotkeys[uid]
        except Exception as e:
            bt.logging.warning(f"Failed to get miner hotkey for UID {uid}: {e}")
        return ""

    # ──────────────────────────────────────────────────────────────────────
    # POST /validators/models/{uid}/screening
    # ──────────────────────────────────────────────────────────────────────
    async def post_screening(
        self,
        uid: int,
        validator_hotkey: str,
        validator_stake: float,
        screening_score: float,
        passed: bool
    ) -> Dict[str, Any]:
        """Submit screening result.

        Args:
            uid: Miner UID
            validator_hotkey: This validator's hotkey (sent in auth headers)
            validator_stake: This validator's stake (not used, backend gets from chain)
            screening_score: Score from 200 private seeds
            passed: Whether model passed screening threshold

        Returns:
            Backend response or error dict.
        """
        return await self._post_signed(
            f"/validators/models/{uid}/screening",
            {
                "score": screening_score,
                "passed": passed
            }
        )

    # ──────────────────────────────────────────────────────────────────────
    # POST /validators/models/{uid}/score
    # ──────────────────────────────────────────────────────────────────────
    async def post_score(
        self,
        uid: int,
        validator_hotkey: str,
        validator_stake: float,
        model_hash: str,
        total_score: float,
        per_type_scores: Dict[str, float],
        seeds_evaluated: int,
        epoch_number: Optional[int] = None
    ) -> Dict[str, Any]:
        data = {
            "score": total_score,
            "per_type_scores": per_type_scores,
            "seeds_evaluated": seeds_evaluated,
            "benchmark_version": BENCHMARK_VERSION,
        }
        if epoch_number is not None:
            data["epoch_number"] = epoch_number
        return await self._post_signed(f"/validators/models/{uid}/score", data)

    # ──────────────────────────────────────────────────────────────────────
    # GET /validators/sync
    # ──────────────────────────────────────────────────────────────────────
    async def sync(self) -> Dict[str, Any]:
        """Get current weights and re-eval queue from backend.

        Returns:
            {
                "current_top": {"uid": 42, "score": 0.847, "model_hash": "..."},
                "weights": {"42": 1.0, "0": 0.0},
                "reeval_queue": [{"uid": 42, "reason": "7_day_reeval"}],
                "leaderboard_version": 15
            }

            If backend is down, returns last known weights (freeze-last behavior).
        """
        try:
            data = await self._get_signed("/validators/sync")

            if "error" not in data:
                # Map backend response to expected format
                current_champion = data.get("current_champion", {})
                current_top = {}
                if current_champion:
                    current_top = {
                        "uid": current_champion.get("uid"),
                        "score": current_champion.get("benchmark_score"),
                        "model_hash": current_champion.get("model_hash")
                    }

                # Map reeval_queue to use uid
                reeval_queue = []
                for item in data.get("reeval_queue", []):
                    reeval_queue.append({
                        "uid": item.get("uid"),
                        "reason": item.get("reason")
                    })

                self._runtime_state["last_weights"] = data.get("weights", {})
                self._runtime_state["reeval_queue"] = reeval_queue
                self._runtime_state["last_sync"] = time.time()
                self._runtime_state["current_top"] = current_top
                _save_runtime_state(self._runtime_state)

                bt.logging.info(f"Backend sync successful: leaderboard v{data.get('leaderboard_version', '?')}")
                return {
                    "current_top": current_top,
                    "weights": data.get("weights", {}),
                    "reeval_queue": reeval_queue,
                    "leaderboard_version": data.get("leaderboard_version", 0)
                }

            raise Exception(data.get("error", "Unknown error"))

        except Exception as e:
            bt.logging.warning(f"Backend API error (sync): {e} - using freeze-last weights")

            return {
                "current_top": self._runtime_state.get("current_top", {}),
                "weights": self._runtime_state.get("last_weights", {}),
                "reeval_queue": self._runtime_state.get("reeval_queue", []),
                "leaderboard_version": 0,
                "fallback": True,
                "error": str(e)
            }

    # ──────────────────────────────────────────────────────────────────────
    # POST /validators/heartbeat
    # ──────────────────────────────────────────────────────────────────────
    async def post_heartbeat(
        self,
        status: str,
        current_uid: Optional[int] = None,
        progress: Optional[int] = None,
        total_seeds: Optional[int] = None
    ) -> Dict[str, Any]:
        """Post validator heartbeat for liveness tracking.

        Args:
            status: "idle", "evaluating_screening", or "evaluating_benchmark"
            current_uid: UID being evaluated (required when evaluating)
            progress: Seeds completed so far
            total_seeds: Total seeds to evaluate (200 or 1200)

        Returns:
            {"recorded": True, "message": "..."} or error dict
        """
        data = {"status": status}
        if current_uid is not None:
            data["current_uid"] = current_uid
        if progress is not None:
            data["progress"] = progress
        if total_seeds is not None:
            data["total_seeds"] = total_seeds

        return await self._post_signed("/validators/heartbeat", data)

    # ──────────────────────────────────────────────────────────────────────
    # POST /validators/epoch/publish
    # ──────────────────────────────────────────────────────────────────────
    async def publish_epoch_seeds(
        self,
        epoch_number: int,
        seeds: list[int],
        started_at: str,
        ended_at: str,
    ) -> Dict[str, Any]:
        return await self._post_signed(
            "/validators/epoch/publish",
            {
                "epoch_number": epoch_number,
                "seeds": seeds,
                "started_at": started_at,
                "ended_at": ended_at,
            }
        )
