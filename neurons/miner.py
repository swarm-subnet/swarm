#!/usr/bin/env python3
"""
Swarm Miner — one-shot model submission to the benchmark backend.

Usage:
    python neurons/miner.py \\
        --netuid 124 \\
        --subtensor.network finney \\
        --wallet.name miner \\
        --wallet.hotkey default \\
        --github_url https://github.com/yourname/your-model

The backend downloads, hashes, and verifies submission.zip from your
GitHub repository server-side.  Validators discover new models
automatically via their sync cycle.  You do NOT need to stay online.
"""

import hashlib
import json
import os
import sys
import time
import uuid

import bittensor as bt
import httpx


DEFAULT_BACKEND_URL = os.environ.get("SWARM_BACKEND_API_URL", "")
REQUEST_TIMEOUT = 120.0


def _sign_request(wallet, method: str, path: str, body: bytes) -> dict[str, str]:
    nonce = str(uuid.uuid4())
    timestamp = str(int(time.time()))
    body_hash = hashlib.sha256(body).hexdigest()
    message = f"{timestamp}:{nonce}:{method}:{path}:{body_hash}"
    signature = wallet.hotkey.sign(message.encode()).hex()
    return {
        "X-Miner-Hotkey": wallet.hotkey.ss58_address,
        "X-Miner-Signature": signature,
        "X-Miner-Nonce": nonce,
        "X-Miner-Timestamp": timestamp,
        "Content-Type": "application/json",
    }


def submit_model(wallet, github_url: str, backend_url: str) -> dict:
    """Submit a model to the backend and return the JSON response."""
    endpoint = "/miners/submit"
    payload = json.dumps({"github_url": github_url}).encode()
    headers = _sign_request(wallet, "POST", endpoint, payload)
    url = backend_url.rstrip("/") + endpoint

    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        resp = client.post(url, content=payload, headers=headers)
        data = resp.json()

        if resp.is_success and data.get("accepted"):
            return data

        detail = data.get("detail", data.get("message", f"HTTP {resp.status_code}"))
        data["_error"] = detail
        data["_status_code"] = resp.status_code
        return data


def _print_rejection_hint(error: str) -> None:
    hints = {
        "already submitted": "This hotkey has already submitted a model. Each hotkey can only submit once.",
        "one submission": "This hotkey has already submitted a model. Each hotkey can only submit once.",
        "already claimed": "This GitHub repository is registered to a different hotkey.",
        "readme": "Your repository must contain the exact template README.md from the Swarm repo.",
        "download failed": "submission.zip could not be downloaded. Verify it exists on your main/master branch.",
        "submission.zip not found": "submission.zip could not be found in your repository.",
        "zip rejected": "submission.zip failed safety checks (zip bomb or path traversal).",
        "already registered": "An identical model (same SHA-256 hash) was already submitted by someone.",
        "not registered on subnet": "This hotkey is not registered on Bittensor subnet 124.",
    }
    error_lower = error.lower()
    for pattern, hint in hints.items():
        if pattern in error_lower:
            bt.logging.error(f"  Hint: {hint}")
            return


def main(argv=None):
    parser = bt.config.parser()
    parser.add_argument("--github_url", type=str, required=True,
                        help="Public GitHub repo URL (https://github.com/owner/repo)")
    parser.add_argument("--backend_url", type=str, default=DEFAULT_BACKEND_URL,
                        help="Backend API URL (or set SWARM_BACKEND_API_URL env var)")
    parser.add_argument("--netuid", type=int, default=124)
    config = bt.config(parser, args=argv)

    backend_url = config.backend_url or DEFAULT_BACKEND_URL
    if not backend_url:
        bt.logging.error("Backend URL required. Set --backend_url or SWARM_BACKEND_API_URL env var.")
        return 1

    wallet = bt.wallet(config=config)
    hotkey = wallet.hotkey.ss58_address

    bt.logging.info(f"Hotkey:      {hotkey}")
    bt.logging.info(f"GitHub URL:  {config.github_url}")
    bt.logging.info(f"Backend:     {backend_url}")
    bt.logging.info("Submitting model...")

    try:
        result = submit_model(wallet, config.github_url, backend_url)
    except httpx.ConnectError:
        bt.logging.error(f"Cannot connect to backend at {backend_url}")
        return 1
    except httpx.TimeoutException:
        bt.logging.error("Backend request timed out. The server may be downloading your model — try again.")
        return 1
    except Exception as e:
        bt.logging.error(f"Request failed: {e}")
        return 1

    if result.get("accepted"):
        bt.logging.info("")
        bt.logging.info("=" * 60)
        bt.logging.info("  MODEL SUBMITTED SUCCESSFULLY")
        bt.logging.info("=" * 60)
        bt.logging.info(f"  Model ID:    {result.get('model_id')}")
        bt.logging.info(f"  Model Hash:  {result.get('model_hash')}")
        bt.logging.info(f"  Status:      {result.get('status')}")
        bt.logging.info("=" * 60)
        bt.logging.info("")
        bt.logging.info("You can now go offline. Validators will pick up your model automatically.")
        return 0
    else:
        bt.logging.error("")
        bt.logging.error("=" * 60)
        bt.logging.error("  SUBMISSION REJECTED")
        bt.logging.error("=" * 60)
        bt.logging.error(f"  Reason: {result.get('_error', 'Unknown error')}")
        _print_rejection_hint(result.get("_error", ""))
        bt.logging.error("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main() or 0)
