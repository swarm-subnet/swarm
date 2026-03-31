#!/usr/bin/env python3
"""
Swarm Miner — commit model GitHub URL to Bittensor chain.

Usage:
    python neurons/miner.py \
        --netuid 124 \
        --subtensor.network finney \
        --wallet.name miner \
        --wallet.hotkey default \
        --github_url https://github.com/yourname/your-model

The GitHub URL is committed on-chain. The backend reads the chain
periodically, downloads submission.zip from your repository, verifies
it, and queues it for evaluation. You do NOT need to stay online.
"""

import argparse
import sys
from urllib.parse import urlparse

import bittensor as bt


def _validate_github_url(raw: str) -> str | None:
    """Return normalized https://github.com/{owner}/{repo} or None."""
    url = (raw or "").strip().rstrip("/")
    parsed = urlparse(url)
    if parsed.scheme != "https" or (parsed.netloc or "").lower() != "github.com":
        return None
    parts = [s for s in (parsed.path or "").split("/") if s]
    if len(parts) != 2:
        return None
    return f"https://github.com/{parts[0]}/{parts[1]}"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Swarm Miner — commit model to Bittensor chain"
    )
    parser.add_argument(
        "--github_url", type=str, required=True,
        help="Public GitHub repo URL (https://github.com/owner/repo)",
    )
    parser.add_argument("--netuid", type=int, default=124)
    parser.add_argument("--wallet.name", type=str, default="default", dest="wallet_name")
    parser.add_argument("--wallet.hotkey", type=str, default="default", dest="wallet_hotkey")
    parser.add_argument("--subtensor.network", type=str, default="finney", dest="network")
    parser.add_argument("--logging.debug", action="store_true", dest="debug")
    args = parser.parse_args(argv)

    if args.debug:
        bt.logging.set_debug(True)

    github_url = _validate_github_url(args.github_url)
    if not github_url:
        bt.logging.error(
            "Invalid GitHub URL. Must be https://github.com/{owner}/{repo}"
        )
        return 1

    try:
        wallet = bt.wallet(name=args.wallet_name, hotkey=args.wallet_hotkey)
        hotkey = wallet.hotkey.ss58_address
    except Exception as e:
        bt.logging.error(f"Wallet error: {e}")
        return 1

    bt.logging.info(f"Hotkey:      {hotkey}")
    bt.logging.info(f"GitHub URL:  {github_url}")
    bt.logging.info(f"Network:     {args.network} (netuid {args.netuid})")

    try:
        subtensor = bt.subtensor(network=args.network)
    except Exception as e:
        bt.logging.error(f"Failed to connect to {args.network}: {e}")
        return 1

    metagraph = subtensor.metagraph(netuid=args.netuid)
    if hotkey not in metagraph.hotkeys:
        bt.logging.error(
            f"Hotkey {hotkey[:16]}... is not registered on subnet {args.netuid}."
        )
        return 1

    bt.logging.info("Committing GitHub URL to chain...")

    try:
        success = subtensor.commit(
            wallet=wallet, netuid=args.netuid, data=github_url,
        )
    except Exception as e:
        bt.logging.error(f"Chain commit failed: {e}")
        return 1

    if success:
        bt.logging.info("")
        bt.logging.info("=" * 60)
        bt.logging.info("  MODEL COMMITTED SUCCESSFULLY")
        bt.logging.info("=" * 60)
        bt.logging.info(f"  GitHub URL:  {github_url}")
        bt.logging.info(f"  Hotkey:      {hotkey}")
        bt.logging.info("=" * 60)
        bt.logging.info("")
        bt.logging.info(
            "Validators will discover your model from the chain automatically."
        )
        bt.logging.info("You can now go offline.")
        return 0
    else:
        bt.logging.error("")
        bt.logging.error("=" * 60)
        bt.logging.error("  COMMITMENT FAILED")
        bt.logging.error("=" * 60)
        bt.logging.error(
            "  Chain commit returned False. Possible causes:"
        )
        bt.logging.error(
            "  - Rate limited (wait ~20 minutes between commits)"
        )
        bt.logging.error(
            "  - Insufficient balance for transaction fee"
        )
        bt.logging.error("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main() or 0)
