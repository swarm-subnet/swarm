#!/usr/bin/env python3
"""
Swarm Miner — commit a model to the Bittensor chain.

Public track (open competition):
    python neurons/miner.py --netuid 124 \
        --wallet.name miner --wallet.hotkey default \
        --github_url https://github.com/yourname/your-model

Private track (model stays secret; only trusted validators ever run it):
    python neurons/miner.py --netuid 124 \
        --wallet.name miner --wallet.hotkey default \
        --family_id cf_search_and_rescue \
        --artifact ./submission.zip \
        --backend_url https://your-backend

Public: the GitHub URL is committed on-chain; the backend pulls submission.zip.
Private: the artifact's sha256 is committed on-chain and the artifact is uploaded
to the operator's private vault — it is never made public.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid
from urllib.parse import urlparse
import zipfile

import bittensor as bt


REQUIRED_ROOT_FILES = ("drone_agent.py",)
FORBIDDEN_SUFFIXES = (".exe", ".so", ".dll", ".sh", ".bat", ".pyc")
MAX_UNCOMPRESSED_BYTES = 50 * 1024 * 1024


def _validate_github_url(raw: str) -> str | None:
    """Return normalized https://github.com/{owner}/{repo} or None."""
    url = (raw or "").strip().rstrip("/")
    parsed = urlparse(url)
    if parsed.scheme != "https" or (parsed.netloc or "").lower() != "github.com":
        return None
    parts = [s for s in (parsed.path or "").split("/") if s]
    if len(parts) != 2:
        return None
    repo = parts[1].removesuffix(".git")
    if not repo:
        return None
    return f"https://github.com/{parts[0]}/{repo}"


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_local_families() -> dict | None:
    """{family_id: visibility} from the repo's domain schema, or None."""
    schema_path = (
        Path(__file__).resolve().parent.parent
        / "swarm" / "domain_model" / "benchmark_domain_model.schema.json"
    )
    try:
        families = json.loads(schema_path.read_text())["challenge_families"]
        return {fid: str(fam.get("visibility", "public")) for fid, fam in families.items()}
    except Exception:
        return None


def _fetch_backend_families(backend_url: str) -> dict | None:
    """Same mapping fetched from the backend registry, or None on any failure."""
    import httpx

    try:
        resp = httpx.get(backend_url.rstrip("/") + "/families/metadata", timeout=30)
        if resp.status_code != 200:
            return None
        families = resp.json()["challenge_families"]
        return {fid: str(fam.get("visibility", "public")) for fid, fam in families.items()}
    except Exception:
        return None


def _validate_artifact_zip(path: str) -> str | None:
    """Return a rejection reason for a structurally invalid artifact, else None."""
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            uncompressed = sum(info.file_size for info in zf.infolist())
    except zipfile.BadZipFile:
        return "not a valid ZIP archive"
    except OSError as exc:
        return f"cannot read artifact: {exc}"
    if uncompressed > MAX_UNCOMPRESSED_BYTES:
        return f"uncompressed size too large ({uncompressed / 1e6:.1f} MB, limit 50 MB)"
    for required in REQUIRED_ROOT_FILES:
        if required not in names:
            return f"missing required file at the zip root: {required}"
    forbidden = sorted({n for n in names if n.endswith(FORBIDDEN_SUFFIXES)})
    if forbidden:
        return f"forbidden file type(s): {', '.join(forbidden)}"
    return None


def _upload_private_artifact(backend_url: str, artifact_path: str, digest: str, wallet) -> bool:
    """Upload the private artifact to the operator vault, signed by the hotkey.

    Retries while the backend has not yet scanned the on-chain commitment (404).
    """
    import httpx

    path = f"/miners/models/{digest}/private-upload"
    url = backend_url.rstrip("/") + path
    with open(artifact_path, "rb") as handle:
        body = handle.read()

    attempts = 12
    for attempt in range(1, attempts + 1):
        nonce = uuid.uuid4().hex
        timestamp = str(int(time.time()))
        message = f"{timestamp}:{nonce}:POST:{path}:{digest}"
        signature = wallet.hotkey.sign(message.encode()).hex()
        headers = {
            "X-Miner-Hotkey": wallet.hotkey.ss58_address,
            "X-Miner-Signature": signature,
            "X-Miner-Nonce": nonce,
            "X-Miner-Timestamp": timestamp,
        }
        try:
            response = httpx.post(
                url,
                headers=headers,
                files={"file": ("submission.zip", body, "application/zip")},
                timeout=180,
            )
        except Exception as exc:
            bt.logging.warning(f"Upload attempt {attempt} transport error: {exc}")
            time.sleep(30)
            continue

        if response.status_code == 200:
            bt.logging.info(f"Artifact uploaded to the private vault: {response.json()}")
            return True
        if response.status_code == 404 or response.status_code >= 500:
            reason = (
                "backend has not scanned the commitment yet"
                if response.status_code == 404
                else f"backend unavailable ({response.status_code})"
            )
            bt.logging.info(
                f"Upload not ready ({reason}); attempt {attempt}/{attempts}, retrying in 30s..."
            )
            time.sleep(30)
            continue
        bt.logging.error(f"Upload rejected ({response.status_code}): {response.text}")
        return False

    bt.logging.error("Upload timed out waiting for the backend to scan the commitment.")
    return False


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Swarm Miner — commit a model to the Bittensor chain"
    )
    parser.add_argument(
        "--github_url", type=str, default=None,
        help="Public track: GitHub repo URL (https://github.com/owner/repo)",
    )
    parser.add_argument(
        "--family_id", type=str, default=None,
        help="Private track: the family this submission targets",
    )
    parser.add_argument(
        "--artifact", type=str, default=None,
        help="Private track: path to the submission.zip to upload privately",
    )
    parser.add_argument(
        "--backend_url", type=str, default=None,
        help="Private track: backend base URL for the private upload",
    )
    parser.add_argument(
        "--upload_only", action="store_true",
        help="Private track: skip the chain commit and only (re)upload the artifact",
    )
    parser.add_argument("--netuid", type=int, default=124)
    parser.add_argument("--wallet.name", type=str, default="default", dest="wallet_name")
    parser.add_argument("--wallet.hotkey", type=str, default="default", dest="wallet_hotkey")
    parser.add_argument("--subtensor.network", type=str, default="finney", dest="network")
    parser.add_argument("--logging.debug", action="store_true", dest="debug")
    args = parser.parse_args(argv)

    if args.debug:
        bt.logging.set_debug(True)

    is_private = bool(args.family_id or args.artifact)
    if is_private and args.github_url:
        bt.logging.error(
            "Provide either --github_url (public) OR --family_id + --artifact (private), not both."
        )
        return 1

    if is_private:
        if not (args.family_id and args.artifact and args.backend_url):
            bt.logging.error(
                "Private submission needs --family_id, --artifact, and --backend_url."
            )
            return 1
        families = _fetch_backend_families(args.backend_url)
        if families is None:
            bt.logging.warning(
                "Could not reach the backend to verify the family list; using the local registry."
            )
            families = _load_local_families()
        if families is None:
            bt.logging.warning("Family registry unavailable; skipping family validation.")
        elif args.family_id not in families:
            bt.logging.error(f"Unknown family_id '{args.family_id}'.")
            bt.logging.error(f"Valid families: {', '.join(sorted(families))}")
            return 1
        elif families[args.family_id] != "private":
            bt.logging.error(
                f"'{args.family_id}' is a public-track family, so it does not accept private submissions."
            )
            bt.logging.error(
                "Public models are submitted by committing a GitHub repository URL instead:"
            )
            bt.logging.error(
                "  python neurons/miner.py --netuid 124 --wallet.name miner --wallet.hotkey default \\"
            )
            bt.logging.error("      --github_url https://github.com/you/your-model")
            bt.logging.error("The full submission guide is in docs/miner.md.")
            return 1

        reason = _validate_artifact_zip(args.artifact)
        if reason is not None:
            bt.logging.error(f"Artifact rejected before committing: {reason}")
            bt.logging.error(
                "Fix the zip and retry; nothing was committed on-chain."
            )
            return 1
        try:
            digest = _sha256_file(args.artifact)
        except OSError as exc:
            bt.logging.error(f"Cannot read artifact: {exc}")
            return 1
        commit_data = json.dumps(
            {
                "v": 1,
                "visibility": "private",
                "family_id": args.family_id,
                "sha256": digest,
            },
            separators=(",", ":"),
        )
        commit_label = f"private {args.family_id} (sha256 {digest[:16]}...)"
    else:
        github_url = _validate_github_url(args.github_url or "")
        if not github_url:
            bt.logging.error(
                "Invalid GitHub URL. Must be https://github.com/{owner}/{repo}"
            )
            return 1
        commit_data = github_url
        commit_label = github_url

    try:
        _WalletCls = bt.Wallet if hasattr(bt, "Wallet") else bt.wallet
        wallet = _WalletCls(name=args.wallet_name, hotkey=args.wallet_hotkey)
        hotkey = wallet.hotkey.ss58_address
    except Exception as e:
        bt.logging.error(f"Wallet error: {e}")
        return 1

    if is_private and args.upload_only:
        bt.logging.info("Upload-only: skipping chain commit, retrying the private vault upload.")
        uploaded = _upload_private_artifact(
            args.backend_url, args.artifact, digest, wallet
        )
        if uploaded:
            bt.logging.info("Private artifact uploaded.")
            return 0
        bt.logging.error("Upload failed. Re-run with --upload_only to retry.")
        return 1

    bt.logging.info(f"Hotkey:      {hotkey}")
    bt.logging.info(f"Commitment:  {commit_label}")
    bt.logging.info(f"Network:     {args.network} (netuid {args.netuid})")

    try:
        _SubtensorCls = bt.Subtensor if hasattr(bt, "Subtensor") else bt.subtensor
        subtensor = _SubtensorCls(network=args.network)
    except Exception as e:
        bt.logging.error(f"Failed to connect to {args.network}: {e}")
        return 1

    metagraph = subtensor.metagraph(netuid=args.netuid)
    if hotkey not in metagraph.hotkeys:
        bt.logging.error(
            f"Hotkey {hotkey[:16]}... is not registered on subnet {args.netuid}."
        )
        return 1

    bt.logging.info("Committing to chain...")

    try:
        response = subtensor.set_commitment(
            wallet=wallet, netuid=args.netuid, data=commit_data,
            mev_protection=False,
        )
        success = response.success
    except Exception as e:
        bt.logging.error(f"Chain commit failed: {e}")
        return 1

    if not success:
        bt.logging.error("")
        bt.logging.error("=" * 60)
        bt.logging.error("  COMMITMENT FAILED")
        bt.logging.error("=" * 60)
        bt.logging.error("  Chain commit returned False. Possible causes:")
        bt.logging.error("  - Rate limited (wait ~20 minutes between commits)")
        bt.logging.error("  - Insufficient balance for transaction fee")
        bt.logging.error("=" * 60)
        return 1

    if is_private:
        uploaded = _upload_private_artifact(
            args.backend_url, args.artifact, digest, wallet
        )
        if not uploaded:
            bt.logging.error(
                "Commitment succeeded but the artifact upload failed. "
                "Re-run with --upload_only to retry the upload without re-committing."
            )
            return 1
        bt.logging.info("")
        bt.logging.info("=" * 60)
        bt.logging.info("  PRIVATE MODEL SUBMITTED SUCCESSFULLY")
        bt.logging.info("=" * 60)
        bt.logging.info(f"  Family:  {args.family_id}")
        bt.logging.info(f"  Digest:  {digest}")
        bt.logging.info(f"  Hotkey:  {hotkey}")
        bt.logging.info("=" * 60)
        bt.logging.info("Your model stays private. You can now go offline.")
        return 0

    bt.logging.info("")
    bt.logging.info("=" * 60)
    bt.logging.info("  MODEL COMMITTED SUCCESSFULLY")
    bt.logging.info("=" * 60)
    bt.logging.info(f"  GitHub URL:  {commit_data}")
    bt.logging.info(f"  Hotkey:      {hotkey}")
    bt.logging.info("=" * 60)
    bt.logging.info("")
    bt.logging.info("Validators will discover your model from the chain automatically.")
    bt.logging.info("You can now go offline.")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
