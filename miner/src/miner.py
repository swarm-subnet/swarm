#!/usr/bin/env python3
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

"""
Swarm Miner — commit a model to the Bittensor chain.

Private track (the default for every family; `swarm model submit` wraps this):
    python miner/src/miner.py --netuid 124 \
        --wallet.name miner --wallet.hotkey default \
        --family_id cf_autopilot \
        --artifact ./submission.zip

The artifact's sha256 is committed on-chain and the archive is uploaded to the
operator's private vault. It is published only if it takes the crown.

Public track (kept for families flagged public in the registry):
    python miner/src/miner.py --netuid 124 \
        --wallet.name miner --wallet.hotkey default \
        --github_url https://github.com/yourname/your-model
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
import uuid
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import httpx

os.environ.setdefault("BT_NO_PARSE_CLI_ARGS", "false")

import bittensor as bt

from swarm.constants import DOCKER_PIP_WHITELIST, MAX_MODEL_BYTES
from swarm.core.submission_policy import validate_submission_zip
from swarm.policy_interface import PolicyInterfaceError, read_policy_contract_from_zip

DEFAULT_BACKEND_URL = "https://api.swarm124.com"
# The backend scans the chain every 3 minutes; a commit inside the pre-epoch
# freeze waits until rollover, so the retry budget covers the normal case and
# the resume command covers the rest.
UPLOAD_RETRY_BUDGET_SEC = 30 * 60
UPLOAD_FIRST_DELAY_SEC = 15
UPLOAD_MAX_DELAY_SEC = 300

REJECTION_HELP = {
    "HASH_COLLISION": (
        "this exact archive is already registered to another miner (the earliest "
        "on-chain commit wins). Change the model, package it again and submit the new digest."
    ),
    "ONE_TASK_PER_HOTKEY": (
        "this hotkey already holds a submission. Every hotkey submits once; "
        "register a new hotkey to submit again."
    ),
    "CHAMPION_LOCKED": "this hotkey holds a champion, and a champion's digest cannot change.",
    "HOTKEY_SLOT_CONFLICT": "another commitment claimed this hotkey's slot at the same moment; try again.",
    "PRIVATE_FAMILY_REQUIRED": "this family only accepts private submissions; use `swarm model submit`.",
    "private_family_target": (
        "public GitHub submissions are no longer accepted for this family; "
        "submit the archive privately with `swarm model submit`."
    ),
    "duplicate_content": (
        "the archive is the same model as an existing submission. Comments, formatting "
        "and zip packaging do not make it a new model; change the code or the weights."
    ),
    "invalid_artifact": "the archive failed the submission checks.",
}


def explain_rejection(reason_code: str | None, detail: str | None) -> str:
    """One plain sentence a miner can act on, for any backend reason code."""
    help_text = REJECTION_HELP.get(reason_code or "")
    if help_text and detail:
        return f"{help_text} ({detail})"
    return help_text or detail or f"rejected with reason {reason_code!r}"


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


def _check_public_manifest(github_url: str) -> tuple[str | None, str | None]:
    """(rejection reason, declared family) from the repo manifest.

    The reason is set when the manifest names more than one family. Fetch
    problems are logged and do not block the commit — the scanner enforces
    the rule — and leave the family unknown."""
    for branch in ("main", "master"):
        try:
            resp = httpx.get(
                f"{github_url}/raw/{branch}/submission_manifest.json",
                timeout=30, follow_redirects=True,
            )
        except Exception as exc:
            bt.logging.warning(
                f"Could not fetch the repo manifest ({exc}); proceeding with the "
                "commit. Make sure the repo is public and packaged correctly."
            )
            return None, None
        if resp.status_code != 200:
            continue
        try:
            artifacts = resp.json().get("artifacts") or []
        except ValueError:
            bt.logging.warning(
                "The repo manifest is not valid JSON; proceeding with the commit."
            )
            return None, None
        families = sorted(
            {str(a.get("family_id")) for a in artifacts if isinstance(a, dict)}
        )
        if len(artifacts) > 1 or len(families) > 1:
            return (
                f"the repo manifest declares {len(artifacts)} artifacts "
                f"({', '.join(families)}); one hotkey competes in one task, "
                "so package exactly one family"
            ), None
        return None, (families[0] if families else None)
    bt.logging.warning(
        "No submission_manifest.json found on main or master; proceeding with "
        "the commit. Make sure the repo is public and pushed before submitting."
    )
    return None, None


def _backend_reachable(backend_url: str) -> bool:
    """True when the backend answers its health endpoint. The private track
    checks this before committing so a digest never lands on-chain without a
    reachable upload target."""
    try:
        resp = httpx.get(f"{backend_url.rstrip('/')}/health", timeout=15)
    except Exception:
        return False
    return resp.status_code == 200


def _submission_window(backend_url: str) -> dict | None:
    """{"open": bool, "reopens_in_seconds": int} from the backend, or None."""
    try:
        resp = httpx.get(f"{backend_url.rstrip('/')}/miners/submission-window", timeout=15)
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception:
        return None


def _submission_status(backend_url: str, digest: str) -> dict | None:
    """The backend's view of a committed digest, or None when it cannot answer."""
    try:
        resp = httpx.get(f"{backend_url.rstrip('/')}/miners/models/{digest}/status", timeout=15)
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception:
        return None


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_local_families() -> dict | None:
    """{family_id: visibility} from the repo's domain schema, or None."""
    schema_path = (
        Path(__file__).resolve().parents[2]
        / "swarm" / "domain_model" / "benchmark_domain_model.schema.json"
    )
    try:
        families = json.loads(schema_path.read_text())["challenge_families"]
        return {fid: str(fam.get("visibility", "public")) for fid, fam in families.items()}
    except Exception:
        return None


def _fetch_backend_families(backend_url: str) -> dict | None:
    """Same mapping fetched from the backend registry, or None on any failure."""
    try:
        resp = httpx.get(backend_url.rstrip("/") + "/families/metadata", timeout=30)
        if resp.status_code != 200:
            return None
        families = resp.json()["challenge_families"]
        return {fid: str(fam.get("visibility", "public")) for fid, fam in families.items()}
    except Exception:
        return None


def _known_families(backend_url: str) -> dict | None:
    families = _fetch_backend_families(backend_url)
    if families is None:
        bt.logging.warning(
            "Could not reach the backend to verify the family list; using the local registry."
        )
        families = _load_local_families()
    if families is None:
        bt.logging.warning("Family registry unavailable; skipping family validation.")
    return families


def _normalize_package_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirements_from_zip(zip_path: Path) -> str | None:
    """The archive's root requirements.txt, or None when it declares none."""
    with zipfile.ZipFile(zip_path) as archive:
        if "requirements.txt" not in archive.namelist():
            return None
        return archive.read("requirements.txt").decode("utf-8", errors="replace")


def _check_requirements(requirements: str) -> str | None:
    """Mirror the validator's whitelist rules so a bad line fails here, not on 1,100 seeds."""
    rejected = []
    for raw_line in requirements.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-"):
            return f"requirements_rejected:installer options are not allowed ({line})"
        if line.startswith(("git+", "http://", "https://", "file:", "./", "/")):
            return f"requirements_rejected:URL and path installs are not allowed ({line})"
        if " @ " in line:
            return f"requirements_rejected:direct references are not allowed ({line})"
        line = line.split("#")[0].split(";")[0].strip()
        if not line:
            continue
        if " " in line:
            return f"requirements_rejected:unexpected token in {raw_line.strip()!r}"
        name = re.split(r"[>=<!~\[]", line)[0].strip()
        if name and _normalize_package_name(name) not in DOCKER_PIP_WHITELIST:
            rejected.append(name)
    if rejected:
        return (
            "requirements_rejected:packages not on the whitelist: "
            f"{', '.join(rejected)} (see the Docker Whitelist in miner/docs/miner.md)"
        )
    return None


def _validate_artifact(path: str, *, family_id: str | None = None) -> str | None:
    """Run the same static submission checks used by validators and the backend."""
    artifact = Path(path)
    try:
        size = artifact.stat().st_size
        accepted, detail = validate_submission_zip(artifact)
    except OSError as exc:
        return f"invalid_artifact:cannot read artifact: {exc}"
    if not accepted:
        return detail
    if size > MAX_MODEL_BYTES:
        return (
            f"artifact_too_large:{size / (1024 * 1024):.1f} MiB compressed; "
            f"the cap is {MAX_MODEL_BYTES // (1024 * 1024)} MiB"
        )
    try:
        requirements = _requirements_from_zip(artifact)
    except (zipfile.BadZipFile, OSError) as exc:
        return f"invalid_artifact:cannot read artifact: {exc}"
    if requirements is not None:
        reason = _check_requirements(requirements)
        if reason is not None:
            return reason
    if family_id is None:
        return None
    try:
        declared = str(read_policy_contract_from_zip(artifact).get("family_id") or "")
    except PolicyInterfaceError:
        return None  # a hand-built zip may omit the contract; nothing to bind against
    if declared and declared != family_id:
        return f"artifact_family_mismatch:artifact family {declared!r} does not match {family_id!r}"
    return None


def _response_detail(response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text.strip()
    if isinstance(payload, dict) and payload.get("detail"):
        return str(payload["detail"])
    return response.text.strip()


def _resume_hint(family_id: str, artifact_path: str) -> str:
    return (
        "Nothing more is needed on-chain. Re-run with --upload_only to retry the upload:\n"
        f"  swarm model submit --family-id {family_id} --artifact {artifact_path} --upload-only"
    )


def _upload_private_artifact(backend_url: str, artifact_path: str, digest: str, wallet) -> bool:
    """Upload the private artifact to the operator vault, signed by the hotkey.

    Retries with backoff while the backend has not yet scanned the commitment
    (404) or answers with a server error (5xx). A 404 that the backend explains
    as a rejection stops at once with the reason; any other status fails
    immediately with the backend's own message.
    """
    path = f"/miners/models/{digest}/private-upload"
    url = backend_url.rstrip("/") + path
    family_id = "<family_id>"
    try:
        family_id = str(read_policy_contract_from_zip(Path(artifact_path)).get("family_id") or family_id)
    except (PolicyInterfaceError, OSError):
        pass

    deadline = time.monotonic() + UPLOAD_RETRY_BUDGET_SEC
    delay = UPLOAD_FIRST_DELAY_SEC
    attempt = 0
    while True:
        attempt += 1
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
            with open(artifact_path, "rb") as handle:
                response = httpx.post(
                    url,
                    headers=headers,
                    files={"file": ("submission.zip", handle, "application/zip")},
                    timeout=180,
                )
        except Exception as exc:
            response = None
            reason = f"transport error: {exc}"
        else:
            if response.status_code == 200:
                bt.logging.info(f"Artifact uploaded to the private vault: {response.json()}")
                return True
            if response.status_code == 404:
                status = _submission_status(backend_url, digest)
                if status and not status.get("registered") and status.get("reason_code"):
                    bt.logging.error(
                        "The backend rejected the commitment: "
                        + explain_rejection(status.get("reason_code"), status.get("detail"))
                    )
                    return False
                reason = "backend has not scanned the commitment yet (it scans every 3 minutes)"
            elif response.status_code >= 500:
                reason = f"backend unavailable ({response.status_code})"
            else:
                bt.logging.error(
                    f"Upload rejected ({response.status_code}): {_response_detail(response)}"
                )
                return False

        if time.monotonic() + delay > deadline:
            bt.logging.error(f"Upload gave up after {attempt} attempts ({reason}).")
            bt.logging.error(_resume_hint(family_id, artifact_path))
            return False
        bt.logging.info(f"Upload not ready ({reason}); attempt {attempt}, retrying in {delay}s...")
        time.sleep(delay)
        delay = min(delay * 2, UPLOAD_MAX_DELAY_SEC)


def _open_wallet(wallet_name: str, wallet_hotkey: str):
    _WalletCls = bt.Wallet if hasattr(bt, "Wallet") else bt.wallet
    return _WalletCls(name=wallet_name, hotkey=wallet_hotkey)


def _commit(wallet, *, netuid: int, network: str, commit_data: str) -> bool:
    """Commit ``commit_data`` on-chain for the wallet's hotkey; False on any failure."""
    try:
        _SubtensorCls = bt.Subtensor if hasattr(bt, "Subtensor") else bt.subtensor
        subtensor = _SubtensorCls(network=network)
    except Exception as e:
        bt.logging.error(f"Failed to connect to {network}: {e}")
        return False

    metagraph = subtensor.metagraph(netuid=netuid)
    hotkey = wallet.hotkey.ss58_address
    if hotkey not in metagraph.hotkeys:
        bt.logging.error(f"Hotkey {hotkey[:16]}... is not registered on subnet {netuid}.")
        return False

    bt.logging.info("Committing to chain...")
    try:
        response = subtensor.set_commitment(
            wallet=wallet, netuid=netuid, data=commit_data, mev_protection=False,
        )
        success = response.success
    except Exception as e:
        bt.logging.error(f"Chain commit failed: {e}")
        return False

    if not success:
        bt.logging.error("")
        bt.logging.error("=" * 60)
        bt.logging.error("  COMMITMENT FAILED")
        bt.logging.error("=" * 60)
        bt.logging.error("  Chain commit returned False. Possible causes:")
        bt.logging.error("  - Rate limited (wait ~20 minutes between commits)")
        bt.logging.error("  - Insufficient balance for transaction fee")
        bt.logging.error("=" * 60)
    return bool(success)


def submit_private(
    *,
    family_id: str,
    artifact: str,
    backend_url: str,
    wallet_name: str,
    wallet_hotkey: str,
    netuid: int = 124,
    network: str = "finney",
    upload_only: bool = False,
) -> int:
    """Commit the artifact's digest on-chain and upload the bytes to the vault.

    Every check that can fail runs before anything touches the chain, so a
    rejected archive costs the miner nothing.
    """
    families = _known_families(backend_url)
    if families is not None and family_id not in families:
        bt.logging.error(f"Unknown family_id '{family_id}'.")
        bt.logging.error(f"Valid families: {', '.join(sorted(families))}")
        return 1
    if families is not None and families[family_id] != "private":
        bt.logging.error(
            f"'{family_id}' is a public-track family, so it does not accept private submissions."
        )
        bt.logging.error("Public models are submitted by committing a GitHub repository URL instead:")
        bt.logging.error(
            "  python miner/src/miner.py --netuid 124 --wallet.name miner --wallet.hotkey default \\"
        )
        bt.logging.error("      --github_url https://github.com/you/your-model")
        return 1

    reason = _validate_artifact(artifact, family_id=family_id)
    if reason is not None:
        bt.logging.error(f"Artifact rejected before committing: {reason}")
        bt.logging.error("Fix the zip and retry; nothing was committed on-chain.")
        return 1
    try:
        digest = _sha256_file(artifact)
    except OSError as exc:
        bt.logging.error(f"Cannot read artifact: {exc}")
        return 1

    if not upload_only:
        if not _backend_reachable(backend_url):
            bt.logging.error(f"Backend {backend_url} is unreachable; nothing was committed on-chain.")
            bt.logging.error("Verify --backend_url and your connection, then retry.")
            return 1
        window = _submission_window(backend_url)
        if window is not None and not window.get("open", True):
            minutes = max(1, int(window.get("reopens_in_seconds", 0)) // 60)
            bt.logging.error(
                "The submission window is closed for the pre-epoch freeze; "
                f"it reopens in about {minutes} minutes. Nothing was committed on-chain."
            )
            return 1
        status = _submission_status(backend_url, digest)
        if status and status.get("registered") and status.get("uid") is not None:
            bt.logging.error(
                f"This exact archive is already registered to UID {status['uid']}; "
                "change the model and package it again."
            )
            return 1

    try:
        wallet = _open_wallet(wallet_name, wallet_hotkey)
        hotkey = wallet.hotkey.ss58_address
    except Exception as e:
        bt.logging.error(f"Wallet error: {e}")
        return 1

    if upload_only:
        bt.logging.info("Upload-only: skipping chain commit, retrying the private vault upload.")
        if _upload_private_artifact(backend_url, artifact, digest, wallet):
            bt.logging.info("Private artifact uploaded.")
            return 0
        return 1

    # Compact keys: the chain commitment field caps at Raw128 (128 bytes).
    commit_data = json.dumps({"v": 1, "f": family_id, "s": digest}, separators=(",", ":"))
    bt.logging.info(f"Hotkey:      {hotkey}")
    bt.logging.info(f"Commitment:  private {family_id} (sha256 {digest[:16]}...)")
    bt.logging.info(f"Network:     {network} (netuid {netuid})")

    if not _commit(wallet, netuid=netuid, network=network, commit_data=commit_data):
        return 1

    if not _upload_private_artifact(backend_url, artifact, digest, wallet):
        bt.logging.error("Commitment succeeded but the artifact upload did not land.")
        bt.logging.error(_resume_hint(family_id, artifact))
        return 1
    bt.logging.info("")
    bt.logging.info("=" * 60)
    bt.logging.info("  PRIVATE MODEL SUBMITTED SUCCESSFULLY")
    bt.logging.info("=" * 60)
    bt.logging.info(f"  Family:  {family_id}")
    bt.logging.info(f"  Digest:  {digest}")
    bt.logging.info(f"  Hotkey:  {hotkey}")
    bt.logging.info("=" * 60)
    bt.logging.info("Your model stays private unless it takes the crown. You can now go offline.")
    return 0


def _submit_public(args: argparse.Namespace) -> int:
    github_url = _validate_github_url(args.github_url or "")
    if not github_url:
        bt.logging.error("Invalid GitHub URL. Must be https://github.com/{owner}/{repo}")
        return 1
    families = _known_families(args.backend_url)
    manifest_reason, declared_family = _check_public_manifest(github_url)
    if manifest_reason is not None:
        bt.logging.error(f"Submission rejected before committing: {manifest_reason}")
        bt.logging.error("Fix the repo and retry; nothing was committed on-chain.")
        return 1
    if families:
        target = declared_family if declared_family in families else None
        private_target = target is not None and families[target] == "private"
        everything_private = all(v == "private" for v in families.values())
        if private_target or (target is None and everything_private):
            bt.logging.error(
                "Public GitHub submissions are not accepted for "
                + (f"'{target}'" if target else "any family")
                + "; models are submitted privately now. Nothing was committed on-chain."
            )
            bt.logging.error("Submit the packaged archive instead:")
            bt.logging.error(
                f"  swarm model submit --family-id {target or '<family_id>'} "
                "--artifact Submission/submission.zip"
            )
            return 1

    try:
        wallet = _open_wallet(args.wallet_name, args.wallet_hotkey)
        hotkey = wallet.hotkey.ss58_address
    except Exception as e:
        bt.logging.error(f"Wallet error: {e}")
        return 1

    bt.logging.info(f"Hotkey:      {hotkey}")
    bt.logging.info(f"Commitment:  {github_url}")
    bt.logging.info(f"Network:     {args.network} (netuid {args.netuid})")
    if not _commit(wallet, netuid=args.netuid, network=args.network, commit_data=github_url):
        return 1

    bt.logging.info("")
    bt.logging.info("=" * 60)
    bt.logging.info("  MODEL COMMITTED SUCCESSFULLY")
    bt.logging.info("=" * 60)
    bt.logging.info(f"  GitHub URL:  {github_url}")
    bt.logging.info(f"  Hotkey:      {hotkey}")
    bt.logging.info("=" * 60)
    bt.logging.info("")
    bt.logging.info("Validators will discover your model from the chain automatically.")
    bt.logging.info("You can now go offline.")
    return 0


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
        "--backend_url", type=str, default=DEFAULT_BACKEND_URL,
        help=f"Backend base URL (default: {DEFAULT_BACKEND_URL})",
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
    if not is_private:
        return _submit_public(args)
    if not (args.family_id and args.artifact):
        bt.logging.error("Private submission needs --family_id and --artifact.")
        return 1
    return submit_private(
        family_id=args.family_id,
        artifact=args.artifact,
        backend_url=args.backend_url,
        wallet_name=args.wallet_name,
        wallet_hotkey=args.wallet_hotkey,
        netuid=args.netuid,
        network=args.network,
        upload_only=args.upload_only,
    )


if __name__ == "__main__":
    sys.exit(main() or 0)
