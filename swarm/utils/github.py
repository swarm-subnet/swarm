from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import httpx
import bittensor as bt

GITHUB_DOWNLOAD_TIMEOUT_SEC = 60.0
GITHUB_CONNECT_TIMEOUT_SEC = 10.0
GITHUB_MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024
GITHUB_MAX_README_BYTES = 64 * 1024

REQUIRED_README_HASH = "0682f31a2c2bfe3343e25d10f542998c3e8b27f930a6c59140561e7394c7d859"


def validate_github_url(raw_url: str, *, uid: Optional[int] = None) -> Optional[str]:
    """Validate a GitHub repository URL.

    Accepts ``https://github.com/{owner}/{repo}`` format only.
    Returns the normalized URL or *None* if invalid.
    """
    if not raw_url or not raw_url.strip():
        return None

    url = raw_url.strip().rstrip("/")
    tag = f" (UID {uid})" if uid is not None else ""

    parsed = urlparse(url)

    if parsed.scheme != "https":
        bt.logging.warning(f"Rejecting github_url: non-HTTPS scheme{tag}")
        return None

    host = (parsed.netloc or "").lower()
    if host != "github.com":
        bt.logging.warning(f"Rejecting github_url: unsupported host{tag}")
        return None

    segments = [s for s in (parsed.path or "").split("/") if s]
    if len(segments) != 2:
        bt.logging.warning(f"Rejecting github_url: expected owner/repo{tag}")
        return None

    return f"https://github.com/{segments[0]}/{segments[1]}"


def build_raw_urls(repo_url: str) -> list[str]:
    """Build candidate raw download URLs for ``submission.zip``.

    Tries ``main`` first, then ``master`` as a fallback.
    """
    base = repo_url.rstrip("/")
    return [
        f"{base}/raw/main/submission.zip",
        f"{base}/raw/master/submission.zip",
    ]


async def download_from_github(
    url: str,
    dest: Path,
    *,
    max_bytes: int = GITHUB_MAX_DOWNLOAD_BYTES,
    timeout_sec: float = GITHUB_DOWNLOAD_TIMEOUT_SEC,
) -> bool:
    """Stream-download a file from a GitHub raw URL.

    Returns *True* on success, *False* on any failure.
    """
    tmp = dest.with_suffix(".part")
    tmp.unlink(missing_ok=True)

    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=httpx.Timeout(timeout_sec, connect=GITHUB_CONNECT_TIMEOUT_SEC),
        ) as client:
            async with client.stream("GET", url) as response:
                if response.status_code != 200:
                    bt.logging.warning(
                        f"GitHub download failed: HTTP {response.status_code}"
                    )
                    return False

                content_length = response.headers.get("content-length")
                if content_length and int(content_length) > max_bytes:
                    bt.logging.error(
                        f"GitHub file too large: {int(content_length)} > {max_bytes} bytes"
                    )
                    return False

                total_bytes = 0
                with tmp.open("wb") as fh:
                    async for chunk in response.aiter_bytes(chunk_size=65536):
                        total_bytes += len(chunk)
                        if total_bytes > max_bytes:
                            bt.logging.error(
                                f"GitHub download exceeded {max_bytes} bytes during stream"
                            )
                            tmp.unlink(missing_ok=True)
                            return False
                        fh.write(chunk)

        bt.logging.info(f"Downloaded {total_bytes} bytes from GitHub")

        if dest.exists() and dest.is_dir():
            shutil.rmtree(dest)
        tmp.replace(dest)
        return True

    except httpx.TimeoutException:
        bt.logging.error(f"GitHub download timed out ({timeout_sec}s)")
        tmp.unlink(missing_ok=True)
        return False
    except Exception as e:
        bt.logging.warning(f"GitHub download error: {e}")
        tmp.unlink(missing_ok=True)
        return False


async def check_readme_matches(
    repo_url: str, *, uid: Optional[int] = None
) -> bool:
    """Verify that the repository contains the required README.md.

    Downloads ``README.md`` from the repo (tries ``main``, then ``master``)
    and checks its SHA-256 against ``REQUIRED_README_HASH``.
    """
    base = repo_url.rstrip("/")
    tag = f" (UID {uid})" if uid is not None else ""
    candidates = [
        f"{base}/raw/main/README.md",
        f"{base}/raw/master/README.md",
    ]

    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=httpx.Timeout(GITHUB_DOWNLOAD_TIMEOUT_SEC, connect=GITHUB_CONNECT_TIMEOUT_SEC),
        ) as client:
            for url in candidates:
                resp = await client.get(url)
                if resp.status_code != 200:
                    continue
                if len(resp.content) > GITHUB_MAX_README_BYTES:
                    bt.logging.warning(f"README.md too large{tag}: {len(resp.content)} bytes")
                    return False
                digest = hashlib.sha256(resp.content).hexdigest()
                if digest == REQUIRED_README_HASH:
                    return True
                branch = url.rsplit("/raw/", 1)[-1].split("/", 1)[0]
                bt.logging.info(
                    f"README.md hash mismatch on branch {branch}{tag}, trying next"
                )
    except Exception as e:
        bt.logging.warning(f"README.md check failed{tag}: {e}")
        return False

    bt.logging.warning(f"README.md not found in repo{tag}")
    return False
