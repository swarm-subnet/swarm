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

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import httpx
import bittensor as bt

GITHUB_DOWNLOAD_TIMEOUT_SEC = 60.0
GITHUB_CONNECT_TIMEOUT_SEC = 10.0
GITHUB_MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024


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

    repo = segments[1].removesuffix(".git")
    if not repo:
        bt.logging.warning(f"Rejecting github_url: empty repo name{tag}")
        return None

    return f"https://github.com/{segments[0]}/{repo}"


def build_raw_urls(repo_url: str, artifact_path: str) -> list[str]:
    """Build candidate raw download URLs for a manifest-declared artifact.

    Tries ``main`` first, then ``master`` as a fallback.
    """
    path = Path(artifact_path)
    if path.is_absolute() or ".." in path.parts or "\\" in artifact_path:
        raise ValueError("invalid artifact_path")
    normalized = path.as_posix()
    base = repo_url.rstrip("/")
    return [
        f"{base}/raw/main/{normalized}",
        f"{base}/raw/master/{normalized}",
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
