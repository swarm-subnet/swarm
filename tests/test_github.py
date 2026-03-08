from __future__ import annotations

import asyncio
import hashlib
from unittest.mock import AsyncMock, MagicMock, patch

from swarm.utils.github import (
    REQUIRED_README_HASH,
    build_raw_urls,
    check_readme_matches,
    validate_github_url,
)


def test_validate_github_url_accepts_valid():
    assert validate_github_url("https://github.com/user/repo") == "https://github.com/user/repo"


def test_validate_github_url_strips_trailing_slash():
    assert validate_github_url("https://github.com/user/repo/") == "https://github.com/user/repo"


def test_validate_github_url_rejects_http():
    assert validate_github_url("http://github.com/user/repo") is None


def test_validate_github_url_rejects_non_github():
    assert validate_github_url("https://gitlab.com/user/repo") is None


def test_validate_github_url_rejects_missing_repo():
    assert validate_github_url("https://github.com/onlyone") is None


def test_validate_github_url_rejects_empty():
    assert validate_github_url("") is None
    assert validate_github_url("   ") is None


def test_build_raw_urls_returns_main_and_master():
    urls = build_raw_urls("https://github.com/user/repo")
    assert len(urls) == 2
    assert "main" in urls[0]
    assert "master" in urls[1]
    assert urls[0].endswith("/submission.zip")
    assert urls[1].endswith("/submission.zip")


def _make_response(status_code: int, content: bytes = b"") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.content = content
    return resp


def _mock_client(get_side_effect):
    client = AsyncMock()
    client.get = AsyncMock(side_effect=get_side_effect)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


def test_check_readme_matches_exact():
    readme_bytes = b"exact match content"
    digest = hashlib.sha256(readme_bytes).hexdigest()
    client = _mock_client([_make_response(200, readme_bytes)])

    with patch("swarm.utils.github.REQUIRED_README_HASH", digest), \
         patch("swarm.utils.github.httpx.AsyncClient", return_value=client):
        result = asyncio.run(check_readme_matches("https://github.com/u/r"))

    assert result is True


def test_check_readme_matches_wrong_content():
    client = _mock_client([_make_response(200, b"wrong content")])

    with patch("swarm.utils.github.httpx.AsyncClient", return_value=client):
        result = asyncio.run(check_readme_matches("https://github.com/u/r"))

    assert result is False


def test_check_readme_matches_not_found():
    client = _mock_client([_make_response(404), _make_response(404)])

    with patch("swarm.utils.github.httpx.AsyncClient", return_value=client):
        result = asyncio.run(check_readme_matches("https://github.com/u/r"))

    assert result is False


def test_check_readme_matches_fallback_to_master():
    readme_bytes = b"fallback content"
    digest = hashlib.sha256(readme_bytes).hexdigest()
    client = _mock_client([_make_response(404), _make_response(200, readme_bytes)])

    with patch("swarm.utils.github.REQUIRED_README_HASH", digest), \
         patch("swarm.utils.github.httpx.AsyncClient", return_value=client):
        result = asyncio.run(check_readme_matches("https://github.com/u/r"))

    assert result is True


def test_required_readme_hash_matches_template():
    from pathlib import Path

    template = Path(__file__).parent.parent / "swarm" / "templates" / "README.md"
    assert template.exists(), "Template README.md not found"
    digest = hashlib.sha256(template.read_bytes()).hexdigest()
    assert digest == REQUIRED_README_HASH, (
        f"REQUIRED_README_HASH is stale: template={digest}, constant={REQUIRED_README_HASH}"
    )
