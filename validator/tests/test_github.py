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

"""Accepting a miner's repository URL, building its raw download links, and keeping the shipped docs' links alive."""
from __future__ import annotations

import re
from pathlib import Path

from swarm.utils.github import build_raw_urls, validate_github_url


def test_validate_github_url_accepts_valid():
    """An https owner/repo link on github.com is handed back unchanged."""
    assert validate_github_url("https://github.com/user/repo") == "https://github.com/user/repo"


def test_validate_github_url_strips_trailing_slash():
    """One repository has one canonical form, whether or not the miner typed the final slash."""
    assert validate_github_url("https://github.com/user/repo/") == "https://github.com/user/repo"


def test_validate_github_url_rejects_http():
    """Plain http is refused, so a submission can only be fetched over TLS."""
    assert validate_github_url("http://github.com/user/repo") is None


def test_validate_github_url_rejects_non_github():
    """A host other than github.com is refused even when it is served over https."""
    assert validate_github_url("https://gitlab.com/user/repo") is None


def test_validate_github_url_rejects_missing_repo():
    """An owner page with no repository segment is refused."""
    assert validate_github_url("https://github.com/onlyone") is None


def test_validate_github_url_rejects_empty():
    """A blank or whitespace-only field is refused rather than treated as a link."""
    assert validate_github_url("") is None
    assert validate_github_url("   ") is None


def test_build_raw_urls_returns_main_and_master():
    """Two candidates come back, main first then master, each ending at the artifact path asked for."""
    urls = build_raw_urls(
        "https://github.com/user/repo", "artifacts/cf_autopilot/submission.zip"
    )
    assert len(urls) == 2
    assert "main" in urls[0]
    assert "master" in urls[1]
    assert urls[0].endswith("/artifacts/cf_autopilot/submission.zip")
    assert urls[1].endswith("/artifacts/cf_autopilot/submission.zip")


# The starter README still ships with published public repositories, so a link in it
# that stops resolving is a broken link in every one of them.
_MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
_CANONICAL_LINK = re.compile(
    r"https://github\.com/swarm-subnet/swarm/(?:tree|blob)/main/([^)\s\"']+)"
)


def _pinned_readme_targets() -> list[str]:
    """Every repository path the starter README links to, deduplicated and sorted."""
    template = Path(__file__).resolve().parents[2] / "swarm" / "templates" / "README.md"
    return sorted(set(_CANONICAL_LINK.findall(template.read_text())))


def test_local_links_in_the_miner_guide_resolve():
    """Moving a page changes what every relative link inside it means."""
    root = Path(__file__).resolve().parents[2]
    source = root / "miner" / "docs" / "miner.md"
    local = [
        t for t in _MARKDOWN_LINK.findall(source.read_text())
        if not t.startswith(("http", "mailto:", "/"))
    ]
    missing = sorted({t for t in local if not (source.parent / t.split("#")[0]).exists()})
    assert missing == [], f"the miner guide links to {missing}, which do not exist"


def test_pinned_readme_links_point_at_paths_that_exist():
    """Every canonical link in the starter README still resolves inside the repository."""
    repo_root = Path(__file__).resolve().parents[2]
    targets = _pinned_readme_targets()
    assert targets, "no canonical repository links found in the starter README"
    missing = [t for t in targets if not (repo_root / t).exists()]
    assert missing == [], (
        f"the starter README links to {missing}, which no longer exist; every repository "
        "already published carries this README and cannot be corrected"
    )
