from __future__ import annotations

from swarm.utils.github import validate_github_url


def test_validate_github_url_strips_git_suffix():
    assert (
        validate_github_url("https://github.com/example/project.git/")
        == "https://github.com/example/project"
    )
