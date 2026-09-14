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

"""Fail when code this branch adds or changes carries no docstring.

Most of the repository predates the rule, so checking the whole tree would fail on every commit
for reasons nobody in the pull request caused. The check reads the diff instead: a definition has
to carry a docstring once this branch touches one of its own lines. Modules, classes, functions
and methods all count, including the underscore-named ones and the ones written inside another
function, which is where a reader arrives with the least context.

Run `--all` to check every tracked file instead, which is what this becomes once the backfill of
the older definitions has landed.
"""
from __future__ import annotations

import argparse
import ast
import os
import re
import subprocess
import sys
from pathlib import Path

DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)

# Mirrors extend-exclude in pyproject.toml: vendored assets and build output are not ours to write.
EXCLUDED_DIRS = ("swarm/assets/", "artifacts/", "miner_env/")

# A test pins this module text-identical to the backend's copy apart from its import roots, so a
# docstring here would have to land in two repositories within the same change.
EXCLUDED_FILES = frozenset({"swarm/submission_manifest/__init__.py"})

HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def git(*args: str, root: Path | None = None) -> str:
    """Return the standard output of a git command, raising CalledProcessError when it fails."""
    location = ("-C", str(root)) if root else ()
    return subprocess.run(("git", *location, *args), capture_output=True, text=True, check=True).stdout


def repo_root() -> Path:
    """Return the top of the working tree this script was invoked inside."""
    return Path(git("rev-parse", "--show-toplevel").strip())


def resolve(ref: str, root: Path) -> str | None:
    """Return the commit a ref points at, or None when the ref does not exist here."""
    try:
        return git("rev-parse", "--verify", "--quiet", ref + "^{commit}", root=root).strip() or None
    except subprocess.CalledProcessError:
        return None


def base_commit(explicit: str | None, root: Path) -> str:
    """Return the commit this branch is measured against, the merge base with the target branch.

    Diffing the merge base rather than the branch tip keeps the scope to what this branch did:
    commits that landed on the target meanwhile are not this pull request's to document.
    """
    target = explicit or os.environ.get("GITHUB_BASE_REF") or "main"
    names = ("origin/" + target, target)
    for name in names:
        if resolve(name, root):
            return git("merge-base", name, "HEAD", root=root).strip()
    raise SystemExit(
        "check_docstrings: cannot find the branch to compare against, tried " + ", ".join(names) + ".\n"
        "  In CI the checkout needs fetch-depth: 0 so the base branch is present.\n"
        "  Locally, pass --base <branch>."
    )


def excluded(rel: str) -> bool:
    """Return whether a repository-relative path is outside what this check governs."""
    return (
        rel in EXCLUDED_FILES
        or rel.startswith(EXCLUDED_DIRS)
        or ".egg-info/" in rel
        or not rel.endswith(".py")
    )


def added_lines(diff: str) -> dict[str, set[int]]:
    """Map each Python file in a unified diff to the line numbers it gains, numbered in the new file.

    Blank lines and whole-line comments are left out: reflowing a file or annotating a line is not
    a reason to demand prose on whatever definition surrounds it. Deletions are left out for the
    same reason, so removing a line from an old function does not drag it into scope, and a file
    that was only moved arrives here with nothing added at all.
    """
    files: dict[str, set[int]] = {}
    current: set[int] | None = None
    cursor = 0
    for line in diff.splitlines():
        if line.startswith("+++ "):
            path = line[4:].strip()
            current = None
            if path != "/dev/null" and path.removeprefix("b/").endswith(".py"):
                current = files.setdefault(path.removeprefix("b/"), set())
        elif current is None or line.startswith(("--- ", "diff --git", "\\")):
            continue
        elif hunk := HUNK.match(line):
            cursor = int(hunk.group(1))
        elif line.startswith("+"):
            body = line[1:].strip()
            if body and not body.startswith("#"):
                current.add(cursor)
            cursor += 1
    return files


def span(node: ast.AST) -> tuple[int, int]:
    """Return the first and last line a definition occupies, its decorators included."""
    start = node.lineno
    for decorator in getattr(node, "decorator_list", ()):
        start = min(start, decorator.lineno)
    return start, getattr(node, "end_lineno", None) or node.lineno


def qualified_names(tree: ast.Module) -> dict[ast.AST, str]:
    """Map every definition in a module to its dotted name, so a report reads like the source."""
    names: dict[ast.AST, str] = {}
    stack: list[tuple[ast.AST, str]] = [(tree, "")]
    while stack:
        node, prefix = stack.pop()
        for child in ast.iter_child_nodes(node):
            if isinstance(child, DEFINITIONS):
                name = prefix + child.name
                names[child] = name
                stack.append((child, name + "."))
            else:
                stack.append((child, prefix))
    return names


def innermost(line: int, spans: list[tuple[ast.AST, int, int]]) -> ast.AST | None:
    """Return the narrowest definition containing a line, or None when the line sits at module level.

    Narrowest rather than outermost is what keeps the check proportionate: editing one line inside
    a nested helper asks for a docstring on that helper, not on every function wrapped around it.
    """
    owner = None
    width = None
    for node, start, end in spans:
        if start <= line <= end and (width is None or end - start < width):
            owner, width = node, end - start
    return owner


def problems(rel: str, source: str, added: set[int] | None) -> list[str]:
    """Return one message per definition in scope that carries no docstring.

    A None line set means every definition is in scope, which is what --all runs.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        return [rel + ":" + str(error.lineno or 1) + ": does not parse: " + (error.msg or "")]

    spans = [(node, *span(node)) for node in ast.walk(tree) if isinstance(node, DEFINITIONS)]
    names = qualified_names(tree)

    if added is None:
        in_scope = {tree, *(node for node, _, _ in spans)}
    else:
        in_scope = {innermost(line, spans) or tree for line in added}

    found = []
    for node in in_scope:
        if ast.get_docstring(node) is not None:
            continue
        if node is tree:
            found.append(rel + ":1: no docstring on the module")
        else:
            found.append(rel + ":" + str(node.lineno) + ": no docstring on " + names[node])
    return sorted(found, key=lambda text: int(text.split(":")[1]))


def main() -> int:
    """Print every definition in scope that lacks a docstring and return 1 when any was found."""
    parser = argparse.ArgumentParser(description="Check docstrings on the code this branch changes.")
    parser.add_argument("--all", action="store_true", help="check every tracked file, not only the diff")
    parser.add_argument("--base", help="branch to compare against, default is the pull request base or main")
    args = parser.parse_args()

    root = repo_root()
    if args.all:
        targets: dict[str, set[int] | None] = dict.fromkeys(git("ls-files", "*.py", root=root).splitlines())
    else:
        base = base_commit(args.base, root)
        targets = added_lines(git("diff", "-U0", "-M", base, root=root))

    failures: list[str] = []
    checked = 0
    for rel, added in sorted(targets.items()):
        path = root / rel
        if excluded(rel) or not path.is_file() or added == set():
            continue
        checked += 1
        failures.extend(problems(rel, path.read_text(encoding="utf-8"), added))

    for failure in failures:
        print(failure)
    if failures:
        print()
        print("Every module, class, function and method needs a docstring, including the")
        print("underscore-named ones and the ones written inside another function.")
    scope = "every tracked file" if args.all else str(checked) + " changed file(s)"
    print("check_docstrings: " + scope + ", " + str(len(failures)) + " without a docstring")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
