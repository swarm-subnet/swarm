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

"""Validate family benchmark declarations against the release's merge base."""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from pathlib import Path

REGISTRY_PATH = 'swarm/domain_model/benchmark_domain_model.schema.json'
VERSION_PATH = 'swarm/__init__.py'


def version_tuple(value: str) -> tuple[int, int, int]:
    """Parse a strictly numeric three-part benchmark label."""
    if not isinstance(value, str) or re.fullmatch(r'[0-9]+\.[0-9]+\.[0-9]+', value) is None:
        raise ValueError(f'invalid benchmark version: {value!r}')
    return tuple(map(int, value.split('.')))


def code_version(source: str) -> str:
    """Read the literal deployment version without importing application dependencies."""
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == '__version__' for target in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('missing __version__ assignment')


def validate_release(registry: dict, release: str, base_registry: dict, base_release: str) -> None:
    """Reject invalid snapshots and family changes that do not declare a matching release."""
    prefix = version_tuple('.'.join(release.split('.')[:3]))
    base_prefix = version_tuple('.'.join(base_release.split('.')[:3]))
    families = registry['challenge_families']
    if set(families) != set(registry['entity_types']['challenge_family']['ids']):
        raise ValueError('challenge family IDs differ from the registry enum')
    values = {family: version_tuple(entry.get('benchmark_version')) for family, entry in families.items()}
    if not values or max(values.values()) != prefix:
        raise ValueError('maximum family version must equal the global benchmark prefix')
    if prefix < base_prefix:
        raise ValueError('global benchmark version decreased')
    changed = []
    for family, value in values.items():
        previous = base_registry['challenge_families'].get(family)
        if previous is None:
            # A family new to the registry has no scores to restart; it declares the bump only when it lands on it.
            if value == prefix and prefix != base_prefix:
                changed.append(family)
            continue
        old = version_tuple(previous.get('benchmark_version', '.'.join(map(str, base_prefix))))
        if value < old:
            raise ValueError(f'{family}: benchmark version decreased')
        if value != old:
            changed.append(family)
            if prefix == base_prefix:
                raise ValueError(f'{family}: family version changed without a global benchmark bump')
            if value != prefix:
                raise ValueError(f'{family}: changed family version must equal the new global prefix')
    if prefix != base_prefix and not changed:
        raise ValueError('global benchmark bump must change at least one family')


def main(argv: list[str] | None = None) -> int:
    """Compare the working snapshot with the merge base of an explicit local or CI ref."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True, help='PR base SHA or local base ref')
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    base = subprocess.check_output(['git', 'merge-base', args.base, 'HEAD'], cwd=root, text=True).strip()
    base_registry = json.loads(subprocess.check_output(['git', 'show', f'{base}:{REGISTRY_PATH}'], cwd=root, text=True))
    base_release = code_version(subprocess.check_output(['git', 'show', f'{base}:{VERSION_PATH}'], cwd=root, text=True))
    try:
        validate_release(json.loads((root / REGISTRY_PATH).read_text()), code_version((root / VERSION_PATH).read_text()), base_registry, base_release)
    except ValueError as exc:
        parser.exit(1, f'family versions: {exc}\n')
    print('family versions: release declarations valid')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
