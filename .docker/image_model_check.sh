#!/usr/bin/env bash
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

# ---------------------------------------------------------------
# image_model_check.sh – run the first-time model check through the validator image.
#
# The image drives the host's Docker daemon with the compose file's mounts, exactly as an
# operator's validator does, and checks the default model with this checkout's code. It
# fails unless the model passes. No wallet is needed.
#
# Not for a box with a live validator: the check may rebuild the evaluator image and
# clear that validator's model containers.
#
#   bash .docker/image_model_check.sh
#   SWARM_VALIDATOR_TAG=5.1.6.4 bash .docker/image_model_check.sh
# ---------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
STATE_DIR="$(mktemp -d /tmp/swarm-image-check.XXXXXX)"
trap 'rm -rf "$STATE_DIR"' EXIT

# Created here so the daemon does not create them as root.
mkdir -p "$STATE_DIR/state" "$STATE_DIR/swarm-state" "$REPO_ROOT/state" "$REPO_ROOT/swarm/state"

SWARM_UID="$(id -u)"
SWARM_GID="$(id -g)"
DOCKER_GID="$(stat -c '%g' /var/run/docker.sock)"
export SWARM_STATE_DIR="$STATE_DIR" SWARM_UID SWARM_GID DOCKER_GID

docker compose -f "$REPO_ROOT/.docker/docker-compose.yml" --profile validator run --rm \
  -v "$REPO_ROOT:/opt/swarm-validator" \
  swarm_validator \
  python /opt/swarm-validator/validator/scripts/image_model_check.py \
  /opt/swarm-validator/validator/tests/default_model/default_model.zip
