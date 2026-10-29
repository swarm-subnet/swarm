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
# image_lockdown_check.sh – cut a container off the network through the validator image.
#
# The validator runs as a plain uid with the compose file's capabilities, so this does the
# same: compose starts it as this user and the real lockdown code adds the rules to a
# throwaway container. It fails unless the rules land and that container then cannot
# reach the internet. Unit tests mock the firewall and never see a missing capability.
#
#   bash .docker/image_lockdown_check.sh
#   SWARM_UID=1000 SWARM_VALIDATOR_TAG=6.0.0 bash .docker/image_lockdown_check.sh
#   SWARM_UID=1000 SWARM_VALIDATOR_IMAGE=swarm-validator SWARM_VALIDATOR_TAG=dev \
#     SWARM_VALIDATOR_PULL=never bash .docker/image_lockdown_check.sh   # a local build
# ---------------------------------------------------------------
set -euo pipefail
REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
COMPOSE_FILE="$REPO_ROOT/.docker/docker-compose.yml"
IMAGE="${SWARM_VALIDATOR_IMAGE:-ghcr.io/swarm-subnet/swarm-validator}:${SWARM_VALIDATOR_TAG:-latest}"
STATE_DIR="$(mktemp -d /tmp/swarm-lockdown-check.XXXXXX)"
TARGET="swarm_lockdown_check_$$"
trap 'docker rm -f "$TARGET" >/dev/null 2>&1 || true; rm -rf "$STATE_DIR"' EXIT
# Created here so the daemon does not create them as root.
mkdir -p "$STATE_DIR/state" "$STATE_DIR/swarm-state" "$REPO_ROOT/state" "$REPO_ROOT/swarm/state"
SWARM_UID="${SWARM_UID:-$(id -u)}"
SWARM_GID="${SWARM_GID:-$SWARM_UID}"
DOCKER_GID="$(stat -c '%g' /var/run/docker.sock)"
export SWARM_STATE_DIR="$STATE_DIR" SWARM_UID SWARM_GID DOCKER_GID
# Root needs no capability at all, so a run as root would pass whatever the image grants.
[[ "$SWARM_UID" != "0" ]] || { echo "[ERR] run as a plain uid, or set SWARM_UID to one" >&2; exit 2; }

compose() {
  docker compose -f "$COMPOSE_FILE" --profile validator run --rm --no-deps \
    --pull "${SWARM_VALIDATOR_PULL:-always}" \
    -v "$REPO_ROOT:/opt/swarm-validator" swarm_validator "$@"
}

reaches_internet() {
  docker exec "$TARGET" python -c "
import socket
socket.create_connection(('1.1.1.1', 443), timeout=5)
" 2>/dev/null
}

# Started the way the evaluator starts a flight: by the host daemon, as the validator's
# uid, on the bridge network.
docker run -d --rm --name "$TARGET" --user "$SWARM_UID:$SWARM_GID" --network bridge \
  "$IMAGE" sleep 300 >/dev/null
PID="$(docker inspect -f '{{.State.Pid}}' "$TARGET")"

# Blocked egress only proves something if the container had egress to begin with.
reaches_internet || { echo "[ERR] the container cannot reach the internet before the lockdown" >&2; exit 1; }

echo "[INFO] validator uid $SWARM_UID, target pid $PID"
compose python -c "
import sys
from swarm.validator.docker.docker_evaluator_parts.networking import _apply_network_lockdown
sys.exit(0 if _apply_network_lockdown(None, $PID, '127.0.0.1') else 1)
" || { echo "[ERR] the validator could not lock the container down as uid $SWARM_UID" >&2; exit 1; }

if reaches_internet; then
  echo "[ERR] the rules were applied but the container still reaches the internet" >&2
  exit 1
fi
echo "[OK] lockdown applied as uid $SWARM_UID and egress is blocked"
