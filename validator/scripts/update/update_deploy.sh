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
# update_deploy.sh – pull the published validator image and run it.
#
# Nothing is compiled or installed on the host any more. The version this script
# replaced reset the checkout and reinstalled the package, which left every
# operator's machine deciding for itself what the validator ran on.
#
# It is deliberately still callable by the watcher that operators already have
# running. That watcher checks a version on main and runs this file from disk, so
# it carries a host onto containers without anyone restarting anything. The git
# sync below is what keeps its check working.
#
#   bash validator/scripts/update/update_deploy.sh
# ---------------------------------------------------------------
set -euo pipefail
IFS=$'\n\t'

###############################################################################
# Run from a copy, because the checkout sync below overwrites this file
#
# bash reads a script as it goes. `git reset --hard` rewrites this very file
# mid-run, and bash then continues from a byte offset that now points at
# different content: it stops partway through and still exits 0, so the caller
# sees a successful update that never pulled anything or restarted anything.
# Re-executing from a copy means the running bytes can no longer move.
###############################################################################
if [[ "${SWARM_UPDATE_FROM_COPY:-}" != "1" ]]; then
  _copy="$(mktemp -t swarm_update_deploy.XXXXXX)"
  cp "${BASH_SOURCE[0]}" "$_copy"
  trap 'rm -f "$_copy"' EXIT
  # The copy lives outside the checkout, so it is told where the checkout is.
  SWARM_UPDATE_FROM_COPY=1 \
  SWARM_UPDATE_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" \
    bash "$_copy" "$@"
  exit $?
fi

SCRIPT_DIR="${SWARM_UPDATE_SCRIPT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )}"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
COMPOSE_FILE="$REPO_ROOT/.docker/docker-compose.yml"
SERVICE="validator"
LEGACY_PM2_PROCESS="${PROCESS_NAME_OVERRIDE:-swarm_validator}"

STEP=0
banner() {
  STEP=$((STEP+1))
  echo -e "\n[STEP ${STEP}] $*\n"
}

banner "Checking docker compose"
docker compose version >/dev/null 2>&1 || {
  echo "[ERR] 'docker compose' is required. Install the Docker Compose plugin." >&2
  exit 1
}

# Keeps the compose file and this script current, and keeps the legacy watcher's
# version check meaningful: it compares the checkout against main, so a checkout
# that never moved would make it fire on every cycle forever.
banner "Syncing the checkout"
PREVIOUS_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
restore_checkout() {
  echo "[ERR] update failed; restoring the checkout to $PREVIOUS_COMMIT" >&2
  git -C "$REPO_ROOT" reset --hard "$PREVIOUS_COMMIT" >/dev/null 2>&1 || true
}
trap restore_checkout ERR
git -C "$REPO_ROOT" fetch --quiet origin main
git -C "$REPO_ROOT" reset --hard origin/main

[[ -f "$COMPOSE_FILE" ]] || { echo "[ERR] missing $COMPOSE_FILE" >&2; exit 1; }

# The container runs as the invoking user rather than root, and needs the host's
# docker group to reach the socket. Resolved here so .env stays about the wallet.
banner "Resolving the user and the docker socket group"
export SWARM_UID="${SWARM_UID:-$(id -u)}"
export SWARM_GID="${SWARM_GID:-$(id -g)}"
if [[ -z "${DOCKER_GID:-}" ]]; then
  DOCKER_GID="$(stat -c '%g' /var/run/docker.sock 2>/dev/null || echo 999)"
fi
export DOCKER_GID
export SWARM_STATE_DIR="${SWARM_STATE_DIR:-/opt/swarm-validator-state}"
export BT_WALLET_HOME="${BT_WALLET_HOME:-$HOME/.bittensor}"
echo "[INFO] uid=$SWARM_UID gid=$SWARM_GID docker_gid=$DOCKER_GID"
echo "[INFO] state=$SWARM_STATE_DIR wallets=$BT_WALLET_HOME"

# The same absolute path is mounted on both sides, so it has to exist on the host
# and be writable by the uid the container runs as.
banner "Preparing the state directory"
mkdir -p "$SWARM_STATE_DIR"
chown -R "$SWARM_UID:$SWARM_GID" "$SWARM_STATE_DIR" 2>/dev/null || true

banner "Pulling the published image"
docker compose -f "$COMPOSE_FILE" --profile "$SERVICE" pull "$SERVICE"

# One hotkey, one validator. A host process left running beside the container is a
# second session on the same hotkey, and the backend fences one of them off.
# Stopped only once the image is in hand, so a failed pull leaves the host running.
banner "Stopping the host validator, if one is still running"
if command -v pm2 >/dev/null 2>&1 && pm2 describe "$LEGACY_PM2_PROCESS" >/dev/null 2>&1; then
  echo "[INFO] stopping pm2 process '$LEGACY_PM2_PROCESS'"
  pm2 stop "$LEGACY_PM2_PROCESS" >/dev/null 2>&1 || true
  pm2 save >/dev/null 2>&1 || true
else
  echo "[INFO] no pm2 host validator found"
fi

# up -d recreates the container only when the image or its configuration changed,
# so an unchanged pull leaves the running validator alone.
banner "Starting the validator container"
docker compose -f "$COMPOSE_FILE" --profile "$SERVICE" up -d "$SERVICE"

banner "Running image"
docker compose -f "$COMPOSE_FILE" --profile "$SERVICE" images "$SERVICE" || true

# The deploy stands; a later failure must not roll the checkout back under it.
trap - ERR

echo -e "\n[INFO] Update complete. Logs: docker compose -f $COMPOSE_FILE logs -f $SERVICE"
exit 0
