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
# auto_update_deploy.sh – watch the registry and redeploy on a newer version.
#
# It compares the version label of the image the validator is running against the
# label of a freshly pulled :latest. No git, no checkout, no reinstall.
#
# Run it under systemd (validator/scripts/update/swarm-validator-updater.service)
# or, as before, under PM2:
#   pm2 start --name auto_update_validator --interpreter /bin/bash \
#             validator/scripts/update/auto_update_deploy.sh
# ---------------------------------------------------------------
set -euo pipefail
IFS=$'\n\t'

###############################################################################
# 1. User-tunable settings – **edit these** ──────────────────────
###############################################################################
SLEEP_INTERVAL=600                      # seconds between version checks
IMAGE="${SWARM_VALIDATOR_IMAGE:-ghcr.io/swarm-subnet/swarm-validator}"
TAG="${SWARM_VALIDATOR_TAG:-latest}"
###############################################################################

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
UPDATE_SCRIPT="$SCRIPT_DIR/update_deploy.sh"
COMPOSE_FILE="$REPO_ROOT/.docker/docker-compose.yml"
VERSION_LABEL="swarm.__version__"

[[ -f "$UPDATE_SCRIPT" ]] || { echo "[ERR] missing $UPDATE_SCRIPT" >&2; exit 1; }

###############################################################################
# Helpers
###############################################################################
image_version() {
  # The version label of a local image reference, or empty when it is absent.
  docker image inspect --format "{{index .Config.Labels \"$VERSION_LABEL\"}}" "$1" 2>/dev/null || true
}

running_version() {
  # What the validator is actually running, read from the container's own image
  # rather than from a tag, so a retagged :latest cannot be mistaken for a redeploy.
  local image_id
  image_id="$(docker compose -f "$COMPOSE_FILE" --profile validator ps -q swarm_validator 2>/dev/null \
              | head -n1 | xargs -r docker inspect --format '{{.Image}}' 2>/dev/null || true)"
  [[ -n "$image_id" ]] && image_version "$image_id"
}

is_remote_newer() {
  # sort -V orders dot-separated versions correctly. Equal is not newer.
  [[ "$1" != "$2" ]] && [[ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -n1)" == "$1" ]]
}

###############################################################################
# Banner
###############################################################################
echo "[INFO] ──────────────────────────────────────────────────────────────"
echo "[INFO] Validator image watcher started"
echo "[INFO] Image          : $IMAGE:$TAG"
echo "[INFO] Compose file   : $COMPOSE_FILE"
echo "[INFO] Check interval : $((SLEEP_INTERVAL/60)) min"
echo "[INFO] ──────────────────────────────────────────────────────────────"

###############################################################################
# Main loop
###############################################################################
while true; do
  LVER="$(running_version || true)"

  if docker pull --quiet "$IMAGE:$TAG" >/dev/null 2>&1; then
    RVER="$(image_version "$IMAGE:$TAG")"
  else
    RVER=""
    echo "[WARN] could not reach the registry; keeping the current version"
  fi

  echo "[INFO] Running v${LVER:-unknown}  –  Published v${RVER:-unknown}"

  if [[ -z "$LVER" && -n "$RVER" ]]; then
    echo "[INFO] Validator is not running yet → deploying"
    NEEDS_UPDATE=1
  elif [[ -n "$RVER" ]] && is_remote_newer "$LVER" "$RVER"; then
    echo "[INFO] Newer version published → redeploying"
    NEEDS_UPDATE=1
  else
    NEEDS_UPDATE=0
  fi

  if (( NEEDS_UPDATE )); then
    # Guarded: unguarded under set -e, one failed deploy would kill this watcher
    # and the host would never update again.
    if bash "$UPDATE_SCRIPT"; then
      echo "[INFO] Update finished – next check in $SLEEP_INTERVAL s."
    else
      echo "[ERR] Update failed – retrying in $SLEEP_INTERVAL s." >&2
    fi
  else
    echo "[INFO] Already up-to-date – next check in $SLEEP_INTERVAL s."
  fi

  sleep "$SLEEP_INTERVAL"
done
