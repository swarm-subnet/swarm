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

# Turn the container's environment into the validator's command line.
#
# An operator sets wallet and netuid in .env and never types a flag. Extra flags passed
# to `docker run` after the image name are appended, so an unusual option is still
# reachable. A command instead of flags runs as it is, with no wallet and no backend:
#
#   docker run --rm swarm-validator pytest validator/tests
#   docker run --rm swarm-validator python validator/scripts/cross_machine_check.py run
#
# which is what a test job or a nightly build wants from this image.
set -euo pipefail

# Compose starts the container as root so this can create the state directories and
# hand them to the uid: Docker creates a missing bind source as root, and nothing else
# in the container could fix that. Then it becomes that uid for good. The two
# capabilities are carried across the switch, since dropping uid would drop them.
if [[ "$(id -u)" == "0" && -n "${SWARM_UID:-}" && "$SWARM_UID" != "0" ]]; then
  gid="${SWARM_GID:-$SWARM_UID}"
  for dir in "${SWARM_STATE_DIR:-}" /opt/swarm-validator/state /opt/swarm-validator/swarm/state; do
    [[ -n "$dir" ]] || continue
    mkdir -p "$dir"
    chown "$SWARM_UID:$gid" "$dir"
  done
  groups="$gid"
  [[ -n "${DOCKER_GID:-}" ]] && groups="$gid,$DOCKER_GID"
  # A container started without the two capabilities, a test run for instance, has
  # nothing to carry across and must not fail on the attempt.
  # shellcheck disable=SC2054  # setpriv takes the capabilities as one comma-separated list
  caps=(--inh-caps +sys_admin,+net_admin --ambient-caps +sys_admin,+net_admin)
  setpriv "${caps[@]}" --reuid "$SWARM_UID" --regid "$gid" --groups "$groups" true 2>/dev/null \
    || caps=()
  exec setpriv --reuid "$SWARM_UID" --regid "$gid" --groups "$groups" "${caps[@]}" \
      bash "$0" "$@"
fi

# A bare uid has no passwd entry, so Docker sets HOME=/ and bittensor, which creates
# ~/.bittensor on import, dies before reading any config. Compose points HOME at the
# state directory; a plain `docker run` gets a scratch home instead.
if [[ ! -w "${HOME:-/}" ]]; then
  HOME="/tmp/swarm-validator-home-$(id -u)"
  export HOME
  mkdir -p "$HOME"
fi

if (( $# )) && [[ "$1" != -* ]]; then
  exec "$@"
fi

NETUID="${SWARM_NETUID:-124}"
WALLET_NAME="${SWARM_WALLET_NAME:-}"
WALLET_HOTKEY="${SWARM_WALLET_HOTKEY:-}"
SUBTENSOR_NETWORK="${SWARM_SUBTENSOR_NETWORK:-finney}"
SUBTENSOR_ENDPOINT="${SWARM_SUBTENSOR_CHAIN_ENDPOINT:-}"
LOGGING="${SWARM_LOGGING:-}"

missing=()
[[ -z "$WALLET_NAME" ]] && missing+=("SWARM_WALLET_NAME")
[[ -z "$WALLET_HOTKEY" ]] && missing+=("SWARM_WALLET_HOTKEY")
[[ -z "${SWARM_BACKEND_API_URL:-}" ]] && missing+=("SWARM_BACKEND_API_URL")
if (( ${#missing[@]} )); then
  echo "[ERR] missing required environment: ${missing[*]}" >&2
  echo "      set them in .env next to the compose file, then start again" >&2
  exit 2
fi

# A wallet the container cannot read is the most common first failure, and the error it
# produces otherwise names a key rather than the mount.
WALLET_ROOT="${BT_WALLET_PATH:-$HOME/.bittensor/wallets}"
if [[ ! -d "$WALLET_ROOT/$WALLET_NAME" ]]; then
  echo "[ERR] wallet '$WALLET_NAME' not found under $WALLET_ROOT" >&2
  echo "      check the ~/.bittensor mount and that the container uid can read it" >&2
  exit 2
fi

# Everything the validator writes goes under the state directory, so a directory the
# uid cannot write fails here with the fix, not later inside a library.
STATE_DIR="${SWARM_STATE_DIR:-$HOME}"
for dir in "$STATE_DIR" /opt/swarm-validator/state /opt/swarm-validator/swarm/state; do
  if [[ ! -w "$dir" ]]; then
    echo "[ERR] $dir is not writable by uid $(id -u)" >&2
    echo "      on the host: sudo chown -R $(id -u):$(id -g) $STATE_DIR" >&2
    exit 2
  fi
done

args=(
  --netuid "$NETUID"
  --wallet.name "$WALLET_NAME"
  --wallet.hotkey "$WALLET_HOTKEY"
  --wallet.path "$WALLET_ROOT"
  --subtensor.network "$SUBTENSOR_NETWORK"
)
[[ -n "$SUBTENSOR_ENDPOINT" ]] && args+=(--subtensor.chain_endpoint "$SUBTENSOR_ENDPOINT")
[[ -n "$LOGGING" ]] && args+=("--logging.$LOGGING")

echo "[INFO] swarm validator $(python -c 'import swarm; print(swarm.__version__)')"
echo "[INFO] netuid $NETUID, wallet $WALLET_NAME/$WALLET_HOTKEY, network $SUBTENSOR_NETWORK"

exec python /opt/swarm-validator/neurons/validator.py "${args[@]}" "$@"
