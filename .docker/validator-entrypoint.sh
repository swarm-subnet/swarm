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
# An operator sets wallet and netuid in .env and never types a flag. Anything passed to
# `docker run` after the image name is appended, so an unusual option is still reachable.
set -euo pipefail

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

args=(
  --netuid "$NETUID"
  --wallet.name "$WALLET_NAME"
  --wallet.hotkey "$WALLET_HOTKEY"
  --subtensor.network "$SUBTENSOR_NETWORK"
)
[[ -n "$SUBTENSOR_ENDPOINT" ]] && args+=(--subtensor.chain_endpoint "$SUBTENSOR_ENDPOINT")
[[ -n "$LOGGING" ]] && args+=("--logging.$LOGGING")

echo "[INFO] swarm validator $(python -c 'import swarm; print(swarm.__version__)')"
echo "[INFO] netuid $NETUID, wallet $WALLET_NAME/$WALLET_HOTKEY, network $SUBTENSOR_NETWORK"

exec python /opt/swarm-validator/neurons/validator.py "${args[@]}" "$@"
