#!/usr/bin/env bash
# ---------------------------------------------------------------
# auto_update_deploy.sh
# ---------------------------------------------------------------
# Periodically check origin/main for a higher __version__
# (defined in swarm/__init__.py).  If found, pull and call the
# local update_deploy.sh to rebuild / restart the running Swarm
# validator or miner.
#
#     bash auto_update_deploy.sh
# ---------------------------------------------------------------
set -euo pipefail
IFS=$'\n\t'

# ───────────────────────── Configuration ─────────────────────────
PROCESS_NAME="swarm-validator"          # pm2 process name
WALLET_NAME="my_wallet"                 # coldkey
WALLET_HOTKEY="my_hotkey"               # hotkey
SUBTENSOR_PARAM="--subtensor.network finney"  # extra flags

SLEEP_INTERVAL=600      # seconds between checks (10 min)

# ───────────────────────── Paths ─────────────────────────────────
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$REPO_ROOT" ]]; then
  echo "[ERR] Not inside a Git repository" >&2; exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPDATE_SCRIPT="$SCRIPT_DIR/update_deploy.sh"

[[ -x "$UPDATE_SCRIPT" ]] || {
  echo "[ERR] update_deploy.sh not found or not executable at $UPDATE_SCRIPT" >&2
  exit 1
}

# ────────────────── Helpers: version handling ────────────────────
get_version_from_file() {
  # $1 = file path
  # prints <major>.<minor>[...]
  grep -Eo "^__version__[[:space:]]*=[[:space:]]*['\"][0-9a-zA-Z\.\-]+" "$1" \
    | head -n1 | cut -d"'" -f2 | cut -d"\"" -f2
}

local_version()   { get_version_from_file "$REPO_ROOT/swarm/__init__.py"; }

remote_version() {
  git -C "$REPO_ROOT" fetch origin main --quiet
  git -C "$REPO_ROOT" show origin/main:swarm/__init__.py 2>/dev/null | \
    get_version_from_file /dev/stdin
}

is_remote_newer() {
  # returns 0 (true) if $2 (remote) is strictly higher than $1 (local)
  # both parameters must be non-empty.
  [[ -z "$1" || -z "$2" ]] && return 1
  [[ "$1" == "$2" ]] &&   return 1
  ver_sorted=$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -n1)
  [[ "$ver_sorted" == "$1" ]]
}

# ────────────────────────── Banner ───────────────────────────────
echo "[INFO] Auto-update watcher started"
echo "[INFO]   repo            : $REPO_ROOT"
echo "[INFO]   process name     : $PROCESS_NAME"
echo "[INFO]   wallet / hotkey  : $WALLET_NAME / $WALLET_HOTKEY"
echo "[INFO]   subtensor params : $SUBTENSOR_PARAM"
echo "[INFO]   check interval   : $((SLEEP_INTERVAL/60)) min"

# ────────────────────────── Main loop ────────────────────────────
while true; do
  LOCAL=$(local_version   || echo "0")
  REMOTE=$(remote_version || echo "0")
  echo "[INFO] Local v${LOCAL}  —  Remote v${REMOTE}"

  if is_remote_newer "$LOCAL" "$REMOTE"; then
    echo "[INFO] Newer version detected, pulling..."
    git -C "$REPO_ROOT" pull --ff-only origin main
    echo "[INFO] Running update_deploy.sh ..."
    bash -x "$UPDATE_SCRIPT" \
         "$PROCESS_NAME" "$WALLET_NAME" "$WALLET_HOTKEY" "$SUBTENSOR_PARAM"
    echo "[INFO] Update script finished."
  else
    echo "[INFO] No update needed."
  fi

  echo "[INFO] Sleeping ${SLEEP_INTERVAL}s ..."
  sleep "$SLEEP_INTERVAL"
done
