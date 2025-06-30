#!/usr/bin/env bash
# ---------------------------------------------------------------
# auto_update_deploy.sh – Watch the repo; upgrade & redeploy when
#                         a higher swarm.__version__ is on origin/main.
#
# Run under PM2/tmux/systemd, e.g.
#   pm2 start --name auto_update_validator \
#      --interpreter /bin/bash scripts/auto_update_deploy.sh
# ---------------------------------------------------------------
set -euo pipefail
IFS=$'\n\t'

###############################################################################
# 1. User‑tunable settings – **edit these** ──────────────────────
###############################################################################
PROCESS_NAME="swarm_validator"          # pm2 name used in your launch cmd
WALLET_NAME="my_cold"                   # coldkey
WALLET_HOTKEY="my_validator"            # hotkey
SUBTENSOR_PARAM="--subtensor.network finney"
SLEEP_INTERVAL=600                      # seconds between version checks
###############################################################################

# Path discovery
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
UPDATE_SCRIPT="$SCRIPT_DIR/update_deploy.sh"

[[ -x "$UPDATE_SCRIPT" ]] || {
  echo "[ERR] update_deploy.sh not executable at $UPDATE_SCRIPT" >&2; exit 1; }

###############################################################################
# Helper – read __version__ strings
###############################################################################
extract_version() {
  grep -Eo '^__version__[[:space:]]*=[[:space:]]*["'\'']([^"'\'']+)["'\'']' "$1" |
  head -n1 | sed -E 's/^__version__[[:space:]]*=[[:space:]]*["'\'']([^"'\'']+)["'\'']/\1/'
}

local_version() {
  extract_version "$REPO_ROOT/swarm/__init__.py" 2>/dev/null || echo "0"
}

remote_version() {
  git -C "$REPO_ROOT" fetch --quiet origin main
  temp=$(mktemp)
  git -C "$REPO_ROOT" show origin/main:swarm/__init__.py > "$temp" 2>/dev/null || { rm -f "$temp"; echo "0"; return; }
  extract_version "$temp" || echo "0"
  rm -f "$temp"
}

is_remote_newer() {
  # sort -V guarantees correct semantic order for dot‑separated numbers
  [[ "$1" != "$2" ]] && [[ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -n1)" == "$1" ]]
}

###############################################################################
# Banner
###############################################################################
echo "[INFO] ──────────────────────────────────────────────────────────────"
echo "[INFO] Auto‑update watcher started"
echo "[INFO] Repo root        : $REPO_ROOT"
echo "[INFO] PM2 process name : $PROCESS_NAME"
echo "[INFO] Wallet / Hotkey  : $WALLET_NAME / $WALLET_HOTKEY"
echo "[INFO] Check interval   : $((SLEEP_INTERVAL/60)) min"
echo "[INFO] ──────────────────────────────────────────────────────────────"

###############################################################################
# Main loop
###############################################################################
while true; do
  LVER="$(local_version)"
  RVER="$(remote_version)"

  echo "[INFO] Local v$LVER  –  Remote v$RVER"

  if is_remote_newer "$LVER" "$RVER"; then
    echo "[INFO] Newer version detected → running update_deploy.sh"
    bash "$UPDATE_SCRIPT" \
         "$PROCESS_NAME" \
         "$WALLET_NAME" \
         "$WALLET_HOTKEY" \
         "$SUBTENSOR_PARAM"
    echo "[INFO] Update finished – next check in $SLEEP_INTERVAL s."
  else
    echo "[INFO] Already up‑to‑date – next check in $SLEEP_INTERVAL s."
  fi

  sleep "$SLEEP_INTERVAL"
done
