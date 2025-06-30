#!/usr/bin/env bash
# ---------------------------------------------------------------
# update_deploy.sh – Pull latest code, reinstall, restart PM2.
#
# Called directly or by auto_update_deploy.sh.
# If everything is already up‑to‑date it still rebuilds / restarts,
# so that environment changes (e.g. new requirements) are picked up.
# ---------------------------------------------------------------
set -euo pipefail
IFS=$'\n\t'

###############################################################################
# 0. Helper – tiny progress banner
###############################################################################
STEP=0
banner() {
  STEP=$((STEP+1))
  echo -e "\n[STEP ${STEP}] $*\n"
}

###############################################################################
# 1. Configuration (env‑vars → CLI‑args → defaults)
###############################################################################
banner "Loading configuration"

# ► Defaults – match the public instructions exactly
PROCESS_NAME="swarm_validator"          # pm2 process name
WALLET_NAME=""                          # coldkey  (empty ⇒ prompt if interactive)
WALLET_HOTKEY=""                        # hotkey   (empty ⇒ prompt if interactive)
SUBTENSOR_PARAM="--subtensor.network finney"

# ◄ Allow overrides from environment
PROCESS_NAME="${PROCESS_NAME_OVERRIDE:-$PROCESS_NAME}"
WALLET_NAME="${WALLET_NAME_OVERRIDE:-$WALLET_NAME}"
WALLET_HOTKEY="${WALLET_HOTKEY_OVERRIDE:-$WALLET_HOTKEY}"
SUBTENSOR_PARAM="${SUBTENSOR_PARAM_OVERRIDE:-$SUBTENSOR_PARAM}"

# ◄ Allow overrides from positional CLI args
[[ $# -ge 1 ]] && PROCESS_NAME="$1"
[[ $# -ge 2 ]] && WALLET_NAME="$2"
[[ $# -ge 3 ]] && WALLET_HOTKEY="$3"
[[ $# -ge 4 ]] && SUBTENSOR_PARAM="$4"

# ◄ Interactive prompts (only if running on TTY and still empty)
if [[ -t 0 ]]; then
  [[ -z "$WALLET_NAME"     ]] && read -rp "Coldkey name            : " WALLET_NAME
  [[ -z "$WALLET_HOTKEY"   ]] && read -rp "Hotkey                  : " WALLET_HOTKEY
fi

[[ -z "$WALLET_NAME"   || -z "$WALLET_HOTKEY" ]] && {
  echo "[ERR] WALLET_NAME or WALLET_HOTKEY not set." >&2
  exit 1
}

###############################################################################
# 2. Locate repo root and virtualenv
###############################################################################
banner "Locating repository root & virtualenv"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
echo "Repository root : $REPO_ROOT"

VENV_DIR="$REPO_ROOT/validator_env"
PYTHON_BIN="$VENV_DIR/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[INFO] Virtualenv not found – running setup script first."
  bash "$REPO_ROOT/scripts/validator/main/setup.sh"
fi

###############################################################################
# 3. Update repository
###############################################################################
banner "Pulling latest code from origin/main"
git -C "$REPO_ROOT" fetch --quiet origin main
git -C "$REPO_ROOT" reset --hard origin/main

###############################################################################
# 4. Re‑install package inside venv & restart validator
###############################################################################
banner "Installing updated Python package"
source "$VENV_DIR/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet -e "$REPO_ROOT"

banner "Restarting PM2 process: $PROCESS_NAME"
if ! pm2 restart "$PROCESS_NAME" &>/dev/null; then
  echo "[WARN] PM2 process not found – starting a fresh one."
  interp="$(command -v python)"        # fallback if venv not on PATH for pm2
  pm2 start "$REPO_ROOT/neurons/validator.py" \
        --name "$PROCESS_NAME" \
        --interpreter "$interp" \
        -- \
          --netuid 124 $SUBTENSOR_PARAM \
          --wallet.name "$WALLET_NAME" \
          --wallet.hotkey "$WALLET_HOTKEY"
fi

banner "Update & redeploy completed – validator running"
exit 0
