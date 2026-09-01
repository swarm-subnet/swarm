#!/usr/bin/env bash
# The auto-updater moved to validator/scripts/update/auto_update_deploy.sh.
#
# This path stays because operators registered it with PM2 by hand, and PM2 keeps
# the path it was given across restarts. Removing it would break the updater on
# the pull that delivered the change, on machines we do not control.
#
# Re-register at the new path when convenient, then this can go:
#   pm2 delete auto_update_validator
#   pm2 start --name auto_update_validator --interpreter /bin/bash \
#             validator/scripts/update/auto_update_deploy.sh
#   pm2 save
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"
exec bash "$REPO_ROOT/validator/scripts/update/auto_update_deploy.sh" "$@"
