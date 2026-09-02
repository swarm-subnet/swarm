#!/bin/bash
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

# setup.sh - Setup miner Python environment and dependencies
set -e

handle_error() {
  echo -e "\e[31m[ERROR]\e[0m $1" >&2
  exit 1
}

success_msg() {
  echo -e "\e[32m[SUCCESS]\e[0m $1"
}

info_msg() {
  echo -e "\e[34m[INFO]\e[0m $1"
}

check_python() {
  info_msg "Checking for Python 3.11..."
  python3.11 --version || handle_error "Python 3.11 is required. Run install_dependencies.sh first."
}

create_activate_venv() {
  VENV_DIR="miner_env"
  info_msg "Creating virtualenv in $VENV_DIR..."
  if [ ! -d "$VENV_DIR" ]; then
    python3.11 -m venv "$VENV_DIR" \
      || handle_error "Failed to create virtualenv"
    success_msg "Virtualenv created."
  else
    info_msg "Virtualenv already exists. Skipping creation."
  fi

  info_msg "Activating virtualenv..."
  source "$VENV_DIR/bin/activate" \
    || handle_error "Failed to activate virtualenv"
}

check_uv() {
  info_msg "Checking for uv..."
  uv --version || handle_error "uv is required. Run install_dependencies.sh first."
}

upgrade_setuptools() {
  info_msg "Upgrading setuptools..."
  uv pip install --upgrade setuptools \
    || handle_error "Failed to upgrade setuptools"
  success_msg "setuptools upgraded."
}

install_python_reqs() {
  info_msg "Installing Python dependencies from requirements.txt..."
  [ -f "requirements.txt" ] || handle_error "requirements.txt not found"

  uv pip install -r requirements.txt \
    || handle_error "Failed to install Python dependencies"

  success_msg "Packages installed"
}

install_modules() {
  info_msg "Installing current package in editable mode..."
  uv pip install -e . --no-deps \
    || handle_error "Failed to install current package"
  success_msg "Main package installed."

}

verify_installation() {
  info_msg "Verifying miner environment setup..."
  
  # Check Bittensor
  python -c "import bittensor; print(f'✓ Bittensor: {bittensor.__version__}')" || \
    info_msg "⚠ Warning: Bittensor import failed"
  
  success_msg "Installation verification completed."
}

show_completion_info() {
  echo
  success_msg "Miner setup completed successfully!"
  echo
  echo -e "\e[33m[INFO]\e[0m Virtual environment: $(pwd)/miner_env"
  echo -e "\e[33m[INFO]\e[0m To activate: source miner_env/bin/activate"
  echo
  echo -e "\e[32m[READY]\e[0m Your miner environment is ready to use!"
  echo
  echo -e "\e[34m[NEXT STEPS]\e[0m"
  echo "1. Start your miner with PM2:"
  echo "   source miner_env/bin/activate"
  echo "   pm2 start miner/src/miner.py --name miner --interpreter python -- \\"
  echo "     --netuid 124 --subtensor.network finney \\"
  echo "     --wallet.name your_coldkey --wallet.hotkey your_hotkey \\"
  echo "     --github_url https://github.com/YOUR_USER/YOUR_REPO"
}

main() {
  check_python
  check_uv
  create_activate_venv
  upgrade_setuptools
  install_python_reqs
  install_modules
  verify_installation
  
  show_completion_info
}

main "$@"
