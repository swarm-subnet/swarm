#!/bin/bash
# setup.sh - Setup miner Python environment and dependencies
# TODO - REVIEW
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

upgrade_pip() {
  info_msg "Upgrading pip and setuptools..."
  python -m pip install --upgrade pip setuptools \
    || handle_error "Failed to upgrade pip/setuptools"
  success_msg "pip and setuptools upgraded."
}

install_python_reqs() {
  info_msg "Installing Python dependencies from requirements.txt..."
  [ -f "requirements.txt" ] || handle_error "requirements.txt not found"
  
  pip install -r requirements.txt \
    || handle_error "Failed to install Python dependencies"

  pip install git+https://github.com/utiasDSL/gym-pybullet-drones.git
  
  success_msg "Packages installed"
}

install_modules() {
  info_msg "Installing current package in editable mode..."
  pip install -e . \
    || handle_error "Failed to install current package"
  success_msg "Main package installed."

}

install_bittensor() {
  info_msg "Installing Bittensor v9.6.0 and CLI v9.4.2..."
  pip install bittensor==9.6.0 bittensor-cli==9.4.2 \
    || handle_error "Failed to install Bittensor"
  success_msg "Bittensor installed."
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
  echo "   pm2 start neurons/miner.py --name miner --interpreter python -- \\"
  echo "     --netuid 124 --subtensor.network finney \\"
  echo "     --wallet.name your_coldkey --wallet.hotkey your_hotkey"
}

main() {
  check_python
  create_activate_venv
  upgrade_pip
  install_python_reqs
  install_modules
  install_bittensor
  verify_installation
  
  show_completion_info
}

main "$@"