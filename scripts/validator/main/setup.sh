#!/bin/bash
# setup.sh — create venv, install Python deps & Playwright
# TODO - REVIEW
set -e

# Print error and exit
handle_error() {
  echo -e "\e[31m[ERROR]\e[0m $1" >&2
  exit 1
}

# Print success message
success_msg() {
  echo -e "\e[32m[SUCCESS]\e[0m $1"
}

check_python() {
  echo -e "\e[34m[INFO]\e[0m Checking for Python 3.11..."
  python3.11 --version || handle_error "Python 3.11 is required. Run install_dependencies.sh first."
}

create_activate_venv() {
  VENV_DIR="validator_env"
  echo -e "\e[34m[INFO]\e[0m Creating virtualenv in $VENV_DIR..."
  if [ ! -d "$VENV_DIR" ]; then
    python3.11 -m venv "$VENV_DIR" \
      || handle_error "Failed to create virtualenv"
    success_msg "Virtualenv created."
  else
    echo -e "\e[32m[INFO]\e[0m Virtualenv already exists. Skipping creation."
  fi

  echo -e "\e[34m[INFO]\e[0m Activating virtualenv..."
  source "$VENV_DIR/bin/activate" \
    || handle_error "Failed to activate virtualenv"
}

upgrade_pip() {
  echo -e "\e[34m[INFO]\e[0m Upgrading pip and setuptools..."
  python -m pip install --upgrade pip setuptools \
    || handle_error "Failed to upgrade pip/setuptools"
  success_msg "pip and setuptools upgraded."
}

install_python_reqs() {
  echo -e "\e[34m[INFO]\e[0m Installing Python dependencies from requirements.txt..."
  [ -f "requirements.txt" ] || handle_error "requirements.txt not found"

  pip install -r requirements.txt \
    || handle_error "Failed to install Python dependencies"

  success_msg "Playwright and browsers installed."
}

install_modules() {
  echo -e "\e[34m[INFO]\e[0m Installing current package in editable mode..."
  pip install -e . \
    || handle_error "Failed to install current package"
  success_msg "Main package installed."

install_bittensor() {
  echo -e "\e[34m[INFO]\e[0m Installing Bittensor v9.6.0 and CLI v9.4.2..."
  pip install bittensor==9.6.0 bittensor-cli==9.4.2 \
    || handle_error "Failed to install Bittensor"
  success_msg "Bittensor installed."
}

main() {
  check_python
  create_activate_venv
  upgrade_pip
  install_python_reqs
  install_modules
  install_bittensor
  success_msg "Setup completed successfully."
  echo -e "\e[33m[INFO]\e[0m Virtual environment: $(pwd)/validator_env"
  echo -e "\e[33m[INFO]\e[0m To activate: source validator_env/bin/activate"
}

main "$@"
