#!/bin/bash
# setup.sh — create venv, install Python deps
set -e

handle_error() {
  echo -e "\e[31m[ERROR]\e[0m $1" >&2
  exit 1
}

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

check_uv() {
  echo -e "\e[34m[INFO]\e[0m Checking for uv..."
  uv --version || handle_error "uv is required. Run install_dependencies.sh first."
}

upgrade_setuptools() {
  echo -e "\e[34m[INFO]\e[0m Upgrading setuptools..."
  uv pip install --upgrade setuptools \
    || handle_error "Failed to upgrade setuptools"
  success_msg "setuptools upgraded."
}

install_python_reqs() {
  echo -e "\e[34m[INFO]\e[0m Installing Python dependencies from requirements.txt..."
  [ -f "requirements.txt" ] || handle_error "requirements.txt not found"

  uv pip install -r requirements.txt \
    || handle_error "Failed to install Python dependencies"
  success_msg "Dependencies installed."
}

install_modules() {
  echo -e "\e[34m[INFO]\e[0m Installing current package in editable mode..."
  uv pip install -e . --no-deps \
    || handle_error "Failed to install current package"
  success_msg "Main package installed."
}

main() {
  check_python
  check_uv
  create_activate_venv
  upgrade_setuptools
  install_python_reqs
  install_modules
  success_msg "Setup completed successfully."
  echo -e "\e[33m[INFO]\e[0m Virtual environment: $(pwd)/validator_env"
  echo -e "\e[33m[INFO]\e[0m To activate: source validator_env/bin/activate"
}

main "$@"
