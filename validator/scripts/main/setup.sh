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
  success_msg "Dependencies installed."
}

install_modules() {
  echo -e "\e[34m[INFO]\e[0m Installing current package in editable mode..."
  pip install -e . --no-deps \
    || handle_error "Failed to install current package"
  success_msg "Main package installed."
}

main() {
  check_python
  create_activate_venv
  upgrade_pip
  install_python_reqs
  install_modules
  success_msg "Setup completed successfully."
  echo -e "\e[33m[INFO]\e[0m Virtual environment: $(pwd)/validator_env"
  echo -e "\e[33m[INFO]\e[0m To activate: source validator_env/bin/activate"
}

main "$@"
