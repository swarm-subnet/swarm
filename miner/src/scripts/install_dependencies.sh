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

# install_dependencies.sh - Install ONLY system dependencies for miner
set -e

UV_VERSION="0.12.8"

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

install_system_dependencies() {
  info_msg "Updating apt package lists..."
  sudo apt update -y || handle_error "Failed to update apt lists"
  sudo apt upgrade -y || handle_error "Failed to upgrade packages"

  info_msg "Installing core tools..."
  sudo apt install -y sudo software-properties-common lsb-release \
    || handle_error "Failed to install core tools"

  info_msg "Adding Python 3.11 PPA..."
  sudo add-apt-repository ppa:deadsnakes/ppa -y \
    || handle_error "Failed to add Python PPA"
  sudo apt update -y || handle_error "Failed to refresh apt lists"

  # Same packages as validator
  COMMON_PACKAGES=(
    python3.11 python3.11-venv python3.11-dev
    build-essential cmake wget unzip sqlite3
    libnss3 libnss3-dev gnupg curl nodejs
  )

  # Ask apt which ALSA runtime exists here; libasound2 is virtual from noble onwards.
  UBUNTU_CODENAME=$(lsb_release -cs)
  if apt-cache show libasound2t64 >/dev/null 2>&1; then
    EXTRA_PACKAGES=(libasound2t64)
  else
    EXTRA_PACKAGES=(libasound2)
  fi

  info_msg "Installing system dependencies for $UBUNTU_CODENAME..."
  sudo apt install -y "${COMMON_PACKAGES[@]}" "${EXTRA_PACKAGES[@]}" \
    || handle_error "Failed to install system dependencies"
}

install_uv() {
  if command -v uv &>/dev/null && uv --version &>/dev/null; then
    info_msg "uv is already installed. Skipping."
  else
    info_msg "Installing uv $UV_VERSION..."
    curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" \
      | sudo env UV_INSTALL_DIR="/usr/local/bin" UV_NO_MODIFY_PATH=1 sh \
      || handle_error "Failed to install uv"
  fi
}

install_pm2() {
  if command -v pm2 &>/dev/null; then
    info_msg "PM2 is already installed. Skipping."
  else
    info_msg "Installing PM2..."
    sudo apt install -y npm || handle_error "Failed to install npm"
    sudo npm install -g pm2 || handle_error "Failed to install PM2"
    pm2 update || handle_error "Failed to update PM2"
  fi
}

verify_installation() {
  info_msg "Verifying system dependencies..."

  # Check Python
  python3.11 --version || handle_error "Python 3.11 verification failed"

  # Check uv
  uv --version || handle_error "uv verification failed"

  # Check PM2
  pm2 --version || handle_error "PM2 verification failed"

  success_msg "System dependencies verification passed"
}

main() {
  info_msg "Installing miner system dependencies..."
  install_system_dependencies
  install_uv
  install_pm2
  verify_installation

  success_msg "System dependencies installed successfully!"
  echo -e "\e[33m[NEXT]\e[0m Run: ./miner/src/scripts/setup.sh"
}

main "$@"
