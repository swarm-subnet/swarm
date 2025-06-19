#!/bin/bash
# install_dependencies.sh - Install ONLY system dependencies for miner
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
  
  # Same packages as validator (miners need browser automation too)
  COMMON_PACKAGES=(
    python3.11 python3.11-venv python3.11-dev
    build-essential cmake wget unzip sqlite3
    libnss3 libnss3-dev
    libatk1.0-0 libatk-bridge2.0-0 libcups2
    libx11-xcb1 libxcomposite1 libxcursor1 libxdamage1 libxrandr2
    libgbm1 libpango-1.0-0 libgtk-3-0
    libvpx-dev libevent-dev libopus0
    libgstreamer1.0-0
    libgstreamer-plugins-base1.0-0 libgstreamer-plugins-good1.0-0 libgstreamer-plugins-bad1.0-0
    libwebp-dev libharfbuzz-dev libsecret-1-dev libhyphen0 libflite1 libgles2-mesa-dev
    libx264-dev gnupg curl
  )
  
  # Add version-specific audio package
  UBUNTU_CODENAME=$(lsb_release -cs)
  case "$UBUNTU_CODENAME" in
    jammy)  EXTRA_PACKAGES=(libasound2)   ;;
    noble)  EXTRA_PACKAGES=(libasound2t64) ;;
    *)      EXTRA_PACKAGES=(libasound2)   ;;
  esac
  
  info_msg "Installing system dependencies for $UBUNTU_CODENAME..."
  sudo apt install -y "${COMMON_PACKAGES[@]}" "${EXTRA_PACKAGES[@]}" \
    || handle_error "Failed to install system dependencies"
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
  
  # Check PM2
  pm2 --version || handle_error "PM2 verification failed"
  
  success_msg "System dependencies verification passed"
}

main() {
  info_msg "Installing miner system dependencies..."
  install_system_dependencies
  install_pm2
  verify_installation
  
  success_msg "System dependencies installed successfully!"
  echo -e "\e[33m[NEXT]\e[0m Run: ./scripts/miner/setup.sh"
}

main "$@"