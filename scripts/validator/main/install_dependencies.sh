#!/bin/bash
# install_dependencies.sh - Install ONLY system dependencies for validator
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
  sudo apt install -y sudo software-properties-common lsb-release curl \
    || handle_error "Failed to install core tools"
  
  sudo apt update -y || handle_error "Failed to refresh apt lists"
  
  # Common packages for all Ubuntu versions
  COMMON_PACKAGES=(
    python3.11 python3.11-venv python3.11-dev
    build-essential cmake wget unzip sqlite3
    libnss3 libnss3-dev gnupg curl nodejs npm
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
    info_msg "PM2 is already installed. Checking if it works..."
    if pm2 --version &>/dev/null; then
      info_msg "PM2 is working correctly. Skipping installation."
      return
    else
      info_msg "PM2 exists but not working. Reinstalling..."
      sudo npm uninstall -g pm2 2>/dev/null || true
    fi
  fi
  
  info_msg "Installing PM2..."
  sudo npm install -g pm2@latest || handle_error "Failed to install PM2"
  
  info_msg "Updating PM2..."
  pm2 update || handle_error "Failed to update PM2"
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
  info_msg "Installing validator system dependencies..."
  install_system_dependencies
  install_pm2
  verify_installation
  
  success_msg "System dependencies installed successfully!"
  echo -e "\e[33m[NEXT]\e[0m Run: ./scripts/validator/main/setup.sh"
}

main "$@"