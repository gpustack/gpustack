#!/bin/bash
set -e
set -o noglob

# Usage:
#   curl ... | ENV_VAR=... sh - [args]
#       or
#   ENV_VAR=... ./install.sh [args]
#
# Example:
#   Installing a server with bootstrap passowrd:
#     curl ... | sh - --bootstrap-password mypassword
#   Installing a worker to point at a server:
#     curl ... | sh - --server-url http://myserver --token mytoken
#
# Environment variables:
#   - INSTALL_PACKAGE_SPEC
#     The package spec to install. Defaults to "gpustack".
#     It supports PYPI package names, git URLs, and local paths.
#
#   - INSTALL_PRE_RELEASE
#     If set to true will install pre-release packages.

INSTALL_PACKAGE_SPEC="${INSTALL_PACKAGE_SPEC:-gpustack}"
INSTALL_PRE_RELEASE="${INSTALL_PRE_RELEASE:-0}"

# --- helper functions for logs ---
info()
{
    echo '[INFO] ' "$@"
}
warn()
{
    echo '[WARN] ' "$@" >&2
}
fatal()
{
    echo '[ERROR] ' "$@" >&2
    exit 1
}

install_complete()
{
    info 'Install complete. Run "gpustack" from the command line.'
}

# --- fatal if no systemd or launchd ---
verify_system() {
    if [ -x /bin/systemctl ] || type systemctl > /dev/null 2>&1; then
        return
    fi
    if [ -x /bin/launchctl ] || type launchctl > /dev/null 2>&1; then
        return
    fi
    fatal 'Can not find systemd or launchd to use as a process supervisor for GPUStack.'
}

# Function to check if the script is run as root or has sudo permissions
SUDO=
check_root() {
  if [ "$EUID" -ne 0 ]; then
    if command -v sudo >/dev/null 2>&1; then
      info "running as non-root, will use sudo for installation."
      SUDO="sudo"
    else
      fatal "This script must be run as root. Please use sudo or run as root."
    fi
  fi
}

# Function to detect the OS and package manager
detect_os() {
  if [ "$(uname)" == "Darwin" ]; then
    OS="macos"
  elif [ -f /etc/os-release ]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    OS=$ID
  else
    fatal "Unsupported OS. Only Linux and macOS are supported."
  fi
}

# Function to check and install Python
check_python() {
  if ! command -v python3 &> /dev/null; then
    info "Python3 could not be found. Attempting to install..."
    if [ "$OS" == "ubuntu" ] || [ "$OS" == "debian" ]; then
      $SUDO apt update && apt install -y python3
    elif [ "$OS" == "centos" ] || [ "$OS" == "rhel" ]; then
      $SUDO yum install -y python3
    elif [ "$OS" == "macos" ]; then
      $SUDO brew install python
    else
      fatal "Unsupported OS for automatic Python installation. Please install Python3 manually."
    fi
  fi
  
  PYTHON_VERSION=$(python3 -c "import sys; print(sys.version_info.major * 10 + sys.version_info.minor)")
  if [ "$PYTHON_VERSION" -lt 40 ]; then
    fatal "Python version is less than 3.10. Please upgrade Python to at least version 3.10."
  fi

  if ! command -v pipx &> /dev/null; then
    info "Pipx could not be found. Attempting to install..."
    pip3 install pipx
    pipx ensurepath
  fi
}

# Function to install dependencies
install_dependencies() {
  DEPENDENCIES=("curl" "sudo")
  for dep in "${DEPENDENCIES[@]}"; do
    if ! command -v "$dep" &> /dev/null; then
      fatal "$dep is required but missing. Please install $dep."
    fi
  done
}

# Function to check CUDA for NVIDIA GPUs
check_cuda() {
  if command -v nvidia-smi &> /dev/null; then
    if ! dpkg-query -W cuda; then
      fatal "NVIDIA GPU detected but CUDA is not installed. Please install CUDA."
    fi
  fi
}

# Function to setup systemd for Linux
setup_systemd() {
  info "Setting up GPUStack as a service using systemd."
  $SUDO tee /etc/systemd/system/gpustack.service > /dev/null <<EOF
[Unit]
Description=GPUStack Service

[Service]
ExecStart=$(which gpustack) start $@
Restart=always

[Install]
WantedBy=multi-user.target
EOF

  $SUDO systemctl daemon-reload
  $SUDO systemctl enable gpustack.service
  $SUDO systemctl start gpustack.service
}

# Function to setup launchd for macOS
setup_launchd() {
  info "Setting up GPUStack as a service using launchd."
  $SUDO tee /Library/LaunchDaemons/ai.gpustack.plist > /dev/null <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple Computer//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>ai.gpustack</string>
  <key>ProgramArguments</key>
  <array>
    <string>$(which gpustack)</string>
    <string>start</string>
EOF

  for arg in "$@"; do
    $SUDO echo "    <string>$arg</string>" >> /Library/LaunchDaemons/ai.gpustack.plist
  done

  $SUDO tee -a /Library/LaunchDaemons/ai.gpustack.plist > /dev/null <<EOF
  </array>
  <key>RunAtLoad</key>
  <true/>
</dict>
</plist>
EOF

  $SUDO launchctl load /Library/LaunchDaemons/ai.gpustack.plist
}

# Function to setup and start the service
setup_and_start() {
  if [ "$OS" == "macos" ]; then
    setup_launchd "$@"
  else
    setup_systemd "$@"
  fi
}

# Function to create uninstall script
create_uninstall_script() {
  $SUDO mkdir -p /var/lib/gpustack
  $SUDO tee /var/lib/gpustack/uninstall.sh > /dev/null <<EOF
#!/bin/bash
pipx uninstall gpustack > /dev/null
rm -rf /var/lib/gpustack
if [ "$OS" == "macos" ]; then
  launchctl unload /Library/LaunchDaemons/ai.gpustack.plist
  rm /Library/LaunchDaemons/ai.gpustack.plist
else
  systemctl stop gpustack.service
  systemctl disable gpustack.service
  rm /etc/systemd/system/gpustack.service
  systemctl daemon-reload
fi
echo "GPUStack has been uninstalled."
EOF
  $SUDO chmod +x /var/lib/gpustack/uninstall.sh
}

# Function to install GPUStack using pipx
install_gpustack() {
  if command -v gpustack &> /dev/null; then
    info "GPUStack is already installed."
    return
  fi
  local install_args=()
  if [ "$INSTALL_PRE_RELEASE" -eq 1 ]; then
    install_args+=(--pip-args='--pre')
  fi
  $SUDO pipx install "${install_args[@]}" "${INSTALL_PACKAGE_SPEC}"
  $SUDO pipx ensurepath
  PIPX_BIN_DIR=$(pipx environment --value PIPX_BIN_DIR)
  export PATH="$PIPX_BIN_DIR:$PATH"
}

# Main install process
{
  check_root
  detect_os
  verify_system
  install_dependencies
  check_python
  check_cuda
  install_gpustack
  create_uninstall_script
  setup_and_start "$@"
  install_complete
}
