#!/bin/sh
# Script updated at: 2025-02-19T08:16:49Z
set -e
set -o noglob

# Usage:
#   curl ... | ENV_VAR=... sh -s - [args]
#       or
#   ENV_VAR=... ./install.sh [args]
#
# Example:
#   Installing a server with bootstrap password:
#     curl ... | sh -s - --bootstrap-password mypassword
#   Installing a worker to point at a server:
#     curl ... | sh -s - --server-url http://myserver --token mytoken
#
# Environment variables:
#   - INSTALL_PACKAGE_SPEC
#     The package spec to install. Defaults to "gpustack".
#     It supports PYPI package names, git URLs, and local paths.
#
#   - INSTALL_INDEX_URL
#     Base URL of the Python Package Index.
#
#   - INSTALL_SKIP_POST_CHECK
#     If set to 1 will skip the post installation check.
#
#   - INSTALL_SKIP_BUILD_DEPENDENCIES
#     If set to 1 will skip the build dependencies.
#
#   - INSTALL_SKIP_IOGPU_WIRED_LIMIT
#     If set to 1 will skip setting the GPU wired memory limit on macOS.
#
#   - INSTALL_IOGPU_WIRED_LIMIT_MB
#     This sets the maximum amount of wired memory that the GPU can allocate on macOS.

INSTALL_PACKAGE_SPEC="${INSTALL_PACKAGE_SPEC:-}"
INSTALL_INDEX_URL="${INSTALL_INDEX_URL:-}"
INSTALL_SKIP_POST_CHECK="${INSTALL_SKIP_POST_CHECK:-0}"
INSTALL_SKIP_BUILD_DEPENDENCIES="${INSTALL_SKIP_BUILD_DEPENDENCIES:-1}"
INSTALL_SKIP_IOGPU_WIRED_LIMIT="${INSTALL_SKIP_IOGPU_WIRED_LIMIT:-}"
INSTALL_IOGPU_WIRED_LIMIT_MB="${INSTALL_IOGPU_WIRED_LIMIT_MB:-}"

BREW_APP_OPENFST_NAME="openfst"
BREW_APP_OPENFST_VERSION="1.8.3"

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

# Get value of a script parameter. The first arg should be the param_name, then pass all script params.
# Return value of the patameter, or "" if not found.
get_param_value() {
    param_name="$1"
    shift
    next_arg=""

    for arg in "$@"; do
        case $arg in
            --"$param_name"=*|-"$param_name"=*) # Handle equal sign passed arguments
                echo "${arg#*=}" # Return equal passed value
                return 0
                ;;
            --"$param_name"|-"$param_name") # Handle space passed arguments
                next_arg="true"
                ;;
            *)
                if [ "$next_arg" = "true" ]; then
                    echo "$arg"
                    return 0
                fi
                ;;
        esac
    done
    echo ""
}

check_command() {
  command -v "$1" > /dev/null 2>&1
}

ACTION="Install"
print_complete_message()
{
    usage_hint=""
    if [ "$ACTION" = "Install" ]; then
        data_dir=$(get_param_value "data-dir" "$@")
        if [ -z "$data_dir" ]; then
            data_dir="/var/lib/gpustack"
        fi
        config_file=$(get_param_value "config-file" "$@")
        server_url=$(get_param_value "server-url" "$@")
        if [ -z "$server_url" ]; then
            server_url=$(get_param_value "s" "$@") # try short form
        fi

        # Skip printing the usage hint for workers and advanced users using config file. We are lazy to parse the config file here.
        if [ -z "$server_url" ] && [ -z "$config_file" ]; then
            server_url="localhost"
            server_host=$(get_param_value "host" "$@")
            if [ -n "$server_host" ]; then
                server_url="$server_host"
            fi
            server_port=$(get_param_value "port" "$@")
            if [ -n "$server_port" ]; then
                server_url="$server_url:$server_port"
            fi
            ssl_enabled=$(get_param_value "ssl-keyfile" "$@")
            if [ -n "$ssl_enabled" ]; then
                server_url="https://$server_url"
            else
                server_url="http://$server_url"
            fi

            password_hint=""
            bootstrap_password=$(get_param_value "bootstrap-password" "$@")
            if [ -z "$bootstrap_password" ]; then
                password_hint="To get the default password, run 'cat $data_dir/initial_admin_password'."
            fi

            usage_hint="\n\nGPUStack UI is available at $server_url.\nDefault username is 'admin'.\n${password_hint}\n"
        fi

    fi
    info "$ACTION complete. ${usage_hint}"
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
  if [ "$(id -u)" -ne 0 ]; then
    if check_command "sudo"; then
      info "running as non-root, will use sudo for installation."
      SUDO="sudo"
    else
      fatal "This script must be run as root. Please use sudo or run as root."
    fi
  fi
}

# Function to detect the OS and package manager
detect_os() {
  if [ "$(uname)" = "Darwin" ]; then
    OS="macos"
  elif [ -f /etc/os-release ]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    OS=$ID
  else
    fatal "Unsupported OS. Only Linux and MacOS are supported."
  fi
}

# Function to detect the OS and package manager
detect_device() {
  if check_command "nvidia-smi"; then
    if ! check_command "nvcc" && ! ($SUDO ldconfig -p | grep -q libcudart) && ! ls /usr/local/cuda >/dev/null 2>&1; then
      warn "NVIDIA GPU detected but CUDA is not installed. Please install CUDA."
    fi
    DEVICE="cuda"
    # Create a symlink for nvidia-smi to allow root users in WSL to detect GPU information.
    if [ -f "/usr/lib/wsl/lib/nvidia-smi" ] && [ ! -e "/usr/local/bin/nvidia-smi" ]; then
      $SUDO ln -s /usr/lib/wsl/lib/nvidia-smi /usr/local/bin/nvidia-smi
    fi
  fi

  if check_command "mthreads-gmi"; then
    if ! check_command "mcc" && ! ($SUDO ldconfig -p | grep -q libmusart) && ! ls /usr/local/musa >/dev/null 2>&1 && ! ls /opt/musa >/dev/null 2>&1; then
      warn "Moore Threads GPU detected but MUSA is not installed. Please install MUSA."
    fi
    DEVICE="musa"
  fi
}

# Function to check if a port is available
check_port() {
  port=$1
  if $SUDO lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
    return 1
  fi
  return 0
}

# Function to check if the server and worker ports are available
check_ports() {
  if check_command "gpustack"; then
    # skip on upgrade
    return
  fi

  config_file=$(get_param_value "config-file" "$@")
  if [ -n "$config_file" ]; then
    return
  fi

  server_port=$(get_param_value "port" "$@")
  worker_port=$(get_param_value "worker-port" "$@")
  ssl_enabled=$(get_param_value "ssl-keyfile" "$@")

  if [ -z "$server_port" ]; then
    server_port="80"
    if [ -n "$ssl_enabled" ]; then
      server_port="443"
    fi
  fi

  if [ -z "$worker_port" ]; then
    worker_port="10150"
  fi

  if ! check_port "$server_port"; then
    fatal "Server port $server_port is already in use! Please specify a different port by using --port <YOUR_PORT>."
  fi

  if ! check_port "$worker_port"; then
    fatal "Worker port $worker_port is already in use! Please specify a different port by using --worker-port <YOUR_PORT>."
  fi
}

# Function to reset wired_limit_mb
check_and_reset_wired_limit_mb() {
  if [ "$INSTALL_SKIP_IOGPU_WIRED_LIMIT" = "1" ] || [ "$OS" != "macos" ]; then
    return
  fi
  if [ -n "$INSTALL_IOGPU_WIRED_LIMIT_MB" ] ; then
    # Manually set the value of the wired_limit_mb parameter
    $SUDO sysctl -w iogpu.wired_limit_mb="$INSTALL_IOGPU_WIRED_LIMIT_MB"
    warn "This operation carries risks. Please proceed only if you fully understand the iogpu.wired_limit_mb."
  else
    # Automatically set the most appropriate wired_limit_mb value in macos
    TOTAL_MEM_MB=$(($(sysctl -n hw.memsize) / 1024 / 1024))
    # Calculate 85% and TOTAL_MEM_GB-5GB in MB
    EIGHTY_FIVE_PERCENT=$((TOTAL_MEM_MB * 85 / 100))
    MINUS_5GB=$((TOTAL_MEM_MB - 5120))
    # Set WIRED_LIMIT_MB to higher value
    if [ "$EIGHTY_FIVE_PERCENT" -gt "$MINUS_5GB" ]; then
      WIRED_LIMIT_MB="$EIGHTY_FIVE_PERCENT"
    else
      WIRED_LIMIT_MB="$MINUS_5GB"
    fi
    info "Total memory: $TOTAL_MEM_MB MB, Maximum limit (iogpu.wired_limit_mb): $WIRED_LIMIT_MB MB"
    # Apply the values with sysctl
    $SUDO sysctl -w iogpu.wired_limit_mb="$WIRED_LIMIT_MB"
  fi
}

# Function to check and install Python tools
PYTHONPATH=""
check_python_tools() {
  if ! check_command "python3"; then
    info "Python3 could not be found. Attempting to install..."
    if [ "$OS" = "ubuntu" ] || [ "$OS" = "debian" ]; then
      $SUDO apt update && $SUDO DEBIAN_FRONTEND=noninteractive apt install -y python3
    elif [ "$OS" = "centos" ] || [ "$OS" = "rhel" ] || [ "$OS" = "almalinux" ] || [ "$OS" = "rocky" ] ; then
      $SUDO yum install -y python3
    elif [ "$OS" = "macos" ]; then
      brew install python@3.12
    else
      fatal "Unsupported OS for automatic Python installation. Please install Python3 manually."
    fi
  fi

  PYTHON_VERSION=$(python3 -c "import sys; print(sys.version_info.major * 10 + sys.version_info.minor)")
  CURRENT_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
  if [ "$PYTHON_VERSION" -lt 40 ] || [ "$PYTHON_VERSION" -ge 43 ]; then
    fatal "Python version $CURRENT_VERSION is not supported. Please use Python 3.10, 3.11, or 3.12."
  fi

  PYTHON_STDLIB_PATH=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['stdlib'])")
  if [ -f "$PYTHON_STDLIB_PATH/EXTERNALLY-MANAGED" ]; then
    # Current Python environment is externally manged by OS distros. Package installation by pip is restricted.
    # Use package manager to install pipx in later step.
    # Ref: https://packaging.python.org/en/latest/specifications/externally-managed-environments
    PYTHON_EXTERNALLY_MANAGED=1
  else
    # Otherwise, install pipx using pip3 which has better compatibility than package manager provided one.
    if ! python3 -c "import ensurepip" > /dev/null 2>&1; then
      info "Python module ensurepip could not be found. Attempting to install the python3-venv package..."
      if [ "$OS" = "ubuntu" ] || [ "$OS" = "debian" ]; then
        $SUDO apt update && $SUDO sh -c "DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv"
      else
        fatal "Unsupported OS for automatic ensurepip installation. Please install the ensurepip module manually."
      fi
    fi

    if ! check_command "pip3"; then
      info "Pip3 could not be found. Attempting to ensure pip..."
      if [ "$OS" = "ubuntu" ] || [ "$OS" = "debian" ]; then
        if python3 -m ensurepip 2>&1 | grep -q "ensurepip is disabled"; then
            $SUDO apt update && $SUDO DEBIAN_FRONTEND=noninteractive apt-get install -y python3-pip
        else
            python3 -m ensurepip --upgrade
        fi
      else
        python3 -m ensurepip --upgrade
      fi
    fi

    PIP_PYTHON_VERSION_READABLE=$(pip3 -V | grep -Eo 'python [0-9]+\.[0-9]+' | head -n 1 | awk '{print $2}')
    PIP_PYTHON_VERSION=$(echo "$PIP_PYTHON_VERSION_READABLE" | awk -F. '{print $1 * 10 + $2}')
    if [ "$PIP_PYTHON_VERSION" -lt 40 ]; then
      fatal "Python version for pip3 is $PIP_PYTHON_VERSION_READABLE which is not supported. Please use Python 3.10, 3.11, or 3.12."
    fi
  fi

  USER_BASE_BIN=$(python3 -m site --user-base || true)/bin
  export PATH="$USER_BASE_BIN:$PATH"

  if ! check_command "pipx"; then
    info "Pipx could not be found. Attempting to install..."
    if [ -z "$PYTHON_EXTERNALLY_MANAGED" ]; then
      pip3 install pipx
    elif [ "$OS" = "ubuntu" ] || [ "$OS" = "debian" ]; then
      $SUDO apt update && $SUDO sh -c "DEBIAN_FRONTEND=noninteractive apt-get install -y pipx"
    elif [ "$OS" = "centos" ] || [ "$OS" = "rhel" ] || [ "$OS" = "almalinux" ] || [ "$OS" = "rocky" ] ; then
      $SUDO yum install -y pipx
    elif [ "$OS" = "macos" ]; then
      brew install pipx
    else
      fatal "Unsupported OS for automatic pipx installation. Please install pipx manually."
    fi

    # In case pipx installation causes python3 PATH changes (e.g., brew link), re-evaluate once.
    check_python_tools
  fi

  pipx ensurepath --force
  PIPX_BIN_DIR=$(pipx environment --value PIPX_BIN_DIR)
  export PATH="$PIPX_BIN_DIR:$PATH"
}

# Function to install a specific version of a Homebrew application
brew_install_with_version() {
  BREW_APP_NAME="$1"
  BREW_APP_VERSION="$2"
  BREW_APP_NAME_WITH_VERSION="$BREW_APP_NAME@$BREW_APP_VERSION"
  TAP_NAME="$USER/local-$BREW_APP_NAME-$BREW_APP_VERSION"
  
  # Check current installed versions
  info "Checking installed versions of $BREW_APP_NAME."
  INSTALLED_VERSIONS=$(brew list --versions | grep "$BREW_APP_NAME" || true)
  INSTALLED_VERSION_COUNT=$(brew list --versions | grep -c "$BREW_APP_NAME" || true)

  if [ -n "$INSTALLED_VERSIONS" ]; then
    # Check if the target version is already installed
    if echo "$INSTALLED_VERSIONS" | grep -q "$BREW_APP_VERSION"; then
      if [ "$INSTALLED_VERSION_COUNT" -eq 1 ]; then
        info "$BREW_APP_NAME $BREW_APP_VERSION is already installed."
        return 0
      elif [ "$INSTALLED_VERSION_COUNT" -gt 1 ]; then
        SINGLE_LINE_INSTALLED_VERSIONS=$(echo "$INSTALLED_VERSIONS" | tr '\n' ' ')
        info "Installed $BREW_APP_NAME versions: $SINGLE_LINE_INSTALLED_VERSIONS"
        info "Multiple versions of $BREW_APP_NAME are installed, relink the target version."
        echo "$INSTALLED_VERSIONS" | awk '{print $1}' | while read -r installed_version; do
            brew unlink "$installed_version"
        done

        NEED_VERSION=$(echo "$INSTALLED_VERSIONS" | grep "$BREW_APP_VERSION" | cut -d ' ' -f 1)
        brew link --overwrite "$NEED_VERSION"
        return 0
      fi
    fi
  fi

  # Create a new Homebrew tap
  if brew tap-info "$TAP_NAME" 2>/dev/null | grep -q "Installed"; then
      info "Tap $TAP_NAME already exists. Skipping tap creation."
  else
      info "Creating a new tap: $TAP_NAME..."
      if ! brew tap-new "$TAP_NAME"; then
          fatal "Failed to create the tap $TAP_NAME."
      fi
  fi

  # Extract the history version of the app
  info "Extracting $BREW_APP_NAME version $BREW_APP_VERSION."
  brew tap homebrew/core --force
  brew extract --force --version="$BREW_APP_VERSION" "$BREW_APP_NAME" "$TAP_NAME"

  # Install the specific version of the application
  info "Unlinking before install $BREW_APP_NAME."
  echo "$INSTALLED_VERSIONS" | awk '{print $1}' | while read -r installed_version; do
    brew unlink "$installed_version" 2>/dev/null || true
  done

  info "Installing $BREW_APP_NAME version $BREW_APP_VERSION."
  if ! brew install "$TAP_NAME/$BREW_APP_NAME_WITH_VERSION"; then
      fatal "Failed to install $BREW_APP_NAME version $BREW_APP_VERSION."
  fi

  info "Installed and linked $BREW_APP_NAME version $BREW_APP_VERSION."
}

# Function to install dependencies
install_dependencies() {
  DEPENDENCIES="curl sudo lsof"
  for dep in $DEPENDENCIES; do
    if ! check_command "$dep"; then
      fatal "$dep is required but missing. Please install $dep."
    fi
  done

  # check SeLinux dependency
  if check_command "getenforce"; then
      if [ "Disabled" != "$(getenforce)" ]; then
          if ! check_command "semanage"; then
              fatal "semanage is required while SeLinux enabled but missing. Please install the appropriate package for your OS (e.g., policycoreutils-python-utils for Rocky/RHEL/Ubuntu/Debian)."
          fi
      fi
  fi

  if [ "$INSTALL_SKIP_BUILD_DEPENDENCIES" != "1" ] && [ "$OS" = "macos" ]; then
    if ! check_command "brew"; then
      fatal "Homebrew is required but missing. Please install Homebrew."
    else
      # audio dependency library
      brew_install_with_version "$BREW_APP_OPENFST_NAME" "$BREW_APP_OPENFST_VERSION"
    fi
  fi
}


# Function to setup SeLinux permissions
setup_selinux_permissions() {
    BIN_PATH=$1
    BIN_REAL_PATH=""

    if ! $SUDO semanage fcontext -l | grep "${BIN_PATH}" > /dev/null 2>&1; then
        $SUDO semanage fcontext -a -t bin_t "${BIN_PATH}"
    fi
    $SUDO restorecon -v "${BIN_PATH}" > /dev/null 2>&1

    if [ -L "$BIN_PATH" ]; then
        BIN_REAL_PATH=$(readlink -f "$BIN_PATH")
        if ! $SUDO semanage fcontext -l | grep "${BIN_REAL_PATH}" > /dev/null 2>&1; then
            $SUDO semanage fcontext -a -t bin_t "${BIN_REAL_PATH}"
        fi
        $SUDO restorecon -v "${BIN_REAL_PATH}" > /dev/null 2>&1
    fi
}

# Function to setup systemd for Linux
setup_systemd() {
  # setup permissions
  if check_command "getenforce"; then
      if [ "Disabled" != "$(getenforce)" ]; then
          info "Setting up SeLinux permissions for Python3."
          PYTHON3_BIN_PATH=$(which python3)
          setup_selinux_permissions "$PYTHON3_BIN_PATH"

          info "Setting up SeLinux permissions for gpustack."
          GPUSTACK_BIN_PATH=$(which gpustack)
          setup_selinux_permissions "$GPUSTACK_BIN_PATH"
      fi
  fi

  # Process the arguments and handle spaces and single quotes
  _args=""
  for x in "$@"; do
      case "$x" in
          *\ *)
              x=$(echo "$x" | sed "s/'/'\\\\''/g")
              x="'$x'"
              ;;
      esac
      _args="$_args $x"
  done

  info "Setting up GPUStack as a service using systemd."
  $SUDO tee /etc/systemd/system/gpustack.service > /dev/null <<EOF
[Unit]
Description=GPUStack Service
Wants=network-online.target
After=network-online.target

[Service]
EnvironmentFile=-/etc/default/%N
ExecStart=$(which gpustack) start $_args
LimitNOFILE=65535
Restart=always
StandardOutput=append:/var/log/gpustack.log
StandardError=append:/var/log/gpustack.log

[Install]
WantedBy=multi-user.target
EOF

  $SUDO systemctl daemon-reload
  $SUDO systemctl enable gpustack.service
  $SUDO systemctl restart gpustack.service
}

# Function to setup launchd for macOS
setup_launchd() {
  info "Setting up GPUStack as a service using launchd."

  # Load environment variables from /etc/default/gpustack if exists
  ENV_FILE="/etc/default/gpustack"
  if [ -f "$ENV_FILE" ]; then
    info "Loading environment variables from $ENV_FILE"
    ENV_VARS=""
    while IFS='=' read -r key value; do
      case "$key" in
        \#*|"") continue ;;  # Skip comments and empty lines
      esac
      # Strip surrounding quotes if present
      value=$(echo "$value" | sed -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//")
      ENV_VARS="$ENV_VARS    <key>$key</key><string>$value</string>\n"
    done < "$ENV_FILE"
  fi

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
    echo "    <string>$arg</string>" | $SUDO tee -a /Library/LaunchDaemons/ai.gpustack.plist > /dev/null
  done

  $SUDO tee -a /Library/LaunchDaemons/ai.gpustack.plist > /dev/null <<EOF
  </array>
EOF

  # Add EnvironmentVariables section if ENV_VARS is not empty
  if [ -n "$ENV_VARS" ]; then
    $SUDO tee -a /Library/LaunchDaemons/ai.gpustack.plist > /dev/null <<EOF
  <key>EnvironmentVariables</key>
  <dict>
EOF
    printf "%b" "$ENV_VARS" | $SUDO tee -a /Library/LaunchDaemons/ai.gpustack.plist > /dev/null
    $SUDO tee -a /Library/LaunchDaemons/ai.gpustack.plist > /dev/null <<EOF
  </dict>
EOF
  fi

  $SUDO tee -a /Library/LaunchDaemons/ai.gpustack.plist > /dev/null <<EOF
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>EnableTransactions</key>
  <true/>
  <key>StandardOutPath</key>
  <string>/var/log/gpustack.log</string>
  <key>StandardErrorPath</key>
  <string>/var/log/gpustack.log</string>
</dict>
</plist>
EOF

  $SUDO launchctl bootstrap system /Library/LaunchDaemons/ai.gpustack.plist
}

# Function to disable the service in launchd
disable_service_in_launchd() {
  if [ -f /Library/LaunchDaemons/ai.gpustack.plist ]; then
    $SUDO launchctl bootout system /Library/LaunchDaemons/ai.gpustack.plist
    $SUDO rm /Library/LaunchDaemons/ai.gpustack.plist
    # wait a grace period for the service to stop
    sleep 10
  fi
}

# Function to disable the service in systemd
disable_service_in_systemd() {
  if [ -f /etc/systemd/system/gpustack.service ]; then
    $SUDO systemctl disable gpustack.service
    $SUDO rm /etc/systemd/system/gpustack.service
    $SUDO systemctl daemon-reload
    # wait a grace period for the service to stop
    sleep 10
  fi
}

# Function to disable the service
disable_service() {
  if [ "$OS" = "macos" ]; then
    disable_service_in_launchd
  else
    disable_service_in_systemd
  fi
}

# Function to setup and start the service
setup_and_start() {
  if [ "$OS" = "macos" ]; then
    setup_launchd "$@"
  else
    setup_systemd "$@"
  fi
}

# Helper function to check service status
is_service_running() {
  if [ "$OS" = "macos" ]; then
    # Get service info
    SERVICE_INFO=$($SUDO launchctl print system/ai.gpustack 2>/dev/null)
    # shellcheck disable=SC2181
    if [ $? -ne 0 ]; then
      return 1
    fi

    # Extract service details
    LAST_EXIT_STATUS=$(echo "$SERVICE_INFO" | grep "last exit code =" | awk -F "= " '{print $2}' | xargs)
    IS_RUNNING=$(echo "$SERVICE_INFO" | grep "state = running")

    # Evaluate service health
    if [ -n "$IS_RUNNING" ]; then
      if [ "$LAST_EXIT_STATUS" = "0" ] || [ "$LAST_EXIT_STATUS" = "(never exited)" ]; then
        return 0
      else
        return 1
      fi
    else
      return 1
    fi

  else
    $SUDO systemctl is-active --quiet gpustack.service
  fi
}

# Function to check service status
check_service() {
  if [ "$INSTALL_SKIP_POST_CHECK" -eq 1 ]; then
    return 0
  fi
  info "Waiting for the service to initialize..."
  sleep 10
  info "Running post-install checks..."

  retries=3
  for i in $(seq 1 $retries); do
    if is_service_running; then
      info "GPUStack service is running."
      return 0
    fi
    info "Service not ready, retrying in 2 seconds ($i/$retries)..."
    sleep 2
  done

  fatal "GPUStack service failed to start. Please check the logs at /var/log/gpustack.log for details."
}

# Function to create uninstall script
create_uninstall_script() {
  PYTHON_BIN="python3"
  PIPX_PYTHON_PATH=$(pipx environment --value PIPX_DEFAULT_PYTHON)
  if [ -n "$PIPX_PYTHON_PATH" ]; then
    PYTHON_BIN="$PIPX_PYTHON_PATH"
  fi
  PYTHONPATH=$("$PYTHON_BIN" -c 'import site, sys; print(":".join(sys.path + [site.getusersitepackages()]))')

  $SUDO mkdir -p /var/lib/gpustack
  $SUDO tee /var/lib/gpustack/uninstall.sh > /dev/null <<EOF
#!/bin/bash
set -e
export PYTHONPATH="$PYTHONPATH"
export PIPX_HOME=$(pipx environment --value PIPX_HOME)
export PIPX_BIN_DIR=$(pipx environment --value PIPX_BIN_DIR)
$(which pipx) uninstall gpustack > /dev/null
if [ "$OS" = "macos" ]; then
  launchctl bootout system /Library/LaunchDaemons/ai.gpustack.plist
  rm -f /Library/LaunchDaemons/ai.gpustack.plist
else
  systemctl stop gpustack.service
  systemctl disable gpustack.service
  rm -f /etc/systemd/system/gpustack.service
  systemctl daemon-reload
fi
rm -rf /var/lib/gpustack /var/log/gpustack.log
echo "GPUStack has been uninstalled."
EOF
  $SUDO chmod +x /var/lib/gpustack/uninstall.sh
}

# Function to install GPUStack using pipx
install_gpustack() {
  if check_command "gpustack"; then
    ACTION="Upgrade"
    info "GPUStack is already installed. Upgrading..."
  else
    info "Installing GPUStack..."
  fi

  install_args=""
  if [ -n "$INSTALL_INDEX_URL" ]; then
    install_args="--index-url $INSTALL_INDEX_URL $install_args"
  fi

  default_package_spec="gpustack[audio]"
  if [ "$OS" != "macos" ] && [ "$(uname -m)" = "x86_64" ] && [ "$DEVICE" = "cuda" ]; then
    # Install optional vLLM dependencies on amd64 Linux
    default_package_spec="gpustack[all]"
  fi

  if [ -z "$INSTALL_PACKAGE_SPEC" ]; then
    INSTALL_PACKAGE_SPEC="$default_package_spec"
  fi

  # shellcheck disable=SC2090,SC2086
  pipx install --force --verbose $install_args --python "$(which python3)" "$INSTALL_PACKAGE_SPEC"
  # Workaround for issue #581
  pipx inject gpustack pydantic==2.9.2 --force > /dev/null 2>&1

  # audio dependencies for macOS
  if [ "$INSTALL_SKIP_BUILD_DEPENDENCIES" != "1" ] && [ "$OS" = "macos" ]; then
    # Check current installed versions
    NEED_VERSION=$(brew list --versions | grep "$BREW_APP_OPENFST_NAME" | grep "$BREW_APP_OPENFST_VERSION" | cut -d ' ' -f 1 || true)
    CPLUS_INCLUDE_PATH="$(brew --prefix "$NEED_VERSION")/include"
    export CPLUS_INCLUDE_PATH
    LIBRARY_PATH="$(brew --prefix "$NEED_VERSION")/lib"
    export LIBRARY_PATH
    pipx inject gpustack pynini==2.1.6
    pipx inject gpustack wetextprocessing==1.0.4.1
  fi
}

# Main install process
{
  check_root
  detect_os
  detect_device
  verify_system
  install_dependencies
  check_python_tools
  check_ports "$@"
  check_and_reset_wired_limit_mb "$@"
  install_gpustack
  create_uninstall_script
  disable_service
  setup_and_start "$@"
  check_service
  print_complete_message "$@"
}
