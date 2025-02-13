#!/bin/sh
set -e
set -o noglob

BREW_APP_OPENFST_NAME="openfst"
BREW_APP_OPENFST_VERSION="1.8.3"

LOAD_DIR=""

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

# --- parse parameters ---
while [ "$#" -gt 0 ]; do
    case $1 in
        --load-dir)
            LOAD_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
    esac
done

if [ -z "$LOAD_DIR" ]; then
    fatal "--load-dir parameter is required."
fi

# --- load dependencies ---

# Function to prepare OpenFST
load_dependency_openfst() {
  info "Loading dependency..."

  if ! command -v brew > /dev/null 2>&1; then
    fatal "Homebrew is required but missing. Please install Homebrew."
  else
    # audio dependency library
    load_brew_tap "$BREW_APP_OPENFST_NAME" "$BREW_APP_OPENFST_VERSION"
    load_brew_app "$BREW_APP_OPENFST_NAME" "$BREW_APP_OPENFST_VERSION"
  fi

  info "dependency loaded."
}

# Function to load the Homebrew application
load_brew_app() {
  BREW_APP_NAME="$1"
  BREW_APP_VERSION="$2"
  TAP_NAME="$USER/local-$BREW_APP_NAME-$BREW_APP_VERSION"
  BREW_APP_NAME_WITH_VERSION="$BREW_APP_NAME@$BREW_APP_VERSION"

  info "Loading brew application $BREW_APP_NAME_WITH_VERSION."

  # Check current installed versions
  info "Checking installed versions of $BREW_APP_NAME."
  INSTALLED_VERSIONS=$(brew list --versions | grep "$BREW_APP_NAME" || true)
  INSTALLED_VERSION_COUNT=$(brew list --versions | grep -c "$BREW_APP_NAME" || true)
  INSTALLED_NEEDED_VERSION_COUNT=$(brew list --versions | grep -c "$BREW_APP_NAME_WITH_VERSION" || true)

  if [ -n "$INSTALLED_VERSIONS" ]; then
    # Check if the target version is already installed
    if [ "$INSTALLED_VERSION_COUNT" -eq 1 ] && [ "$INSTALLED_NEEDED_VERSION_COUNT" -eq 1 ]; then
      info "$BREW_APP_NAME_WITH_VERSION is already installed."
      return 0
    elif [ "$INSTALLED_VERSION_COUNT" -gt 1 ]; then
      SINGLE_LINE_INSTALLED_VERSIONS=$(echo "$INSTALLED_VERSIONS" | tr '\n' ' ')
      info "Installed $BREW_APP_NAME versions: $SINGLE_LINE_INSTALLED_VERSIONS"
      info "Multiple versions of $BREW_APP_NAME are installed, relink the target version."
      echo "$INSTALLED_VERSIONS" | awk '{print $1}' | while read -r installed_version; do
          brew unlink "$installed_version"
      done

      NEED_VERSION=$(echo "$INSTALLED_VERSIONS" | grep "$BREW_APP_NAME_WITH_VERSION" | cut -d ' ' -f 1)
      if [ -n "$NEED_VERSION" ]; then
        info "Relinking $NEED_VERSION..."
        brew link --overwrite "$NEED_VERSION"
        return 0
      fi
    fi
  fi

  info "Copying $BREW_APP_NAME_WITH_VERSION to the brew cache directory..."
  BACKUP_BREW_APP_FILE_SUFFIX="--$BREW_APP_NAME-$BREW_APP_VERSION.tar.gz"
  # shellcheck disable=SC2010
  BACKUP_BREW_APP_PATH="$(ls "$LOAD_DIR" | grep -F -- "$BACKUP_BREW_APP_FILE_SUFFIX")"
  BREW_CACHE_DIR="$(brew --cache)/downloads"
  mkdir -p "$BREW_CACHE_DIR"
  cp "$LOAD_DIR/$BACKUP_BREW_APP_PATH" "$BREW_CACHE_DIR"

  info "Installing $BREW_APP_NAME_WITH_VERSION..."
  HOMEBREW_NO_AUTO_UPDATE=1 brew install "$TAP_NAME/$BREW_APP_NAME_WITH_VERSION"
}

# Function to load the Homebrew tap
load_brew_tap() {
  BREW_APP_NAME="$1"
  BREW_APP_VERSION="$2"
  TAP_NAME="$USER/local-$BREW_APP_NAME-$BREW_APP_VERSION"
  TAP_SHORT_NAME=$(echo "$TAP_NAME" | cut -d '/' -f 2)


  info "Loading brew tap $TAP_NAME."

  if brew tap-info "$TAP_NAME" 2>/dev/null | grep -q "Installed"; then
      info "Tap $TAP_NAME already exists. Skipping tap creation."
  else
      info "Creating tap: $TAP_NAME..."

      TAP_PATH_DIRNAME="$(brew --prefix)/Library/Taps/$USER"
      # shellcheck disable=SC2010
      TAP_BACKUP_PATH="$(ls "$LOAD_DIR" | grep "$TAP_SHORT_NAME")"
      mkdir -p "$TAP_PATH_DIRNAME"
      tar -xzf "$TAP_BACKUP_PATH" -C "$TAP_PATH_DIRNAME"
  fi
}

# Main process
{
  load_dependency_openfst
}
