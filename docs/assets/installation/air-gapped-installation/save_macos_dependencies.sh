#!/bin/sh
set -e
set -o noglob

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

# --- save the dependencies ---

# Function to save OpenFST
save_dependency_openfst() {
  info "Saving dependency..."

  if ! command -v brew > /dev/null 2>&1; then
    fatal "Homebrew is required but missing. Please install Homebrew."
  else
    # audio dependency library
    brew_install_with_version "$BREW_APP_OPENFST_NAME" "$BREW_APP_OPENFST_VERSION"
    save_brew_app "$BREW_APP_OPENFST_NAME" "$BREW_APP_OPENFST_VERSION"
    save_brew_tap "$BREW_APP_OPENFST_NAME" "$BREW_APP_OPENFST_VERSION"
  fi

  info "Dependency saved."
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

# Function to save the Homebrew application
save_brew_app() {
  BREW_APP_NAME="$1"
  BREW_APP_VERSION="$2"
  BREW_APP_NAME_WITH_VERSION="$BREW_APP_NAME@$BREW_APP_VERSION"
  TAP_NAME="$USER/local-$BREW_APP_NAME-$BREW_APP_VERSION"

  info "Saving brew application $BREW_APP_NAME_WITH_VERSION."

  # Save the app
  brew fetch --deps "$TAP_NAME/$BREW_APP_NAME_WITH_VERSION"

  # Get the cache path
  CACHE_PATH=$(brew --cache openfst@1.8.3 "$BREW_APP_NAME_WITH_VERSION")
  cp "$CACHE_PATH" .
}

# Function to save the Homebrew tap
save_brew_tap() {
  BREW_APP_NAME="$1"
  BREW_APP_VERSION="$2"
  BREW_APP_NAME_WITH_VERSION="$BREW_APP_NAME@$BREW_APP_VERSION"
  TAP_NAME="$USER/local-$BREW_APP_NAME-$BREW_APP_VERSION"

  info "Saving brew tap $TAP_NAME."

  # TAP_PATH example: /opt/homebrew/Library/Taps/$USER/homebrew-local-openfst-1.8.3
  TAP_PATH=$(brew --repo "$TAP_NAME")
  TAP_PATH_DIRNAME=$(dirname "$TAP_PATH")
  TAP_PATH_BASENAME=$(basename "$TAP_PATH")
  TAP_SHORT_NAME=$(echo "$TAP_NAME" | cut -d '/' -f 2)
  tar -czf "$TAP_SHORT_NAME.tar.gz" -C "$TAP_PATH_DIRNAME" "$TAP_PATH_BASENAME"
}

# Main process
{
  save_dependency_openfst
}
