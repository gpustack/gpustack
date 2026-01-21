#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
source "${ROOT_DIR}/hack/lib/init.sh"

function build() {
  uv build
}

function prepare_dependencies() {
  DEPS_ONLY=true bash "${ROOT_DIR}/hack/install.sh"
}

function set_version() {
  local version_file="${ROOT_DIR}/gpustack/__init__.py"
  local pyproject_file="${ROOT_DIR}/pyproject.toml"
  local git_commit="${GIT_COMMIT:-HEAD}"
  local git_commit_short="${git_commit:0:7}"

  gpustack::log::info "setting version to $GIT_VERSION"
  gpustack::log::info "setting git commit to $git_commit_short"

  # Replace the __version__ variable in the __init__.py file
  gpustack::util::sed "s/__version__ = .*/__version__ = '${GIT_VERSION}'/" "${version_file}"
  gpustack::util::sed "s/__git_commit__ = .*/__git_commit__ = '${git_commit_short}'/" "${version_file}"

  # Update the version in pyproject.toml
  gpustack::util::sed "s/^version = .*/version = \"${GIT_VERSION}\"/" "${pyproject_file}"
}

function restore_version_file() {
  local version_file="${ROOT_DIR}/gpustack/__init__.py"
  local pyproject_file="${ROOT_DIR}/pyproject.toml"
  git checkout -- "${version_file}" "${pyproject_file}"
}

#
# main
#

gpustack::log::info "+++ BUILD +++"
prepare_dependencies
set_version
build
restore_version_file
gpustack::log::info "--- BUILD ---"
