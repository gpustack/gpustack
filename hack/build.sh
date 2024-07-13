#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
source "${ROOT_DIR}/hack/lib/init.sh"

function build() {
  poetry build

  dist_dir="$ROOT_DIR/dist"
  whl_files=$(find "$dist_dir" -name "*.whl")
  if [ -z "$whl_files" ]; then
      gpustack::log::fatal "no wheel files found in $dist_dir"
  fi

  for whl_file in $whl_files; do
      original_name=$(basename "$whl_file")
      if gpustack::util::is_darwin; then
        new_name=${original_name/any/macosx_11_0_universal2}
        mv -f "$whl_file" "$dist_dir/$new_name"
        gpustack::log::info "renamed $original_name to $new_name"
      elif gpustack::util::is_linux; then
        new_name_amd=${original_name/any/manylinux2014_x86_64}
        new_name_arm=${original_name/any/manylinux2014_aarch64}
        cp -f "$whl_file" "$dist_dir/$new_name_amd"
        cp -f "$whl_file" "$dist_dir/$new_name_arm"
        rm -rf "$whl_file"
        gpustack::log::info "renamed $original_name to $new_name_amd"
        gpustack::log::info "renamed $original_name to $new_name_arm"
      fi
  done
}

function prepare_dependencies() {
  bash "${ROOT_DIR}/hack/install.sh"
}

function set_version() {  
  local version_file="${ROOT_DIR}/gpustack/__init__.py"
  local git_commit="${GIT_COMMIT:-HEAD}"
  local git_commit_short="${git_commit:0:7}"

  gpustack::log::info "setting version to $GIT_VERSION"
  gpustack::log::info "setting git commit to $git_commit_short"

  # Replace the __version__ variable in the __init__.py file
  gpustack::util::sed "s/__version__ = .*/__version__ = '${GIT_VERSION}'/" "${version_file}"
  gpustack::util::sed "s/__git_commit__ = .*/__git_commit__ = '${git_commit_short}'/" "${version_file}"

  # Update the poetry version
  poetry version "${GIT_VERSION}"
}

function restore_version_file() {
  local version_file="${ROOT_DIR}/gpustack/__init__.py"
  git checkout -- "${version_file}"
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
