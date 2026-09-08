#!/usr/bin/env bash

# Set error handling
set -o errexit
set -o nounset
set -o pipefail

# Get the root directory and third_party directory
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

# Include the common functions
source "${ROOT_DIR}/hack/lib/init.sh"

function download_deps() {
  if ! command -v uv &> /dev/null; then
    pip install uv
  fi
  # uv sync --all-extras to install all dependencies
  uv sync --locked
  if [[ "${DEPS_ONLY:-false}" == "false" ]]; then
    uv pip install pre-commit==3.7.1
    uv run pre-commit install
  fi
}

function download_ui() {
  if [[ "${UI_DOWNLOAD:-true}" != "true" ]]; then
    gpustack::log::info "skipping UI assets download"
    return
  fi

  local default_tag="latest"
  local ui_path="${ROOT_DIR}/gpustack/ui"
  local tmp_ui_path="${ui_path}/tmp"
  local tag="latest"

  if [[ "${GIT_VERSION}" != "v0.0.0" ]]; then
    tag="${GIT_VERSION}"
  fi

  rm -rf "${ui_path}"
  mkdir -p "${tmp_ui_path}/ui"

  gpustack::log::info "downloading '${tag}' UI assets"

  if ! curl --retry 3 --retry-connrefused --retry-delay 3 -sSfL "https://gpustack-ui-1303613262.cos.accelerate.myqcloud.com/releases/${tag}.tar.gz" 2>/dev/null |
    tar -xzf - --directory "${tmp_ui_path}/ui" 2>/dev/null; then

    if [[ "${tag:-}" =~ ^v([0-9]+)\.([0-9]+)(\.[0-9]+)?(-[0-9A-Za-z.-]+)?(\+[0-9A-Za-z.-]+)?$ ]]; then
      gpustack::log::fatal "failed to download '${tag}' ui archive"
    fi

    gpustack::log::warn "failed to download '${tag}' ui archive, fallback to '${default_tag}' ui archive"
    if ! curl --retry 3 --retry-connrefused --retry-delay 3 -sSfL "https://gpustack-ui-1303613262.cos.accelerate.myqcloud.com/releases/${default_tag}.tar.gz" |
      tar -xzf - --directory "${tmp_ui_path}/ui" 2>/dev/null; then
      gpustack::log::fatal "failed to download '${default_tag}' ui archive"
    fi
  fi
  cp -a "${tmp_ui_path}/ui/dist/." "${ui_path}"

  rm -rf "${tmp_ui_path}"
}

# Copy extra static files to ui including catalog icons
function copy_extra_static() {
  local extra_static_path="${ROOT_DIR}/static"
  local ui_static_path="${ROOT_DIR}/gpustack/ui/static"
  if [ -d "${extra_static_path}" ]; then
    cp -a "${extra_static_path}/." "${ui_static_path}"
  fi
}

# Package the Helm chart into the UI static tree, where the server already
# serves it: gpustack/ui/static is mounted at /static, so the packaged chart
# needs no route of its own. The registration manifest points the in-cluster
# bootstrap Job at that URL, which is why this is a build dependency and not a
# release-only step — a wheel without it hands out a manifest that 404s.
#
# Runs after copy_extra_static, never before: download_ui does `rm -rf` on the
# whole ui directory, so anything written there earlier is lost.
#
# The chart is packaged as committed, with only its dependencies vendored. Its
# own version and appVersion are left alone: the values file the server
# generates names every image explicitly, so the chart's defaults never decide
# what gets deployed, and the served URL stays version-free.
function package_chart() {
  if [[ "${CHART_PACKAGE:-true}" != "true" ]]; then
    gpustack::log::info "skipping Helm chart packaging"
    return
  fi

  local chart_path="${ROOT_DIR}/charts/gpustack-chart"
  local target_dir="${ROOT_DIR}/gpustack/ui/static/charts"

  if ! command -v helm >/dev/null 2>&1; then
    gpustack::log::fatal "helm is required to package the chart; install it or set CHART_PACKAGE=false"
  fi

  gpustack::log::info "packaging Helm chart"

  # Vendor the dependencies only when the vendored copies do not already satisfy
  # what Chart.yaml pins: a build box that has them stays offline, and packaging
  # does not turn a reachability problem at somebody else's chart repository
  # into a failed build.
  #
  # Whether they satisfy it is helm's answer, not a test of our own — `helm
  # dependency list` reports each dependency as `ok`, `missing` or `wrong
  # version`. Asking only whether charts/ is non-empty is not enough: a moved
  # pin with a stale tgz still sitting there packages the old dependency and
  # says `Successfully packaged`, and nothing before a running cluster would
  # show it.
  #
  # `dependency update`, not `build`: Chart.lock is not tracked, and `build`
  # refuses a repository that was never `helm repo add`-ed.
  local unsatisfied
  if ! unsatisfied=$(helm dependency list "${chart_path}" 2>/dev/null |
    awk 'NR > 1 && NF && $NF != "ok" { print $1 }'); then
    # No answer is not the same as "they are fine": vendor rather than package
    # whatever happens to be sitting in charts/.
    unsatisfied="(could not read the chart's dependencies)"
  fi
  if [[ -n "${unsatisfied}" ]]; then
    gpustack::log::info "vendoring the chart's dependencies: ${unsatisfied//$'\n'/ }"
    if ! helm dependency update "${chart_path}" >/dev/null; then
      gpustack::log::fatal "failed to vendor the chart's dependencies"
    fi
  fi

  rm -rf "${target_dir}"
  mkdir -p "${target_dir}"

  # A fixed file name, not the chart's version: the manifest that references it
  # is regenerated by the same server that serves it, so a version in the URL
  # would be a second thing to keep in step for no gain.
  local packaged
  packaged=$(helm package "${chart_path}" --destination "${target_dir}" | awk -F': ' '{print $NF}')
  if [[ ! -f "${packaged}" ]]; then
    gpustack::log::fatal "failed to package the chart"
  fi
  mv "${packaged}" "${target_dir}/gpustack-chart.tgz"

  gpustack::log::info "packaged Helm chart to ${target_dir}/gpustack-chart.tgz"
}

# Update community backends
function make_community_backends() {
  local tmp_dir
  tmp_dir=$(mktemp -d -t gpustack-community-backends.XXXXXX)

  # shellcheck disable=SC2064
  trap "rm -rf \"${tmp_dir}\"" EXIT

  local target_dir="${ROOT_DIR}/gpustack/assets/"

  gpustack::log::info "pulling community backends"

  # Clone the repository
  git clone https://github.com/gpustack/community-inference-backends "${tmp_dir}"

  # Build the community backends
  (
    cd "${tmp_dir}"
    if [[ "${UV_SYSTEM_PYTHON:-}" == "1" ]]; then
      # In Docker build, use system Python directly
      uv pip install PyYAML && uv run make
    else
      # For local development, use virtual environment
      uv venv && source .venv/bin/activate && uv pip install PyYAML && uv run make
    fi
  )

  # Create target directory and copy the yaml file
  mkdir -p "${target_dir}"
  cp "${tmp_dir}/dist/community-inference-backends.yaml" "${target_dir}/community-inference-backends.yaml"

  gpustack::log::info "community backends updated successfully"
}

#
# main
#

gpustack::log::info "+++ DEPENDENCIES +++"
download_deps
download_ui
copy_extra_static
package_chart
make_community_backends
gpustack::log::info "--- DEPENDENCIES ---"
