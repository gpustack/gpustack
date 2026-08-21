#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

# Hosts the public site may fetch from and the bundled one may not. Checking the
# output rather than the config is what makes this hard to regress: the ways a
# page starts reaching out again — a theme override, a new markdown extension, a
# plugin that inlines a CDN script — are not visible in mkdocs.offline.yml.
EXTERNAL_HOSTS="unpkg\.com|fonts\.googleapis\.com|fonts\.gstatic\.com|buttons\.github\.io|img\.shields\.io|raw\.githubusercontent\.com"

function build_public() {
  uv run mkdocs build
}

function build_offline() {
  local site_dir="${ROOT_DIR}/gpustack/help"

  uv run mkdocs build \
    --config-file "${ROOT_DIR}/mkdocs.offline.yml" \
    --site-dir "${site_dir}"

  # The search index is the one file where compressing pays — ~1.2 MB, and it is
  # requested by its own name, so PrecompressedStaticFiles finds the .gz sibling
  # with no change to the server. Pages are requested as directories, whose .gz
  # cannot exist, so gzipping them would produce files nothing ever asks for.
  gzip --best --force --keep "${site_dir}/search/search_index.json"

  if grep -rlE "https://(${EXTERNAL_HOSTS})" "${site_dir}" --include="*.html"; then
    echo "error: the pages above still fetch from the network" >&2
    return 1
  fi
}

#
# main
#

case "${1:-}" in
'')
  build_public
  ;;
--offline | offline)
  build_offline
  ;;
*)
  echo "usage: $(basename "${0}") [--offline]" >&2
  exit 1
  ;;
esac
