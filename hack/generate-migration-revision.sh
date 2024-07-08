#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

function show_help {
    echo "Usage: $0 MESSAGE"
    exit 1
}

if [ $# -eq 0 ]; then
    show_help
fi
MESSAGE="$*"
DEFAULT_DATABASE_URL="sqlite:////var/lib/gpustack/database.db"
DATABASE_URL=${DATABASE_URL:-$DEFAULT_DATABASE_URL}

export DATABASE_URL
alembic revision --autogenerate -m "$MESSAGE"
