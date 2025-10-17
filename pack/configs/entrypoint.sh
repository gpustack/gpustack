#!/bin/bash

set -e

if [ "$1" = "start" ]; then
	# shellcheck disable=SC2124
	export GPUSTACK_EXTRA_ARGS="$@"
	exec supervisord -c /etc/supervisor/conf.d/supervisord.conf
else
	exec gpustack "$@"
fi
