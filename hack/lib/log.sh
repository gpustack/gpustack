#!/usr/bin/env bash

##
# Borrowed from github.com/kubernetes/kubernetes/hack/lib/logging.sh
##

# -----------------------------------------------------------------------------
# Logger variables helpers. These functions need the
# following variables:
#
#    LOG_LEVEL  -  The level of logger, default is "debug".

log_level="${LOG_LEVEL:-"debug"}"
log_colorful="${LOG_COLORFUL:-"true"}"

# Handler for when we exit automatically on an error.
gpustack::log::errexit() {
  local err="${PIPESTATUS[*]}"

  # if the shell we are in doesn't have errexit set (common in subshells) then
  # don't dump stacks.
  set +o | grep -qe "-o errexit" || return

  set +o xtrace
  gpustack::log::panic "${BASH_SOURCE[1]}:${BASH_LINENO[0]} '${BASH_COMMAND}' exited with status ${err}" "${1:-1}"
}

gpustack::log::install_errexit() {
  # trap ERR to provide an error handler whenever a command exits nonzero, this
  # is a more verbose version of set -o errexit
  trap 'gpustack::log::errexit' ERR

  # setting errtrace allows our ERR trap handler to be propagated to functions,
  # expansions and subshells
  set -o errtrace
}

# Debug level logging.
gpustack::log::debug() {
  [[ ${log_level} == "debug" ]] || return 0
  local message="${2:-}"

  local timestamp
  timestamp="$(date +"[%m%d %H:%M:%S]")"
  echo -e "[DEBG] ${timestamp} ${1-}" >&2
  shift 1
  for message; do
    echo -e "       ${message}" >&2
  done
}

# Info level logging.
gpustack::log::info() {
  [[ ${log_level} == "debug" ]] || [[ ${log_level} == "info" ]] || return 0
  local message="${2:-}"

  local timestamp
  timestamp="$(date +"[%m%d %H:%M:%S]")"
  if [[ ${log_colorful} == "true" ]]; then
    echo -e "\033[34m[INFO]\033[0m ${timestamp} ${1-}" >&2
  else
    echo -e "[INFO] ${timestamp} ${1-}" >&2
  fi
  shift 1
  for message; do
    echo -e "       ${message}" >&2
  done
}

# Warn level logging.
gpustack::log::warn() {
  local message="${2:-}"

  local timestamp
  timestamp="$(date +"[%m%d %H:%M:%S]")"
  if [[ ${log_colorful} == "true" ]]; then
    echo -e "\033[33m[WARN]\033[0m ${timestamp} ${1-}" >&2
  else
    echo -e "[WARN] ${timestamp} ${1-}" >&2
  fi
  shift 1
  for message; do
    echo -e "       ${message}" >&2
  done
}

# Error level logging, log an error but keep going, don't dump the stack or exit.
gpustack::log::error() {
  local message="${2:-}"

  local timestamp
  timestamp="$(date +"[%m%d %H:%M:%S]")"
  if [[ ${log_colorful} == "true" ]]; then
    echo -e "\033[31m[ERRO]\033[0m ${timestamp} ${1-}" >&2
  else
    echo -e "[ERRO] ${timestamp} ${1-}" >&2
  fi
  shift 1
  for message; do
    echo -e "       ${message}" >&2
  done
}

# Fatal level logging, log an error but exit with 1, don't dump the stack or exit.
gpustack::log::fatal() {
  local message="${2:-}"

  local timestamp
  timestamp="$(date +"[%m%d %H:%M:%S]")"
  if [[ ${log_colorful} == "true" ]]; then
    echo -e "\033[41;33m[FATA]\033[0m ${timestamp} ${1-}" >&2
  else
    echo -e "[FATA] ${timestamp} ${1-}" >&2
  fi
  shift 1
  for message; do
    echo -e "       ${message}" >&2
  done

  exit 1
}

# Panic level logging, dump the error stack and exit.
# Args:
#   $1 Message to log with the error
#   $2 The error code to return
#   $3 The number of stack frames to skip when printing.
gpustack::log::panic() {
  local message="${1:-}"
  local code="${2:-1}"

  local timestamp
  timestamp="$(date +"[%m%d %H:%M:%S]")"
  if [[ ${log_colorful} == "true" ]]; then
    echo -e "\033[41;33m[FATA]\033[0m ${timestamp} ${message}" >&2
  else
    echo -e "[FATA] ${timestamp} ${message}" >&2
  fi

  # print out the stack trace described by $function_stack
  if [[ ${#FUNCNAME[@]} -gt 2 ]]; then
    if [[ ${log_colorful} == "true" ]]; then
      echo -e "\033[31m       call stack:\033[0m" >&2
    else
      echo -e "       call stack:" >&2
    fi
    local i
    for ((i = 1; i < ${#FUNCNAME[@]} - 2; i++)); do
      echo -e "       ${i}: ${BASH_SOURCE[${i} + 2]}:${BASH_LINENO[${i} + 1]} ${FUNCNAME[${i} + 1]}(...)" >&2
    done
  fi

  if [[ ${log_colorful} == "true" ]]; then
    echo -e "\033[41;33m[FATA]\033[0m ${timestamp} exiting with status ${code}" >&2
  else
    echo -e "[FATA] ${timestamp} exiting with status ${code}" >&2
  fi

  popd >/dev/null 2>&1 || exit "${code}"
  exit "${code}"
}
