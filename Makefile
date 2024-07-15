# Detect operating system
ifeq ($(OS),Windows_NT)
    PLATFORM_SHELL := powershell
    SCRIPT_EXT := .ps1
    SCRIPT_DIR := hack/windows
else
    PLATFORM_SHELL := /bin/bash
    SCRIPT_EXT := .sh
    SCRIPT_DIR := hack
endif

# Borrowed from https://stackoverflow.com/questions/18136918/how-to-get-current-relative-directory-of-your-makefile
curr_dir := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))

# Borrowed from https://stackoverflow.com/questions/2214575/passing-arguments-to-make-run
rest_args := $(wordlist 2, $(words $(MAKECMDGOALS)), $(MAKECMDGOALS))

$(eval $(rest_args):;@:)

# List targets based on script extension and directory
ifeq ($(OS),Windows_NT)
    targets := $(shell powershell -Command "Get-ChildItem -Path $(curr_dir)/$(SCRIPT_DIR) | Select-Object -ExpandProperty BaseName")
else
	targets := $(shell ls $(curr_dir)/$(SCRIPT_DIR) | grep $(SCRIPT_EXT) | sed 's/$(SCRIPT_EXT)$$//')
endif

$(targets):
	@$(eval TARGET_NAME=$@)
ifeq ($(PLATFORM_SHELL),/bin/bash)
	$(curr_dir)/$(SCRIPT_DIR)/$(TARGET_NAME)$(SCRIPT_EXT) $(rest_args)
else
	powershell "$(curr_dir)/$(SCRIPT_DIR)/$(TARGET_NAME)$(SCRIPT_EXT) $(rest_args)"
endif

help:
	#
	# Usage:
	#
	#   * [dev] `make install`, install all dependencies.
	#
	#   * [dev] `make generate`, generate codes.
	#
	#   * [dev] `make lint`, check style.
	#
	#   * [dev] `make test`, execute unit testing.
	#
	#   * [dev] `make build`, execute building.
	#
	#   * [dev] `make build-docs`, build docs, not supported on Windows.
	#
	#   * [dev] `make serve-docs`, serve docs, not supported on Windows.
	#
	#   * [ci]  `make ci`, execute `make install`, `make lint`, `make test`, `make build`.
	#
	@echo

.DEFAULT_GOAL := build
.PHONY: $(targets)
