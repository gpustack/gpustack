SHELL := /bin/bash

# Borrowed from https://stackoverflow.com/questions/18136918/how-to-get-current-relative-directory-of-your-makefile
curr_dir := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))

# Borrowed from https://stackoverflow.com/questions/2214575/passing-arguments-to-make-run
rest_args := $(wordlist 2, $(words $(MAKECMDGOALS)), $(MAKECMDGOALS))
$(eval $(rest_args):;@:)

targets := $(shell ls $(curr_dir)/hack | grep '.sh' | sed 's/\.sh//g')
$(targets):
	@$(curr_dir)/hack/$@.sh $(rest_args)

help:
	#
	# Usage:
	#
	#   * [dev] `make deps`, install all dependencies.
	#
	#   * [dev] `make generate`, generate something.
	#
	#   * [dev] `make lint`, check style.
	#
	#   * [dev] `make test`, execute unit testing.
	#
	#   * [dev] `make build`, execute building.
	#
	#   * [ci]  `make ci`, execute `make deps`, `make lint`, `make test`, `make build`.
	#
	@echo

.DEFAULT_GOAL := build
.PHONY: $(targets)