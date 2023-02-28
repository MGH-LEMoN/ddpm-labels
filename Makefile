# Run all commands in one shell
.ONESHELL:

# Default target
.DEFAULT_GOAL := help

.PHONY : help
## help: run 'make help" at commandline
help : Makefile
	@sed -n 's/^##//p' $<

.PHONY: list
## list: list all targets in the current make file
list:
	@LC_ALL=C $(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$'

# Generic Variables
USR := $(shell whoami | head -c 2)
DT := $(shell date +"%Y%m%d-%H%M%S")

# ddpm-train: training a model from scratch
ddpm-train:
	sbatch --job-name=$(DT) submit.sh python scripts/ddpm-main.py

# ddpm-resume: resume training
ddpm-resume:
# TODO:

# ddpm-sample: generate samples from a trained model
ddpm-sample:
# TODO: