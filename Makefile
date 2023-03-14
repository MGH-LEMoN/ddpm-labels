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
DT := $(shell date +"%Y%m%d")
# -%H%M%S

# ddpm-train: training a model from scratch
# Training parameters
model_idx = 1
time_steps = 800
beta_schedule = 'linear'
epochs = 50
jei_flag = 1
group_labels = 1
learning_rate = 5e-5
image_size = (192, 224)
results_dir = M$(model_idx)T$(time_steps)$(beta_schedule)G$(group_labels)J$(jei_flag)
results_dir = test

ddpm-train:
	# sbatch --job-name=$(DT)-$(results_dir) submit.sh \
		python -u scripts/main.py train \
			--model_idx $(model_idx) \
			--time_steps $(time_steps) \
			--beta_schedule $(beta_schedule) \
			--results_dir $(results_dir) \
			--epochs $(epochs) \
			--jei_flag $(jei_flag) \
			--group_labels $(group_labels) \
			--learning_rate $(learning_rate) \
			--image_size '$(image_size)' \
			;

# ddpm-resume: resume training
ddpm-resume:
	# sbatch --job-name=$(DT)-$(results_dir) submit.sh \
		python -u scripts/main.py resume-train \
			/space/calico/1/users/Harsha/ddpm-labels/logs/20230313-test \
			;

# ddpm-sample: generate samples from a trained model
ddpm-sample:
# TODO:

# ddpm-test: change self.DEBUG to TRUE in ddpm_config.py
# please refer to ddpm-labels/models/model1.py and model2.py for more info
# on how to modify the model parameters to work with fashion mnist data
ddpm-test:
	python -u scripts/main.py train \
		--model_idx 1 \
		--time_steps 300 \
		--beta_schedule linear \
		--results_dir mnist \
		--epochs 10 \
		--image_size '(28, 28)' \
		;