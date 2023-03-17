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
model_idx = 1 2
time_steps = 800
beta_schedule = linear cosine quadratic sigmoid
loss_type = l1 l2 huber
epochs = 1000
jei_flag = 1
group_labels = 0
lr = 5e-5
im_size = (96, 112)

# ddpm-resume: train a model from scratch
ddpm-train:
# sbatch --job-name=$$logdir submit.sh 
	for model in $(model_idx); do \
		for schedule in $(beta_schedule); do \
			for loss in $(loss_type); do \
				logdir=M$$model\T$(time_steps)$$schedule\G$(group_labels)J$(jei_flag)D1
				sbatch --job-name=$$logdir submit.sh python -u scripts/main.py train \
					--model_idx $$model \
					--time_steps $(time_steps) \
					--beta_schedule $$schedule \
					--loss_type $$loss \
					--logdir $$logdir \
					--epochs $(epochs) \
					--jei_flag $(jei_flag) \
					--group_labels $(group_labels) \
					--lr $(lr) \
					--im_size '$(im_size)' \
					--downsample; \
			done; \
		done; \
	done;


# ddpm-resume: resume training
ddpm-resume:
	# sbatch --job-name=$(DT)-$(results_dir) submit.sh \
		python -u scripts/main.py resume-train \
			/space/calico/1/users/Harsha/ddpm-labels/logs/mnist \
			--debug \
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
		--logdir mnist \
		--epochs 25 \
		--im_size '(28, 28)' \
		--debug \
		;
