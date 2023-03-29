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

# Notes: Since accelerate is not working, I recommend using the following values
# | Model | Batch | CPU | Memory |
# | 1 | 688 | 3 | 96 | (ignore this model going forward)
# | 2 | 240 | 3 | 96 |

# Training parameters
model_idx = 1 2
time_steps = 800
beta_schedule = linear cosine quadratic sigmoid
loss_type = l1 l2 huber
epochs = 500
jei_flag = 1
group_labels = 2
# { 0 | 1 | 2}
lr = 5e-5
im_size = (192, 224)
# {(192, 224) | (96, 112)}
downsampled = 0
# {0 | 1}
batch_size = 240


## ddpm-train: train a model from scratch
ddpm-train:
	for model in $(model_idx); do \
		for schedule in $(beta_schedule); do \
			for loss in $(loss_type); do \
				logdir=M$$model\T$(time_steps)$$schedule\L$$loss\G$(group_labels)J$(jei_flag)D$(downsampled)
				sbatch --job-name=$$logdir submit.sh python -u scripts/main.py train \
					--model_idx $$model \
					--time_steps $(time_steps) \
					--beta_schedule $$schedule \
					--loss_type $$loss \
					--logdir $$logdir \
					--epochs $(epochs) \
					--batch_size $(batch_size) \
					--jei_flag $(jei_flag) \
					--group_labels $(group_labels) \
					--lr $(lr) \
					--im_size '$(im_size)' \
					--downsample; \
			done; \
		done; \
	done;


## ddpm-resume: resume training
ddpm-resume:
	for model in $(model_idx); do \
		for schedule in $(beta_schedule); do \
			for loss in $(loss_type); do \
				logdir=test-M$$model\T$(time_steps)$$schedule\L$$loss\G$(group_labels)J$(jei_flag)D1
				sbatch --job-name=$$logdir submit.sh python -u scripts/main.py resume-train \
					/space/calico/1/users/Harsha/ddpm-labels/logs/$$logdir; \
			done; \
		done; \
	done;


## ddpm-sample: generate samples from a trained model
ddpm-sample: model_filter = M1* # other examples: {M2* | M2*G2*D0 | ...}
ddpm-sample:
	for model_dir in `ls -d1 -r /space/calico/1/users/Harsha/ddpm-labels/logs/$(model_filter)`; do \
		model=`basename $$model_dir`
	 	sbatch --job-name=sample-$$model submit.sh python scripts/plot_samples.py $$model_dir
	done;


## ddpm-images-to-pdf: collect all images into a pdf
ddpm-images-to-pdf:
	python -c "from scripts import combine_images_to_pdf; combine_images_to_pdf()"


## ddpm-test: test changes to code using fashion-mnist data
# please refer to ddpm-labels/models/model1.py and model2.py for more info
# on how to modify the model parameters to work with fashion mnist data
ddpm-test:
	python -u scripts/main.py train \
		--model_idx 1 \
		--time_steps 30 \
		--beta_schedule linear \
		--logdir mnist \
		--epochs 10 \
		--im_size '(28, 28)' \
		--debug \
		;


## model-summary: print model summary
model-summary:
	python ddpm_labels/models/model1.py
	python ddpm_labels/models/model2.py
