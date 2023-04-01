#!/usr/bin/env python
# coding: utf-8
import argparse
import ast
import glob
import json
import os
import sys

sys.path.append(os.getcwd())

import numpy as np
import torch
from beta_schedule import closed_form_equations
from data_loaders import DDPMLabelsDataset, FashionMnistDataset
from ddpm_config import Configuration
from ddpm_parser import parse_cmdline_arguments
from losses import forward_diffusion_sample
from plotting import plot_diffusion_process, show_images
from training import auto_train, train
from utils import load_labelmap_names
from yael_funcs import logit_to_image

from ext.utils import set_seed


def main():
    args = parse_cmdline_arguments()

    set_seed()

    if sys.argv[1] == "train":
        try:
            args.im_size = ast.literal_eval(args.im_size)
        except ValueError:
            pass

        config = Configuration(args)
        train_main(config)

    elif sys.argv[1] == "resume-train":
        chkpt_folder = args.logdir

        config_file = os.path.join(chkpt_folder, "config.json")
        assert os.path.exists(config_file), "Configuration file not found"

        with open(config_file) as json_file:
            data = json.load(json_file)

        assert isinstance(data, dict), "Invalid Object Type"

        dice_list = sorted(glob.glob(os.path.join(chkpt_folder, "model*")))

        if dice_list:
            data["checkpoint"] = dice_list[-1]
            data["start_epoch"] = int(os.path.basename(dice_list[-1]).split("_")[-1])
        else:
            sys.exit("No checkpoints exist to resume training")

        args = argparse.Namespace(**data)
        config = Configuration(args, "config_resume.json")

        train_main(config, True)
    else:
        raise Exception("Invalid Sub-command")

    return args


def get_noisy_image(config, x_start, t, cf_results):
    # add noise
    x_noisy, _ = forward_diffusion_sample(x_start, t, cf_results)

    # turn back into RGB image
    if config.debug:
        noisy_image = x_noisy
    else:
        noisy_image = logit_to_image(config, x_noisy)

    return noisy_image


def train_main(config, resume_flag=False):
    if config.debug:
        training_set = FashionMnistDataset()
    else:
        training_set = DDPMLabelsDataset(
            config,
            load_labelmap_names("ddpm_files_padded.txt"),
        )

    # closed form results
    cf_results = closed_form_equations(config)

    if not resume_flag:
        show_images(config, training_set, num_samples=15, cols=5)

        # get a random image to simulate forward diffusion
        image = training_set[np.random.permutation(len(training_set))[0]]

        # plot forward diffusion
        plot_diffusion_process(
            config,
            [
                get_noisy_image(config, image, torch.tensor([t]), cf_results)
                for t in config.plot_time_steps
            ],
            file_name="forward_process.png",
        )

    train(config, training_set, cf_results)


if __name__ == "__main__":
    main()
