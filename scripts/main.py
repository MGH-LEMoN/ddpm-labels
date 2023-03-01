#!/usr/bin/env python
# coding: utf-8
import os
import random
import sys

sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
import numpy as np
import torch
from beta_schedule import closed_form_equations
from data_loaders import DDPMLabelsDataset
from ddpm_config import Configuration
from losses import forward_diffusion_sample
from plotting import plot_forward_process, show_images
from training import auto_train, train
from utils import load_labelmap_names
from yael_funcs import logit_to_image


def get_noisy_image(config, x_start, t, cf_results):
    # add noise
    x_noisy, _ = forward_diffusion_sample(x_start, t, cf_results)

    # turn back into RGB image
    noisy_image = logit_to_image(config, x_noisy)
    return noisy_image


if __name__ == "__main__":
    config = Configuration()

    # Fix random seed
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    training_set = DDPMLabelsDataset(
        config,
        load_labelmap_names("ddpm_files_padded.txt"),
    )

    show_images(
        config, training_set, num_samples=15, cols=5, jei_flag=config.jei_flag
    )

    # closed form results
    cf_results = closed_form_equations(config)

    # get an image to simulate forward diffusion
    image = training_set[0]

    # plot forward diffusion
    plot_forward_process(
        config,
        [
            get_noisy_image(config, image, torch.tensor([t]), cf_results)
            for t in config.plot_time_steps
        ],
        file_name="forward_process.png",
    )

    auto_train(config, training_set, cf_results)
    # sample_plot_image(config, epoch)
