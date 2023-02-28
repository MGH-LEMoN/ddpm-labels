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
from losses import forward_diffusion_sample, get_index_from_list, p_losses
from plotting import plot_forward_process, show_images
from torch.optim import Adam
from torch.utils.data import DataLoader
from utils import load_labelmap_names
from yael_funcs import (
    color_map_for_data,
    logit_to_image,
    prob_to_rgb,
    softmax_jei,
    softmax_yael,
)

from ddpm_labels.models.model1 import SimpleUnet
from ddpm_labels.models.model2 import Unet


@torch.no_grad()
def sample_timestep(model, x, t, closed_form_results):
    """
    Calls the model to predict the noise in the image and returns
    the denoised image.
    Applies noise to this image, if we are not in the last step yet.
    """
    betas = closed_form_results["betas"]
    sqrt_one_minus_alphas_cumprod = closed_form_results[
        "sqrt_one_minus_alphas_cumprod"
    ]
    sqrt_recip_alphas = closed_form_results["sqrt_recip_alphas"]
    posterior_variance = closed_form_results["posterior_variance"]

    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    # Call model (current image - noise prediction)
    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t == 0:
        return model_mean
    else:
        posterior_variance_t = get_index_from_list(
            posterior_variance, t, x.shape
        )
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise


def show_tensor_image(config, image, save=0, jei_flag=False):
    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :]

    if jei_flag:
        img = softmax_jei(image)
    else:
        img = softmax_yael(image)

    img = prob_to_rgb(img, implicit=False, colormap=color_map_for_data())
    plt.imshow(img, interpolation="nearest")

    plt.subplots_adjust(wspace=0.025)
    if save:
        save_file = os.path.join(
            config.logdir,
            "forward-process.png",
        )
        plt.savefig(save_file, bbox_inches="tight")


@torch.no_grad()
def sample_plot_image(config, model, device, epoch):
    T = config.T
    # Sample noise
    img_size = config.IMG_SIZE
    img = torch.randn((1, config.image_channels, *img_size), device=device)
    plt.figure(figsize=(15, 15))
    num_images = 10
    stepsize = int(T / num_images)

    for i in range(0, T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(model, img, t)
        if i % stepsize == 0:
            plt.subplot(1, num_images, i // stepsize + 1)
            plt.tick_params(
                axis="both",  # changes apply to the x-axis
                which="both",  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                left=False,
                right=False,
                labelbottom=False,
                labelleft=False,
            )  # labels along the bottom edge are off
            show_tensor_image(img.detach().cpu(), save=0)
    plt.subplots_adjust(wspace=0.025)
    save_file = os.path.join(
        "/space/calico/1/users/Harsha/ddpm-labels/logs/experiment_02",
        f"reverse-image_{epoch:03d}.png",
    )
    plt.savefig(save_file, bbox_inches="tight")


def get_noisy_image(x_start, t, cf_results, jei_flag=False):
    # add noise
    x_noisy, _ = forward_diffusion_sample(x_start, t, cf_results)

    # turn back into RGB image
    noisy_image = logit_to_image(x_noisy, jei_flag=jei_flag)
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
        load_labelmap_names("ddpm_files_padded.txt"),
        jei_flag=config.jei_flag,
    )
    training_generator = DataLoader(training_set, **config.params)

    show_images(
        config, training_set, num_samples=15, cols=5, jei_flag=config.jei_flag
    )

    # closed form results
    cf_results = closed_form_equations(config)

    # get an image to simulate forward diffusion
    image = next(iter(training_generator))[0]

    # plot forward diffusion
    plot_forward_process(
        config,
        [
            get_noisy_image(
                image, torch.tensor([t]), cf_results, config.jei_flag
            )
            for t in list(np.arange(0, config.T, 100)) + [config.T - 1]
        ],
    )

    model = SimpleUnet(config.image_channels)
    # model = Unet(
    #     dim=16, channels=config.image_channels, dim_mults=(2, 4, 8, 16, 32, 64)
    # )

    model.to(config.DEVICE)
    optimizer = Adam(model.parameters(), lr=config.learning_rate)

    for epoch in range(1, config.EPOCHS + 1):
        epoch_loss = 0
        for step, batch in enumerate(training_generator):
            optimizer.zero_grad()

            batch_size = batch.shape[0]
            batch = batch.to(config.DEVICE)

            t = torch.randint(
                0, config.T, (batch_size,), device=config.DEVICE
            ).long()
            loss = p_losses(
                model, batch, t, cf_results, loss_type=config.loss_type
            )
            epoch_loss += loss.item() * batch_size

            loss.backward()
            optimizer.step()

        # writing loss value to writer object
        print(
            f"Epoch {epoch:03d} | Loss: {epoch_loss/training_set.n_files:0.5f}"
        )

        with config.writer as fn:
            fn.add_scalar(
                "training_loss", epoch_loss / len(training_set), epoch
            )

        # Saving model every 25 epochs
        if epoch == 1 or epoch % 25 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(config.logdir, f"model_{epoch:04d}"),
            )

    # sample_plot_image(config.device, epoch)
