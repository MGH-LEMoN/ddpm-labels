import os
import random
from datetime import datetime

import numpy as np
import torch
from accelerate import Accelerator, find_executable_batch_size
from losses import p_losses, reverse_diffusion_sample, sample
from plotting import plot_diffusion_process
from torch.optim import Adam
from torch.utils.data import DataLoader
from yael_funcs import logit_to_image

from ddpm_labels.models.model1 import SimpleUnet
from ddpm_labels.models.model2 import Unet

# from torchvision.utils import save_image

# https://towardsdatascience.com/a-batch-too-large-finding-the-batch-size-that-fits-on-gpus-aef70902a9f1


def auto_train(args, dataset, closed_form_calculations):
    accelerator = Accelerator()

    @find_executable_batch_size(starting_batch_size=4096 * 4)
    def inner_training_loop(batch_size=128):
        nonlocal accelerator  # Ensure they can be used in our context
        accelerator.free_memory()  # Free all lingering references

        accelerator.print(f"Trying batch size: {batch_size}")
        args.batch_size = batch_size

        train(args, dataset, closed_form_calculations)

    try:
        inner_training_loop()
    except RuntimeError:
        print("No executable batch size found")
        exit()


def select_model(config):
    if config.model_idx == 1:
        model = SimpleUnet(config.im_channels)
    elif config.model_idx == 2:
        if config.debug:
            model = Unet(
                dim=16,
                channels=config.im_channels,
                dim_mults=(1, 2, 4),
            )
        else:
            model = Unet(
                dim=16,
                channels=config.im_channels,
                dim_mults=(2, 4, 8, 16, 32),
            )
    else:
        print("Invalid Model ID")
        exit()
    return model


def train(config, training_set, cf_results):
    model = select_model(config)

    model.to(config.device)
    optimizer = Adam(model.parameters(), lr=config.lr)

    if config.checkpoint:
        model.load_state_dict(torch.load(config.checkpoint))

    params = {
        "batch_size": config.batch_size,
        "shuffle": True,
        "num_workers": 4,
        # "worker_init_fn": np.random.seed(42),
    }

    training_generator = DataLoader(training_set, **params)

    for epoch in range(config.start_epoch + 1, config.epochs + 1):
        epoch_loss = 0
        for batch in training_generator:
            optimizer.zero_grad()

            batch_size = batch.shape[0]
            batch = batch.to(config.device)

            t = torch.randint(
                0, config.time_steps, (batch_size,), device=config.device
            ).long()

            loss = p_losses(model, batch, t, cf_results, loss_type=config.loss_type)
            epoch_loss += loss.item() * batch_size

            loss.backward()
            optimizer.step()

        # writing loss value to writer object
        print(
            f"{datetime.now().strftime('%Y%m%d-%H:%M:%S')} Epoch {epoch:04d} | Loss: {epoch_loss/len(training_set):0.5f}"
        )
        config.writer.add_scalar("training_loss", epoch_loss / len(training_set), epoch)

        # Saving model every 25 epochs
        if config.save_checkpoint:
            if epoch == 1 or epoch % config.checkpoint_freq == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(config.logdir, f"model_{epoch:04d}"),
                )

        if config.save_images:
            if epoch == 1 or epoch % config.sampling_freq == 0:
                # sample batch_size images
                samples = sample(
                    config,
                    model,
                    cf_calculations=cf_results,
                )

                imgs = [
                    samples[time_step][random.randint(0, config.batch_size - 1)]
                    for time_step in config.plot_time_steps
                ]
                denoised_images = [
                    logit_to_image(config, torch.Tensor(img)) for img in imgs
                ]

                # plot reverse diffusion
                file_name = f"reverse-{epoch:04d}.png"
                plot_diffusion_process(config, denoised_images, file_name)

    config.writer.close()
