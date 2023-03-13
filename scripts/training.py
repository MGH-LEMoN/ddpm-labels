import os
import random

import numpy as np
import torch
from accelerate import Accelerator, find_executable_batch_size
from ddpm_labels.models.model1 import SimpleUnet
from ddpm_labels.models.model2 import Unet
from torch.optim import Adam
from torch.utils.data import DataLoader

from losses import p_losses, reverse_diffusion_sample, sample
from plotting import plot_diffusion_process
from yael_funcs import logit_to_image

# from torchvision.utils import save_image

# https://towardsdatascience.com/a-batch-too-large-finding-the-batch-size-that-fits-on-gpus-aef70902a9f1


def auto_train(args, dataset, closed_form_calculations):
    accelerator = Accelerator()

    @find_executable_batch_size(starting_batch_size=4096 * 4)
    def inner_training_loop(batch_size=128):
        nonlocal accelerator  # Ensure they can be used in our context
        accelerator.free_memory()  # Free all lingering references

        accelerator.print(f"Trying batch size: {batch_size}")
        args.BATCH_SIZE = batch_size

        train(args, dataset, closed_form_calculations)

    try:
        inner_training_loop()
    except RuntimeError:
        print("No executable batch size found")
        exit()


def select_model(config):
    if config.model_idx == 1:
        model = SimpleUnet(config.image_channels)
    elif config.model_idx == 2:
        if config.DEBUG:
            model = Unet(
                dim=16,
                channels=config.image_channels,
                dim_mults=(1, 2, 4),
            )
        else:
            model = Unet(
                dim=16,
                channels=config.image_channels,
                dim_mults=(2, 4, 8, 16, 32, 64),
            )
    else:
        print("Invalid Model ID")
        exit()
    return model


def train(config, training_set, cf_results):
    model = select_model(config)

    model.to(config.DEVICE)
    optimizer = Adam(model.parameters(), lr=config.learning_rate)

    params = {
        "batch_size": config.BATCH_SIZE,
        "shuffle": True,
        # "num_workers": 0,
        # "worker_init_fn": np.random.seed(42),
    }

    training_generator = DataLoader(training_set, **params)

    for epoch in range(1, config.EPOCHS + 1):
        epoch_loss = 0
        for batch in training_generator:
            optimizer.zero_grad()

            batch_size = batch.shape[0]
            batch = batch.to(config.DEVICE)

            t = torch.randint(0, config.T, (batch_size,), device=config.DEVICE).long()

            loss = p_losses(model, batch, t, cf_results, loss_type=config.loss_type)
            epoch_loss += loss.item() * batch_size

            loss.backward()
            optimizer.step()

        # writing loss value to writer object
        try:
            print(f"Epoch {epoch:03d} | Loss: {epoch_loss/training_set.n_files:0.5f}")
        except:
            print(f"Epoch {epoch:03d} | Loss: {epoch_loss/len(training_set):0.5f}")

        config.writer.add_scalar("training_loss", epoch_loss / len(training_set), epoch)

        # Saving model every 25 epochs
        if config.save_checkpoint:
            if epoch == 1 or epoch % 25 == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(config.logdir, f"model_{epoch:04d}"),
                )

        if config.save_images:
            if epoch == 1 or epoch % 5 == 0:
                # sample batch_size images
                samples = sample(
                    config,
                    model,
                    image_size=config.image_size,
                    batch_size=config.batch_size,
                    channels=config.image_channels,
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
                file_name = f"{config.dir_flag}-reverse-{epoch:04d}.png"
                plot_diffusion_process(config, denoised_images, file_name)

    config.writer.close()
