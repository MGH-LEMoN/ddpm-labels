import os

import numpy as np
import torch
from accelerate import Accelerator, find_executable_batch_size
from data_loaders import DDPMLabelsDataset
from ddpm_config import Configuration
from losses import p_losses
from torch.optim import Adam
from torch.utils.data import DataLoader
from utils import load_labelmap_names

from ddpm_labels.models.model1 import SimpleUnet
from ddpm_labels.models.model2 import Unet

# from torchvision.utils import save_image

# https://towardsdatascience.com/a-batch-too-large-finding-the-batch-size-that-fits-on-gpus-aef70902a9f1


def auto_train(args, dataset, closed_form_calculations):
    accelerator = Accelerator()

    @find_executable_batch_size(starting_batch_size=2048)
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


def train(config, training_set, cf_results):
    model = SimpleUnet(config.image_channels)
    model = Unet(
        dim=16, channels=config.image_channels, dim_mults=(2, 4, 8, 16, 32, 64)
    )

    model.to(config.DEVICE)
    optimizer = Adam(model.parameters(), lr=config.learning_rate)

    params = {
        "batch_size": config.BATCH_SIZE,
        "shuffle": True,
        "num_workers": 0,
        "worker_init_fn": np.random.seed(42),
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
        print(f"Epoch {epoch:03d} | Loss: {epoch_loss/training_set.n_files:0.5f}")

        with config.writer as fn:
            fn.add_scalar("training_loss", epoch_loss / len(training_set), epoch)

        # Saving model every 25 epochs
        if config.save_checkpoint:
            if epoch == 1 or epoch % 25 == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(config.logdir, f"model_{epoch:04d}"),
                )

    # sample_plot_image(config.device, epoch)