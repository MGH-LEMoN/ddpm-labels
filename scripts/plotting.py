import os

import matplotlib.pyplot as plt
import numpy as np

from scripts.ddpm_transforms import reverse_rgb_transform
from scripts.yael_funcs import logit_to_image


# source: https://pytorch.org/vision/stable/auto_examples/plot_transforms.html#sphx-glr-auto-examples-plot-transforms-py
def plot_diffusion_process(
    config, imgs, file_name, with_orig=False, row_title=None, **imshow_kwargs
):
    """Demonstrate forward/reverse process on

    Args:
        imgs (_type_): _description_
        with_orig (bool, optional): _description_. Defaults to False.
        row_title (_type_, optional): _description_. Defaults to None.
    """
    logdir = config.logdir if config else "logs"

    if "reverse" in file_name:
        time_steps = config.plot_time_steps[::-1]
    else:
        time_steps = config.plot_time_steps

    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(
        figsize=(10, 10), nrows=num_rows, ncols=num_cols, squeeze=False
    )
    plt.tight_layout()
    for row_idx, row in enumerate(imgs):
        # row = [image] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]

            img = img.squeeze()

            cmap = "gray" if img.dim() == 2 else "viridis"
            ax.imshow(np.asarray(img), cmap=cmap, **imshow_kwargs)

            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            ax.set(title=rf"{str(time_steps[col_idx])}")
            ax.title.set_size(10)

    if with_orig:
        axs[0, 0].set(title="Original image")
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.subplots_adjust(wspace=0.025)
    save_file = os.path.join(
        logdir,
        file_name,
    )
    plt.savefig(save_file, bbox_inches="tight")


# display a few images to check the label maps
def show_images(config, data, num_samples=20, cols=4):
    """Plots some samples from the dataset"""

    fig = plt.figure(figsize=(10, 10))
    plt.tight_layout()

    # randomly sample num_samples indices
    sample_ids = np.random.permutation(len(data))[:num_samples]

    for i, sample_id in enumerate(sample_ids):
        plt.subplot(num_samples // cols + 3, cols, i + 1)

        img = data[sample_id]

        if config.debug:
            img = img.squeeze()

        if config.rgb_flag:
            img = reverse_rgb_transform(img)

        if not config.rgb_flag and not config.debug:
            img = logit_to_image(config, img)

        cmap = "gray" if img.dim() == 2 else "viridis"
        plt.imshow(img, interpolation="nearest", cmap=cmap)

        plt.xticks([])
        plt.yticks([])

    plt.subplots_adjust(wspace=0.025, hspace=0.025)
    save_file = os.path.join(
        config.logdir,
        "sample_labels.png",
    )
    plt.savefig(save_file, bbox_inches="tight")
