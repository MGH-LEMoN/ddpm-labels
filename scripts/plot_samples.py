#!/usr/bin/env python
# coding: utf-8
import glob
import os
import sys

sys.path.append(os.getcwd())

import torch
from ddpm_config import Configuration

import matplotlib.pyplot as plt
import numpy as np
from beta_schedule import closed_form_equations
from losses import sample
from PIL import Image
from plotting import plot_diffusion_process
from training import select_model

# turn back into RGB image
from yael_funcs import logit_to_image


def collect_images_into_pdf(target_dir_str):
    """[summary]
    Args:
        target_dir_str ([str]): string relative to RESULTS_DIR
    """
    target_dir = os.path.join(target_dir_str, "images")
    base_dir = os.path.basename(target_dir_str)

    out_file = os.path.join(target_dir_str, base_dir + ".pdf")
    images = sorted(glob.glob(os.path.join(target_dir, "*.png")))

    pdf_img_list = [Image.open(image).convert("RGB") for image in images]
    pdf_img_list[0].save(out_file, save_all=True, append_images=pdf_img_list[1:])


def combine_images_to_pdf():
    model_dirs = sorted(
        # glob.glob("/space/calico/1/users/Harsha/ddpm-labels/logs/*G1*D1")
    )

    for model_dir in model_dirs:
        print(os.path.basename(model_dir))
        collect_images_into_pdf(model_dir)


def samples_from_epochs(model_dirs):
    # list of all models trained so far in logs directory

    if model_dirs is None:
        model_dirs = sorted(
            glob.glob("/space/calico/1/users/Harsha/ddpm-labels/logs/*RGB")
        )

    if isinstance(model_dirs, str):
        model_dirs = [model_dirs]

    for model_dir in model_dirs:
        base_dir = os.path.basename(model_dir)
        print(base_dir)

        # create directory to save samples (as images)
        img_dir = os.path.join(model_dir, "images")
        os.makedirs(img_dir, exist_ok=True)

        # read config.json from the model/experiment
        config = Configuration.read_config(os.path.join(model_dir, "config.json"))

        # add attribute (as it is missing)
        config.sampling_batch_size = (num_samples := 1)

        model = select_model(config)
        model.to(config.device)

        # closed form equations for forward process
        cf_calculations = closed_form_equations(config)

        # load all checkpoints
        model_chkpts = sorted(glob.glob(os.path.join(model_dir, "model_*")))

        for model_chkpt in model_chkpts:
            epoch_num = os.path.basename(model_chkpt).split("_")[-1]

            print(f"Running Checkpoint: {epoch_num}")

            out_file_name = os.path.join(img_dir, f"{base_dir}-reverse-{epoch_num}.png")

            # Skip if file already exists
            if os.path.isfile(out_file_name):
                continue

            model.load_state_dict(torch.load(model_chkpt))

            # sample batch_size images
            samples = sample(
                config,
                model,
                cf_calculations=cf_calculations,
            )

            imgs = []
            for choice in range(num_samples):
                select_imgs = [
                    samples[time_step][choice] for time_step in config.plot_time_steps
                ]
                denoised_images = [
                    logit_to_image(config, torch.Tensor(select_img))
                    for select_img in select_imgs
                ]

                imgs.append(denoised_images)

            # plot_diffusion_process(config, denoised_images, save_file)
            with_orig = False
            row_title = None
            imshow_kwargs = {}

            if "reverse" in out_file_name:
                time_steps = config.plot_time_steps[::-1]

            if not isinstance(imgs[0], list):
                # Make a 2d grid even if there's just 1 row
                imgs = [imgs]

            num_rows = len(imgs)
            num_cols = len(imgs[0]) + with_orig
            fig, axs = plt.subplots(
                figsize=(13.5, 24.5),
                nrows=num_rows,
                ncols=num_cols,
                squeeze=False,
            )
            plt.tight_layout()

            for row_idx, row in enumerate(imgs):
                # row = [image] + row if with_orig else row
                for col_idx, img in enumerate(row):
                    ax = axs[row_idx, col_idx]
                    if config.debug:
                        ax.imshow(np.asarray(img[0]), cmap="gray", **imshow_kwargs)
                    else:
                        ax.imshow(np.asarray(img), **imshow_kwargs)
                    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

                    # print title
                    if row_idx == 0:
                        ax.set(title=rf"{str(time_steps[col_idx])}")
                        ax.title.set_size(10)

                    # print sample number (for each row)
                    if col_idx == 0:
                        ax.set_ylabel(f"Sample {row_idx + 1}")

                    # print epoch number (at bottom row and center column)
                    if row_idx == (num_samples - 1) and col_idx == len(time_steps) // 2:
                        ax.set_xlabel(f"Epoch {epoch_num}")

            if with_orig:
                axs[0, 0].set(title="Original image")
                axs[0, 0].title.set_size(8)
            if row_title is not None:
                for row_idx in range(num_rows):
                    axs[row_idx, 0].set(ylabel=row_title[row_idx])

            plt.subplots_adjust(wspace=0, hspace=0)
            plt.savefig(out_file_name, bbox_inches="tight")
            plt.close()


if __name__ == "__main__":
    samples_from_epochs(
        sys.argv[1] if len(sys.argv) > 1 else None
    )  # see ddpm-sample target in the Makefile
