#!/usr/bin/env python
# coding: utf-8
import argparse
import glob
import json
import os
import sys

import torch
from tqdm.auto import tqdm
from ddpm_config import Configuration

sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
import numpy as np
from beta_schedule import closed_form_equations
from losses import sample
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from plotting import plot_diffusion_process

# turn back into RGB image
from yael_funcs import logit_to_image

from ddpm_labels.models.model1 import SimpleUnet
from ddpm_labels.models.model2 import Unet
from training import select_model


def plot_from_pickle_files():
    # list of all models trained so far in logs directory
    pkl_files = sorted(
        glob.glob(
            "/space/calico/1/users/Harsha/ddpm-labels/logs/20230306-M2T800linearG1J1/*.pkl"
        )
    )

    config = Configuration(25, 800, (192, 224), "linear", 4, 1)

    import pickle

    for pkl_file in pkl_files:
        base_dir = os.path.basename(pkl_file)

        epoch_num = os.path.splitext(base_dir)[0].split("-")[-1]
        epoch_num_int = int(epoch_num)

        save_file = base_dir.replace("pkl", "png")

        with open(pkl_file, "rb") as fh:
            samples = pickle.load(fh)

        plot_time_steps = [0, 100, 200, 300, 400, 500, 600, 700, 799]

        imgs = []
        for choice in range(25):
            select_imgs = [samples[time_step][choice] for time_step in plot_time_steps]
            denoised_images = [
                logit_to_image(config, torch.Tensor(img)).numpy() for img in select_imgs
            ]
            imgs.append(denoised_images)

        # plot_diffusion_process(config, denoised_images, save_file)
        file_name = save_file
        with_orig = False
        row_title = None
        imshow_kwargs = {}
        logdir = "./"

        time_steps = plot_time_steps[::-1]

        if not isinstance(imgs[0], list):
            # Make a 2d grid even if there's just 1 row
            imgs = [imgs]

        num_rows = len(imgs)
        num_cols = len(imgs[0]) + with_orig
        fig, axs = plt.subplots(
            figsize=(15, 15), nrows=num_rows, ncols=num_cols, squeeze=False
        )
        plt.tight_layout()
        for row_idx, row in enumerate(imgs):
            # row = [image] + row if with_orig else row
            for col_idx, img in enumerate(row):
                ax = axs[row_idx, col_idx]
                if False:
                    ax.imshow(np.asarray(img[0]), cmap="gray", **imshow_kwargs)
                else:
                    ax.imshow(np.asarray(img), **imshow_kwargs)
                ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
                if row_idx == 0:
                    ax.set(title=rf"{str(time_steps[col_idx])}")
                    ax.title.set_size(10)

        # if with_orig:
        #     axs[0, 0].set(title="Original image")
        #     axs[0, 0].title.set_size(8)
        # if row_title is not None:
        #     for row_idx in range(num_rows):
        #         axs[row_idx, 0].set(ylabel=row_title[row_idx])

        plt.subplots_adjust(wspace=0.025, hspace=0.025)
        save_file = os.path.join(
            logdir,
            file_name,
        )
        plt.savefig(
            save_file,
            format="png",
            transparent=True,
            bbox_inches="tight",
            pad_inches=0,
            dpi=300,
        )
        # plt.savefig(save_file, bbox_inches="tight")


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
    else:
        model_dirs = [model_dirs]

    for model_dir in model_dirs[:1]:
        base_dir = os.path.basename(model_dir)

        if not os.path.isdir(os.path.join(model_dir, "images")):
            os.makedirs(os.path.join(model_dir, "images"), exist_ok=True)

        print(base_dir)

        config = Configuration.read_config(os.path.join(model_dir, "config.json"))
        config.sampling_batch_size = (num_samples := 1)

        model = select_model(config)
        model.to(config.device)

        cf_calculations = closed_form_equations(config)

        model_chkpts = sorted(glob.glob(os.path.join(model_dir, "model_*")))

        for model_chkpt in model_chkpts[:1]:
            epoch_num = os.path.basename(model_chkpt).split("_")[-1]
            epoch_num_int = int(epoch_num)

            print(f"Running Checkpoint: {epoch_num}")

            save_file = f"{model_dir}/images/{base_dir}-reverse-{epoch_num}.png"

            # Skip if file already exists
            if os.path.isfile(save_file):
                curr_file = f"{model_dir}/images/reverse_process-{epoch_num[1:]}.png"
                if os.path.isfile(curr_file):
                    os.remove(curr_file)
                continue

            model.load_state_dict(torch.load(model_chkpt))

            # sample batch_size images
            samples = sample(
                config,
                model,
                cf_calculations=cf_calculations,
            )

            #     import pickle
            #     save_pkl = f"{model_dir}/{base_dir}-reverse-{epoch_num}.pkl"
            #     with open(save_pkl, "wb") as fh:
            #         pickle.dump(samples, fh)

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
            file_name = f"{model_dir}/images/{base_dir}-reverse-{epoch_num}.png"

            if not os.path.isdir(os.path.dirname(file_name)):
                os.makedirs(os.path.dirname(file_name), exist_ok=True)

            with_orig = False
            row_title = None
            imshow_kwargs = {}

            if "reverse" in file_name:
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

                    if row_idx == 0:
                        ax.set(title=rf"{str(time_steps[col_idx])}")
                        ax.title.set_size(10)

                    if col_idx == 0:
                        ax.set_ylabel(f"Sample {row_idx + 1}")

                    if row_idx == (num_samples - 1) and col_idx == 4:
                        ax.set_xlabel(f"Epoch {epoch_num_int}")

            if with_orig:
                axs[0, 0].set(title="Original image")
                axs[0, 0].title.set_size(8)
            if row_title is not None:
                for row_idx in range(num_rows):
                    axs[row_idx, 0].set(ylabel=row_title[row_idx])

            plt.subplots_adjust(wspace=0, hspace=0)
            plt.savefig(file_name, bbox_inches="tight")
            plt.close()


if __name__ == "__main__":
    samples_from_epochs(
        sys.argv[1] if len(sys.argv) > 1 else None
    )  # see ddpm-sample target in the Makefile
