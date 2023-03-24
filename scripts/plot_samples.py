#!/usr/bin/env python
# coding: utf-8

import glob
import os
import sys

import torch
from tqdm.auto import tqdm

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

DEBUG = False


class Configuration:
    """
    This configuration object is a collection of all variables relevant to the analysis
    """

    def __init__(
        self,
        sampling_batch_size,
        time_steps,
        im_size,
        beta_schedule,
        im_channels,
        jei_flag,
        downsample,
    ):
        self.time_steps = time_steps
        self.im_size = im_size
        self.beta_schedule = beta_schedule
        self.im_channels = im_channels
        self.jei_flag = jei_flag
        self.logdir = "./"
        self.DEBUG = False
        self.downsample = downsample

        self.sampling_batch_size = sampling_batch_size

        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.plot_time_steps = [0, 100, 200, 300, 400, 500, 600, 700, 799]


def select_model(config, model_idx, image_channels):
    if model_idx == 1:
        model = SimpleUnet(image_channels)
    elif model_idx == 2:
        if DEBUG:
            model = Unet(
                dim=16,
                channels=image_channels,
                dim_mults=(1, 2, 4),
            )
        else:
            if not config.downsample:
                dim_mults = (2, 4, 8, 16, 32, 64)
            else:
                dim_mults = (2, 4, 8, 16, 32)
            model = Unet(
                dim=16,
                channels=image_channels,
                dim_mults=dim_mults,
            )
    else:
        print("Invalid Model ID")
        exit()
    return model


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
    images = sorted(glob.glob(os.path.join(target_dir, "*")))

    pdf_img_list = []
    for image in images:
        img = Image.open(image)
        img = img.convert("RGB")
        pdf_img_list.append(img)

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
    # model_dirs = reversed(
    #     sorted(glob.glob("/space/calico/1/users/Harsha/ddpm-labels/logs/*G2*"))
    # )
    model_dirs = [model_dirs]

    for model_dir in model_dirs:
        base_dir = os.path.basename(model_dir)

        if not os.path.isdir(os.path.join(model_dir, "images")):
            os.makedirs(os.path.join(model_dir, "images"), exist_ok=True)

        print(base_dir)

        # model_idx = int(base_dir[1])
        if "M1" in base_dir:
            model_idx = 1

        if "M2" in base_dir:
            model_idx = 2

        if "20230306" in base_dir:
            jei_flag = int(base_dir[-1])
            group_labels = int(base_dir[-3])

            downsample = 0
            im_size = (192, 224)

        else:
            jei_flag = int(base_dir[-3])
            group_labels = int(base_dir[-5])

            downsample = int(base_dir[-1])
            im_size = (96, 112)

        if DEBUG:
            im_channels = 1
        else:
            group_labels_flag_dict = {0: 24, 1: 4, 2: 14}
            im_channels = group_labels_flag_dict[group_labels] - (1 - jei_flag)

        if "linear" in base_dir:
            beta_schedule = "linear"

        if "cosine" in base_dir:
            beta_schedule = "cosine"

        if "quadratic" in base_dir:
            beta_schedule = "quadratic"

        if "sigmoid" in base_dir:
            beta_schedule = "sigmoid"

        config = Configuration(
            25, 800, im_size, beta_schedule, im_channels, jei_flag, downsample
        )

        model = select_model(config, model_idx, im_channels)
        model.to(config.DEVICE)

        cf_calculations = closed_form_equations(config)

        model_chkpts = sorted(glob.glob(os.path.join(model_dir, "model_*")))

        for model_chkpt in model_chkpts:
            epoch_num = os.path.basename(model_chkpt).split("_")[-1]
            epoch_num_int = int(epoch_num)

            print(f"Running Checkpoint: {epoch_num}")

            save_file = f"{model_dir}/images/{base_dir}-reverse-{epoch_num}.png"

            # Skip if file already exists
            if os.path.isfile(save_file):
                curr_file = (
                    f"{model_dir}/images/reverse_process-{epoch_num[1:]}.png"
                )
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
            for choice in range(num_samples := 20):
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
                    if config.DEBUG:
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
    samples_from_epochs(sys.argv[1])  # see ddpm-sample target in the Makefile
