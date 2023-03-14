#!/usr/bin/env python
# coding: utf-8
import argparse
import ast
import glob
import json
import os
import random
import sys
from argparse import ArgumentParser

sys.path.append(os.getcwd())


import numpy as np
import torch
from beta_schedule import closed_form_equations
from data_loaders import DDPMLabelsDataset
from ddpm_config import Configuration
from losses import forward_diffusion_sample
from plotting import plot_diffusion_process, show_images
from training import auto_train, train
from utils import load_labelmap_names
from yael_funcs import logit_to_image

from ext.lab2im.utils import infer


def parse_cmdline_arguments():
    parser = ArgumentParser()

    subparsers = parser.add_subparsers(help="sub-command help")

    # create the parser for the "resume-train" command
    parser_resume = subparsers.add_parser(
        "resume-train", help="Use this sub-command for resuming training"
    )
    parser_resume.add_argument(
        "results_dir",
        type=str,
        help="""Folder containing previous checkpoints""",
    )

    # create the parser for the "train" command
    parser_train = subparsers.add_parser(
        "train", help="Use this sub-command for training"
    )

    parser_train.add_argument(
        "--model_idx", type=int, dest="model_idx", default=1
    )
    parser_train.add_argument("--epochs", type=int, dest="EPOCHS", default=1000)
    parser_train.add_argument("--time_steps", type=int, dest="T", default=500)
    parser_train.add_argument(
        "--beta_schedule", type=str, dest="beta_schedule", default="linear"
    )
    parser_train.add_argument(
        "--results_dir", type=str, dest="results_dir", default="test"
    )
    parser_train.add_argument(
        "--jei_flag", type=int, dest="jei_flag", default=0
    )
    parser_train.add_argument(
        "--group_labels", type=int, dest="group_labels", default=0
    )
    parser_train.add_argument(
        "--learning_rate", type=float, dest="learning_rate", default=5e-5
    )
    parser_train.add_argument(
        "--image_size", nargs="?", type=infer, dest="IMG_SIZE", default=None
    )
    parser_train.add_argument(
        "--image_channels", type=int, dest="image_channels", default=1
    )
    parser_train.add_argument(
        "--loss_type", type=str, dest="loss_type", default="huber"
    )

    # If running the code in debug mode (vscode)
    gettrace = getattr(sys, "gettrace", None)

    if gettrace():
        # sys.argv = [
        #     "main.py",
        #     "train",
        #     "--learning_rate",
        #     "1e-3",
        #     "--time_steps",
        #     "750",
        #     "--jei_flag",
        #     "1",
        #     "--group_labels",
        #     "1",
        #     "--results_dir",
        #     "test",
        #     "--image_size",
        #     "(192, 224)",
        #     "--epochs",
        #     "10",
        #     "--beta_schedule",
        #     "linear",
        # ]

        # sys.argv = [
        #     "main.py",
        #     "resume-train",
        #     "/space/calico/1/users/Harsha/ddpm-labels/logs/20230313-test",
        # ]

        sys.argv = [
            "main.py",
            "train",
            "--model_idx",
            "1",
            "--time_steps",
            "300",
            "--beta_schedule",
            "linear",
            "--results_dir",
            "mnist",
            "--epochs",
            "10",
            "--image_size",
            "(28, 28)",
        ]

    args = parser.parse_args()

    if sys.argv[1] == "train":
        try:
            args.IMG_SIZE = ast.literal_eval(args.IMG_SIZE)
        except ValueError:
            pass

        config = Configuration(args)
        train_main(config)

    elif sys.argv[1] == "resume-train":
        chkpt_folder = args.results_dir

        config_file = os.path.join(chkpt_folder, "config.json")
        assert os.path.exists(config_file), "Configuration file not found"

        with open(config_file) as json_file:
            data = json.load(json_file)

        assert isinstance(data, dict), "Invalid Object Type"

        data.pop("message", "Key Error")

        dice_list = sorted(glob.glob(os.path.join(chkpt_folder, "model*")))

        if dice_list:
            data["checkpoint"] = dice_list[-1]
            data["start_epoch"] = int(
                os.path.basename(dice_list[-1]).split("_")[-1]
            )
            data["results_dir"] = os.path.basename(chkpt_folder)
        else:
            sys.exit("No checkpoints exist to resume training")

        args = argparse.Namespace(**data)
        config = Configuration(args, "config_resume.json")

        train_main(config, True)

    else:
        raise Exception("Invalid Sub-command")

    return args


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def get_noisy_image(config, x_start, t, cf_results):
    # add noise
    x_noisy, _ = forward_diffusion_sample(x_start, t, cf_results)

    # turn back into RGB image
    if config.DEBUG:
        noisy_image = x_noisy
    else:
        noisy_image = logit_to_image(config, x_noisy)

    return noisy_image


def train_main(config, resume_flag=False):
    if config.DEBUG:
        from datasets import load_dataset
        from torchvision import transforms
        from torchvision.transforms import Compose

        dataset = load_dataset("fashion_mnist")
        transform = Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: (t * 2) - 1),
            ]
        )

        # define function
        def transforms(examples):
            examples["pixel_values"] = [
                transform(image.convert("L")) for image in examples["image"]
            ]
            del examples["image"]
            return examples

        transformed_set = dataset.with_transform(transforms).remove_columns(
            "label"
        )
        training_set = transformed_set["train"][:]["pixel_values"]
    else:
        training_set = DDPMLabelsDataset(
            config,
            load_labelmap_names("ddpm_files_padded.txt")[:256],
        )

    # closed form results
    cf_results = closed_form_equations(config)

    if not resume_flag:
        show_images(config, training_set, num_samples=15, cols=5)

        # get an image to simulate forward diffusion
        image = training_set[0]

        # plot forward diffusion
        plot_diffusion_process(
            config,
            [
                get_noisy_image(config, image, torch.tensor([t]), cf_results)
                for t in config.plot_time_steps
            ],
            file_name="forward_process.png",
        )

    auto_train(config, training_set, cf_results)


if __name__ == "__main__":
    set_seed()
    parse_cmdline_arguments()
