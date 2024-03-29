import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from ext import utils as ext_utils
from ext.numpyencoder import NumpyEncoder


class Configuration:
    """
    This configuration object is a collection of all variables relevant to the analysis
    """

    def __init__(self, args, config_file_name=None):
        self.logdir = getattr(args, "logdir", os.getcwd())

        if not os.path.isabs(self.logdir):
            self.logdir = os.path.join(
                os.getcwd(),
                "logs",
                self.logdir,
            )

        if not os.path.isdir(self.logdir):
            os.makedirs(self.logdir)

        self.writer = SummaryWriter(self.logdir)

        self.epochs = getattr(args, "epochs", 1000)
        self.batch_size = getattr(args, "batch_size", 32)
        self.time_steps = getattr(args, "time_steps", 300)
        self.im_size = getattr(args, "im_size", (28, 28))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.downsample = getattr(args, "downsample", False)

        self.rgb_flag = getattr(args, "rgb_flag", False)
        self.model_idx = getattr(args, "model_idx", 1)
        self.jei_flag = getattr(args, "jei_flag", 1)
        self.group_labels = getattr(
            args, "group_labels", 0
        )  # see group_labels() in utils.py
        self.beta_schedule = getattr(args, "beta_schedule", "linear")

        self.save_images = getattr(args, "save_images", False)
        self.save_checkpoint = getattr(args, "save_checkpoint", True)

        self.debug = getattr(args, "debug", 0)

        if self.debug:
            self.im_channels = 1
        elif self.rgb_flag:
            self.im_channels = 3
        else:
            # {idx: # of labels}
            # 0: no grouping; 1: WM/GM/CSF/BG; 2: merge left/right
            group_labels_flag_dict = {0: 24, 1: 4, 2: 14}
            self.im_channels = group_labels_flag_dict[self.group_labels] - (
                1 - self.jei_flag
            )

        self.lr = getattr(args, "lr", 1e-5)
        self.loss_type = getattr(args, "loss_type", "l2")

        self.plot_time_steps = list(np.arange(0, self.time_steps, 100)) + [
            self.time_steps - 1
        ]

        if sys.argv[1] == "train":
            file_name = "config.json"

        if sys.argv[1] == "resume-train":
            file_name = "config_resume.json"

        config_file_name = os.path.join(self.logdir, file_name)

        self.start_epoch = getattr(args, "start_epoch", 0)
        self.checkpoint = getattr(args, "checkpoint", None)

        self.sampling_batch_size = getattr(args, "sampling_batch_size", 16)
        self.sampling_freq = getattr(args, "sampling_freq", 10)
        self.checkpoint_freq = getattr(args, "checkpoint_freq", 10)

        self.COMMIT_HASH = ext_utils.get_git_revision_short_hash()
        self.CREATED_ON = f'{datetime.now().strftime("%A %m/%d/%Y %H:%M:%S")}'

        self.write_config(config_file_name)

    def write_config(self, file_name=None):
        """Write configuration to a file
        Args:
            CONFIG (dict): configuration
        """
        file_name = file_name if file_name else "config.json"

        dictionary = self.__dict__
        json_object = json.dumps(
            dictionary,
            sort_keys=True,
            indent=4,
            separators=(", ", ": "),
            ensure_ascii=False,
            cls=NumpyEncoder,
        )

        config_file = os.path.join(dictionary["logdir"], file_name)

        with open(config_file, "w") as outfile:
            outfile.write(json_object)

    @classmethod
    def read_config(self, file_name):
        """Read configuration from a file"""
        with open(file_name, "r") as fh:
            config_dict = json.load(fh)

        return argparse.Namespace(**config_dict)


if __name__ == "__main__":
    config = Configuration()
