import ast
import json
import os
from datetime import datetime

import numpy as np
import torch
from ext.numpyencoder import NumpyEncoder
from torch.utils.tensorboard import SummaryWriter


class Configuration:
    """
    This configuration object is a collection of all variables relevant to the analysis
    """

    def __init__(self, args):
        now = datetime.now()
        self.dir_flag = now.strftime("%Y%m%d") + "-" + args.results_dir  # -%H%M%S
        self.logdir = os.path.join(
            "/space/calico/1/users/Harsha/ddpm-labels/logs", self.dir_flag
        )

        if not os.path.isdir(self.logdir):
            os.makedirs(self.logdir)
        self.writer = SummaryWriter(self.logdir)

        self.EPOCHS = args.epochs
        self.BATCH_SIZE = 128
        self.T = args.time_steps
        self.IMG_SIZE = args.image_size
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_idx = args.model_idx
        self.jei_flag = args.jei_flag
        self.group_labels = args.group_labels  # see group_labels() in utils.py
        self.beta_schedule = args.beta_schedule
        # "cosine" | "linear" | "quadratic" | "sigmoid"

        self.save_images = True
        self.save_checkpoint = True

        if self.jei_flag and self.group_labels:
            self.image_channels = 4

        if self.jei_flag and not self.group_labels:
            self.image_channels = 24

        if not self.jei_flag and self.group_labels:
            self.image_channels = 3

        if not self.jei_flag and not self.group_labels:
            self.image_channels = 23

        self.learning_rate = args.learning_rate
        self.loss_type = args.loss_type
        # "l1" | "l2" | "huber"

        self.plot_time_steps = list(np.arange(0, self.T, 100)) + [self.T - 1]

        self._write_config()

        self.DEBUG = False
        if self.DEBUG:
            self.image_channels = 1

    def _write_config(self, file_name=None):
        """Write configuration to a file
        Args:
            CONFIG (dict): configuration
        """
        file_name = "config.json" if file_name is None else file_name

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


if __name__ == "__main__":
    config = Configuration()
