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

    def __init__(self, args, config_file_name=None):
        now = datetime.now()
        if getattr(args, "dir_flag", None):
            self.dir_flag = args.dir_flag
        else:
            self.dir_flag = now.strftime("%Y%m%d") + "-" + args.results_dir  # -%H%M%S

        if getattr(args, "logdir", None):
            self.logdir = args.logdir
        else:
            self.logdir = os.path.join(
                "/space/calico/1/users/Harsha/ddpm-labels/logs", self.dir_flag
            )

        if not os.path.isdir(self.logdir):
            os.makedirs(self.logdir)
        self.writer = SummaryWriter(self.logdir)

        self.EPOCHS = args.EPOCHS
        self.BATCH_SIZE = 128
        self.T = args.T
        self.IMG_SIZE = args.IMG_SIZE
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_idx = args.model_idx
        self.jei_flag = args.jei_flag
        self.group_labels = args.group_labels  # see group_labels() in utils.py
        self.beta_schedule = args.beta_schedule
        # "cosine" | "linear" | "quadratic" | "sigmoid"

        self.save_images = False
        self.save_checkpoint = True

        self.DEBUG = True

        if self.DEBUG:
            self.image_channels = 1
        else:
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

        if config_file_name:
            config_file_name = os.path.join(self.logdir, config_file_name)

        self._write_config(config_file_name)

        self.start_epoch = getattr(args, "start_epoch", 0)
        self.checkpoint = getattr(args, "checkpoint", None)

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
