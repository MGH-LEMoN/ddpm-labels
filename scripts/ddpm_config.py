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
        PRJCT_FOLDER = os.getcwd()

        self.logdir = args.logdir

        if not os.path.isabs(self.logdir):
            self.logdir = os.path.join(
                PRJCT_FOLDER,
                "logs",
                self.logdir,
            )

        if not os.path.isdir(self.logdir):
            os.makedirs(self.logdir)

        self.writer = SummaryWriter(self.logdir)

        self.epochs = args.epochs
        self.batch_size = None
        self.time_steps = args.time_steps
        self.im_size = args.im_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_idx = args.model_idx
        self.jei_flag = args.jei_flag
        self.group_labels = args.group_labels  # see group_labels() in utils.py
        self.beta_schedule = args.beta_schedule

        self.save_images = False
        self.save_checkpoint = True

        self.debug = args.debug

        if self.debug:
            self.im_channels = 1
        else:
            self.im_channels = 24 - 20 * self.group_labels - (1 - self.jei_flag)

            # if self.jei_flag and self.group_labels:
            #     self.im_channels = 4

            # if self.jei_flag and not self.group_labels:
            #     self.im_channels = 24

            # if not self.jei_flag and self.group_labels:
            #     self.im_channels = 3

            # if not self.jei_flag and not self.group_labels:
            #     self.im_channels = 23

        self.lr = args.lr
        self.loss_type = args.loss_type

        self.plot_time_steps = list(np.arange(0, self.time_steps, 100)) + [
            self.time_steps - 1
        ]

        if config_file_name:
            config_file_name = os.path.join(self.logdir, config_file_name)

        self._write_config(config_file_name)

        self.start_epoch = getattr(args, "start_epoch", 0)
        self.checkpoint = getattr(args, "checkpoint", None)

        self.sampling_batch_size = 16
        self.sampling_freq = 10
        self.checkpoint_freq = 10

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
