import json
import os
from collections import namedtuple
from datetime import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


class Configuration:
    """
    This configuration object is a collection of all variables relevant to the analysis
    """

    def __init__(self):

        now = datetime.now()
        dir_flag = now.strftime("%Y%m%d")  # -%H%M%S
        self.logdir = os.path.join(
            "/space/calico/1/users/Harsha/ddpm-labels/logs", dir_flag
        )

        if not os.path.isdir(self.logdir):
            os.makedirs(self.logdir)
        self.writer = SummaryWriter(self.logdir)

        self.EPOCHS = 1000
        self.BATCH_SIZE = 64
        self.T = 500
        self.IMG_SIZE = (192, 224)
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        self.params = {
            "batch_size": self.BATCH_SIZE,
            "shuffle": True,
            "num_workers": 0,
            "worker_init_fn": np.random.seed(42),
        }

        self.jei_flag = True
        self.beta_schedule = "linear"
        # "cosine" | "linear" | "quadratic" | "sigmoid"

        if self.jei_flag:
            self.image_channels = 24
        else:
            self.image_channels = 23

        self.learning_rate = 1e-5
        self.loss_type = "l2"
        # "l1" | "l2" | "huber"

    def _write_config(self, file_name=None):
        """Write configuration to a file
        Args:
            CONFIG (dict): configuration
        """
        file_name = "config.json" if file_name is None else file_name

        dictionary = self.__dict__
        json_object = json.dumps(dictionary, sort_keys=True, indent=4)

        config_file = os.path.join(dictionary["logdir"], file_name)

        with open(config_file, "w") as outfile:
            outfile.write(json_object)


if __name__ == "__main__":
    config = Configuration()