import pathlib

import numpy as np
import torch
import torch.nn.functional as F
from yael_funcs import image_to_logit

from ext.lab2im.utils import load_volume


# iterator dataset (for use with pathlib.Path generator as it is quick)
class DDPMLabelsIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self):
        self.files = pathlib.Path("/cluster/vxmdata1/FS_Slim/proc/cleaned").glob(
            "*/aseg_23*.mgz"
        )
        self.n_files = len(self.files)

    def __iter__(self):
        self.source = iter(self.data)
        for _, item in enumerate(self.source):
            vol = load_volume(item)
            resized_vol = torch.unsqueeze(torch.Tensor(vol.astype(np.uint8)), -1)
            yield resized_vol


# Dataset class for use with list(pathlib.Path). This is really slow
class DDPMLabelsDataset(torch.utils.data.Dataset):
    def __init__(self, files, jei_flag=False):
        self.files = files
        self.n_files = len(self.files)
        self.jei_flag = jei_flag

    def __getitem__(self, index):
        vol = load_volume(self.files[index])
        return image_to_logit(vol, self.jei_flag)

    def __len__(self):
        return self.n_files
