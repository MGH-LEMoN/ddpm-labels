import pathlib

import numpy as np
import torch
from utils import group_labels, group_left_right
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
    def __init__(self, config, files):
        self.files = files
        self.n_files = len(self.files)
        self.config = config

    def __getitem__(self, index):
        vol = load_volume(self.files[index])
        if self.config.group_labels == 1:
            vol = np.vectorize(group_labels().get)(vol)

        if self.config.group_labels == 2:
            vol = np.vectorize(group_left_right().get)(vol)

        return image_to_logit(self.config, vol)

    def __len__(self):
        return self.n_files


class FashionMnistDataset:
    def __init__(self):
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

        transformed_set = dataset.with_transform(transforms).remove_columns("label")
        self.training_set = transformed_set["train"][:]["pixel_values"]
