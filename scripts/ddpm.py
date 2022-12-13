#!/usr/bin/env python
# coding: utf-8
import os
import sys


sys.path.append("/space/calico/1/users/Harsha/ddpm-labels")
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from ext.lab2im.utils import load_volume
from ext.mindboggle.labels import extract_numbers_names_colors
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms
from matplotlib import colors

from model import SimpleUnet
from utils import load_file_list


# iterator dataset (for use with pathlib.Path generator as it is quick)
class DDPMLabelsIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self):
        self.files = pathlib.Path(
            "/cluster/vxmdata1/FS_Slim/proc/cleaned"
        ).glob("*/aseg_23*.mgz")
        self.n_files = len(self.files)

    def __iter__(self):
        self.source = iter(self.data)
        for _, item in enumerate(self.source):
            vol = load_volume(item)
            resized_vol = torch.unsqueeze(
                torch.Tensor(vol.astype(np.uint8)), -1
            )
            yield resized_vol


# Dataset class for use with list(pathlib.Path). This is really slow
class DDPMLabelsDataset(torch.utils.data.Dataset):
    def __init__(self, files):
        self.files = files
        self.n_files = len(self.files)

    def __getitem__(self, index):
        vol = load_volume(self.files[index])
        resized_vol = torch.unsqueeze(torch.Tensor(vol.astype(np.uint8)), 0)
        return resized_vol

    def __len__(self):
        return self.n_files


def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)


def get_index_from_list(vals, t, x_shape):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def forward_diffusion_sample(x_0, t, device="cpu"):
    """
    Takes an image and a timestep as input and
    returns the noisy version of it
    """
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(
        sqrt_alphas_cumprod, t, x_0.shape
    )
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(
        device
    ) + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(
        device
    )


def load_transformed_dataset():
    data_transforms = [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # Scales data into [0,1]
        # transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1]
    ]
    data_transform = transforms.Compose(data_transforms)

    # train = torchvision.datasets.StanfordCars(root=".", download=True,
    #                                      transform=data_transform)

    # test = torchvision.datasets.StanfordCars(root=".", download=True,
    #                                      transform=data_transform, split='test')
    return torch.utils.data.ConcatDataset([training_set, validation_set])


def get_loss(model, x_0, t):
    x_noisy, noise = forward_diffusion_sample(
        x_0, t, device="cuda" if torch.cuda.is_available() else "cpu"
    )
    noise_pred = model(x_noisy, t)
    return F.l1_loss(noise, noise_pred)


@torch.no_grad()
def sample_timestep(x, t):
    """
    Calls the model to predict the noise in the image and returns
    the denoised image.
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


def color_map_for_data():
    _, fs_names, fs_colors = extract_numbers_names_colors(
        "/usr/local/freesurfer/dev/FreeSurferColorLUT.txt"
    )

    with open("/cluster/vxmdata1/FS_Slim/proc/readme", "r") as f:
        voxmorph_label_index = f.read().splitlines()

    voxmorph_label_index = [
        item.strip().split(":") for item in voxmorph_label_index if item != ""
    ][
        -24:
    ]  # HACK
    voxmorph_label_index = [
        [int(item[0]), item[1].strip()] for item in voxmorph_label_index
    ]
    voxmorph_label_index_dict = dict(voxmorph_label_index)
    my_colors = [
        fs_colors[fs_names.index(item)]
        for item in voxmorph_label_index_dict.values()
    ]
    cmap = colors.ListedColormap(np.array(my_colors) / 255)
    # plt.imshow(np.arange(max(voxmorph_label_index_dict.keys()))[None], cmap=cmap)

    return cmap


# display a few images to check the label maps
def show_images(data, num_samples=20, cols=4):
    """Plots some samples from the dataset"""
    plt.figure(figsize=(10, 10))
    plt.tight_layout()
    for i, img in enumerate(data):
        if i == num_samples:
            break
        plt.subplot(num_samples // cols + 3, cols, i + 1)
        plt.imshow(img[0], cmap=color_map_for_data(), interpolation="nearest")
        plt.xticks([])
        plt.yticks([])
    plt.subplots_adjust(wspace=0.025, hspace=0.025)
    save_file = os.path.join(
        "/space/calico/1/users/Harsha/ddpm-labels/output", "sample_labels.png"
    )
    plt.savefig(save_file, bbox_inches="tight")


def show_tensor_image(image, save=0):
    reverse_transforms = transforms.Compose(
        [
            # transforms.Lambda(lambda t: (t + 1) / 2),
            # transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
            # transforms.Lambda(lambda t: t * 255.),
            transforms.Lambda(lambda t: np.squeeze(t.numpy().astype(np.uint8))),
            transforms.ToPILImage(mode="L"),
        ]
    )

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    plt.imshow(
        np.squeeze(reverse_transforms(image)),
        cmap=color_map_for_data(),
        interpolation="nearest",
    )
    plt.subplots_adjust(wspace=0.025)
    if save:
        save_file = os.path.join(
            "/space/calico/1/users/Harsha/ddpm-labels/output",
            "forward-preocess.png",
        )
        plt.savefig(save_file, bbox_inches="tight")


@torch.no_grad()
def sample_plot_image(device, epoch):
    # Sample noise
    img_size = IMG_SIZE
    img = torch.randn((1, 1, img_size, img_size), device=device)
    plt.figure(figsize=(15, 15))
    num_images = 10
    stepsize = int(T / num_images)

    for i in range(0, T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(img, t)
        if i % stepsize == 0:
            plt.subplot(1, num_images, i // stepsize + 1)
            plt.tick_params(
                axis="both",  # changes apply to the x-axis
                which="both",  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                left=False,
                right=False,
                labelbottom=False,
                labelleft=False,
            )  # labels along the bottom edge are off
            show_tensor_image(img.detach().cpu(), save=0)
    plt.subplots_adjust(wspace=0.025)
    save_file = os.path.join(
        "/space/calico/1/users/Harsha/ddpm-labels/output",
        f"reverse-image_{epoch:03d}.png",
    )
    plt.savefig(save_file, bbox_inches="tight")


if __name__ == "__main__":
    os.makedirs(
        "/space/calico/1/users/Harsha/ddpm-labels/output", exist_ok=True
    )

    IMG_SIZE, BATCH_SIZE, T, EPOCHS = 256, 16, 300, 200

    params = {
        "batch_size": BATCH_SIZE,
        "shuffle": True,
        "num_workers": 1,
        "worker_init_fn": np.random.seed(42),
    }

    filename = load_file_list()[:128]  # HACK

    partition = {}
    partition["train"], partition["validation"] = train_test_split(
        filename, test_size=0.5, random_state=42
    )

    # Generators
    training_set = DDPMLabelsDataset(partition["train"])
    training_generator = data.DataLoader(training_set, **params)

    validation_set = DDPMLabelsDataset(partition["validation"])
    validation_generator = data.DataLoader(validation_set, **params)

    show_images(training_set, num_samples=14, cols=5)

    # Define beta schedule
    betas = linear_beta_schedule(timesteps=T)

    # Pre-calculate different terms for closed form
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    posterior_variance = (
        betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    )

    transformed_data = load_transformed_dataset()
    dataloader = DataLoader(
        training_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )

    # Simulate forward diffusion
    image = next(iter(dataloader))

    plt.figure(figsize=(15, 15))
    num_images = 10
    stepsize = int(T / num_images)

    for idx in range(0, T, stepsize):
        t = torch.Tensor([idx]).type(torch.int64)
        plt.subplot(1, num_images + 1, (idx // stepsize) + 1)
        plt.tick_params(
            axis="both",
            which="both",  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False,
        )  # labels along the bottom edge are off
        image, noise = forward_diffusion_sample(image, t)
        show_tensor_image(image, save=1)

    model = SimpleUnet()
    print("Num params: ", sum(p.numel() for p in model.parameters()))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    epochs = 200  # Try more!

    for epoch in range(EPOCHS):
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()

            t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
            loss = get_loss(model, batch, t)
            loss.backward()
            optimizer.step()

            if epoch % 25 == 0 and step == 0:
                print(
                    f"Epoch {epoch:03d} | step {step:03d} Loss: {loss.item():0.5f} "
                )
                sample_plot_image(device, epoch)
