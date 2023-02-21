#!/usr/bin/env python
# coding: utf-8


import math
import os
from datetime import datetime
from functools import partial
from inspect import isfunction

# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from beta_schedule import linear_beta_schedule
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from torch import einsum, nn
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from scripts.model2 import Unet

now = datetime.now()

logdir = os.path.join("logs", now.strftime("%Y%m%d-%H%M%S"))

if not os.path.isdir(logdir):
    os.makedirs(logdir)
writer = SummaryWriter(logdir)


timesteps = 500

# define beta schedule
betas = linear_beta_schedule(timesteps=timesteps)

# define alphas
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = (
    betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
)


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


# from PIL import Image
# import requests

# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# image = Image.open(requests.get(url, stream=True).raw)
# image


# take a sample file and plot it
import glob
import os

import matplotlib.pyplot as plt
import utils as my_utils
from yael_funcs import color_map_for_data, image_to_logit

from ext.lab2im import edit_volumes, utils

files = sorted(
    glob.glob(os.path.join(my_utils.DATA_DIR, "test-maps-padded", "*.mgz"))
)
image = utils.load_volume(files[0])
# fig = plt.figure()
# plt.imshow(image, cmap=color_map_for_data(), interpolation="nearest")

# writer.add_figure('sample image', fig)
# writer.close()


# from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize

# image_size = 128
# transform = Compose([
#     Resize(image_size),
#     CenterCrop(image_size),
#     ToTensor(), # turn into Numpy array of shape HWC, divide by 255
#     Lambda(lambda t: (t * 2) - 1),

# ])

# x_start = transform(image).unsqueeze(0)
# x_start.shape

from torchvision.transforms import Compose, Lambda, ToTensor

transform = Compose(
    [
        Lambda(lambda t: image_to_logit(t)),
    ]
)

x_start = transform(image).unsqueeze(0)
x_start.shape


# import numpy as np

# reverse_transform = Compose([
#      Lambda(lambda t: (t + 1) / 2),
#      Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
#      Lambda(lambda t: t * 255.),
#      Lambda(lambda t: t.numpy().astype(np.uint8)),
#      ToPILImage(),
# ])

from yael_funcs import logit_to_image

reverse_transform = Compose(
    [
        Lambda(lambda t: logit_to_image(t)),
    ]
)


# plt.imshow(reverse_transform(x_start.squeeze()), interpolation="nearest")
# writer.add_figure('sample image', fig)
# writer.close()


# forward diffusion (using the nice property)
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return (
        sqrt_alphas_cumprod_t * x_start
        + sqrt_one_minus_alphas_cumprod_t * noise
    )


def get_noisy_image(x_start, t):
    # add noise
    x_noisy = q_sample(x_start, t=t)

    # turn back into PIL image
    noisy_image = reverse_transform(x_noisy.squeeze())

    return noisy_image


# take time step
t = torch.tensor([475])

_ = get_noisy_image(x_start, t)


import matplotlib.pyplot as plt

# use seed for reproducability
torch.manual_seed(0)

# source: https://pytorch.org/vision/stable/auto_examples/plot_transforms.html#sphx-glr-auto-examples-plot-transforms-py
def plot(imgs, with_orig=False, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(
        figsize=(200, 200), nrows=num_rows, ncols=num_cols, squeeze=False
    )
    for row_idx, row in enumerate(imgs):
        row = [image] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title="Original image")
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()


# plot(
#     [
#         get_noisy_image(x_start, torch.tensor([t]))
#         for t in [0, 100, 200, 300, 400, 499]
#     ]
# )
# writer.add_figure('forward process', fig)
# writer.close()


def p_losses(denoise_model, x_start, t, noise=None, loss_type="l1"):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    predicted_noise = denoise_model(x_noisy, t)

    if loss_type == "l1":
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == "l2":
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss


from datasets import load_dataset

# load dataset from the hub
# dataset = load_dataset("fashion_mnist")
image_size = (192, 224)
channels = 24
batch_size = 32

from ext.lab2im.utils import load_volume


class DDPMLabelsDataset(torch.utils.data.Dataset):
    def __init__(self, files):
        self.files = files
        self.n_files = len(self.files)

    def __getitem__(self, index):
        vol = load_volume(self.files[index])
        resized_vol = torch.Tensor(vol.astype(np.uint8))
        resized_vol = torch.movedim(
            F.one_hot(resized_vol.to(torch.int64), num_classes=24),
            -1,
            0,
        )

        ## YB-20230209
        ### # # K one hot -> (K-1) logits
        ref_onehot = resized_vol
        ref_logit = 7 * (ref_onehot[1:] - ref_onehot[0])
        logit = torch.zeros([ref_logit.shape[0] + 1, *ref_logit.shape[1:]])
        logit[1:] = ref_logit
        ## YB-20230209

        return logit.float()

    def __len__(self):
        return self.n_files


from torch.utils.data import DataLoader
from torchvision import transforms
from utils import load_labelmap_names

# # define image transformations (e.g. using torchvision)
# transform = Compose([
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Lambda(lambda t: (t * 2) - 1)
# ])

# # define function
# def transforms(examples):
#    examples["pixel_values"] = [transform(image.convert("L")) for image in examples["image"]]
#    del examples["image"]

#    return examples

# transformed_dataset = dataset.with_transform(transforms).remove_columns("label")

# # create dataloader
# dataloader = DataLoader(transformed_dataset["train"], batch_size=batch_size, shuffle=True)


filename = load_labelmap_names("ddpm_files_padded.txt")
train_dataset = DDPMLabelsDataset(filename)
dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


@torch.no_grad()
def p_sample(model, x, t, t_index):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise


# Algorithm 2 (including returning all images)
@torch.no_grad()
def p_sample_loop(model, shape):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(
        reversed(range(0, timesteps)),
        desc="sampling loop time step",
        total=timesteps,
    ):
        img = p_sample(
            model, img, torch.full((b,), i, device=device, dtype=torch.long), i
        )
        imgs.append(img.cpu())  # removed .numpy()
    return imgs


@torch.no_grad()
def sample(model, image_size, batch_size=16, channels=3):
    return p_sample_loop(model, shape=(batch_size, channels, *image_size))


from pathlib import Path


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


results_folder = Path("./results")
results_folder.mkdir(exist_ok=True)
save_and_sample_every = 5


from torch.optim import Adam

model = Unet(dim=16, channels=24, dim_mults=(2, 4, 8, 16, 32, 64))
optimizer = Adam(model.parameters(), lr=1e-3)

print("Num params: ", sum(p.numel() for p in model.parameters()))

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# torch.save(model.state_dict(), '/space/calico/1/users/Harsha/ddpm-labels/models/unet1')
# model2 = Unet(
#     dim=16,
#     channels=24,
#     dim_mults=(2, 4, 8, 16, 32)
# )
# model2.load_state_dict(torch.load('/space/calico/1/users/Harsha/ddpm-labels/models/unet1'))
# optimizer2 = Adam(model2.parameters(), lr=1e-3)

# from torchvision.utils import save_image

epochs = 1000

for epoch in range(1, epochs + 1):
    epoch_loss = 0
    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()

        # batch_size = batch["pixel_values"].shape[0]
        # batch = batch["pixel_values"].to(device)

        batch_size = batch.shape[0]
        batch = batch.to(device)

        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        t = torch.randint(0, timesteps, (batch_size,), device=device).long()

        loss = p_losses(model, batch, t, loss_type="huber")
        epoch_loss += loss.item() * batch_size

        loss.backward()
        optimizer.step()

        # save generated images
        # if step != 0 and step % save_and_sample_every == 0:
        #   milestone = step // save_and_sample_every
        #   batches = num_to_groups(4, batch_size)
        #   all_images_list = list(map(lambda n: sample(model, image_size, batch_size=n, channels=channels), batches))
        #   all_images = torch.cat(*all_images_list, dim=0)
        #   all_images = (all_images + 1) * 0.5
        #   save_image(all_images, str(results_folder / f'sample-{milestone}.png'), nrow = 6)

    print(f"Epoch: {epoch:04d} Loss: {loss.item():1.6f}")

    if epoch == 1 or epoch % 100 == 0:
        torch.save(
            model.state_dict(), os.path.join(logdir, f"model_{epoch:04d}")
        )

    writer.add_scalar("training_loss", epoch_loss / 64, epoch)

writer.close()


# # sample 64 images
# samples = sample(model, image_size=image_size, batch_size=batch_size, channels=channels)

# # # show a random one
# # random_index = 5
# # plt.imshow(samples[-1][random_index].reshape(image_size, image_size, channels), cmap="gray")

# def test_plot(img):
#     img = softmax0(img)
#     img = prob_to_rgb(img, implicit=False, colormap=color_map_for_data())
#     return img

# for i in range(batch_size):
#     plot([test_plot(samples[t][i]) for t in [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 499]])
