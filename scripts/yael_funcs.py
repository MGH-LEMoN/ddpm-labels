import glob
import os
import random
from warnings import warn

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import utils as my_utils

from ext.lab2im import utils
from ext.mindboggle.labels import extract_numbers_names_colors


def softmax_jei(logit):
    """K logits -> K probabilities"""
    return logit.softmax(0)


def softmax_yael(logit0):
    """(K-1) logits -> K probabilities"""
    logit = torch.zeros([logit0.shape[0] + 1, *logit0.shape[1:]])
    logit[1:] = logit0
    return logit.softmax(0)


def logit_yael(prob):
    """K probabilities -> (K-1) logits"""
    logit = prob.clamp_min(1e-8).log()
    logit = logit[1:] - logit[0]
    return logit


def rgb_map_for_data():
    _, fs_names, fs_colors = extract_numbers_names_colors(
        "/usr/local/freesurfer/dev/FreeSurferColorLUT.txt"
    )

    with open("/cluster/vxmdata1/FS_Slim/proc/readme", "r") as f:
        voxmorph_label_index = f.read().splitlines()

    # get the last 24 lines of the readme file (format--> id: name)
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
        fs_colors[fs_names.index(item)] for item in voxmorph_label_index_dict.values()
    ]

    return my_colors


def color_map_for_data():
    my_colors = rgb_map_for_data()
    cmap = mcolors.ListedColormap(np.array(my_colors) / 255)

    # fig = plt.figure()
    # plt.imshow(np.arange(max(voxmorph_label_index_dict.keys()))[None], cmap=cmap)
    # plt.show()

    return cmap


def _get_colormap_cat(colormap, nb_classes, dtype=None, device=None):
    if colormap is None:
        if not plt:
            raise ImportError("Matplotlib not available")
        if nb_classes <= 10:
            colormap = plt.get_cmap("tab10")
        elif nb_classes <= 20:
            colormap = plt.get_cmap("tab20")
        else:
            warn("More than 20 classes: multiple classes will share" "the same color.")
            colormap = plt.get_cmap("tab20")
    elif isinstance(colormap, str):
        colormap = plt.get_cmap(colormap)
    if isinstance(colormap, mcolors.Colormap):
        n = nb_classes
        colormap = [colormap(i)[:3] for i in range(n)]
    colormap = torch.as_tensor(colormap, dtype=dtype, device=device)
    return colormap


def prob_to_rgb(image, implicit=False, colormap=None):
    """Convert soft probabilities to an RGB image.
    Parameters
    ----------
    image : (*batch, K, H, W)
        A (batch of) 2D image, with categories along the 'K' dimension.
    implicit : bool, default=False
        Whether the background class is implicit.
        Else, the first class is assumed to be the background class.
    colormap : (K, 3) tensor or str, optional
        A colormap or the name of a matplotlib colormap.
    Returns
    -------
    image : (*batch, H, W, 3)
        A (batch of) RGB image.
    """

    if not implicit:
        image = image[..., 1:, :, :]

    # added by YB
    if not image.dtype.is_floating_point:
        image = image.to(torch.get_default_dtype())

    *batch, nb_classes, height, width = image.shape
    shape = (height, width)
    colormap = _get_colormap_cat(colormap, nb_classes, image.dtype, image.device)

    cimage = image.new_zeros([*batch, *shape, 3])
    for i in range(nb_classes):
        cimage += image[..., i, :, :, None] * colormap[i % len(colormap)]

    return cimage.clamp_(0, 1)


def image_to_logit(args, image):
    resized_vol = torch.Tensor(image.astype(np.uint8))

    # one-hot encode the label map and
    # HWC to CHW and add batch dimension
    resized_vol = torch.movedim(
        F.one_hot(resized_vol.to(torch.int64), num_classes=args.im_channels),
        -1,
        0,
    )

    if args.downsample:
        resized_vol = torch.unsqueeze(resized_vol, 0)
        resized_vol = resized_vol.to(torch.uint8)
        resized_vol = F.interpolate(
            resized_vol,
            scale_factor=0.5,
            mode="bilinear",
            align_corners=True,
            recompute_scale_factor=False,
            antialias=False,
        )
        resized_vol = torch.squeeze(resized_vol)
        resized_vol = torch.argmax(resized_vol, dim=0)
        resized_vol = torch.movedim(
            F.one_hot(resized_vol.to(torch.int64), num_classes=args.im_channels),
            -1,
            0,
        )

    if args.jei_flag:
        logit = resized_vol * 7 - 3.5
    else:
        logit = 7 * (resized_vol[1:] - resized_vol[0])

    return logit.float()


def logit_to_image(config, img):
    func = softmax_jei if config.jei_flag else softmax_yael
    return prob_to_rgb(func(img), implicit=True, colormap=color_map_for_data())


def plot_sample_label_map():
    # take a sample file and plot it
    file = random.choice(
        glob.glob(os.path.join(my_utils.DATA_DIR, "test-maps-padded", "*.mgz"))
    )
    curr_vol = utils.load_volume(file)
    plt.imshow(curr_vol, cmap=color_map_for_data(), interpolation="nearest")
    plt.show()

    return curr_vol


if __name__ == "__main__":
    curr_vol = plot_sample_label_map()

    # one-hot encode the label map
    output = F.one_hot(
        torch.Tensor(curr_vol.astype(np.uint8)).to(torch.long), num_classes=24
    )
    print(output.shape)

    # HWC to CHW
    output1 = output.permute(2, 0, 1)

    # add batch dimension
    output = output1.unsqueeze(0)

    rgb_output = prob_to_rgb(output, implicit=True, colormap=color_map_for_data())
    print(rgb_output.shape)
    fig2 = plt.figure()
    plt.imshow(rgb_output[0])

    plt.imshow(logit_to_image(image_to_logit(curr_vol, False), False))
    plt.show()

    plt.imshow(logit_to_image(image_to_logit(curr_vol, True), True))
    plt.show()
