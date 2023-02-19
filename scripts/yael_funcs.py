from warnings import warn

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colors

from ext.mindboggle.labels import extract_numbers_names_colors


def softmax0(logit0):
    """(K-1) logits -> K probabilities"""
    logit = torch.zeros([logit0.shape[0] + 1, *logit0.shape[1:]])
    logit[1:] = logit0
    return logit.softmax(0)


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


def _get_colormap_cat(colormap, nb_classes, dtype=None, device=None):
    if colormap is None:
        if not plt:
            raise ImportError("Matplotlib not available")
        if nb_classes <= 10:
            colormap = plt.get_cmap("tab10")
        elif nb_classes <= 20:
            colormap = plt.get_cmap("tab20")
        else:
            warn(
                "More than 20 classes: multiple classes will share"
                "the same color."
            )
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
    colormap = _get_colormap_cat(
        colormap, nb_classes, image.dtype, image.device
    )

    cimage = image.new_zeros([*batch, *shape, 3])
    for i in range(nb_classes):
        cimage += image[..., i, :, :, None] * colormap[i % len(colormap)]

    return cimage.clamp_(0, 1)
