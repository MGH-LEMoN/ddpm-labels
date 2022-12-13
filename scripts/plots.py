from warnings import warn

import matplotlib.pyplot as plt
import torch
from matplotlib import colors as mcolors


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
        colormap = [colormap(i / (n - 1))[:3] for i in range(n)]
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

    *batch, nb_classes, height, width = image.shape
    shape = (height, width)
    colormap = _get_colormap_cat(
        colormap, nb_classes, image.dtype, image.device
    )

    cimage = image.new_zeros([*batch, *shape, 3])
    for i in range(nb_classes):
        cimage += image[..., i, :, :, None] * colormap[i % len(colormap)]

    return cimage.clamp_(0, 1)
