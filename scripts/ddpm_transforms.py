import torch
import torchvision.transforms as T


class NoneTransform(object):
    """Does nothing to the image, to be used instead of None

    Args:
        image in, image out, nothing is done
    """

    def __call__(self, image):
        return image


rgb_transform = T.Compose(
    [
        T.ToTensor(),
        T.Lambda(lambda t: (t - t.min()) / (t.max() - t.min())),
        T.Lambda(lambda t: (t * 2) - 1),
    ]
)

reverse_rgb_transform = T.Compose(
    [
        T.Lambda(lambda t: (t + 1) / 2),
        T.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        T.Lambda(lambda t: t * 255.0),
        T.Lambda(lambda t: t.to(torch.uint8)),
    ]
)
