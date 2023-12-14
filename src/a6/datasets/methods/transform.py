import itertools
import logging
import random

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import ImageFilter

import a6.datasets.methods.normalization as normalization

logger = logging.getLogger(__name__)


class MinMaxScale:
    def __init__(self, min_max: list[normalization.VariableMinMax]):
        self.min_max = min_max

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        channels = [
            normalization.min_max_scale(channel, min_max=min_max)
            for channel, min_max in zip(x, itertools.cycle(self.min_max))
        ]
        return torch.stack(channels)


class PILRandomGaussianBlur:
    """
    Apply Gaussian Blur to the PIL image. Take the radius and probability of
    application as the parameter.
    This transform was used in SimCLR - https://arxiv.org/abs/2002.05709
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = np.random.rand() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


def color_distortion(strength: float = 1.0) -> transforms.Compose:
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(
        0.8 * strength, 0.8 * strength, 0.8 * strength, 0.2 * strength
    )
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    return transforms.Compose([rnd_color_jitter, rnd_gray])
