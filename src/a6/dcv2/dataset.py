# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found
# [here](https://github.com/facebookresearch/swav/blob/06b1b7cbaf6ba2a792300d79c7299db98b93b7f9/LICENSE)  # noqa: E501
#
import copy
import logging
import pathlib
from collections.abc import Iterable

import torch
import torchvision.datasets
import torchvision.transforms
import xarray as xr

import a6.datasets.coordinates as _coordinates
import a6.datasets.methods.normalization as normalization
import a6.datasets.methods.transform as transform

logger = logging.getLogger(__name__)

SizeCropsRelative = list[float | tuple[float, float]]
SizeCropsSpecific = list[int | tuple[int, int]]


class Base(torchvision.datasets.VisionDataset):
    _n_channels: int

    @property
    def n_channels(self) -> int:
        return self._n_channels


class MultiCropDataset(Base, torchvision.datasets.ImageFolder):
    def __init__(
        self,
        data_path: pathlib.Path,
        size_crops: SizeCropsRelative,
        nmb_crops: list[int],
        min_scale_crops: list[float],
        max_scale_crops: list[float],
        size_dataset: int = -1,
        return_index: bool = False,
    ):
        super().__init__(data_path.as_posix())
        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)
        if size_dataset >= 0:
            self.samples = self.samples[:size_dataset]
        self.return_index = return_index

        path, _ = self.samples[0]
        image = self.loader(path)
        size_x, size_y = image.size
        size_crops = _convert_relative_to_specific_crop_size(
            size_crops, size_x=size_x, size_y=size_y
        )

        self._n_channels = 3

        # TODO: Calculate mean and std on-the-fly from ``self.samples``
        # E.g. via running mean/std, see https://stackoverflow.com/a/17637351
        mean = [0.485, 0.456, 0.406]
        std = [0.228, 0.224, 0.225]

        trans = []
        for i in range(len(size_crops)):
            trans.extend(
                [
                    torchvision.transforms.Compose(
                        [
                            torchvision.transforms.RandomResizedCrop(
                                size_crops[i],
                                scale=(min_scale_crops[i], max_scale_crops[i]),
                            ),
                            # In original DCv2 paper, the following data
                            # augmentation strategies are implemented
                            # torchvision.transforms.RandomHorizontalFlip(p=0.5),
                            # transforms.Compose(
                            #     [
                            #         transform.color_distortion(),
                            #         transform.PILRandomGaussianBlur(),
                            #     ]
                            # ),
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize(
                                mean=mean, std=std
                            ),
                        ]
                    )
                ]
                * nmb_crops[i]
            )
        self.trans = trans

    def __getitem__(self, index):
        path, _ = self.samples[index]
        image = self.loader(path)
        multi_crops = list(map(lambda trans: trans(image), self.trans))
        if self.return_index:
            return index, multi_crops
        return multi_crops


class MultiCropXarrayDataset(Base, torchvision.datasets.VisionDataset):
    def __init__(
        self,
        data_path: pathlib.Path,
        dataset: xr.Dataset,
        nmb_crops: list[int],
        size_crops: SizeCropsRelative,
        min_scale_crops: list[float],
        max_scale_crops: list[float],
        return_index: bool = False,
        coordinates: _coordinates.Coordinates = _coordinates.Coordinates(),
    ):
        super().__init__(data_path.as_posix())

        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)

        size_crops = _convert_relative_to_specific_crop_size(
            size_crops,
            size_x=dataset[coordinates.latitude].size,
            size_y=dataset[coordinates.longitude].size,
        )

        if (n_levels := dataset[coordinates.level].size) > 1:
            raise ValueError(
                "Only training with single-level data supported, but data "
                f"contain {n_levels} levels ({dataset[coordinates.level]})"
            )
        self.dataset = dataset
        self._coordinates = coordinates
        self._n_channels = len(dataset.data_vars)

        self.return_index = return_index

        mean = [
            normalization.calculate_mean(self.dataset, variable=variable)
            for variable in self.dataset.data_vars
        ]
        std = [
            normalization.calculate_std(self.dataset, variable=variable)
            for variable in self.dataset.data_vars
        ]

        min_max_values = normalization.get_min_max_values(self.dataset)

        trans = []
        for i in range(len(size_crops)):
            trans.extend(
                [
                    torchvision.transforms.Compose(
                        [
                            torchvision.transforms.RandomResizedCrop(
                                size_crops[i],
                                scale=(min_scale_crops[i], max_scale_crops[i]),
                            ),
                            # In original DCv2 paper, the following data
                            # augmentation strategies are implemented
                            # torchvision.transforms.RandomHorizontalFlip(p=0.5),
                            # torchvision.transforms.Compose(
                            #     [
                            #         transform.color_distortion(),
                            #         transform.PILRandomGaussianBlur(),
                            #     ]
                            # ),
                            # torchvision.transforms.ToTensor(),
                            torchvision.transforms.Compose(
                                [transform.MinMaxScale(min_max=min_max_values)]
                            ),
                            torchvision.transforms.Normalize(
                                mean=mean, std=std
                            ),
                        ]
                    )
                ]
                * nmb_crops[i]
            )
        self.trans = trans

    def __len__(self) -> int:
        return len(self.dataset[self._coordinates.time])

    def __getitem__(
        self, index: int
    ) -> list[torch.Tensor] | tuple[int, list[torch.Tensor]]:
        sample = (
            self.dataset.isel({self._coordinates.time: index})
            .to_array()
            .to_numpy()
        )
        image = torch.from_numpy(sample)
        multi_crops = list(map(lambda trans: trans(image), self.trans))
        if self.return_index:
            return index, multi_crops
        return multi_crops


def _convert_relative_to_specific_crop_size(
    size_crops: SizeCropsRelative, size_x: int, size_y: int
) -> SizeCropsSpecific:
    for size in size_crops:
        if not isinstance(size, float | Iterable):
            raise ValueError(
                f"Crop size must be float or tuple[float], "
                f"but {type(size)} given"
            )

        if (
            isinstance(size, float)
            and not 1.0 >= size >= 0.0
            or (
                isinstance(size, Iterable)
                and not all(1.0 >= s >= 0.0 for s in size)
            )
        ):
            raise ValueError(
                f"Crop size must be in the range [0; 1], but {size} given"
            )

    size_crops_old = copy.deepcopy(size_crops)
    if isinstance(size_crops[0], tuple):
        size_crops = [
            (int(scale_lat * size_x), int(scale_lon * size_y))
            for scale_lat, scale_lon in size_crops_old
        ]
    else:
        max_crop_size = min(size_x, size_y)
        size_crops = [int(scale * max_crop_size) for scale in size_crops_old]
    logger.warning(
        "Converted relative crop sizes %s to specific crop sizes %s",
        size_crops_old,
        size_crops,
    )

    return size_crops
