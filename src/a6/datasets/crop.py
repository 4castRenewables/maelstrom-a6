# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found
# [here](https://github.com/facebookresearch/swav/blob/06b1b7cbaf6ba2a792300d79c7299db98b93b7f9/LICENSE)  # noqa: E501
#
import copy
import json
import logging
import pathlib
from collections.abc import Iterable
from typing import TypeAlias

import torch
import torchvision.datasets
import torchvision.transforms
import xarray as xr
from PIL import Image

import a6.datasets.coordinates as _coordinates
import a6.datasets.methods as methods
import a6.datasets.transforms as transforms

logger = logging.getLogger(__name__)

SizeCropsRelative: TypeAlias = list[float | tuple[float, float]]
SizeCropsSpecific: TypeAlias = list[int | tuple[int, int]]


class Base(torchvision.datasets.VisionDataset):
    _n_channels: int

    def __init__(
        self,
        data_path: pathlib.Path,
        nmb_crops: list[int],
        size_crops: SizeCropsRelative,
        min_scale_crops: list[float],
        max_scale_crops: list[float],
        return_index: bool = False,
    ):
        super().__init__(data_path.as_posix())

        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)

        self.return_index = return_index

    @property
    def n_channels(self) -> int:
        return self._n_channels


class MultiCropMnistDataset(Base, torchvision.datasets.VisionDataset):
    _n_channels = 1

    def __init__(
        self,
        data_path: pathlib.Path,
        size_crops: SizeCropsRelative,
        nmb_crops: list[int],
        min_scale_crops: list[float],
        max_scale_crops: list[float],
        return_index: bool = False,
        download: bool = True,
    ):
        if not data_path.exists():
            data_path.mkdir(parents=True, exist_ok=True)

        super().__init__(
            data_path=data_path,
            nmb_crops=nmb_crops,
            size_crops=size_crops,
            min_scale_crops=min_scale_crops,
            max_scale_crops=max_scale_crops,
            return_index=return_index,
        )

        self.dataset = torchvision.datasets.MNIST(
            root=data_path.as_posix(),
            train=True,
            download=download,
        )

        image, _ = self.dataset[0]
        size_x, size_y = image.size
        size_crops = convert_relative_to_absolute_crop_size(
            size_crops, size_x=size_x, size_y=size_y
        )

        self.transform = _create_transformations(
            nmb_crops=nmb_crops,
            size_crops=size_crops,
            min_scale_crops=min_scale_crops,
            max_scale_crops=max_scale_crops,
            mean=[0.1306],
            std=[0.3081],
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(
        self, index: int
    ) -> list[torch.Tensor] | tuple[list[torch.Tensor], int]:
        image, _ = self.dataset[index]
        multi_crops = list(map(lambda trans: trans(image), self.transform))
        if self.return_index:
            return multi_crops, index
        return multi_crops


class MultiCropImageNet(Base, torchvision.datasets.VisionDataset):
    def __init__(
        self,
        data_path: pathlib.Path,
        size_crops: SizeCropsRelative,
        nmb_crops: list[int],
        min_scale_crops: list[float],
        max_scale_crops: list[float],
        split: str = "train",
        return_index: bool = False,
    ):
        if not data_path.exists():
            data_path.mkdir(parents=True, exist_ok=True)

        super().__init__(
            data_path=data_path,
            nmb_crops=nmb_crops,
            size_crops=size_crops,
            min_scale_crops=min_scale_crops,
            max_scale_crops=max_scale_crops,
            return_index=return_index,
        )

        self.samples = []
        self.targets = []
        self.syn_to_class = {}

        self.transform = _create_transformations(
            nmb_crops=nmb_crops,
            size_crops=size_crops,
            min_scale_crops=min_scale_crops,
            max_scale_crops=max_scale_crops,
            # These are the mean and std for the RGB channels
            # of the ImageNet dataset.
            mean=[0.485, 0.456, 0.406],
            std=[0.228, 0.224, 0.225],
        )

        with open(data_path / "imagenet_class_index.json", "rb") as f:
            json_file = json.load(f)
            for class_id, v in json_file.items():
                self.syn_to_class[v[0]] = int(class_id)

        with open(data_path / "ILSVRC2012_val_labels.json", "rb") as f:
            self.val_to_syn = json.load(f)

        samples_dir = data_path / "ILSVRC/Data/CLS-LOC" / split

        for entry in samples_dir.glob("*"):
            if split == "train":
                syn_id = entry
                target = self.syn_to_class[syn_id]
                syn_folder = samples_dir / syn_id

                for sample in syn_folder.glob("*"):
                    sample_path = syn_folder / sample
                    self.samples.append(sample_path)
                    self.targets.append(target)
            elif split == "val":
                syn_id = self.val_to_syn[entry]
                target = self.syn_to_class[syn_id]
                sample_path = samples_dir / entry
                self.samples.append(sample_path)
                self.targets.append(target)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, index: int
    ) -> list[torch.tensor] | tuple[list[torch.tensor], int]:
        image = Image.open(self.samples[index]).convert("rgb")
        multi_crops = list(map(lambda trans: trans(image), self.trans))
        if self.return_index:
            return multi_crops, index
        return multi_crops


class MultiCropDataset(Base, torchvision.datasets.ImageFolder):
    _n_channels = 3

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
        super().__init__(
            data_path=data_path,
            nmb_crops=nmb_crops,
            size_crops=size_crops,
            min_scale_crops=min_scale_crops,
            max_scale_crops=max_scale_crops,
            return_index=return_index,
        )

        if size_dataset >= 0:
            self.samples = self.samples[:size_dataset]

        path, _ = self.samples[0]
        image = self.loader(path)
        size_x, size_y = image.size
        size_crops = convert_relative_to_absolute_crop_size(
            size_crops, size_x=size_x, size_y=size_y
        )

        # TODO: Calculate mean and std on-the-fly from ``self.samples``
        # E.g. via running mean/std, see https://stackoverflow.com/a/17637351
        mean = [0.485, 0.456, 0.406]
        std = [0.228, 0.224, 0.225]

        self.transform = _create_transformations(
            nmb_crops=nmb_crops,
            size_crops=size_crops,
            min_scale_crops=min_scale_crops,
            max_scale_crops=max_scale_crops,
            mean=mean,
            std=std,
            random_horizontal_flip=True,
            color_distortion=True,
            gaussian_blur=True,
        )

    def __getitem__(
        self, index: int
    ) -> list[torch.Tensor] | tuple[list[torch.Tensor], int]:
        path, _ = self.samples[index]
        image = self.loader(path)
        multi_crops = list(map(lambda trans: trans(image), self.transform))
        if self.return_index:
            return multi_crops, index
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
        super().__init__(
            data_path=data_path,
            nmb_crops=nmb_crops,
            size_crops=size_crops,
            min_scale_crops=min_scale_crops,
            max_scale_crops=max_scale_crops,
            return_index=return_index,
        )

        size_crops = convert_relative_to_absolute_crop_size(
            size_crops,
            size_x=dataset[coordinates.latitude].size,
            size_y=dataset[coordinates.longitude].size,
        )

        self.dataset = dataset
        # For single-level, levels is 0-D array and has to
        # be converted to a list
        levels = self.dataset[coordinates.level].to_numpy().tolist()
        self._levels = levels if isinstance(levels, list) else [levels]

        self._coordinates = coordinates
        self._n_channels = len(dataset.data_vars) * len(self._levels)

        self.return_index = return_index

        if "mean" in self.dataset.attrs:
            mean = self.dataset.attrs["mean"]
            logger.info("Reading mean from dataset attribute 'mean': %s", mean)
        else:
            logger.info("Calculating mean from dataset")
            mean = methods.statistics.get_statistics(
                self.dataset,
                method=methods.normalization.calculate_mean,
                levels=self._levels,
                coordinates=self._coordinates,
            )

        if "std" in self.dataset.attrs:
            std = self.dataset.attrs["std"]
            logger.info(
                "Reading standard deviations from dataset attribute 'std': %s",
                std,
            )
        else:
            logger.info("Calculating std from dataset")
            std = methods.statistics.get_statistics(
                self.dataset,
                method=methods.normalization.calculate_std,
                levels=self._levels,
                coordinates=self._coordinates,
            )

        logger.info("Calculated mean %s and standard deviation %s", mean, std)

        self.transforms = _create_transformations(
            nmb_crops=nmb_crops,
            size_crops=size_crops,
            min_scale_crops=min_scale_crops,
            max_scale_crops=max_scale_crops,
            mean=mean,
            std=std,
            to_tensor=False,
        )

    def __len__(self) -> int:
        return len(self.dataset[self._coordinates.time])

    def __getitem__(
        self, index: int
    ) -> list[torch.Tensor] | tuple[list[torch.Tensor], int]:
        sample = transforms.xarray.concatenate_levels_to_channels(
            self.dataset,
            time_index=index,
            levels=self._levels,
            coordinates=self._coordinates,
        )
        multi_crops = list(
            map(lambda transform: transform(sample), self.transforms)
        )

        if self.return_index:
            return multi_crops, index
        return multi_crops


def convert_relative_to_absolute_crop_size(
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
        "Converted relative crop sizes %s to absolute crop sizes %s",
        size_crops_old,
        size_crops,
    )

    return size_crops


def _create_transformations(
    nmb_crops: list[int],
    size_crops: list[float | tuple[float, float]],
    min_scale_crops: list[float],
    max_scale_crops: list[float],
    mean: list[float],
    std: list[float],
    min_max_values: list[methods.normalization.VariableMinMax] | None = None,
    to_tensor: bool = True,
    random_horizontal_flip: bool = False,
    color_distortion: bool = False,
    gaussian_blur: bool = False,
) -> list[torchvision.transforms.Compose]:
    return [
        # Define transform pipeline
        torchvision.transforms.Compose(
            list(
                filter(
                    None,
                    [
                        torchvision.transforms.RandomResizedCrop(
                            size,
                            scale=(min_scale, max_scale),
                            antialias=True,
                        ),
                        ###
                        # In original DCv2 paper, the following data
                        # augmentation strategies are implemented
                        (
                            torchvision.transforms.RandomHorizontalFlip(p=0.5)
                            if random_horizontal_flip
                            else None
                        ),
                        (
                            methods.transform.color_distortion()
                            if color_distortion
                            else None
                        ),
                        (
                            methods.transform.PILRandomGaussianBlur()
                            if gaussian_blur
                            else None
                        ),
                        ###
                        (
                            torchvision.transforms.ToTensor()
                            if to_tensor
                            else None
                        ),
                        (
                            methods.transform.MinMaxScale(
                                min_max=min_max_values
                            )
                            if min_max_values is not None
                            else None
                        ),
                        torchvision.transforms.Normalize(mean=mean, std=std),
                    ],
                )
            )
        )
        for n_crops, size, min_scale, max_scale in zip(
            nmb_crops, size_crops, min_scale_crops, max_scale_crops, strict=True
        )
        # Repeat transform for `n_crops` to achieve the given number of crops
        for _ in range(n_crops)
    ]
