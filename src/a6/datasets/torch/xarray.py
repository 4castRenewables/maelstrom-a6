import logging
import pathlib

import torch
import xarray as xr

import a6.datasets.coordinates as _coordinates
import a6.datasets.methods as methods
import a6.datasets.torch._base as _base
import a6.datasets.transforms as transforms
import a6.datasets.variables as _variables

logger = logging.getLogger(__name__)


class Default(_base.Base):
    def __init__(
        self,
        data_path: pathlib.Path,
        dataset: xr.Dataset,
        return_index: bool = False,
        coordinates: _coordinates.Coordinates = _coordinates.Coordinates(),
    ):
        super().__init__(
            data_path=data_path,
            return_index=return_index,
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

        self.transforms = transforms.xarray.default(
            mean=mean,
            std=std,
            to_tensor=False,
        )

    def __len__(self) -> int:
        return len(self.dataset[self._coordinates.time])

    def __getitem__(
        self, index: int
    ) -> torch.Tensor | tuple[int, torch.Tensor]:
        sample = transforms.xarray.concatenate_levels_to_channels(
            self.dataset,
            time_index=index,
            levels=self._levels,
            coordinates=self._coordinates,
        )

        transformed = self.transforms(sample)

        if self.return_index:
            return index, transformed
        return transformed


class WithGWLTarget(Default):
    def __init__(
        self,
        data_path: pathlib.Path,
        weather_dataset: xr.Dataset,
        gwl_dataset: xr.Dataset,
        coordinates: _coordinates.Coordinates = _coordinates.Coordinates(),
        variables: _variables.GWL = _variables.GWL(),
    ):
        super().__init__(
            data_path=data_path,
            dataset=weather_dataset,
            return_index=True,
            coordinates=coordinates,
        )

        self.gwl_dataset = gwl_dataset
        self._variables = variables

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        index, sample = super().__getitem__(index=index)
        time_index = self.dataset[self._coordinates.time].isel(
            {self._coordinates.time: index}
        )
        # `method="ffill"` selects closest backwards timestep
        gwl = self.gwl_dataset.sel(
            {self._coordinates.time: time_index}, method="ffill"
        )[self._variables.gwl]
        return sample, int(gwl)
