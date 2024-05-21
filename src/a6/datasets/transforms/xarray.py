import numpy as np
import torch
import torchvision
import xarray as xr

import a6.datasets.coordinates as _coordinates
import a6.datasets.methods as methods


def default(
    mean: list[float],
    std: list[float],
    min_max_values: list[methods.normalization.VariableMinMax] | None = None,
    to_tensor: bool = True,
) -> torchvision.transforms.Compose:
    """Create transforms to min-max scale and normalize."""
    return torchvision.transforms.Compose(
        list(
            filter(
                None,
                [
                    torchvision.transforms.ToTensor() if to_tensor else None,
                    (
                        methods.transform.MinMaxScale(min_max=min_max_values)
                        if min_max_values is not None
                        else None
                    ),
                    torchvision.transforms.Normalize(mean=mean, std=std),
                ],
            )
        )
    )


def concatenate_levels_to_channels(
    data: xr.Dataset,
    time_index: int,
    levels: list[int],
    coordinates: _coordinates.Coordinates = _coordinates.Coordinates(),
) -> torch.Tensor:
    """Concatenate all levels to appear as input channels."""
    time_step = data.isel({coordinates.time: time_index})
    without_nans = methods.mask.set_nans_to_mean(
        time_step, coordinates=coordinates
    )

    if len(levels) == 1:
        # If only single level given, argument for `level` to `xr.Dataset.sel`
        # must be a single integer, otherwise the data will have an additional
        # dimension.
        result = (
            without_nans.sel({coordinates.level: levels[0]})
            .to_array()
            .to_numpy()
        )
    else:
        result = np.concatenate(
            [
                without_nans.sel({coordinates.level: level})
                .to_array()
                .to_numpy()
                for level in levels
            ]
        )

    if np.isnan(result).any():
        raise ValueError(
            f"Sample at index {time_index} ({time_step[coordinates.time]}) "
            f"has NaNs: {result}"
        )

    return torch.from_numpy(result)
