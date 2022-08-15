import typing as t

import lifetimes.modes.methods.pca.pca as pca
import lifetimes.utils as utils
import numpy as np
import xarray as xr


class Reshaper:
    """Reshapes given PCA data to the according dimensions."""

    def __init__(self, dimensions: utils.dimensions.Dimensions):
        self._dimensions = dimensions

    def __call__(
        self,
        data: xr.DataArray,
        includes_time_dimension: bool = True,
        rename_dim_0: bool = False,
    ) -> xr.Dataset:
        """Reshape the given data.

        Parameters
        ----------
        data : xr.DataArray
            Data to reshape.
        includes_time_dimension : bool, default=True
            Whether the data include a temporal dimension.
            This has an effect on how the data have to be reshaped.
        rename_dim_0 : bool, default=False
            Whether the 0th dimension should be renamed to `entry`.

        """
        if rename_dim_0:
            data = _rename_1d_data_array_dimension(data)

        variables = self._vars_to_xarray(
            data, includes_time_dimension=includes_time_dimension
        )
        return xr.Dataset(
            data_vars={str(da.name): da for da in variables},
        )

    def _vars_to_xarray(
        self, data: xr.DataArray, includes_time_dimension: bool
    ) -> t.Iterator[xr.DataArray]:
        for name, slice_ in self._dimensions.to_variable_name_and_slices():
            values = data.sel({pca.PC_VALUES_DIM: slice_})
            if includes_time_dimension:
                reshaped = values.data.reshape(self._dimensions.to_tuple())
                # The PCs are flipped along axis 1.
                yield xr.DataArray(
                    np.flip(reshaped, axis=1),
                    dims=[
                        pca.PC_DIM,
                        *self._dimensions.spatial_dimension_names,
                    ],
                    name=name,
                )
            else:
                reshaped = values.data.reshape(
                    self._dimensions.to_tuple(include_time_dim=False)
                )
                # The PCs are flipped along axis 0.
                yield xr.DataArray(
                    np.flip(reshaped, axis=0),
                    dims=self._dimensions.spatial_dimension_names,
                    name=name,
                )


def _rename_1d_data_array_dimension(data: xr.DataArray) -> xr.DataArray:
    try:
        [dim] = data.dims
    except ValueError:
        raise ValueError(f"Data has more than 1 dimension: {data.dims}")
    return data.rename({dim: pca.PC_VALUES_DIM})
