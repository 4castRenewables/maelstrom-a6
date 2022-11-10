from collections.abc import Iterator

import numpy as np
import xarray as xr

import a6.datasets.dimensions as _dimensions
import a6.modes.methods.pca.pca as pca


class Reshaper:
    """Reshapes given PCA data to the according dimensions."""

    def __init__(self, dimensions: _dimensions.SpatioTemporalDimensions):
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

        reshaped_entries = self._reshape_entries(
            data, includes_time_dimension=includes_time_dimension
        )
        return xr.Dataset(
            data_vars={str(da.name): da for da in reshaped_entries},
        )

    def _reshape_entries(
        self, data: xr.DataArray, includes_time_dimension: bool
    ) -> Iterator[xr.DataArray]:
        """Reshape the entries of a PCA.

        Notes
        -----
        Before performing a PCA, the gridded `(m x n)` data of each variable in
        a dataset is flattened and concatenated for each time step. For N
        variables and T time steps, this results in a `(T x m * n * N)` matrix.
        Hence, to get the PC of an individual variable `i of N` in the shape of
        the `(m x n)` grid, this can be undone by taking the
        `(i - 1) * (m * n)`th until the `i * (m * n)`th entry of a PC and
        reshaping it into the original grid shape `(m x n)`.

        """
        for name, slice_ in self._dimensions.to_variable_name_and_slices():
            values = data.sel({pca.PC_VALUES_DIM: slice_})
            if includes_time_dimension:
                yield self._reshape_spatio_temporal_entries(
                    data=values,
                    name=name,
                )
            else:
                yield self._reshape_spatial_entries(
                    data=values,
                    name=name,
                )

    def _reshape_spatio_temporal_entries(
        self, data: xr.DataArray, name: str
    ) -> xr.DataArray:
        reshaped = data.data.reshape(self._dimensions.shape())
        # The PCs are flipped along axis 1.
        return xr.DataArray(
            np.flip(reshaped, axis=1),
            dims=[
                pca.PC_DIM,
                *self._dimensions.spatial_dimension_names,
            ],
            name=name,
        )

    def _reshape_spatial_entries(
        self, data: xr.DataArray, name: str
    ) -> xr.DataArray:
        reshaped = data.data.reshape(
            self._dimensions.shape(include_time_dim=False)
        )
        # The PCs are flipped along axis 0.
        return xr.DataArray(
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
