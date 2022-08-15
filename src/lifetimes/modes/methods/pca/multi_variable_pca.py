import typing as t

import lifetimes.modes.methods.pca.pca_abc as pca_abc
import lifetimes.modes.methods.pca.single_variable_pca as single_variable_pca
import numpy as np
import xarray as xr


class MultiVariablePCA(single_variable_pca.SingleVariablePCA):
    """Wrapper for `sklearn.decomposition.PCA` for multi-variable data."""

    def _to_original_shape(
        self, data: xr.DataArray, includes_time_dimension: bool = True
    ) -> xr.Dataset:
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
            values = data.sel({pca_abc.PC_VALUES_DIM: slice_})
            if includes_time_dimension:
                reshaped = values.data.reshape(self._dimensions.to_tuple())
                # The PCs are flipped along axis 1.
                yield xr.DataArray(
                    np.flip(reshaped, axis=1),
                    dims=[
                        pca_abc.PC_DIM,
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
