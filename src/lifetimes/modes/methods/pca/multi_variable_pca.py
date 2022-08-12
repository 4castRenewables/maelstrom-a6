import lifetimes.modes.methods.pca.pca_abc as pca_abc
import lifetimes.modes.methods.pca.single_variable_pca as single_variable_pca
import numpy as np
import xarray as xr


class MultiVariablePCA(single_variable_pca.SingleVariablePCA):
    """Wrapper for `sklearn.decomposition.PCA` for multi-variable data."""

    def _to_original_shape(
        self, data: xr.DataArray, includes_time_dimension: bool = True
    ) -> xr.DataArray:
        # x
        if includes_time_dimension:
            reshaped = data.data.reshape(self._dimensions.to_tuple())
            # The PCs are flipped along axis 1.
            return xr.DataArray(
                np.flip(reshaped, axis=1),
                dims=[
                    pca_abc.PC_DIM,
                    *self._dimensions.spatial_dimension_names,
                ],
            )
        reshaped = data.data.reshape(
            self._dimensions.to_tuple(include_time_dim=False)
        )
        # The PCs are flipped along axis 0.
        return xr.DataArray(
            np.flip(reshaped, axis=0),
            dims=self._dimensions.spatial_dimension_names,
        )
