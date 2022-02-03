import typing as t
import abc
import xarray as xr

class AbstractCoordianteTransformation(abc.ABC):

    def __init__(components: t.Optional[xr.DataSet], eignevalues: t.Optional[xr.DataSet]):
        self.components: t.Optional[xr.DataSet] = components
        self.eigenvalues: t.Optional[xr.DataSet] = eigenvalues

    def transform(data: xr.DataSet) -> xr.DataSet:
        if self.components is None:
            raise RuntimeError("Transformation matrix is not available.")
        return self._transform(data)

    @abc.abstractmethod
    def _transform(data: xr.DataSet) -> xr.DataSet:
        pass

    def inverse_transform(data: xr.DataSet) -> xr.DataSet:
        if self.components is None:
            raise RuntimeError("Transformation matrix is not available.")
        return self._inverse_transform(data)

    @abc.abstractmethod
    def _inverse_transform(data: xr.DataSet) -> xr.DataSet:
        pass

