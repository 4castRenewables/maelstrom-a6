import xarray as xr
import lifetimes.coordinate_transformations as transformations


class Varimax(transformations.BaseTransformation):
    def __init__(self, as_dataset: xr.Dataset):
        super().__init__(as_dataset=as_dataset)

    '''
    @property
    @functools.lru_cache
    def components_varimax_rotated(
            self,
    ) -> np.ndarray:
        """Return the principal components (EOFs)."""
        return _perform_varimax_rotation(self.components)


    @property
    @functools.lru_cache
    def components_varimax_rotated_in_original_shape(
            self,
    ) -> np.ndarray:
        """Return the principal components (EOFs)."""
        return self._to_original_shape(self.components_varimax_rotated)

    @functools.lru_cache
    def transform_with_varimax_rotation(
            self, n_components: t.Optional[int] = None
    ) -> np.ndarray:
        """Transform the given data into the vector space of the varimax-rotated PCs."""
        return self._transform(
            self.components_varimax_rotated, n_components=n_components
        )
    '''


def perform_varimax_rotation(
    matrix: xr.Dataset,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> Varimax:
    n_eigenvectors = len(matrix["transformation_matrix"])
    matrix_as_np_array = matrix.values.reshape(n_eigenvectors, -1)
    transposed = matrix_as_np_array.T
    n_row, n_col = transposed.shape
    rotation_matrix = np.eye(n_col)
    var = 0

    for _ in range(max_iter):
        comp_rot = np.dot(transposed, rotation_matrix)
        tmp = comp_rot * np.transpose((comp_rot**2).sum(axis=0) / n_row)
        u, s, v = np.linalg.svd(np.dot(matrix, comp_rot**3 - tmp))
        rotation_matrix = np.dot(u, v)
        var_new = np.sum(s)
        if var != 0 and var_new < var * (1 + tol):
            break
        var = var_new

    as_dataset = _to_dataset(rotation_matrix)
    return Varimax(as_dataset=as_dataset)


def _rotation_matrix_to_dataset(
    roatation_matrix: np.ndarray, input_dataset: xr.Dataset
) -> xr.Dataset:
    x_coordinate, y_coordinate = list(input_dataset.coords.keys())
    x_coordinates = input_dataset.coords[x_coordinate].values
    y_coordinates = input_dataset.coords[y_coordinate].values
    training_dimensions = {x_coordinate: x_coordinates, y_coordinate: y_coordinates}
    training_dimension_shapes = [x_coordinates.shape[0], y_coordinates.shape[0]]
    return xr.Dataset(
        data_vars={
            "transformation_matrix": (
                ["eigenvector_number", *training_dimensions],
                roatation_matrix.reshape(-1, *training_dimension_shapes),
            ),
        },
        coords=coords,
    )
