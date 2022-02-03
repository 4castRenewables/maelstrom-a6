import .abstract_transformation

class OrthogonalTransformation(abstract_transformation.AbstractTransformation):

    def __init__(components: xr.DataSet, eigenvalues: xr.DataSet):
        super().__init__(components, eigenvalues)

    @classmethod
    def from_varimax(rotation_matrix: xr.DataSet):
        pass #TODO varimax result to Dataset

    def _transform(data: xr.DataSet) -> xr.DataSet:
        pass

    def _inverse_transform(data: xr.DataSet) -> xr.DataSet:
        pass
