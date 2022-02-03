import sklearn.decomposition
import .abstract_transformation

class PCATransformation(abstract_transformation.AbstractTransformation):

    def __init__(pca: sklearn.decomposition.PCA):
        super().__init__(components=pca.components_, eigenvalues=pca.explained_variance_)

    def _transform(data: xr.DataSet) -> xr.DataSet:
        pass #TODO solange ich davon ausgehen kann, dass ich eine Matrix habe, kann ich die Trafo auch in der Vaterklasse implementieren

    def _inverse_transform(data: xr.DataSet) -> xr.DataSet:
        pass

