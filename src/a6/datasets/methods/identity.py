import a6.types as types
import a6.utils as utils


@utils.functional.make_functional
def identity(data: types.DataND, *args, **kwargs) -> types.DataND:
    """Return the unchanged input (identity method)."""
    return data
