import a6.types as types
import a6.utils as utils


@utils.make_functional
def drop_variables(
    data: types.XarrayData,
    names: str | list[str],
) -> types.XarrayData:
    """Drop variables from dataset."""
    return data.drop_vars(names)
