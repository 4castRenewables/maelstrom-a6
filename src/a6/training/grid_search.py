import sklearn.model_selection as model_selection

import a6.features.methods.reshape as reshape
import a6.types as types
import a6.utils as utils


@utils.log_consumption
def perform_grid_search(  # noqa: CFQ002
    model: types.Model,
    parameters: dict[str, list],
    training_data: types.XarrayData,
    target_data: types.XarrayData,
    cv: model_selection.BaseCrossValidator,
    groups: list[int],
    scorers: types.Scorers,
    refit: str,
) -> model_selection.GridSearchCV:
    """Perform a parameter grid search on a dataset."""
    gs = model_selection.GridSearchCV(
        estimator=model,
        param_grid=parameters,
        scoring=scorers,
        cv=cv,
        refit=refit,
        n_jobs=utils.get_cpu_count(),
    )
    return gs.fit(
        X=reshape.sklearn.transpose(training_data),
        y=reshape.sklearn.transpose(target_data),
        groups=groups,
    )
