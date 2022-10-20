import a6.types as types
import a6.utils as utils
import sklearn.model_selection as model_selection


def perform_grid_search(
    model: types.Model,
    parameters: dict[str, list],
    X: types.XarrayData,
    y: types.XarrayData,
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
    return gs.fit(X=utils.transpose(X), y=utils.transpose(y), groups=groups)
