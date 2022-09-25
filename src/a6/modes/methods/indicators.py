"""
Calculation of the indicators d (local dimensionality) and theta (persistence)
for dynamical systems.

See e.g.
- Faranda et al. 2017: Dynamical proxies of North Atlantic predictability and
  extremes
- Messori et al. 2017: A dynamical systems approach to studying midlatitude
  weather extremes
- Falasca and Bracco 2021: Exploring the climate system through manifold
  learning

Code from https://github.com/FabriFalasca/climate-and-dynamical-systems
"""
import sys
import typing as t

import numpy as np
import scipy.stats
import sklearn.metrics.pairwise


def indicators(
    X: np.ndarray,
    Y: t.Optional[np.ndarray] = None,
    metric: t.Union[str, t.Callable] = "euclidean",
    q: float = 0.98,
    pareto_fit: str = "scipy",
    theta_fit: str = "ferro",
    distances: t.Optional[np.ndarray] = None,
):
    """Fit a dataset to find its local dimension and persistence index.

    Parameters
    ----------
    X : np.ndarray
        Dataset to fit.
        Shape (n_samples, n_dimensions).
    Y : np.ndarray, optional
        Reference to estimate local dimension.
        Shape (n_samples_2, n_dimensions).
        If `None`, Y = X.
    metric : str or callable, default="euclidean"
        Metric used between sample of X and Y.
        See sklearn.metrics.pairwise.pairwise_distances.
    q : float, default=0.98
        Quantile for the threshold for generalized pareto distribution.
    pareto_fit : str, default="scipy"
        Method to fit the scale of generalized pareto law.
        Options:
            - "scipy" (scale estimation, slower)
            - "mean" (assume shape = 0)
    theta_fit : str = "ferro"
        Method to fit the theta.
        "ferro" or "sueveges".
    distances : np.ndarray, optional
        Pairwise distance between X and Y.
        Must be of shape (n_samples, n_samples)
        If None, `sklearn.metrics.pairwise.pairwise_distances` is used.

    Returns
    -------
    local_dimension : np.ndarray
        Local dimension `d` of the elements of X.
        Is of shape (n_samples).
    persistence : np.ndarray
        Persistence (also called extremal index `Theta`) of the elements of X
        Is of shape (n_samples).

    """
    distances_logarithmic = _calculate_logarithmic_distances(
        X=X,
        Y=Y,
        metric=metric,
        distances=distances,
    )
    thresholds = np.percentile(distances_logarithmic, 100 * q, axis=1)
    size = thresholds.size

    local_dimension = np.zeros_like(thresholds)
    persistence = np.zeros_like(thresholds)

    for i in range(size):
        threshold = np.array(
            np.argwhere(distances_logarithmic[i, :] > thresholds[i])
        ).ravel()
        distances_within_threshold = (
            distances_logarithmic[i, threshold] - thresholds[i]
        )
        local_dimension[i] = _calculate_local_dimension_fit(
            distances=distances_within_threshold, pareto_fit=pareto_fit
        )
        persistence[i] = _calculate_persistence_fit(
            thresholds=threshold, q=q, theta_fit=theta_fit
        )

    return local_dimension, persistence


def _calculate_logarithmic_distances(
    X: np.ndarray,
    Y: t.Optional[np.ndarray],
    metric: str,
    distances: t.Optional[np.ndarray],
) -> np.ndarray:
    if distances is None:
        distances = sklearn.metrics.pairwise.pairwise_distances(
            X, Y=Y, metric=metric
        )
    distances[distances == 0] = sys.float_info.max
    distances_logarithmic = -np.log(distances)
    return distances_logarithmic


def _calculate_local_dimension_fit(distances, pareto_fit):
    return 1.0 / _genpareto_fit(distances, pareto_fit)


def _genpareto_fit(distances, pareto_fit):
    if pareto_fit == "scipy":
        return scipy.stats.genpareto.fit(distances, floc=0)[2]
    else:
        return np.mean(distances)


def _calculate_persistence_fit(thresholds, q, theta_fit):
    if theta_fit == "sueveges":
        return _theta_sueveges_fit(thresholds, q)
    else:
        return _theta_ferro_fit(thresholds)


def _theta_sueveges_fit(threshold, q):
    Nc = np.count_nonzero((threshold[1:] - threshold[:-1] - 1) > 0)
    N = threshold.size - 1
    tmp = (1.0 - q) * (threshold[-1] - threshold[0])
    return (
        tmp + N + Nc - np.sqrt(np.power(tmp + N + Nc, 2.0) - 8.0 * Nc * tmp)
    ) / (2.0 * tmp)


def _theta_ferro_fit(threshold):
    Ti = threshold[1:] - threshold[:-1]
    if np.max(Ti) > 2:
        res = (
            2
            * (np.sum(Ti - 1) ** 2)
            / ((Ti.size - 1) * np.sum((Ti - 1) * (Ti - 2)))
        )
    else:
        res = 2 * (np.sum(Ti) ** 2) / ((Ti.size - 1) * np.sum(Ti**2))
    res = min(1, res)
    return res
