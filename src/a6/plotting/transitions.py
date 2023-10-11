import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import seaborn.matrix as sns_matrix

import a6.types as types


def plot_transition_matrix_heatmap(
    data: types.TimeSeries,
    name: str = "transition-matrix-heatmap",
    output_dir: pathlib.Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    transitions = _calculate_markov_transition_matrix(data)

    fig, ax1 = plt.subplots()
    sns.heatmap(
        transitions, ax=ax1, annot=True, cmap="Reds", annot_kws={"size": 8}
    )
    plt.setp(ax1.get_xticklabels(), rotation=60)
    fig.suptitle("Transition probabilities")

    if output_dir is not None:
        plt.savefig(output_dir / f"{name}.pdf")

    return fig, ax1


def plot_transition_matrix_clustermap(
    data: types.TimeSeries,
    name: str = "transition-matrix-clustermap",
    output_dir: pathlib.Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    transitions = _calculate_markov_transition_matrix(data)

    cluster_map: sns_matrix.ClusterGrid = sns.clustermap(
        transitions,
        method="complete",
        annot=True,
        cmap="Reds",
        annot_kws={"size": 8},
    )
    fig: plt.Figure = cluster_map.fig
    ax: plt.Axes = cluster_map.ax_heatmap
    plt.setp(ax.get_xticklabels(), rotation=60)
    fig.suptitle("Transition probabilities")

    if output_dir is not None:
        plt.savefig(output_dir / f"{name}.pdf")

    return fig, ax


def _calculate_markov_transition_matrix(data: types.TimeSeries) -> np.ndarray:
    """Calculate Markov transition matrix for integer time series.

    Notes
    -----
    The method only works for time series with integer values.
    E.g. ``[1, 1, 3, 4, 0, 2, ...]``.

    """
    n_states = 1 + int(max(data))
    matrix = np.zeros((n_states, n_states))

    for i, j in zip(data[:-1], data[1:], strict=True):
        # Add 1 to matrix at index (i, j) inplace
        np.add.at(matrix, (i, j), 1)

    # convert to probabilities by dividing each row by the sum of its states
    as_probabilities = matrix / matrix.sum(axis=1, keepdims=True)

    # If a state never transitions into another (e.g. it never occurs
    # or is at the end of the time series), it's row gets divided by 0,
    # resulting in NaN. Hence, replace NaN with 0.
    nan_to_zero = np.nan_to_num(as_probabilities, copy=True, nan=0.0)
    return nan_to_zero
