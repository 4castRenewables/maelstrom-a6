import a6.modes.methods.clustering as clustering
import a6.modes.methods.pca as _pca
import a6.plotting.clustering as _plotting_clustering
import a6.plotting.pca as _plotting_pca
import matplotlib.pyplot as plt

import mlflow


def create_plots_and_log_to_mantik(
    pca: _pca.PCA,
    clusters: clustering.ClusterAlgorithm,
):
    """Create all desired plots and logs them to mantik."""

    figures_and_axs: list[tuple[tuple[plt.Figure, plt.Axes], str]] = [
        (_plotting_pca.plot_scree_test(pca, display=False), "scree_test.pdf"),
        (
            _plotting_clustering.plot_first_three_components_timeseries_clusters(  # noqa
                clusters, display=False
            ),
            "pc_space_clusters.pdf",
        ),
    ]

    if isinstance(clusters, clustering.HDBSCAN):
        figures_and_axs.extend(
            [
                (
                    _plotting_clustering.plot_condensed_tree(clusters),
                    "condensed_tree.pdf",
                ),
                (
                    _plotting_clustering.plot_single_linkage_tree(clusters),
                    "single_linkage_tree.pdf",
                ),
            ]
        )

    for (fig, _), name in figures_and_axs:
        mlflow.log_figure(fig, name)
