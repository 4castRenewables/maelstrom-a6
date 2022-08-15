import lifetimes.plotting.clustering as clustering
import pytest


@pytest.fixture(params=["kmeans", "hdbscan"], scope="session")
def clusters(request, kmeans, hdbscan):
    return request.getfixturevalue(request.param)


def test_plot_first_three_components_timeseries_clusters(clusters):
    clustering.plot_first_three_components_timeseries_clusters(clusters)


def test_plot_condensed_tree(hdbscan):
    clustering.plot_condensed_tree(hdbscan)


def test_plot_single_linkage_tree(hdbscan):
    clustering.plot_single_linkage_tree(hdbscan)
