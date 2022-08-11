import lifetimes.plotting.pca as _pca
import pytest


@pytest.mark.parametrize("variance_ratio", [None, 0.8])
def test_plot_scree_test(pca, variance_ratio):
    _pca.plot_scree_test(pca, variance_ratio=variance_ratio)


def test_plot_first_three_components_timeseries(pca):
    _pca.plot_first_three_components_timeseries(pca)
