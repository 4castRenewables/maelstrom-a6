import pytest

import a6.plotting.pca as _pca


@pytest.mark.parametrize("variance_ratio", [None, 0.8])
def test_plot_scree_test(pca, variance_ratio):
    _pca.plot_scree_test(pca, variance_ratio=variance_ratio, display=False)


def test_plot_first_three_components_timeseries(pca):
    _pca.plot_first_three_components_timeseries(pca, display=False)
