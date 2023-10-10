import a6.plotting.autocorrelation as autocorrelation


def test_plot_autocorrelation(time_series_data):
    autocorrelation.plot_autocorrelation(time_series_data)


def test_plot_partial_autocorrelation(time_series_data):
    autocorrelation.plot_partial_autocorrelation(time_series_data)
