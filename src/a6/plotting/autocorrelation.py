import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots as tsaplots

import a6.types as types


def plot_autocorrelation(data: types.TimeSeries) -> tuple[plt.Figure, plt.Axes]:
    lags = _get_lags_for_autocorrelation(data)

    fig, ax1 = plt.subplots()
    tsaplots.plot_acf(
        x=data,
        ax=ax1,
        lags=lags,
        title=f"Autocorrelation (lags={lags})",
    )

    return fig, ax1


def plot_partial_autocorrelation(
    data: types.TimeSeries,
) -> tuple[plt.Figure, plt.Axes]:
    lags = _get_lags_for_partial_autocorrelation(data)

    fig, ax1 = plt.subplots()
    tsaplots.plot_pacf(
        x=data,
        ax=ax1,
        lags=lags,
        title=f"Partial autocorrelation (lags={lags})",
    )

    return fig, ax1


def _get_lags_for_partial_autocorrelation(data: types.TimeSeries) -> int:
    lags = _get_lags_for_autocorrelation(data)
    size = len(data)
    if lags > (half_size := int(0.5 * size)):
        # Partial correlations can only be computed for lags up to
        # 50% of the sample size.
        return half_size - 1
    return lags


def _get_lags_for_autocorrelation(data: types.TimeSeries) -> int:
    size = len(data)
    if size > 1e3:
        # If n_samples > 1e3, show every hundredth data sample
        # in autocorrelation plot
        return int(size / 1e2)
    elif size > 1e2:
        # If 1e3 > n_samples > 1e2, show ever tenth data sample
        # in autocorrelation plot
        return int(size / 10)
    return size - 1
