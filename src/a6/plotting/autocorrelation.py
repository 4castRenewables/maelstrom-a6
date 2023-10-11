import pathlib

import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots as tsaplots

import a6.types as types


def plot_autocorrelation(
    data: types.TimeSeries,
    name: str = "autocorrelation",
    output_dir: pathlib.Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    lags = _get_lags_for_autocorrelation(data)

    fig, ax1 = plt.subplots()
    tsaplots.plot_acf(
        x=data,
        ax=ax1,
        lags=lags,
        title=f"Autocorrelation (lags={lags})",
    )

    if output_dir is not None:
        plt.savefig(output_dir / f"{name}.pdf")

    return fig, ax1


def plot_partial_autocorrelation(
    data: types.TimeSeries,
    name: str = "partial-autocorrelation",
    output_dir: pathlib.Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    lags = _get_lags_for_partial_autocorrelation(data)

    fig, ax1 = plt.subplots()
    tsaplots.plot_pacf(
        x=data,
        ax=ax1,
        lags=lags,
        title=f"Partial autocorrelation (lags={lags})",
    )

    if output_dir is not None:
        plt.savefig(output_dir / f"{name}.pdf")

    return fig, ax1


def _get_lags_for_partial_autocorrelation(data: types.TimeSeries) -> int | None:
    lags = _get_lags_for_autocorrelation(data)
    size = len(data)

    if lags is None:
        return (size // 2) - 1

    if lags > (half_size := size // 2):
        # Partial correlations can only be computed for lags up to
        # 50% of the sample size.
        return half_size - 1
    return None


def _get_lags_for_autocorrelation(data: types.TimeSeries) -> int | None:
    size = len(data)
    if size > 1e3:
        # If n_samples > 1e3, show every hundredth data sample
        # in autocorrelation plot
        return int(size / 1e2)
    elif size > 1e2:
        # If 1e3 > n_samples > 1e2, show ever tenth data sample
        # in autocorrelation plot
        return int(size / 10)
    return None
