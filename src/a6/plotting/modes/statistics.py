import datetime

import a6.modes.methods.appearances as appearances
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np


def plot_modes_durations(
    modes: appearances.Modes,
    figsize: tuple[int, int] = (12, 6),
    display: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the mode mean durations and standard deviation."""
    fig, ax = plt.subplots(figsize=figsize)
    n_modes_ticks = list(range(1, modes.size + 1))

    durations, stds, zero = _calculate_mean_durations_and_standard_deviations(
        modes
    )

    ax.bar(
        n_modes_ticks,
        durations,
        yerr=stds,
        align="center",
        alpha=0.5,
        ecolor="black",
        capsize=10,
        bottom=zero,
    )

    ax.set_xlabel("LSWR")
    ax.set_xticks(n_modes_ticks)
    ax.set_xticklabels(n_modes_ticks)

    ax.set_ylabel("Mean duration")

    ax.set_title("Duration of LSWRs")

    ax.yaxis.grid(True)

    ax.yaxis_date()
    ylim = ax.get_ylim()
    ax.set_ylim(None, ylim[1] + 0.1 * np.diff(ylim))

    def set_label(t, i):
        t._text = f"{i}d"
        return t

    ax.set_yticklabels(
        [set_label(t, i) for i, t in enumerate(ax.get_yticklabels())]
    )

    plt.tight_layout()

    if display:
        plt.show()
    return fig, ax


def _calculate_mean_durations_and_standard_deviations(
    modes: appearances.Modes,
) -> tuple[list[float], list[float], datetime.datetime]:
    # Specify a random date to use for the times.
    zero = datetime.datetime(2018, 1, 1)

    means = [zero + mode.statistics.duration.mean for mode in modes.modes]
    stds = [zero + mode.statistics.duration.std for mode in modes.modes]

    zero = mdates.date2num(zero)

    stds = [t - zero for t in mdates.date2num(stds)]
    means = [t - zero for t in mdates.date2num(means)]

    return means, stds, zero
