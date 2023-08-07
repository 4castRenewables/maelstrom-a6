import matplotlib.pyplot as plt

import a6.modes.methods.appearances as appearances


def plot_modes_durations(
    modes: appearances.Modes,
    figsize: tuple[int, int] = (12, 6),
    colors: list[str] | None = None,
    display: bool = True,
    start_at_index_0: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the mode mean durations and standard deviation."""
    fig, ax = plt.subplots(figsize=figsize)
    n_modes_ticks = (
        list(range(1, modes.size + 1))
        if not start_at_index_0
        else list(range(modes.size))
    )

    durations, stds = _calculate_mean_durations_and_standard_deviations(modes)

    ax.bar(
        n_modes_ticks,
        durations,
        yerr=stds,
        color=colors,
        align="center",
        alpha=1,
        ecolor="black",
        capsize=10,
    )

    ax.set_xlabel("LSWR")
    ax.set_xticks(n_modes_ticks)
    ax.set_xticklabels(n_modes_ticks)

    ax.set_ylabel("Mean duration [days]")

    ax.set_title("Duration of LSWRs")

    ax.yaxis.grid(True)

    plt.tight_layout()

    if display:
        plt.show()
    return fig, ax


def _calculate_mean_durations_and_standard_deviations(
    modes: appearances.Modes,
) -> tuple:
    def convert(x):
        return x.total_seconds() / 60 / 60 / 24

    means = [convert(mode.statistics.duration.mean) for mode in modes]
    stds = [convert(mode.statistics.duration.std) for mode in modes]

    return means, stds
