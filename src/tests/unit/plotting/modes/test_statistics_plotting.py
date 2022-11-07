import a6.plotting.modes.statistics as statistics


def test_plot_modes_durations(mode_appearances):
    statistics.plot_modes_durations(mode_appearances)


def test_calculate_mean_durations_and_standard_deviations(mode_appearances):
    statistics._calculate_mean_durations_and_standard_deviations(
        mode_appearances
    )
