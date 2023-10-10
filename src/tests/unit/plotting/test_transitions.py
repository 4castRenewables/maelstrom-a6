import matplotlib.pyplot as plt

import a6.plotting.transitions as transitions


def test_plot_transition_matrix_heatmap(time_series_data):
    transitions.plot_transition_matrix_heatmap(time_series_data)
    plt.show()


def test_plot_transition_matrix_clustermap(time_series_data):
    transitions.plot_transition_matrix_clustermap(time_series_data)
    plt.show()