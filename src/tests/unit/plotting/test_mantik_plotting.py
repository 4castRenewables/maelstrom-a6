import a6.plotting.mantik as mantik

import mlflow


def test_create_plots_and_log_to_mantik(monkeypatch, pca, hdbscan):
    result = []

    def grab_plots(fig, name):
        result.append((fig, name))

    expected = [
        "scree_test.pdf",
        "pc_space_clusters.pdf",
        "condensed_tree.pdf",
        "single_linkage_tree.pdf",
    ]
    monkeypatch.setattr(mlflow, "log_figure", grab_plots)
    mantik.create_plots_and_log_to_mantik(pca=pca, clusters=hdbscan)

    for (_, res), exp in zip(result, expected):
        assert res == exp
