import lifetimes.modes.methods.clustering as clustering
import lifetimes.modes.methods.pca as _pca


def test_find_principal_component_clusters(ds):
    da = ds["ellipse"]
    n_timesteps = len(da.coords["time"])
    pca = _pca.spatio_temporal_principal_component_analysis(
        data=da,
        time_coordinate="time",
        latitude_coordinate="lat",
        x_coordinate="lat",
        y_coordinate="lon",
        variance_ratio=None,
    )
    n_components = 2
    n_clusters = 2

    clusters = clustering.find_principal_component_clusters(
        pca=pca,
        n_components=n_components,
        n_clusters=n_clusters,
    )

    # I expect 2 clusters that have positions given in a
    # dimension that depends on my number of PCs (2 here).
    assert clusters.centers.shape == (n_clusters, n_components)
    # Each time step should be assigned a label and thus belong
    # to one of the two clusters.
    assert len(clusters.labels) == n_timesteps
