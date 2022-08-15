def test_find_pc_space_clusters_with_kmeans(kmeans):
    # Expect 2 clusters that have positions given in a
    # dimension that depends on my number of PCs (3 here).
    assert kmeans.centers.shape == (2, 3)
    # Each time step should be assigned a label and thus belong
    # to one of the two clusters.
    assert len(kmeans.labels) == 5
    assert kmeans.n_clusters == 2


def test_find_pc_space_clusters_with_hdbscan(hdbscan):
    assert hdbscan.n_clusters == 2

    cluster = hdbscan.inverse_transformed_cluster(0)

    n_vars = len(cluster.data_vars)
    assert n_vars == len(hdbscan.pca._dimensions.variables)

    shape = tuple(cluster.sizes.values())
    assert shape == (10, 10)
