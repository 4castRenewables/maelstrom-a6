from unittest import mock

import pytest


@pytest.mark.skip
@mock.patch("dask_jobqueue.SLURMCluster")
@mock.patch("dask.distributed.Client")
def test_execute_benchmark(
    mock_client, mock_cluster, dask_slurm_cluster_kwargs, log_directory
):
    # Note: Test of this function is not possible since
    # SLURMCluster requires a SLURM installation
    pass
