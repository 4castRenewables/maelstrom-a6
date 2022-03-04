import dask.distributed
import dask_jobqueue
import lifetimes.benchmark as bench
import pytest


@pytest.fixture(scope="session")
def log_directory(tmpdir_factory):
    log_dir = tmpdir_factory.mktemp("log")
    return log_dir


@pytest.fixture
def cluster(log_directory):
    kwargs = {
        "queue": "test",
        "cores": 1,
        "memory": "1GB",
        "interface": "lo",
        "log_directory": str(log_directory),
    }

    return dask_jobqueue.SLURMCluster(job_name="test", **kwargs)


@pytest.fixture
def client(cluster):
    return dask.distributed.Client(cluster)


@pytest.fixture
def benchmarking_context(client, log_directory):
    # Note: Dask client is initialized globally and an implicit dependency of the
    # context
    return bench.DaskBenchmarkingContext(job_name="test", log_directory=log_directory)
