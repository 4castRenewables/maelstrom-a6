import pathlib
import os
import typing as t
import unittest.mock

import dask.distributed
import dask.utils
import dask_jobqueue
import lifetimes.parallel.slurm as slurm
import pytest

FILE_DIR = pathlib.Path(__file__).parent


class DummyJob:
    def __init__(self):
        pass

    @property
    def log_directory(self) -> str:
        return "test_log_directory"


class FakeSLURMCluster(dask_jobqueue.SLURMCluster):


    @property
    def _dummy_job(self):
        return DummyJob()

    def job_script(self):
        return "test_job_script"

    def scale(self, n=None, jobs=0, memory=None, cores=None):
        # Passing a not-interpretable byte string causes a ValueError
        # due to the below function, which has to be caught in the tests.
        if memory is not None:
            dask.utils.parse_bytes(memory)
        pass

    def close(self, timeout=None):
        pass


class FakeDaskClient(dask.distributed.Client):
    pass

    def close(self, timeout=None):
        pass


@pytest.fixture(scope="session")
def log_directory(tmpdir_factory):
    log_dir = tmpdir_factory.mktemp("log")
    return log_dir


@pytest.fixture
@unittest.mock.patch("dask_jobqueue.SLURMCluster", FakeSLURMCluster)
@unittest.mock.patch("dask.distributed.Client", FakeDaskClient)
def client(log_directory) -> slurm.DaskSlurmClient:
    log_directory = FILE_DIR / "test"
    # SINGULARITY_IMAGE_ENV_VAR has to be set for the default Python executable
    os.environ[slurm.SINGULARITY_IMAGE_ENV_VAR] = "test"
    _class = slurm.DaskSlurmClient
    # Need to set interface to an available network interface due to
    # ` ValueError: 'ib0' is not a valid network interface.`
    _class._interface = "lo"
    _client = _class(
        queue="test",
        project="test",
        python_executable="python",
        # Log directory has to be set. Otherwise the client will
        # create a folder in the $HOME directory.
        log_directory=log_directory.as_posix(),
    )
    os.unsetenv(slurm.SINGULARITY_IMAGE_ENV_VAR)
    log_directory.rmdir()
    return _client


def dummy_benchmark_method(inputs: t.Iterable[str]):
    for input in inputs:
        print(input)


class TestDaskSlurmClient:
    def test_python_executable_string(self, client):
        result = client._python_executable.format("test")
        expected = "srun singularity run test python"

        assert result == expected

    def test_enter(self, client):
        with client.scale(workers=1):
            assert client.ready

    def test_job_script(self, client):
        assert client.job_script == "test_job_script"

    @pytest.mark.parametrize(
        ("_extra_job_commands_attribute", "extra_job_commands", "expected"),
        [
            (None, None, []),
            (None, ["test"], ["test"]),
            (["test"], None, ["test"]),
            (["test_1"], ["test_2"], ["test_2", "test_1"]),
        ]
    )
    def test_update_extra_job_commands(self, client, _extra_job_commands_attribute,extra_job_commands, expected):
        client._extra_job_commands = _extra_job_commands_attribute

        result = client._update_extra_job_commands(extra_job_commands)

        assert result == expected
