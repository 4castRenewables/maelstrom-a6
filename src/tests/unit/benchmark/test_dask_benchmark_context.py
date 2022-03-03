import os
import time

import dask.distributed as distributed


def test_dask_benchmarking_context(benchmarking_context, log_directory):
    assert benchmarking_context.job_name == "test"
    assert benchmarking_context.log_directory == log_directory
    assert isinstance(
        benchmarking_context.memory_sampler, distributed.diagnostics.MemorySampler
    )
    assert isinstance(
        benchmarking_context.performance_report, distributed.performance_report
    )


def test_dask_benchmarking_context_enter_and_exit(benchmarking_context, log_directory):
    with benchmarking_context:
        time.sleep(1)
    assert os.path.exists(log_directory + "/dask_memory_sample_test.csv")
    assert os.path.exists(log_directory + "/dask_performance_report_test.html")


def test_save_memory_sample(benchmarking_context, log_directory):
    benchmarking_context._save_memory_sample(log_directory + "/test")
    assert os.path.exists(log_directory + "/test")
