from collections.abc import Callable
from collections.abc import Iterable

import a6.benchmark.dask_benchmark_context as _context
import a6.parallel


def execute_benchmark(
    method: Callable,
    job_name: str,
    method_args: Iterable[tuple],
    cluster_scale: dict,
    dask_slurm_cluster_kwargs: dict,
):
    """
    Wrapper for benchmark execution.

    Parameters
    ----------
    method: Method that should be benchmarked. Note: It is advised to use below
     functions to add logging and make the function lazy for better dask support
    job_name: Job name; used as prefix in saved files.
    method_args: Iterable of argument tuples to hand over to method.
    cluster_scale: Kwargs to dask_jobqueue.SLURMCluster(...).scale(**kwargs).
    dask_slurm_cluster_kwargs: Kwargs to dask_jobqueue.SLURMCluster.

    Returns
    -------

    """
    client = a6.parallel.DaskSlurmClient(**dask_slurm_cluster_kwargs)

    # Write configuration to file.
    with open(f"{client.log_directory}/{job_name}_config.json", "w") as f:
        f.write(str(locals()))

    with _context.DaskBenchmarkingContext(
        job_name=job_name, log_direcotry=client.log_directory
    ):
        with client(**cluster_scale):
            client.execute(method, method_args)
