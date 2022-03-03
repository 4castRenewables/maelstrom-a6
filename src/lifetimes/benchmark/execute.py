import typing as t

import lifetimes.benchmark.dask_benchmark_context as _context
import lifetimes.parallel


def execute_benchmark(
    method: t.Callable,
    job_name: str,
    method_args: t.Iterable[t.Tuple],
    cluster_scale: t.Dict,
    dask_slurm_cluster_kwargs: t.Dict,
):
    """
    Wrapper for benchmark execution.

    Parameters
    ----------
    method: Method that should be benchmarked. Note: It is advised to use below
      functions to add logging and make the function lazy for better dask support.
    job_name: Job name; used as prefix in saved files.
    method_args: Iterable of argument tuples to hand over to method.
    cluster_scale: Kwargs to dask_jobqueue.SLURMCluster(...).scale(**kwargs).
    dask_slurm_cluster_kwargs: Kwargs to dask_jobqueue.SLURMCluster.

    Returns
    -------

    """
    client = lifetimes.parallel.DaskSlurmClient(**dask_slurm_cluster_kwargs)

    # Write configuration to file.
    with open(f"{client.log_directory}/{job_name}_config.json", "w") as f:
        f.write(str(locals()))

    with _context.DaskBenchmarkingContext(
        job_name=job_name, log_direcotry=client.log_directory
    ):
        with client(**cluster_scale):
            client.execute(method, method_args)
