import typing as t

import dask.distributed as distributed


class DaskBenchmarkingContext:
    """Class providing a context manager to enable dask benchmarking utility usage."""

    def __init__(
        self,
        job_name: str,
        memory_sampler: t.Optional[
            distributed.diagnostics.MemorySampler
        ] = None,
        log_directory: str = ".",
    ):
        """
        Initialize.

        Parameters
        ----------
        job_name: Job name. Used as prefix in all files that are written to disk.
        memory_sampler: dask.distributed.diagnostics.MemorySampler instance.
        log_directory: Dorectory to which logs are written.
        """
        self.job_name = job_name
        self.log_directory = log_directory
        if memory_sampler is not None:
            self.memory_sampler = memory_sampler
            self.memory_sampler_context = memory_sampler.sample(self.job_name)
        else:
            self.memory_sampler = distributed.diagnostics.MemorySampler()
            self.memory_sampler_context = self.memory_sampler.sample(
                self.job_name
            )
        self.performance_report = distributed.performance_report(
            filename=f"{self.log_directory}/dask_performance_report_{job_name}.html"
        )

    def __enter__(self):
        """Enter both attribute context managers."""
        self.memory_sampler_context.__enter__()
        self.performance_report.__enter__()

    def __exit__(self, type, value, traceback):
        """Exit both attribute context managers and save memory sample.

        Note: Arguments are standard arguments for context manager exit.
        """
        self.memory_sampler_context.__exit__(type, value, traceback)
        self.performance_report.__exit__(type, value, traceback)
        self._save_memory_sample(
            memory_sample_filename=f"{self.log_directory}/dask_memory_sample_{self.job_name}.csv"
        )

    def _save_memory_sample(self, memory_sample_filename: str):
        """
        Save memory sample.

        Parameters
        ----------
        memory_sample_filename: Filename to save to.

        Returns
        -------
            None.
        """
        as_dataframe = self.memory_sampler.to_pandas()
        as_dataframe.to_csv(memory_sample_filename)
