import pathlib

import a6.datasets._base as _base


class Era5(_base.Dataset):
    """Represents the ERA5 data from ECMWF."""

    def __init__(  # noqa: CFQ002
        self,
        path: pathlib.Path,
        pattern: str | None = "**/era5_*.nc",
        slice_time_dimension: bool = False,
        slice_time_time_dimension_after: int | None = None,
        preprocessing: _base.Processing | None = None,
        postprocessing: _base.Processing | None = None,
        parallel_loading: bool = True,
    ):
        """Initialize without opening the files.

        Parameters
        ----------
        path : pathlib.Path
            Paths to the data file or folder.
        pattern : str, default="**/*.grb"
            Pattern for the data files to read.

            If path is a file, the pattern will be ignored.
        slice_time_dimension : bool, default=True
            Whether the files are temporarily overlapping and should be sliced.
            The ECMWF models are usually run e.g. at 00am, 06am, 12pm, 18pm
            for 48 hours. As a consequence, the data of new models overlap
            with data from older models by some hours (6 in this case).
        slice_time_time_dimension_after : int, optional
            The number of time steps after which to slice each data file.

            Defaults to ``None`` since we typically only use data files
            with a single time step at 12pm.
        preprocessing : Callable, optional
            Pre-processing to apply to each data file before appending i
        postprocessing : Callable, optional
            Post-processing to apply to the entire dataset after reading
            individual files.
        parallel_loading : bool, default True
            Whether to load the data files parallely.

        """
        super().__init__(
            path=path,
            pattern=pattern,
            slice_time_dimension=slice_time_dimension,
            slice_time_time_dimension_after=slice_time_time_dimension_after,
            preprocessing=preprocessing,
            postprocessing=postprocessing,
            parallel_loading=parallel_loading,
        )
