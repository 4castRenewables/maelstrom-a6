import pathlib

import a6.datasets._base as _base


class EcmwfIfsHres(_base.Dataset):
    """Represents the IFS HRES data from ECMWF."""

    def __init__(  # noqa: CFQ002
        self,
        path: pathlib.Path,
        pattern: str | None = "*.nc",
        slice_time_dimension: bool = True,
        slice_time_time_dimension_after: int | None = 12,
        preprocessing: _base.Processing | None = None,
        postprocessing: _base.Processing | None = None,
        parallel_loading: bool = True,
    ):
        """Initialize without opening the files.

        Parameters
        ----------
        path : pathlib.Path
            Paths to the data file or folder.
        pattern : str, default="*.nc"
            Pattern for the data files to read.
            If path is a file, the pattern will be ignored.
        slice_time_dimension : bool, default=True
            Whether the files are temporarily overlapping and should be sliced.
            The ECMWF models are usually run e.g. at 00am, 06am, 12pm, 18pm
            for 48 hours. As a consequence, the data of new models overlap
            with data from older models by some hours (6 in this case).
        slice_time_time_dimension_after : int, default=12
            The number of time steps after which to slice each data file.

            Defaults to 12 since the typical data used in A6 is from
            the model runs at 00am and 12pm, hence they have to be sliced
            after 12 time steps.
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
