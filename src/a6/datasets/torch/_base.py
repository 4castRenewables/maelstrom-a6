import pathlib

import torchvision


class Base(torchvision.datasets.VisionDataset):
    _n_channels: int

    def __init__(
        self,
        data_path: pathlib.Path,
        return_index: bool = False,
    ):
        super().__init__(data_path.as_posix())
        self.return_index = return_index

    @property
    def n_channels(self) -> int:
        return self._n_channels
