import pytest
import torch
import xarray as xr

import a6.datasets as datasets


@pytest.fixture()
def dataset(
    era5_ds, era5_path, gwl_path
) -> datasets.torch.xarray.WithGWLTarget:
    gwl = xr.open_dataset(gwl_path)
    return datasets.torch.xarray.WithGWLTarget(
        data_path=era5_path,
        weather_dataset=era5_ds,
        gwl_dataset=gwl,
    )


class TestTorchWithGWL:
    def test_len(self, dataset):
        result = len(dataset)

        assert result == 4

    def test_getitem(self, dataset):
        data, target = dataset[0]

        assert not torch.isnan(data).any()
        assert target == 10
