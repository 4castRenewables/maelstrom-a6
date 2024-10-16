import pytest
import torch

import a6.datasets.crop as dataset


@pytest.fixture
def era5_torchvision_dataset(era5_path, era5) -> dataset.MultiCropXarrayDataset:
    # Example data is 5x5 grid, hence set size of crops to 3 (0.6)
    era5_ds = era5.to_xarray(levels=[500])
    return dataset.MultiCropXarrayDataset(
        data_path=era5_path,
        dataset=era5_ds,
        nmb_crops=[2],
        size_crops=[0.6],
        min_scale_crops=[0.08],
        max_scale_crops=[1.0],
    )


@pytest.fixture
def era5_torchvision_dataset_multi_level(
    era5_path, era5
) -> dataset.MultiCropXarrayDataset:
    # Example data is 5x5 grid, hence set size of crops to 3 (0.6)
    era5_ds = era5.to_xarray(levels=[500, 950])
    return dataset.MultiCropXarrayDataset(
        data_path=era5_path,
        dataset=era5_ds,
        nmb_crops=[2],
        size_crops=[0.6],
        min_scale_crops=[0.08],
        max_scale_crops=[1.0],
    )


class TestMultiCropXarrayDataset:
    def test_init_crop_size_too_large(self, era5_path, era5_ds):
        # Example data is 5x5 grid, hence size of crops must be smaller than 5
        expected = "Crop size must be in the range [0; 1], but 1.1 given"

        era5_ds = era5_ds.sel({"level": 500})

        with pytest.raises(ValueError) as e:
            dataset.MultiCropXarrayDataset(
                data_path=era5_path,
                dataset=era5_ds,
                nmb_crops=[2],
                size_crops=[1.1],
                min_scale_crops=[0.08],
                max_scale_crops=[1.0],
            )

        result = str(e.value)

        assert result == expected

    def test_len(self, era5_torchvision_dataset):
        # ``era5_ds`` contains 4 days, hence ``__len__`` is
        # expected to return 4.
        expected = 4

        result = len(era5_torchvision_dataset)

        assert result == expected

    def test_n_channels(self, era5_torchvision_dataset):
        assert era5_torchvision_dataset.n_channels == 5

    def test_n_channels_multi_level(self, era5_torchvision_dataset_multi_level):
        # 2 levels selected should result in 10 input channels
        assert era5_torchvision_dataset_multi_level.n_channels == 2 * 5

    @pytest.mark.parametrize(
        # The test dataset contains 4 days of data
        "index",
        [0, 1, 2, 3],
    )
    def test_getitem(self, era5_torchvision_dataset, index):
        result: list[torch.Tensor] = era5_torchvision_dataset[index]

        for sample in result:
            assert list(sample.size()) == [5, 3, 3]

    @pytest.mark.parametrize(
        # The test dataset contains 4 days of data
        "index",
        [0, 1, 2, 3],
    )
    def test_getitem_multi_level(
        self, era5_torchvision_dataset_multi_level, index
    ):
        result: list[torch.Tensor] = era5_torchvision_dataset_multi_level[index]

        for sample in result:
            # 2 levels selected should result in 10 input channels
            assert list(sample.size()) == [2 * 5, 3, 3]


@pytest.mark.parametrize(
    ("size_crops", "expected"),
    [
        # Test case: scales to 0.5 of ``size_lat``
        ([0.5], [2]),
        # Test case: scales to 0.5 of ``size_lat`` and 0.25 of
        # ``size_lon``, respectively
        ([(0.5, 0.4)], [(2, 4)]),
    ],
)
def test_convert_relative_to_absolute_crop_size(size_crops, expected):
    result = dataset.convert_relative_to_absolute_crop_size(
        size_crops=size_crops, size_x=5, size_y=10
    )

    assert result == expected


@pytest.mark.parametrize(
    ("size_crops", "expected"),
    [
        # Test case: crop size < 0.0
        ([-0.1], "-0.1"),
        # Test case: crop size > 1.0
        ([1.1], "1.1"),
        # Test case: second crop size < 0.0
        ([0.1, -0.5], "-0.5"),
        # Test case: second crop size > 1.0
        ([0.1, 1.5], "1.5"),
        # Test case: one of fist crop sizes < 0
        ([(-0.1, 0.5)], "(-0.1, 0.5)"),
        # Test case: one of fist crop sizes > 1.0
        ([(0.1, 1.5)], "(0.1, 1.5)"),
        # Test case: one of second crop sizes < 0
        ([(0.1, 0.5), (-0.1, 0.5)], "(-0.1, 0.5)"),
        # Test case: one of second crop sizes > 1.0
        ([(0.1, 0.5), (0.1, 1.5)], "(0.1, 1.5)"),
    ],
)
def test_convert_relative_to_absolute_crop_size_raises(size_crops, expected):
    expected = f"Crop size must be in the range [0; 1], but {expected} given"
    with pytest.raises(ValueError) as e:
        dataset.convert_relative_to_absolute_crop_size(
            size_crops=size_crops, size_x=5, size_y=10
        )

    result = str(e.value)

    assert result == expected
