import pytest
import torch

import a6.datasets.methods.normalization as normalization
import a6.datasets.methods.transform as transform


class TestMinMaxScale:
    @pytest.mark.parametrize(
        ("data", "expected"),
        [
            # Test case: single-level data
            (
                torch.Tensor(
                    [
                        [3],
                        [7.5],
                    ]
                ),
                torch.Tensor(
                    [
                        [(3 - 2) / 4],
                        [(7.5 - 5) / 10],
                    ]
                ),
            ),
            # Test case: multi-level data
            (
                torch.Tensor(
                    [
                        [3],
                        [7.5],
                        [3.5],
                        [7],
                    ]
                ),
                torch.Tensor(
                    [
                        [(3 - 2) / 4],
                        [(7.5 - 5) / 10],
                        [(3.5 - 2) / 4],
                        [(7 - 5) / 10],
                    ]
                ),
            ),
        ],
    )
    def test_call(self, data, expected):
        min_max_values = [
            normalization.VariableMinMax(
                name="test",
                min=2,
                max=4,
            ),
            normalization.VariableMinMax(
                name="test",
                min=5,
                max=10,
            ),
        ]
        scaler = transform.MinMaxScale(min_max=min_max_values)

        result = scaler(data)

        torch.testing.assert_close(result, expected)
