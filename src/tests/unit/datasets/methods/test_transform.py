import torch

import a6.datasets.methods.normalization as normalization
import a6.datasets.methods.transform as transform


class TestMinMaxScale:
    def test_call(self):
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

        data = torch.Tensor(
            [
                [3],
                [7.5],
            ]
        )

        expected = torch.Tensor(
            [
                [(3 - 2) / 4],
                [(7.5 - 5) / 10],
            ]
        )

        result = scaler(data)

        torch.testing.assert_close(result, expected)
