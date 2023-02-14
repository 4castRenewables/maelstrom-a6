import pytest

import a6.features.features as features
import a6.testing as testing


class TestFeature:
    @pytest.mark.parametrize(
        ("variables", "generator", "expected"),
        [
            # Test case: feature cannot be generated if no variable given
            ([], None, ValueError()),
            # Test case: feature cannot be generated if multiple variables were
            # given but no generator method
            (["test_var_1", "test_var_2"], None, ValueError()),
        ],
    )
    def test_initialization(self, variables, generator, expected):
        with testing.expect_raise_if_exception(expected):
            features.Feature(
                name="ellipse", variables=variables, generator=generator
            )
