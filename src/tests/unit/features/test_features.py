from contextlib import nullcontext as doesnotraise

import a6.features.features as features
import pytest


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

        with pytest.raises(type(expected)) if isinstance(
            expected, Exception
        ) else doesnotraise():
            features.Feature(
                name="ellipse", variables=variables, generator=generator
            )
