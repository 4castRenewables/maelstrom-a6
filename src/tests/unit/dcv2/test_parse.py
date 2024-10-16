import pytest

import a6.dcv2.parse as _parse


@pytest.mark.parametrize(
    ("attribute", "args", "expected"),
    [
        # Test case: parsing of multiple values
        (
            "drop_variables",
            ["--drop-variables", "a", "b", "c"],
            ["a", "b", "c"],
        ),
        # Test case: parsing of multiple values as signle-quoted string
        ("drop_variables", ["--drop-variables", "'a b c'"], ["a", "b", "c"]),
        # Test case: parsing of multiple values as double-quoted string
        ("drop_variables", ["--drop-variables", '"a b c"'], ["a", "b", "c"]),
        # Test case: parsing of multiple values as double-quoted string
        ("drop_variables", ["--drop-variables", "None"], None),
        # Test case: parsing of ``--no`` prefix
        ("enable_tracking", ["--no-enable-tracking"], False),
        # Test case: parsing of ``None``
        ("pattern", ["--pattern", "None"], None),
        ("levels", ["--levels", "1", "2", "3"], [1, 2, 3]),
        # Test case: parsing of multiple values as signle-quoted string
        ("levels", ["--levels", "'1 2 3'"], [1, 2, 3]),
        # Test case: parsing of multiple values as double-quoted string
        ("levels", ["--levels", '"1 2 3"'], [1, 2, 3]),
        # Test case: passing of float or tuple of floats
        # Default should remain if no args given
        ("nmb_crops", [], [2]),
        ("nmb_crops", ["--nmb-crops", "3"], [3]),
        ("nmb_crops", ["--nmb-crops", "4", "5"], [4, 5]),
        ("nmb_crops", ["--nmb-crops", "'4 5'"], [4, 5]),
        ("nmb_crops", ["--nmb-crops", "'4 (5,6)'"], [4, (5, 6)]),
    ],
)
def test_create_argparser(attribute, args, expected):
    parsed = _parse.create_argparser().parse_args(args)

    result = getattr(parsed, attribute)

    assert result == expected
