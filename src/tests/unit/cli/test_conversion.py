import a6.cli.conversion as conversion
import a6.testing as testing
import pytest


@pytest.mark.parametrize(
    ("s", "expected"),
    [
        ("true", True),
        ("True", True),
        ("TRUE", True),
        ("T", True),
        ("t", True),
        ("1", True),
        ("y", True),
        ("Y", True),
        ("yes", True),
        ("Yes", True),
        ("YES", True),
        ("false", False),
        ("False", False),
        ("FALSE", False),
        ("F", False),
        ("f", False),
        ("0", False),
        ("n", False),
        ("N", False),
        ("no", False),
        ("No", False),
        ("NO", False),
        ("invalid", ValueError()),
    ],
)
def test_string_to_bool(s, expected):
    with testing.expect_raise_if_exception(expected):
        result = conversion.string_to_bool(s)

        assert result == expected


@pytest.mark.parametrize(
    ("s", "type_", "expected"),
    [
        ("1", int, 1),
        ("None", int, None),
        ("none", int, None),
        ("invalid", int, ValueError()),
    ],
)
def test_parse_optional_value(s, type_, expected):
    with testing.expect_raise_if_exception(expected):
        func = conversion.cast_optional(type_)
        result = func(s)

        assert result == expected
