import contextlib

import lifetimes.cli.conversion as conversion
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
    with pytest.raises(type(expected)) if isinstance(
        expected, Exception
    ) else contextlib.nullcontext():
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
    with pytest.raises(type(expected)) if isinstance(
        expected, Exception
    ) else contextlib.nullcontext():
        func = conversion.cast_optional(type_)
        result = func(s)

        assert result == expected
