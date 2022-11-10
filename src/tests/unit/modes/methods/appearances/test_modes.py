import datetime

import pytest

import a6.testing as testing


class TestModes:
    def test_size(self, mode_appearances):
        expected = 2

        result = mode_appearances.size

        assert result == expected

    def test_labels(self, mode_appearances):
        expected = [0, 1]

        result = list(mode_appearances.labels)

        assert result == expected

    @pytest.mark.parametrize(
        ("label", "expected"), [(0, 0), (1, 1), (2, ValueError())]
    )
    def test_get_mode(self, mode_appearances, label, expected):

        with testing.expect_raise_if_exception(expected):
            result = mode_appearances.get_mode(label)
            expected = mode_appearances.modes[expected]

            assert result == expected

    @pytest.mark.parametrize(
        ("date", "expected"),
        [
            (datetime.datetime(2000, 1, 1), 0),
            (datetime.datetime(2000, 1, 2), 0),
            (datetime.datetime(2000, 1, 3), 0),
            (datetime.datetime(2000, 1, 4), 1),
            (datetime.datetime(2000, 1, 5), 1),
            (datetime.datetime(2000, 1, 6), 1),
            (datetime.datetime(2000, 1, 7), ValueError()),
        ],
    )
    def test_get_appearance(self, mode_appearances, date, expected):
        with testing.expect_raise_if_exception(expected):
            result = mode_appearances.get_appearance(date)

            expected = mode_appearances.modes[expected].appearances[0]
            assert result == expected
