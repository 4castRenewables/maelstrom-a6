import datetime

import pytest


@pytest.fixture
def mode(mode_appearances):
    return mode_appearances.modes[0]


class TestMode:
    @pytest.mark.parametrize(
        ("start", "end", "expected"),
        [
            (
                None,
                None,
                [
                    datetime.datetime(2000, 1, 1),
                    datetime.datetime(2000, 1, 2),
                    datetime.datetime(2000, 1, 3),
                ],
            ),
            (
                datetime.datetime(2000, 1, 2),
                None,
                [
                    datetime.datetime(2000, 1, 2),
                    datetime.datetime(2000, 1, 3),
                ],
            ),
            (
                None,
                datetime.datetime(2000, 1, 2),
                [
                    datetime.datetime(2000, 1, 1),
                    datetime.datetime(2000, 1, 2),
                ],
            ),
            (
                datetime.datetime(2000, 1, 2),
                datetime.datetime(2000, 1, 2),
                [
                    datetime.datetime(2000, 1, 2),
                ],
            ),
        ],
    )
    def test_get_dates(self, mode, start, end, expected):
        result = list(mode.get_dates(start, end))

        assert result == expected
