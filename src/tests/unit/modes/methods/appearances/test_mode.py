import datetime

import pytest


@pytest.fixture
def mode(mode_appearances):
    return mode_appearances.modes[0]


class TestMode:
    def test_dates(self, mode):
        expected = [
            datetime.datetime(2000, 1, 1),
            datetime.datetime(2000, 1, 2),
            datetime.datetime(2000, 1, 3),
        ]

        result = list(mode.dates)

        assert result == expected
