import a6.plotting.extents as extents


class TestGermany:
    def test_to_dict(self):
        expected = {
            "longitude": slice(5.5, 15.3),
            "latitude": slice(55, 47),
        }

        result = extents.Germany.to_dict()

        assert result == expected
