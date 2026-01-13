import pytest

from taqsim.geo import haversine


class TestHaversine:
    def test_same_point_returns_zero(self):
        assert haversine(0, 0, 0, 0) == 0.0
        assert haversine(45.0, 90.0, 45.0, 90.0) == 0.0

    def test_known_distance_london_paris(self):
        # London (51.5074, -0.1278) to Paris (48.8566, 2.3522)
        # Expected: ~344 km
        distance = haversine(51.5074, -0.1278, 48.8566, 2.3522)
        assert 340_000 < distance < 350_000

    def test_known_distance_new_york_los_angeles(self):
        # NYC (40.7128, -74.0060) to LA (34.0522, -118.2437)
        # Expected: ~3944 km
        distance = haversine(40.7128, -74.0060, 34.0522, -118.2437)
        assert 3_900_000 < distance < 4_000_000

    def test_equator_one_degree_longitude(self):
        # 1 degree longitude at equator ~111 km
        distance = haversine(0, 0, 0, 1)
        assert 110_000 < distance < 112_000

    def test_symmetry(self):
        d1 = haversine(10, 20, 30, 40)
        d2 = haversine(30, 40, 10, 20)
        assert d1 == pytest.approx(d2)

    def test_crossing_date_line(self):
        # Points on either side of the international date line
        distance = haversine(0, 179, 0, -179)
        # Should be about 222 km (2 degrees at equator)
        assert 220_000 < distance < 225_000
