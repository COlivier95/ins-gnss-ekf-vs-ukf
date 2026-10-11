import numpy as np

from insnav.geodesy import wgs84 as g

RNG = np.random.default_rng(0)


def test_lla_ecef_roundtrip():
    lla = np.column_stack([RNG.uniform(-1.5, 1.5, 200), RNG.uniform(-np.pi, np.pi, 200),
                           RNG.uniform(-500, 10000, 200)])
    back = g.ecef_to_lla(g.lla_to_ecef(lla))
    assert np.allclose(back[:, :2], lla[:, :2], atol=1e-11)   # ~ 0.06 mm
    assert np.allclose(back[:, 2], lla[:, 2], atol=1e-4)


def test_ecef_known_points():
    assert np.allclose(g.lla_to_ecef([0, 0, 0]), [g.A, 0, 0])
    assert np.allclose(g.lla_to_ecef([np.pi / 2, 0, 0]), [0, 0, g.B], atol=1e-6)


def test_ned_roundtrip_and_axes():
    o = np.array([np.radians(33.7), np.radians(-117.9), -23.0])
    ned = np.array([[100.0, 0, 0], [0, 100.0, 0], [0, 0, 10.0], [1234.5, -987.6, 3.2]])
    lla = g.ned_to_lla(ned, o)
    assert np.allclose(g.lla_to_ned(lla, o), ned, atol=1e-6)
    assert lla[0, 0] > o[0]          # north increases latitude
    assert lla[1, 1] > o[1]          # east increases longitude
    assert lla[2, 2] < o[2]          # down decreases height


def test_radii_small_displacement():
    lat = np.radians(33.7)
    r_n, r_e = g.radii_of_curvature(lat)
    o = np.array([lat, 0.3, 0.0])
    dlat, dlon = 1e-5, 1e-5
    ned = g.lla_to_ned(np.array([[lat + dlat, 0.3, 0.0], [lat, 0.3 + dlon, 0.0]]), o)
    assert abs(ned[0, 0] - r_n * dlat) < 1e-3
    assert abs(ned[1, 1] - r_e * np.cos(lat) * dlon) < 1e-3


def test_normal_gravity():
    assert abs(g.normal_gravity(0.0, 0.0) - 9.7803253359) < 1e-9
    assert abs(g.normal_gravity(np.pi / 2, 0.0) - 9.8321849378) < 1e-6
    assert g.normal_gravity(0.6, 1000.0) < g.normal_gravity(0.6, 0.0)       # ~ -3.1e-3 per km
    assert abs((g.normal_gravity(0.6, 0.0) - g.normal_gravity(0.6, 1000.0)) - 3.08e-3) < 1e-4


def test_linear_geoid():
    from insnav.geodesy.geoid import LinearGeoid
    pts = [[33.7003, -117.9083, -35.2447], [33.6730, -117.9177, -35.2936]]
    geo = LinearGeoid(pts)
    for lat, lon, n in pts:
        assert abs(geo(np.radians(lat), np.radians(lon)) - n) < 1e-9
    mid = geo(np.radians(33.68665), np.radians(-117.913))
    assert abs(mid - (-35.26915)) < 1e-4
    assert abs(LinearGeoid(pts[:1])(0.1, 0.2) - pts[0][2]) < 1e-12
