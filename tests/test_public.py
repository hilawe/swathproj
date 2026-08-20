"""Public test suite: the forward and inverse maps, and construction validation.

Runs without any data files. The verification/ scripts cover agreement with real products.
"""
import numpy as np
import pytest

from swathproj import SwathGeometry
from swathproj.conical import ConicalGeometry

NSCAN = 2000
TIMES = np.linspace(0.0, 101.0 / 60.0, NSCAN)


def make_cross_track():
    return SwathGeometry(
        ref_lat=10.0, ref_lon=-75.0, heading=261.28, n_scan=NSCAN,
        scan_time_hours=TIMES, full_revolution=True,
    )


def test_cross_track_roundtrip():
    # interior scans only, since the orbit-overlap seam (j near 0 or n_scan) is deliberately
    # two-valued and is exercised separately below via inverse_all
    geo = make_cross_track()
    for i0 in (0, 200, 400, 600, 800):
        for j0 in (400, 800, 1200, 1600):
            lat, lon = geo.forward(i0, j0)
            i, j = geo.inverse(lat, lon)
            assert (int(i), int(j)) == (i0, j0)


def test_seam_recovers_on_one_branch():
    # at the self-overlap a point belongs to two scans, and the original must be one of them
    geo = make_cross_track()
    for j0 in (0, 1, NSCAN - 2, NSCAN - 1):
        lat, lon = geo.forward(400, j0)
        i, j, i2, j2 = geo.inverse_all(lat, lon)
        assert (400, j0) in {(int(i), int(j)), (int(i2), int(j2))}


def test_off_swath_returns_sentinel():
    geo = make_cross_track()
    for lat, lon in [(-45.0, -45.0), (60.0, 60.0)]:
        i, j = geo.inverse(lat, lon)
        assert (int(i), int(j)) == (-1, -1)


def test_forward_nan_for_out_of_range_scan():
    geo = make_cross_track()
    lat, lon = geo.forward(400, -1)
    assert np.isnan(float(lat)) and np.isnan(float(lon))


def test_cf_pole_is_stable():
    geo = make_cross_track()
    pole = geo.cf_rotated_pole()
    assert len(pole) == 3
    assert all(np.isfinite(p) for p in pole)


def test_construction_rejects_bad_parameters():
    with pytest.raises(ValueError):
        SwathGeometry(0.0, 0.0, 0.0, 4, scan_time_hours=np.zeros(4))  # under-specified axis
    with pytest.raises(ValueError):
        SwathGeometry(0.0, 0.0, 0.0, 4, n_pixel=3, nadir_index=1, cell_size_deg=180.0,
                      scan_time_hours=np.zeros(4), full_revolution=True)  # extent past a pole


def test_conical_forward_and_inverse_status():
    geo = ConicalGeometry(
        ref_lat=10.0, ref_lon=-75.0, heading=261.28, n_scan=NSCAN, n_pixel=90,
        scan_angle_deg=360.0 / NSCAN, cone_radius_deg=8.1, azimuth_origin_deg=-72.0,
        azimuth_step_deg=1.6, scan_time_hours=TIMES, full_revolution=True,
    )
    lat, lon = geo.forward(45, 1000)
    assert np.isfinite(float(lat)) and np.isfinite(float(lon))
    with pytest.raises(NotImplementedError):
        geo.inverse(lat, lon)
