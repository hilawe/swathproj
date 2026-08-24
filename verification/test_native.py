"""Validate the native (unresampled) cross-track whiskbroom geometry.

Run from the repo root with the project venv:

    .venv/bin/python verification/test_native.py

No data files are needed: these are synthetic checks of the slant-range projection and the
NativeCrossTrackGeometry forward/inverse. A separate script (verify_native_crosstrack.py) does the
real-file comparison once a native VIIRS geolocation granule is present.

The load-bearing check is test 1. The slant-range formula
gamma(theta) = arcsin(k sin theta) - theta is checked against a wholly independent derivation:
a 3-D ray cast from the satellite at the look angle, intersected with the Earth sphere, whose
central angle from the sub-satellite point is measured directly. The two share no algebra, so
the oracle cannot pass a formula that is merely self-consistent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "verification"))

from swathproj.native_geometry import (  # noqa: E402
    NativeCrossTrackGeometry,
    central_angle_to_look_angle,
    look_angle_to_central_angle,
)
from swathproj.geometry import EARTH_RADIUS_KM  # noqa: E402
from verify_native_crosstrack import (  # noqa: E402
    central_from_zenith,
    fit_crosstrack_index,
    fit_crosstrack_profile,
)
from swathproj.rotated_pole import rotated_to_geographic  # noqa: E402


def _central_angle_by_ray(look_deg, altitude_km, R=EARTH_RADIUS_KM):
    """Independent oracle: cast a ray from the satellite and intersect the Earth sphere.

    Earth centre at the origin, sphere radius R. Sub-satellite point at (R, 0, 0), satellite at
    (R + h, 0, 0). The cross-track scan sweeps the x-z plane, and a look angle theta rotates the
    nadir direction (-1, 0, 0) about the y axis toward +z, giving d = (-cos theta, 0, sin theta).
    The nearest sphere intersection P has central angle arccos(P_x / R) from the sub-satellite
    point. This uses only vector algebra and the quadratic for a line meeting a sphere, sharing
    nothing with arcsin(k sin theta) - theta.
    """
    theta = np.asarray(look_deg, float) * np.pi / 180.0
    h = np.asarray(altitude_km, float)
    sx = R + h
    d = np.stack([-np.cos(theta), np.zeros_like(theta), np.sin(theta)], axis=-1)
    # |S + t d|^2 = R^2  ->  t^2 - 2 (R+h) cos theta t + ((R+h)^2 - R^2) = 0
    b = sx * np.cos(theta)
    c = sx**2 - R**2
    disc = b**2 - c
    t = b - np.sqrt(disc)                       # nearest intersection
    px = sx - t * np.cos(theta)
    gamma = np.degrees(np.arccos(np.clip(px / R, -1.0, 1.0)))
    return gamma


def test_gamma_matches_independent_ray_oracle():
    R = EARTH_RADIUS_KM
    for h in (450.0, 705.0, 833.0, 870.0):
        looks = np.linspace(0.0, 60.0, 61)
        # keep only look angles that actually reach the Earth at this altitude
        s = ((R + h) / R) * np.sin(np.radians(looks))
        looks = looks[np.abs(s) <= 1.0 - 1e-6]
        gamma_formula = look_angle_to_central_angle(looks, h, R)
        gamma_oracle = _central_angle_by_ray(looks, h, R)
        err = np.max(np.abs(gamma_formula - gamma_oracle))
        assert err < 1e-9, f"h={h}: closed-form gamma disagrees with ray oracle by {err} deg"
    print("test 1 gamma vs independent ray oracle: agree to < 1e-9 deg")


def test_look_central_roundtrip():
    h = np.array([700.0, 720.0, 760.0])[:, None]
    looks = np.linspace(-55.0, 55.0, 45)[None, :]
    gamma = look_angle_to_central_angle(looks, h)
    back = central_angle_to_look_angle(gamma, h)
    err = np.max(np.abs(back - np.broadcast_to(looks, gamma.shape)))
    assert err < 1e-9, f"look -> central -> look error {err} deg"
    # oddness: gamma(-theta) = -gamma(theta)
    g_pos = look_angle_to_central_angle(np.array([10.0, 30.0, 50.0]), 800.0)
    g_neg = look_angle_to_central_angle(np.array([-10.0, -30.0, -50.0]), 800.0)
    assert np.max(np.abs(g_pos + g_neg)) < 1e-12, "gamma is not odd in the look angle"
    print("test 2 look/central round trip and oddness: exact")


def test_native_spacing_grows_off_nadir():
    """The native rotated-latitude step must grow toward the swath edge, and the native model
    must diverge sharply from the equal-angle model, which is the whole reason native needs its
    own formula."""
    h = 833.0
    step = 0.32                       # look-angle step, deg
    looks = np.arange(0.0, 56.0, step)
    gamma = look_angle_to_central_angle(looks, h)
    dgamma = np.diff(gamma)           # rotated-latitude step per sample
    nadir_step, edge_step = dgamma[0], dgamma[-1]
    assert edge_step > 4.0 * nadir_step, (
        f"edge step {edge_step:.4f} deg is not much larger than the nadir step "
        f"{nadir_step:.4f} deg, and the native non-uniformity is missing")
    # an equal-angle model pinned to the nadir step would place the edge sample here:
    equal_angle_edge = nadir_step * (len(looks) - 1)
    native_edge = gamma[-1]
    gap_km = abs(native_edge - equal_angle_edge) * (np.pi / 180.0) * EARTH_RADIUS_KM
    assert gap_km > 100.0, (
        f"native and equal-angle edge differ by only {gap_km:.1f} km, and a resampled grid would "
        "not have exposed the native requirement")
    print(f"test 3 native spacing grows off nadir: edge/nadir step "
          f"{edge_step / nadir_step:.1f}x, native vs equal-angle edge gap {gap_km:.0f} km")


def test_altitude_moves_the_edge():
    edge_low = look_angle_to_central_angle(55.0, 800.0)
    edge_high = look_angle_to_central_angle(55.0, 837.0)   # +37 km, the measured pass variation
    move_km = abs(edge_high - edge_low) * (np.pi / 180.0) * EARTH_RADIUS_KM
    assert move_km > 20.0, (
        f"a 37 km altitude change moves the edge central angle by only {move_km:.1f} km, and a "
        "constant altitude would then be adequate, contradicting the per-scan design")
    print(f"test 4 altitude drives the edge: 37 km altitude change moves the edge {move_km:.0f} km")


def _make_native(n_scan=40, n_pixel=51, altitude=None):
    times = np.linspace(0.0, 12.0 / 60.0, n_scan)      # 12 minutes, hours
    if altitude is None:
        altitude = 833.0
    return NativeCrossTrackGeometry(
        ref_lat=5.0, ref_lon=-40.0, heading=190.0,
        n_scan=n_scan, n_pixel=n_pixel, scan_angle_deg=0.24,
        look_angle_step_deg=56.0 / (n_pixel // 2),   # nadir at the centre sample
        nadir_index=n_pixel // 2, altitude_km=altitude,
        scan_time_hours=times,
    )


def test_forward_inverse_roundtrip():
    """Forward every cell of a synthetic native swath, then invert. Containment must recover the
    originating cell exactly. This exercises the closed-form cross-track inverse."""
    geo = _make_native()
    ii, jj = np.meshgrid(np.arange(geo.n_pixel), np.arange(geo.n_scan), indexing="ij")
    lat, lon = geo.forward(ii, jj)
    ri, rj = geo.inverse(lat, lon)
    miss = np.count_nonzero((ri != ii) | (rj != jj))
    assert miss == 0, f"{miss} of {ii.size} native cells did not round-trip through the inverse"
    print(f"test 5 forward/inverse round trip: {ii.size} cells, 100% recovered")


def test_per_scan_altitude_roundtrips():
    """A per-scan altitude array (the real case) must also round-trip, not just a scalar."""
    n_scan = 30
    alt = 815.0 + 20.0 * np.sin(np.linspace(0, np.pi, n_scan))   # varies ~20 km across the pass
    geo = _make_native(n_scan=n_scan, altitude=alt)
    ii, jj = np.meshgrid(np.arange(geo.n_pixel), np.arange(geo.n_scan), indexing="ij")
    lat, lon = geo.forward(ii, jj)
    ri, rj = geo.inverse(lat, lon)
    assert np.array_equal(ri, ii) and np.array_equal(rj, jj), "per-scan altitude did not round-trip"
    # the altitude really enters the FORWARD mapping: freezing it to the mean must move the
    # edge pixels' geographic positions. (The discrete inverse index is deliberately robust to a
    # sub-half-step perturbation, so the divergence is checked on positions, not indices.)
    geo_const = _make_native(n_scan=n_scan, altitude=float(alt.mean()))
    lat_c, lon_c = geo_const.forward(ii, jj)
    sep_km = _separation_km(lat[0], lon[0], lat_c[0], lon_c[0])   # edge column i = 0
    assert np.max(sep_km) > 10.0, (
        f"freezing altitude to the mean moved the edge pixels by at most {np.max(sep_km):.1f} "
        "km, so the per-scan altitude is barely used in the mapping")
    print(f"test 6 per-scan altitude round trip: exact, and freezing altitude moves edge "
          f"pixels up to {np.max(sep_km):.0f} km")


def _separation_km(lat1, lon1, lat2, lon2, R=EARTH_RADIUS_KM):
    """Great-circle separation in km between matched geographic points."""
    dlat = np.radians(np.asarray(lat2, float) - np.asarray(lat1, float))
    dlon = np.radians(np.asarray(lon2, float) - np.asarray(lon1, float))
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2)
    return 2.0 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def test_off_swath_returns_minus_one():
    geo = _make_native()
    # the antipode of the sub-satellite region is certainly off swath
    ri, rj = geo.inverse(-5.0, 140.0)
    assert int(ri) == -1 and int(rj) == -1, f"off-swath point returned ({ri}, {rj}), not (-1, -1)"
    print("test 7 off-swath point returns (-1, -1)")


def test_nadir_maps_to_sub_satellite_point():
    """forward at the nadir sample of scan 0 must equal the independently computed sub-satellite
    point (rotated (0, scan_origin) through the pole, then this scan's Earth-rotation shift)."""
    geo = _make_native()
    lat0, lon0 = geo.forward(geo.nadir_index, 0)
    pole_lat, pole_lon, npgl = geo.cf_rotated_pole()
    exp_lat, exp_lon = rotated_to_geographic(0.0, geo.scan_origin_deg, pole_lat, pole_lon, npgl)
    exp_lon = (exp_lon - geo.rotation_rate * geo.scan_time_hours[0] + 180.0) % 360.0 - 180.0
    sep = _separation_km(lat0, lon0, exp_lat, exp_lon)
    assert sep < 1e-6, f"nadir sample is {sep:.4g} km from the sub-satellite point, not on it"
    print("test 8 nadir sample lands on the independently computed sub-satellite point")


def test_physical_branch_is_enforced():
    """Beyond the Earth limb both helpers must return NaN, not a spurious in-range answer. Without
    this the inverse maps points far outside the swath onto edge pixels."""
    h = 833.0
    k = (EARTH_RADIUS_KM + h) / EARTH_RADIUS_KM
    theta_limb = np.degrees(np.arcsin(1.0 / k))
    gamma_max = np.degrees(np.arccos(1.0 / k))
    # a 130 deg look has sin within range but is non-physical
    assert not np.isfinite(look_angle_to_central_angle(130.0, h)), "130 deg look accepted"
    assert not np.isfinite(look_angle_to_central_angle(theta_limb + 0.5, h)), "past-limb look accepted"
    assert np.isfinite(look_angle_to_central_angle(theta_limb - 0.5, h)), "in-limb look rejected"
    # central angles past the limb value must not invert to a look angle
    assert not np.isfinite(central_angle_to_look_angle(gamma_max + 5.0, h)), "past-limb gamma accepted"
    assert not np.isfinite(central_angle_to_look_angle(55.0, h)), "gamma=55 (non-physical) accepted"
    assert np.isfinite(central_angle_to_look_angle(gamma_max - 1.0, h)), "in-range gamma rejected"
    print(f"test 9 physical branch enforced: limb look {theta_limb:.1f} deg, max central "
          f"{gamma_max:.1f} deg, beyond both -> NaN")


def test_crosstrack_fit_recovers_slantrange_and_rejects_equal_angle():
    """The verify_native_crosstrack harness logic, on a synthetic NON-uniformly sampled native profile
    (mimicking VIIRS aggregation): the slant-range fit, using per-pixel satellite zenith, must
    reproduce it and recover the altitude, while the equal-angle fit cannot. (Checks the fit CODE
    using the file-style zenith path. Test 1 checks the physics.)"""
    h = 833.0
    k = (EARTH_RADIUS_KM + h) / EARTH_RADIUS_KM
    # deliberately NON-uniform look angles across track, as an aggregated instrument delivers
    steps = np.concatenate([np.full(12, 0.9), np.full(8, 0.6), np.full(10, 0.3)])
    looks = np.concatenate([-np.cumsum(steps[::-1])[::-1], [0.0], np.cumsum(steps)])
    nadir = len(steps)
    gamma = look_angle_to_central_angle(looks, h)                 # signed central angles
    zen = np.degrees(np.arcsin(np.clip(k * np.sin(looks * np.pi / 180.0), -1.0, 1.0)))  # satellite zenith
    lat_row, lon_row = gamma, np.zeros_like(gamma)                # pixels along a meridian
    f = fit_crosstrack_profile(lat_row, lon_row, np.abs(zen), nadir)
    assert f["rms_native_km"] < 0.05, f"slant-range fit left {f['rms_native_km']:.3f} km on exact data"
    assert f["rms_equal_km"] > 5.0, (
        f"equal-angle fit residual is only {f['rms_equal_km']:.1f} km, so the non-uniform profile "
        "did not defeat the linear model")
    assert f["rms_native_km"] < 0.1 * f["rms_equal_km"], "slant-range did not clearly beat equal-angle"
    assert abs(f["fitted_altitude_km"] - h) < 15.0, (
        f"fitted altitude {f['fitted_altitude_km']:.0f} km is far from the true {h:.0f} km")
    print(f"test 10 cross-track fit (aggregated sampling): slant-range {f['rms_native_km']:.3f} km "
          f"vs equal-angle {f['rms_equal_km']:.0f} km, altitude {f['fitted_altitude_km']:.0f} km")


def test_index_fit_recovers_uniform_step():
    """fit_crosstrack_index (the non-circular pixel-index model) must recover a UNIFORM-step native
    profile's look-angle step and altitude from the array index alone, using no zenith. This is the
    check that AVHRR exercises against real geolocation."""
    n_pixel, nadir, step, h = 101, 50, 0.27, 828.0
    off = np.arange(n_pixel) - nadir
    gamma = look_angle_to_central_angle(off * step, h)   # signed central angles, uniform steps
    lat_row, lon_row = gamma, np.zeros_like(gamma)        # pixels along a meridian
    f = fit_crosstrack_index(lat_row, lon_row, nadir)
    assert f["rms_km"] < 0.05, f"pixel-index fit left {f['rms_km']:.3f} km on exact data"
    assert abs(f["look_step_deg"] - step) < 0.005, f"look step off: {f['look_step_deg']:.4f}"
    assert abs(f["fitted_altitude_km"] - h) < 15.0, f"altitude off: {f['fitted_altitude_km']:.0f}"
    print(f"test 11 pixel-index fit recovers uniform step {f['look_step_deg']:.4f} deg/pixel and "
          f"altitude {f['fitted_altitude_km']:.0f} km ({f['rms_km']:.3f} km residual)")


if __name__ == "__main__":
    test_gamma_matches_independent_ray_oracle()
    test_look_central_roundtrip()
    test_native_spacing_grows_off_nadir()
    test_altitude_moves_the_edge()
    test_forward_inverse_roundtrip()
    test_per_scan_altitude_roundtrips()
    test_off_swath_returns_minus_one()
    test_nadir_maps_to_sub_satellite_point()
    test_physical_branch_is_enforced()
    test_crosstrack_fit_recovers_slantrange_and_rejects_equal_angle()
    test_index_fit_recovers_uniform_step()
    print("\nnative geometry: all synthetic tests passed")
