"""The VGAC stored coordinates match the analytic mapping to the file's storage precision.

Run from the repo root with the project venv:

    .venv/bin/python verification/verify_vgac_rotation_rate.py

WHY THIS EXISTS. The proposal reported a VGAC residual of about 0.33 km and described it as model
error, and the coordinate-semantics analysis was built on the premise that the mapping and the
stored coordinates disagree at the kilometre scale by design. That residual is accounted for by the
library using the nominal Earth-rotation rate of 15.0 degrees per hour instead of the rate this
orbit requires. With that one scalar fitted, the mapping matches every stored coordinate in the file
to about 0.3 m, which is the float32 quantum of the stored values.

WHAT IT DOES NOT SHOW. Agreement at storage precision is consistent with the coordinates having been
produced by this mapping. It does not establish that they were, and it does not establish that
14.994075 deg/hr is a parameter the producer holds explicitly rather than a value this fit uses to
absorb longitudinal drift. Only the production algorithm settles that. Both numbers below are real
and answer different questions: 0.3 m is the reconstruction error after fitting the rate from the
orbit, and 0.3326 km is the error using the nominal 15.0.

WHAT MAKES THIS EVIDENCE RATHER THAN A FIT. The rate is fitted on the FIRST HALF of the scans and
scored on the SECOND HALF, which the fit never saw, and then on every valid pixel in the file.
Everything else the mapping needs is either read from the file (the three projection origin
parameters and the per-scan times) or is a nominal constant fixed before this test (3.9 km cells,
nadir index 400, the along-track step implied by orbit closure). One free parameter, 8.26 million
scored points, disjoint fit and test.

SCOPE. One orbit, one platform, one date. This says nothing about whether the rate is constant
across orbits, and nothing whatever about the native cross-track or conical products, whose
residuals are measured against genuine CF auxiliary coordinates and remain unexplained.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DATA = REPO / "data" / "VGAC_sample_orbit.nc"
EARTH_KM = 6371.0

# The rate this orbit requires, established by the fit below. Held as a constant so that a
# regression in the geometry code moves the fitted value away from it and fails, rather than
# quietly refitting to whatever the new code produces.
EXPECTED_RATE = 14.994075
NOMINAL_RATE = 15.0


def great_circle_km(lat1, lon1, lat2, lon2):
    d2r = np.pi / 180.0
    a1, a2 = lat1 * d2r, lat2 * d2r
    dlon = (lon2 - lon1) * d2r
    h = np.sin((a2 - a1) / 2) ** 2 + np.cos(a1) * np.cos(a2) * np.sin(dlon / 2) ** 2
    return 2 * np.arcsin(np.sqrt(np.clip(h, 0, 1))) * EARTH_KM


def main():
    if not DATA.exists():
        print(f"SKIP: {DATA} not present.")
        return 0

    import netCDF4
    from swathproj import SwathGeometry
    from swathproj.rotated_pole import rotated_to_geographic

    ds = netCDF4.Dataset(str(DATA))
    lat = np.asarray(ds["lat"][:], float)
    lon = np.asarray(ds["lon"][:], float)
    scan_time = np.asarray(ds["time"][:], float)
    n_scan, n_pixel = lat.shape

    geo = SwathGeometry(float(ds["proj_lat0"][:]), float(ds["proj_lon0"][:]),
                        float(ds["proj_rot"][:]), n_scan, n_pixel=n_pixel,
                        scan_time_hours=scan_time, full_revolution=True)
    pole_lat, pole_lon, npgl = geo.cf_rotated_pole()

    def residual(jj, ii, rate):
        rot_lat = (ii - geo.nadir_index) * geo.cell_size_deg
        rot_lon = jj * geo._scan_angd
        la, lo = rotated_to_geographic(rot_lat, rot_lon, pole_lat, pole_lon, npgl)
        lo = (lo - rate * scan_time[jj] + 180.0) % 360.0 - 180.0
        return great_circle_km(lat[jj, ii], lon[jj, ii], la, lo)

    valid = np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) <= 90)
    rng = np.random.default_rng(1)
    half = n_scan // 2

    def sample(lo_scan, hi_scan, n):
        jj = rng.integers(lo_scan, hi_scan, n)
        ii = rng.integers(0, n_pixel, n)
        keep = valid[jj, ii]
        return jj[keep], ii[keep]

    fit_j, fit_i = sample(0, half, 40000)
    test_j, test_i = sample(half, n_scan, 40000)

    # --- fit the single free parameter on the FIRST half only ------------------------------
    grid = np.linspace(14.97, 15.02, 2001)
    fitted = float(grid[int(np.argmin([np.median(residual(fit_j, fit_i, r)) for r in grid]))])
    print(f"rotation rate fitted on scans 0..{half}: {fitted:.6f} deg/hr")
    assert abs(fitted - EXPECTED_RATE) < 5e-5, (
        f"the fitted rotation rate moved to {fitted:.6f}, away from the recorded "
        f"{EXPECTED_RATE}. Either the geometry changed or the file did.")

    # --- score on the SECOND half, which the fit never saw ---------------------------------
    held_out = residual(test_j, test_i, fitted)
    ho_median = float(np.median(held_out))
    print(f"HELD-OUT scans {half}..{n_scan}: median {ho_median * 1000:.1f} m, "
          f"p95 {np.percentile(held_out, 95) * 1000:.1f} m, max {held_out.max() * 1000:.1f} m")
    assert ho_median < 0.002, (
        f"held-out median is {ho_median:.4f} km. The claim that the mapping reproduces the "
        f"stored coordinates to the storage precision does not hold on data the fit never saw.")

    # --- score on EVERY valid pixel --------------------------------------------------------
    jj, ii = np.nonzero(valid)
    full = residual(jj, ii, fitted)
    full_median = float(np.median(full))
    quantum_m = float(np.spacing(np.float32(60.0)) * 111.19 * 1000)
    print(f"FULL ARRAY n={full.size}: median {full_median * 1000:.1f} m, "
          f"p95 {np.percentile(full, 95) * 1000:.1f} m, max {full.max() * 1000:.1f} m")
    print(f"float32 quantum on a stored latitude near 60 deg: {quantum_m:.1f} m")
    assert full_median < 0.002, (
        f"full-array median is {full_median:.4f} km, not at storage precision")
    assert full.max() < 0.010, f"full-array max is {full.max():.4f} km, larger than expected"

    # --- and that the NOMINAL rate is what produced the previously published figure ---------
    nominal = residual(jj, ii, NOMINAL_RATE)
    span_h = float(scan_time.max() - scan_time.min())
    predicted_km = (NOMINAL_RATE - fitted) * span_h * 111.19
    print(f"with the nominal {NOMINAL_RATE} deg/hr: median {np.median(nominal):.4f} km, "
          f"max {nominal.max():.4f} km")
    print(f"predicted from the rate error alone: "
          f"{(NOMINAL_RATE - fitted) * span_h:.5f} deg = {predicted_km:.3f} km at the equator")
    assert np.median(nominal) > 0.1, (
        "the nominal rate no longer reproduces the previously published kilometre-scale "
        "residual, so the explanation for that figure is no longer demonstrated")
    assert abs(nominal.max() - predicted_km) < 0.05, (
        f"the worst residual under the nominal rate is {nominal.max():.3f} km but the rate "
        f"error alone predicts {predicted_km:.3f} km. The published figure is then NOT fully "
        f"explained by the rate, and the claim that none of it is physical does not hold.")

    print("\nvgac rotation rate: the stored coordinates match the mapping to storage precision "
          "once one orbit-specific rate is fitted, and the previously published kilometre-scale "
          "residual is accounted for by the nominal rate constant")
    return 0


if __name__ == "__main__":
    sys.exit(main())
