"""Is the fitted along-track parameter identifiable, and does it reproduce a VGAC orbit?

Run from the repo root with the project venv:

    .venv/bin/python verification/verify_vgac_alongtrack.py

THE QUESTION. Fitting one along-track parameter brings the mapping into agreement with the stored
coordinates at about 0.3 m, against a float32 storage quantum of 0.4 m. Before that parameter may
be called an Earth-rotation rate, it has to be shown that the data can tell it apart from the other
along-track quantity in the model, the per-scan angular step. If a compensating change in the step
reproduced the orbit equally well, the fit would determine only a combination and the name would be
unearned.

THE CONCERN IS REAL FOR THE ABSTRACT FORM. Write the Earth-fixed along-track angle as

    along(j) = j * scan_step - rate * t(j)

with both terms in one angular variable. In this product t is exactly linear in j (worst departure
8.2e-13 s, intercept -2.3e-16 h, zero to floating point), so that expression collapses to
`(-rate*a) + j*(scan_step - rate*b)` and only the combined slope is determined. Part 2 below
confirms that: the two parameterisations evaluate within 1e-13 degrees of each other.

BUT THIS MODEL DOES NOT HAVE THAT FORM, and Part 3 measures the difference. The scan step is a
ROTATED-FRAME longitude applied BEFORE the rotated-pole transformation, so changing it moves a cell
ALONG THE GROUND TRACK, altering geographic latitude and longitude together. The rate is subtracted
from GEOGRAPHIC longitude AFTER the transformation, a pure zonal shear that leaves latitude
untouched. Those are different displacements, and latitude is what separates them. Measured on the
reference orbit:

    scan_step = 360/n_scan, fitted rate 14.994075     ->    0.34 m
    scan_step = 360/n_scan, nominal rate 15.0         ->  330.72 m
    compensating step from the algebra, nominal 15.0  ->  808.39 m

The substitution the abstract algebra prescribes makes the fit WORSE than leaving the rate at its
nominal value. The parameters are not interchangeable in this model, so the rate is identifiable
from a single orbit and the fitted value is a property of the orbit rather than of a convention.

WHAT REMAINS UNSETTLED. That the parameter is identifiable does not establish that the producer
holds it, or that it is what generates the archived coordinates. Agreement at storage precision is
consistent with that and does not demonstrate it. A cross-orbit transfer test, fitting on one orbit
and applying the value unchanged to another, is the next discriminating measurement.

SCOPE. One orbit, one platform (NOAA-20, from the archive path and orbit series, not the file's own
platform attribute, which is wrong), one date. Nothing here bears on the native cross-track or
conical products.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DATA = REPO / "data" / "VGAC_sample_orbit.nc"
EARTH_KM = 6371.0
NOMINAL_RATE = 15.0
EXPECTED_FITTED = 14.994075   # under the scan_step = 360/n_scan convention, and only under it


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

    # --- PART 1: the time axis is exactly linear, which is what creates the degeneracy -------
    idx = np.arange(n_scan, dtype=float)
    design = np.vstack([np.ones(n_scan), idx]).T
    (a, b), *_ = np.linalg.lstsq(design, scan_time, rcond=None)
    departure_s = float(np.abs(scan_time - (a + b * idx)).max() * 3600.0)
    print(f"time axis: t = {a:.3e} + {b:.12f}*index hours")
    print(f"   max departure from linear = {departure_s:.2e} s")
    assert departure_s < 1e-9, (
        f"the scan time departs from a straight line by {departure_s:.2e} s. The identifiability "
        f"argument below assumes exact linearity and no longer applies.")

    # --- PART 2: the ABSTRACT form is degenerate, which is the concern worth testing ------
    step_a, rate_a = geo._scan_angd, EXPECTED_FITTED
    slope = step_a - rate_a * b
    step_b, rate_b = slope + NOMINAL_RATE * b, NOMINAL_RATE

    along_a = idx * step_a - rate_a * scan_time
    along_b = idx * step_b - rate_b * scan_time
    eval_diff = float(np.abs((along_a - along_b + 180.0) % 360.0 - 180.0).max())
    theory_deg = abs(rate_a - rate_b) * abs(a)
    print(f"abstract along-track form, A: step={step_a:.12f} rate={rate_a}")
    print(f"                           B: step={step_b:.12f} rate={rate_b}")
    print(f"   theoretical remainder (via the intercept) = {theory_deg:.2e} deg")
    print(f"   double-precision evaluation difference    = {eval_diff:.2e} deg")
    assert theory_deg < 1e-15, (
        f"theoretical remainder {theory_deg:.2e} deg: the intercept is large enough to separate "
        f"the terms even in the abstract form, so Part 2's premise no longer holds")
    assert eval_diff < 1e-10, (
        f"the abstract forms differ by {eval_diff:.2e} deg, so they are NOT degenerate and this "
        f"script's framing needs re-deriving")

    # --- PART 3: the ACTUAL model is not degenerate, and this is the identifiability result -
    # The step enters as a rotated-frame longitude before the rotated-pole transform, and the rate
    # is subtracted from geographic longitude after it. A step change therefore moves a cell along the
    # ground track (latitude AND longitude) while a rate change is a pure zonal shear. If the
    # substitution Part 2 licenses were also valid here, parameterisation B would fit as well as A.
    valid = np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) <= 90)
    rng = np.random.default_rng(1)
    probe_j = rng.integers(0, n_scan, 30000)
    probe_i = rng.integers(0, n_pixel, 30000)
    keep = valid[probe_j, probe_i]
    probe_j, probe_i = probe_j[keep], probe_i[keep]

    def median_residual(step, rate, jj=None, ii=None):
        jj = probe_j if jj is None else jj
        ii = probe_i if ii is None else ii
        la, lo = rotated_to_geographic((ii - geo.nadir_index) * geo.cell_size_deg,
                                       jj * step, pole_lat, pole_lon, npgl)
        lo = (lo - rate * scan_time[jj] + 180.0) % 360.0 - 180.0
        return float(np.median(great_circle_km(lat[jj, ii], lon[jj, ii], la, lo)))

    m_fitted = median_residual(step_a, rate_a)
    m_nominal = median_residual(step_a, NOMINAL_RATE)
    m_substituted = median_residual(step_b, rate_b)
    print(f"\nactual model, median residual:")
    print(f"   step=360/n_scan, fitted rate {rate_a}      {m_fitted * 1000:9.2f} m")
    print(f"   step=360/n_scan, nominal rate {NOMINAL_RATE}          {m_nominal * 1000:9.2f} m")
    print(f"   compensating step, nominal rate {NOMINAL_RATE}        {m_substituted * 1000:9.2f} m")
    assert m_substituted > 1.5 * m_nominal, (
        f"the compensating step reaches {m_substituted * 1000:.1f} m against {m_nominal * 1000:.1f} m "
        f"for simply leaving the rate nominal. The substitution is supposed to make things WORSE, "
        f"because the step also moves latitude. If it does not, the two parameters may be "
        f"interchangeable in the real model and the rate would be unidentifiable.")
    assert m_fitted < 0.01 * m_substituted, (
        f"the fitted rate ({m_fitted * 1000:.1f} m) is not decisively better than the algebraic "
        f"substitution ({m_substituted * 1000:.1f} m), so the rate is not clearly identifiable")

    # --- PART 4: the reconstruction result, fitted on one half and scored on the other -------
    def residual(jj, ii, rate):
        rot_lat = (ii - geo.nadir_index) * geo.cell_size_deg
        rot_lon = jj * geo._scan_angd
        la, lo = rotated_to_geographic(rot_lat, rot_lon, pole_lat, pole_lon, npgl)
        lo = (lo - rate * scan_time[jj] + 180.0) % 360.0 - 180.0
        return great_circle_km(lat[jj, ii], lon[jj, ii], la, lo)

    half = n_scan // 2

    def sample(lo_scan, hi_scan, n):
        jj = rng.integers(lo_scan, hi_scan, n)
        ii = rng.integers(0, n_pixel, n)
        keep = valid[jj, ii]
        return jj[keep], ii[keep]

    fit_j, fit_i = sample(0, half, 40000)
    test_j, test_i = sample(half, n_scan, 40000)

    grid = np.linspace(14.97, 15.02, 2001)
    fitted = float(grid[int(np.argmin([np.median(residual(fit_j, fit_i, r)) for r in grid]))])
    print(f"\nfitted along-track parameter (as a rate, under scan_step=360/n_scan): {fitted:.6f}")
    assert abs(fitted - EXPECTED_FITTED) < 5e-5, (
        f"the fitted along-track parameter moved to {fitted:.6f}, away from the recorded "
        f"{EXPECTED_FITTED}. Either the geometry changed or the file did.")

    held = residual(test_j, test_i, fitted)
    ho_median = float(np.median(held))
    print(f"HELD-OUT scans {half}..{n_scan}: median {ho_median * 1000:.1f} m, "
          f"p95 {np.percentile(held, 95) * 1000:.1f} m, max {held.max() * 1000:.1f} m")
    assert ho_median < 0.002, (
        f"held-out median is {ho_median:.4f} km, so the fit does not transfer even across halves "
        f"of the same orbit")

    jj, ii = np.nonzero(valid)
    full = residual(jj, ii, fitted)
    quantum_m = float(np.spacing(np.float32(60.0)) * 111.19 * 1000)
    print(f"FULL ARRAY n={full.size}: median {np.median(full) * 1000:.1f} m, "
          f"max {full.max() * 1000:.1f} m  (float32 quantum {quantum_m:.1f} m)")
    assert np.median(full) < 0.002, "full-array median is not at storage precision"
    assert full.max() < 0.010, (
        f"full-array MAX is {full.max() * 1000:.1f} m. The median can stay at storage precision "
        f"while a minority of cells regress, so the tail is asserted separately.")

    nominal = residual(jj, ii, NOMINAL_RATE)
    print(f"with the nominal {NOMINAL_RATE}: median {np.median(nominal):.4f} km, "
          f"max {nominal.max():.4f} km")
    assert np.median(nominal) > 0.1, (
        "the nominal setting no longer reproduces the previously published kilometre-scale "
        "residual, so the explanation for that figure is no longer demonstrated")

    print("\nvgac along-track: one fitted along-track parameter reproduces this orbit to storage "
          "precision, and it IS identifiable as a zonal shear rate. The abstract additive form is "
          "degenerate, but this model applies the scan step before the rotated-pole transform, so "
          "the step also moves latitude and cannot absorb a rate change.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
