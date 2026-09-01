"""One fitted along-track degree of freedom reproduces a VGAC orbit to its storage precision.

Run from the repo root with the project venv:

    .venv/bin/python verification/verify_vgac_alongtrack.py

WHAT THIS MEASURES, AND WHAT IT REFUSES TO NAME. The library previously left the Earth-rotation
rate at its nominal 15 degrees per hour and reported the resulting 0.33 km disagreement with the
stored coordinates as this model's accuracy. Fitting a single along-track degree of freedom instead
brings the agreement to about 0.3 m, which is the float32 quantum of the stored values. That result
is real and is asserted below on a held-out half of the orbit.

The thing that is fitted CANNOT be called an Earth-rotation rate on this evidence. The Earth-fixed
along-track angle of scan j is

    rot_lon(j) = j * scan_step - rate * t(j)

and in this product t(j) is exactly linear in j: t = a + b*j with the departure from that line at
the 1e-12 second level and a itself zero to floating point. Substituting,

    rot_lon(j) = (-rate * a) + j * (scan_step - rate * b)

so the data constrain only the intercept and the combined slope. With a = 0 the intercept carries
nothing, and a change in `rate` is indistinguishable from a compensating change in `scan_step`.
This script exhibits the degeneracy explicitly: the fitted-rate parameterisation and a nominal-rate
parameterisation with an adjusted scan step agree to about 1e-18 degrees across the whole orbit,
which is 1e-13 metres at the equator. They are the same model written two ways.

CONSEQUENCES worth carrying into the proposal.

- The defensible finding is "one fitted along-track degree of freedom reproduces this orbit to
  storage precision", not "the orbit requires a rotation rate of 14.994075 deg/hr".
- The identifiable quantity is the per-scan Earth-fixed along-track angle itself. Decomposing it
  into an inertial scan step and a rotation rate introduces a parameter the data do not determine,
  which is an argument for binding the combined angle in any encoding.
- Fitting the value separately on more orbits would not resolve this, since the degeneracy is
  present in every orbit with a linear time axis. What would test it is TRANSFER: fit on one orbit
  and apply the value unchanged to another.
- The earlier puzzle over why 14.994075 matches neither the sidereal rate nor the sun-synchronous
  nominal dissolves. The number depends on having assumed scan_step = 360/n_scan exactly, and it
  moves if that assumption moves.

SCOPE. One orbit, one platform (NOAA-20, from the archive path and orbit series, not from the
file's own platform attribute, which is wrong), one date. Nothing here bears on the native
cross-track or conical products.
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

    # --- PART 2: the degeneracy, shown by construction --------------------------------------
    step_a = geo._scan_angd                       # 360 / n_scan
    slope = step_a - EXPECTED_FITTED * b          # the identifiable combination
    step_b = slope + NOMINAL_RATE * b             # the step that makes the NOMINAL rate fit
    worst_deg = abs((-NOMINAL_RATE * a) - (-EXPECTED_FITTED * a)) + \
        abs((step_b - NOMINAL_RATE * b) - slope) * (n_scan - 1)
    print(f"parameterisation A: scan_step={step_a:.12f}, rate={EXPECTED_FITTED}")
    print(f"parameterisation B: scan_step={step_b:.12f}, rate={NOMINAL_RATE}")
    print(f"   worst along-track difference over the orbit = {worst_deg:.2e} deg "
          f"= {worst_deg * 111190:.2e} m")
    assert worst_deg < 1e-12, (
        f"the two parameterisations differ by {worst_deg:.2e} deg, so they are NOT degenerate "
        f"and the rate may be identifiable after all. Re-derive before claiming otherwise.")

    # --- PART 3: the reconstruction result, fitted on one half and scored on the other -------
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

    nominal = residual(jj, ii, NOMINAL_RATE)
    print(f"with the nominal {NOMINAL_RATE}: median {np.median(nominal):.4f} km, "
          f"max {nominal.max():.4f} km")
    assert np.median(nominal) > 0.1, (
        "the nominal setting no longer reproduces the previously published kilometre-scale "
        "residual, so the explanation for that figure is no longer demonstrated")

    print("\nvgac along-track: one fitted along-track degree of freedom reproduces this orbit to "
          "storage precision, and that degree of freedom is NOT identifiable as a rotation rate "
          "from a single orbit with a linear time axis")
    return 0


if __name__ == "__main__":
    sys.exit(main())
