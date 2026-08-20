"""Verify the projection against a real VGAC orbit file, the authoritative ground truth.

Run from the repo root with the project venv:

    .venv/bin/python verification/verify_real_vgac.py

Needs a production VGAC orbit at data/VGAC_sample_orbit.nc (about 475 MB, gitignored). Fetch
one from NOAA's public archive, for example:

    curl -o data/VGAC_sample_orbit.nc \\
      https://noaa-vgac-pds.s3.amazonaws.com/noaa-20/2026/01/01/VGAC_VN2002MOD_A2026001_0030_n42076_K005.nc

The script skips with a clear message when the file is absent, so the rest of the suite still
runs on a fresh clone.

Why this file settles things the synthetic tests cannot. Every other test in this project
compares one formulation against another, so a convention shared by both is invisible. A
production file carries the projection parameters AND the per-cell latitude and longitude the
producer actually published, so it can adjudicate between formulations. Two questions about the published formulation are answered here.

RESULT 1, the cross-track convention. Fitting the only free parameter, the cross-track offset,
against the file's own coordinates gives a sharp minimum at cell centre = (i - 400) * beta with
i a ZERO-BASED array index. Residual 0.34 km, flat across the whole swath, which is the
sub-cell scatter expected because the published coordinates are means of the contributing VIIRS
pixels inside a 3.9 km cell. Moving the offset costs about 0.97 km per quarter cell, linearly.
Two nearby conventions are therefore wrong:
  - (i - 401 + 0.5), which this library used before this check, is half a cell out, 1.95 km.
  - (i - 401), the paper's Sect. 3.2 formula read with a zero-based i, is a full cell out,
    3.89 km. It is consistent only if i is one-based, which the paper does not state.

RESULT 2, which forward formulation is authoritative. Against the file, the spherical rotation
used here has a median error of 0.34 km, while the tutorial's ellipsoidal `vgac_to_earth` has a
median error of 18 km. The spherical formulation is the one that reproduces the published data.
Earlier rounds of this project described the gap between them as "a tutorial-internal
inconsistency rather than an error in the archived data. The measurement here establishes the
direction with evidence.

RESULT 3, a consequence for the published inverse. `earth_to_vgac` uses NADIR = 401 with
half-cell centres, which places the file's true cell centres exactly on ITS cell boundaries.
Its output at real cell centres is therefore ambiguous, tipping either way on floating-point
rounding, and it recovers the exact cell for only about half of them. This library recovers
about 99.8%.

None of this is a criticism of the dataset itself. The geolocation in the file is
self-consistent, and what differs is the indexing convention assumed by the published helper code
and by the formula as literally written in the preprint.
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "reference" / "knappweathersat-VGAC-tutorial-bb3556d"))

DATA = REPO / "data" / "VGAC_sample_orbit.nc"
EARTH_KM = 6371.0


def great_circle_km(lat1, lon1, lat2, lon2):
    d2r = np.pi / 180.0
    a1, a2 = lat1 * d2r, lat2 * d2r
    dlon = (lon2 - lon1) * d2r
    h = np.sin((a2 - a1) / 2) ** 2 + np.cos(a1) * np.cos(a2) * np.sin(dlon / 2) ** 2
    return 2 * np.arcsin(np.sqrt(np.clip(h, 0, 1))) * EARTH_KM


def load():
    import netCDF4
    ds = netCDF4.Dataset(str(DATA))
    return {
        "ref_lat": float(ds["proj_lat0"][:]),
        "ref_lon": float(ds["proj_lon0"][:]),
        "heading": float(ds["proj_rot"][:]),
        "scan_time": np.asarray(ds["time"][:], float),
        "lat": np.asarray(ds["lat"][:], float),
        "lon": np.asarray(ds["lon"][:], float),
    }


def main():
    if not DATA.exists():
        print(f"SKIP: {DATA} not present. See this file's docstring for the download command.")
        return 0

    from swathproj import SwathGeometry
    from swathproj.rotated_pole import rotated_to_geographic
    from vgac_projection import earth_to_vgac, vgac_to_earth

    d = load()
    n_scan, n_pixel = d["lat"].shape
    geo = SwathGeometry(d["ref_lat"], d["ref_lon"], d["heading"], n_scan,
                        n_pixel=n_pixel, scan_time_hours=d["scan_time"], full_revolution=True)
    pole_lat, pole_lon, npgl = geo.cf_rotated_pole()
    rng = np.random.default_rng(0)
    jj = rng.integers(0, n_scan, 4000)
    ii = rng.integers(0, n_pixel, 4000)
    tgt_lat, tgt_lon = d["lat"][jj, ii], d["lon"][jj, ii]

    def residual_for_offset(offset):
        rot_lat = (ii - offset) * geo.cell_size_deg
        rot_lon = jj * geo._scan_angd
        la, lo = rotated_to_geographic(rot_lat, rot_lon, pole_lat, pole_lon, npgl)
        lo = (lo - geo.rotation_rate * d["scan_time"][jj] + 180.0) % 360.0 - 180.0
        return great_circle_km(tgt_lat, tgt_lon, la, lo)

    # RESULT 1: the cross-track offset is fitted, not assumed
    offsets = np.arange(399.0, 402.01, 0.25)
    medians = [float(np.median(residual_for_offset(o))) for o in offsets]
    best = offsets[int(np.argmin(medians))]
    print(f"1: fitted cross-track offset = i - {best:.2f} "
          f"(residual {min(medians):.3f} km), library uses i - {geo.nadir_index}")
    assert best == 400.0, f"the file says the offset is {best}, not 400"
    assert geo.nadir_index == 400, "library nadir_index disagrees with the fitted offset"
    assert min(medians) < 0.6, "best residual is larger than the expected sub-cell scatter"

    # RESULT 2: which forward formulation reproduces the file
    rot_lat = (ii - geo.nadir_index) * geo.cell_size_deg
    rot_lon = jj * geo._scan_angd
    la_s, lo_s = geo.forward(ii, jj)
    la_e, lo_e = vgac_to_earth(d["ref_lat"], d["ref_lon"], d["heading"], rot_lat, rot_lon)
    lo_e = (lo_e - geo.rotation_rate * d["scan_time"][jj] + 180.0) % 360.0 - 180.0
    err_spherical = float(np.median(great_circle_km(tgt_lat, tgt_lon, la_s, lo_s)))
    err_ellipsoid = float(np.median(great_circle_km(tgt_lat, tgt_lon, la_e, lo_e)))
    print(f"2: median error vs the file: this library {err_spherical:.3f} km, "
          f"tutorial vgac_to_earth {err_ellipsoid:.3f} km")
    assert err_spherical < 0.6, "the spherical rotation no longer reproduces the file"
    assert err_ellipsoid > 5.0 * err_spherical, \
        "vgac_to_earth is no longer clearly worse, re-examine which formulation is authoritative"

    # RESULT 3: inverse recovery from the file's own coordinates, both implementations
    sub = slice(0, 400)   # the inverse is exhaustive by design, so keep the sample modest
    fi, fj = geo.inverse(tgt_lat[sub], tgt_lon[sub])
    ours = float(np.mean((fi == ii[sub]) & (fj == jj[sub])))
    pi, pj = earth_to_vgac(d["ref_lat"], d["ref_lon"], d["heading"], d["scan_time"],
                           tgt_lat[sub], tgt_lon[sub])
    theirs = float(np.mean((np.asarray(pi) == ii[sub]) & (np.asarray(pj) == jj[sub])))
    print(f"3: exact-cell recovery from the file's own coordinates: "
          f"this library {ours:.1%}, published earth_to_vgac {theirs:.1%}")
    assert ours > 0.99, "inverse recovery from real data regressed"
    assert theirs < 0.75, "published inverse recovery changed, re-check the convention finding"

    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
