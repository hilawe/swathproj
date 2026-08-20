"""Synthetic round-trip checks of the VGAC projection tutorial code.

Run from the repo root with the project venv:

    .venv/bin/python verification/roundtrip_synthetic.py

Two tests against reference/knappweathersat-VGAC-tutorial-bb3556d/vgac_projection.py,
using a synthetic orbit (proj_lat0=10, proj_lon0=-75, proj_rot=360-98.7, 101-minute
period, linear scan time). No data download needed.

Test A (tutorial forward vs tutorial inverse): vgac_to_earth() followed by
earth_to_vgac() misplaces cells by up to ~7 cross-track and ~4 along-track indices.
The two functions use different Earth models (vgac_to_earth converts through the WGS84
ellipsoid via earth2xyz, while earth_to_vgac is deliberately spherical to match the
production forward transform) and different nadir conventions (400 vs 401), so their
disagreement measures implementation inconsistency inside the tutorial, not the
accuracy of the projection itself.

Test B (algebraic forward vs tutorial inverse): inverting earth_to_vgac's own
spherical rotation sequence in closed form and feeding cell centers through it, the
iterative inverse (5 iterations) recovers the exact (i, j) for 94.9% of 731 sampled
cells. Every failure is at the orbit seam (j = 0 or j >= 10250), where the swath
overlaps itself after one revolution (~25 deg of Earth rotation) and the Earth-to-cell
mapping is genuinely one-to-many, and the iteration lands on the earlier scan. Away from
the seam the inverse is exact at cell resolution.

Both behaviors were observed on 2026-08-12 with numpy under Python 3.9.
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "reference" / "knappweathersat-VGAC-tutorial-bb3556d"))

from vgac_projection import ANGD, NADIR, earth_to_vgac, vgac_to_earth  # noqa: E402

D2R = np.pi / 180.0

PROJ_LAT0, PROJ_LON0, PROJ_ROT = 10.0, -75.0, 360.0 - 98.7
NSCN = 10313
SCANTIME = np.linspace(0.0, 101.0 / 60.0, NSCN)
SCAN_ANGD = 360.0 / NSCN


def forward_from_inverse(i, j):
    """Closed-form inversion of earth_to_vgac's spherical rotation sequence."""
    lat2 = (i - NADIR + 0.5) * ANGD
    lon2 = j * SCAN_ANGD
    la, lo = lat2 * D2R, lon2 * D2R
    xpp = np.cos(la) * np.cos(lo)
    yppp = np.cos(la) * np.sin(lo)
    zppp = np.sin(la)
    sg, cg = np.sin(-PROJ_ROT * D2R), np.cos(-PROJ_ROT * D2R)
    yp = yppp * cg - zppp * sg
    zpp = yppp * sg + zppp * cg
    sw, cw = np.sin(-PROJ_LAT0 * D2R), np.cos(-PROJ_LAT0 * D2R)
    xp = xpp * cw + zpp * sw
    zp = -xpp * sw + zpp * cw
    lat = np.degrees(np.arctan2(zp, np.hypot(xp, yp)))
    lon = np.degrees(np.arctan2(yp, xp)) + PROJ_LON0 - 15.0 * SCANTIME[j]
    return lat, (lon + 180.0) % 360.0 - 180.0


def test_a_tutorial_forward_vs_inverse():
    ii = np.array([0, 200, 400, 600, 800])
    jj = np.array([50, 1000, 2578, 5156, 7734, 10000])
    I, J = (a.ravel() for a in np.meshgrid(ii, jj))
    alpha = (I - (NADIR - 1)) * ANGD
    lam_s = J * SCAN_ANGD
    lat, lon = vgac_to_earth(PROJ_LAT0, PROJ_LON0, PROJ_ROT, alpha, lam_s)
    lon = (lon - 15.0 * SCANTIME[J] + 180.0) % 360.0 - 180.0
    i2, j2 = earth_to_vgac(PROJ_LAT0, PROJ_LON0, PROJ_ROT, SCANTIME, lat, lon)
    di, dj = i2 - I, j2 - J
    print(f"A: tutorial forward vs inverse over {len(I)} cells: "
          f"di in [{di.min()},{di.max()}], dj in [{dj.min()},{dj.max()}]")
    assert np.abs(di).max() >= 2, "expected the documented forward/inverse inconsistency"


def test_b_selfconsistent_forward_vs_inverse():
    results = []
    for i in range(0, 801, 50):
        for j in list(range(0, NSCN, 250)) + [NSCN - 1]:
            lat, lon = forward_from_inverse(i, j)
            i2, j2 = earth_to_vgac(PROJ_LAT0, PROJ_LON0, PROJ_ROT, SCANTIME, lat, lon)
            results.append((i, j, int(i2), int(j2)))
    fails = [(i, j, i2, j2) for i, j, i2, j2 in results if (i2, j2) != (i, j)]
    frac = 1.0 - len(fails) / len(results)
    seam = all(j == 0 or j >= 10250 for _, j, _, _ in fails)
    print(f"B: self-consistent forward vs inverse: {frac:.1%} of {len(results)} cells "
          f"exact; {len(fails)} failures, all at the orbit seam: {seam}")
    assert frac > 0.94
    assert seam, "off-seam round-trip failure: the inverse is not exact where it should be"


if __name__ == "__main__":
    test_a_tutorial_forward_vs_inverse()
    test_b_selfconsistent_forward_vs_inverse()
    print("done")
