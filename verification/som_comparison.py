"""Can PROJ's Space Oblique Mercator already express the VGAC swath geometry?

Run from the repo root with the project venv:

    .venv/bin/python verification/som_comparison.py

Why this matters. Space Oblique Mercator (`+proj=som`, Snyder, built for Landsat) is the
closest thing in the registered projection canon to an orbit-parameterized projection, and
its defining property is that a satellite ground track maps to a straight line. If the VGAC
swath geometry were a reparameterization of SOM, then no new projection would be needed and
the whole proposal would reduce to a binding exercise. This script tests that.

Method. Build a synthetic VGAC orbit (sun-synchronous, inclination 98.7 degrees, 101-minute
period) using the forward map that is self-consistent with the published inverse in
reference/knappweathersat-VGAC-tutorial-bb3556d/vgac_projection.py. Take the nadir line,
which is the ground track by construction, and push it through SOM. Under a correct SOM
parameterization the along-track coordinate x must be constant. Search all three SOM
parameters rather than assuming the nominal ones, because a negative result from a
mis-parameterized comparison is not evidence.

A sanity assertion guards the synthetic orbit itself: an inclination of 98.7 degrees must
produce a ground track reaching 81.3 degrees latitude. An earlier version of this comparison
used a wrong rotation convention, produced a near-equatorial track, and would have yielded a
confident but meaningless negative.

Result observed 2026-08-12 (pyproj 3.6.1, PROJ 9.3.0). Over 1400 parameter combinations the
best fit still leaves about 95 km of scatter in a quantity that a genuine match would hold
constant to within metres, and cross-track scan lines depart from straight by tens to
hundreds of km. SOM and the VGAC formulation are different idealizations of a satellite
swath, and neither is a reparameterization of the other.

Interpretation, stated narrowly. This shows the two models disagree. It does not show either
is wrong. VGAC idealizes the track as a great circle carrying a uniform longitude shear for
Earth rotation, while SOM models the ground track of an orbiting platform over a rotating
Earth. The comparison bounds what an existing registered projection can do for this data,
which is the question the proposal has to answer.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
from pyproj import Proj

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "reference" / "knappweathersat-VGAC-tutorial-bb3556d"))

from vgac_projection import ANGD, NADIR  # noqa: E402

D2R = np.pi / 180.0
INC = 98.7
NSCN = 10313
PERIOD_MIN = 101.0
SCANTIME = np.linspace(0.0, PERIOD_MIN / 60.0, NSCN)
SCAN_ANGD = 360.0 / NSCN
PROT = 360.0 - INC  # rotation convention as used in roundtrip_synthetic.py


def vgac_forward(i, j, prot=PROT):
    """VGAC cell indices to lat/lon, self-consistent with the published inverse."""
    la = ((i - NADIR + 0.5) * ANGD) * D2R
    lo = (j * SCAN_ANGD) * D2R
    xpp = np.cos(la) * np.cos(lo)
    yppp = np.cos(la) * np.sin(lo)
    zppp = np.sin(la)
    sg, cg = np.sin(-prot * D2R), np.cos(-prot * D2R)
    yp = yppp * cg - zppp * sg
    zpp = yppp * sg + zppp * cg
    lat = np.degrees(np.arctan2(zpp, np.hypot(xpp, yp)))
    lon = np.degrees(np.arctan2(yp, xpp)) - 15.0 * SCANTIME[j]
    return lat, (lon + 180.0) % 360.0 - 180.0


def test_orbit_is_polar():
    """Guard the synthetic orbit before drawing any conclusion from it."""
    lat, _ = vgac_forward(400, np.arange(0, NSCN, 50))
    reached = np.abs(lat).max()
    print(f"sanity: ground track reaches {reached:.1f} deg latitude "
          f"(inclination {INC} requires 81.3)")
    assert abs(reached - 81.3) < 1.5, "synthetic orbit is not the intended polar orbit"


def test_som_cannot_straighten_the_track():
    seg = np.arange(200, 2800, 40)  # ascending quarter, away from the orbit seam
    lat_s, lon_s = vgac_forward(400, seg)
    best = None
    for asc in np.arange(80.0, 120.0, 2.0):
        for inc in np.arange(94.0, 104.0, 1.0):
            for rev in np.arange(85.0, 120.0, 5.0):
                try:
                    proj = Proj(proj="som", asc_lon=asc, inc_angle=inc,
                                ps_rev=rev / 1440.0, R=6378137.0)
                    x = np.asarray(proj(lon_s, lat_s)[0])
                except Exception:
                    continue
                if not np.all(np.isfinite(x)):
                    continue
                spread = x.std() / 1e3
                if best is None or spread < best[-1]:
                    best = (asc, inc, rev, spread)
    assert best is not None, "no SOM parameterization produced finite coordinates"
    asc, inc, rev, spread = best
    print(f"best SOM fit: asc_lon={asc:.0f} inc_angle={inc:.0f} period={rev:.0f} min "
          f"leaves {spread:.1f} km of scatter in the along-track coordinate")
    assert spread > 5.0, ("SOM reproduced the VGAC track closely, which would change the "
                          "premise of the proposal and must be investigated")


if __name__ == "__main__":
    test_orbit_is_polar()
    test_som_cannot_straighten_the_track()
    print("done")
