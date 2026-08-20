"""Does the VGAC rotation stage reduce to a transform CF already has?

Run from the repo root with the project venv:

    .venv/bin/python verification/rotated_pole_decomposition.py

Why this is the pivotal question. The VGAC forward map is two stages: a rotation of a unit
vector by the orbit parameters (theta_o, phi_o), then a longitude shift of -15t for Earth
rotation during the orbit. If the rotation stage is nothing more than a rotated-pole
transform, then CF already standardizes it as the `rotated_latitude_longitude` grid mapping,
and the proposal shrinks from "a new satellite projection" to "an existing CF grid mapping
plus one additional time term". That is a far easier change to argue and to adopt.

Any product of rotation matrices is a single rotation, so on paper the rotation stage must be
expressible as a rotated pole. This script confirms it numerically and, more usefully, shows
the rotated pole is not a free fit: it is the image of the satellite-frame pole (0, 0, 1)
under the VGAC rotation, computed directly from theta_o and phi_o. Only the rotated-longitude
origin lam0 is solved, and it is a single scalar per orbit.

Method. For several (theta_o, phi_o), transform 3000 random satellite-frame points by the
VGAC rotation and by a textbook rotated-pole transform whose pole is derived, not fitted, and
compare. The rotated-longitude origin lam0 is found by a refining grid search, so its
residual is solver-limited rather than fundamental. Where lam0 happens to land on a search
node (the theta_o = 0 case), the agreement is 6e-14 degrees, machine precision, which is the
real number. The other cases sit near 1e-6 degrees (sub-metre), limited by the lam0 grid.

Result observed 2026-08-12: worst-case agreement 2.2e-6 degrees, about 0.25 m on the ground,
solver-limited. The VGAC rotation stage is the CF rotated_latitude_longitude transform.

What this does NOT claim. It does not claim the whole VGAC mapping is a rotated pole. The
Earth-rotation term -15t is a separate, along-track-dependent longitude shear and is the part
with no existing CF counterpart. Isolating it as the sole novel element is exactly the value
of this decomposition, and characterizing it is the next design question.

Scope of this script versus the CF-convention check. This script shows the rotation stage is
*a* rotated pole, using a self-contained rotated-pole formula. It does not by itself prove
the rotation matches CF's specific parameter convention (grid_north_pole_latitude and the
180 - north_pole_grid_longitude term). That stronger claim is tested in the test suite, which feeds the derived CF parameters to
PROJ, an independent CF implementation, and compares against the raw physical rotation.
"""

import sys

import numpy as np

D2R = np.pi / 180.0


def vgac_rotation(alpha, lam, theta_o, phi_o):
    """VGAC rotation stage r_g = R2(theta_o) R1(phi_o) r_s, Earth-rotation term off."""
    a = np.asarray(alpha, float) * D2R
    l = np.asarray(lam, float) * D2R
    o = np.ones_like(a)
    rs = np.array([np.cos(a) * np.cos(l), np.cos(a) * np.sin(l), np.sin(a) * o])
    p, t = phi_o * D2R, theta_o * D2R
    r1 = np.array([[1, 0, 0], [0, np.cos(p), np.sin(p)], [0, -np.sin(p), np.cos(p)]])
    r2 = np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])
    rg = r2 @ r1 @ rs
    return (np.degrees(np.arctan2(rg[2], np.hypot(rg[0], rg[1]))),
            np.degrees(np.arctan2(rg[1], rg[0])))


def rotated_pole(alpha, lam, pole_lat, pole_lon, lam0):
    """Textbook rotated-pole transform, the CF rotated_latitude_longitude family."""
    a = np.asarray(alpha, float) * D2R
    l = (np.asarray(lam, float) - lam0) * D2R
    o = np.ones_like(a)
    x = np.cos(a) * np.cos(l)
    y = np.cos(a) * np.sin(l)
    z = np.sin(a) * o
    pl, po = pole_lat * D2R, pole_lon * D2R
    c, s = np.cos(np.pi / 2 - pl), np.sin(np.pi / 2 - pl)
    x1, z1, y1 = x * c + z * s, -x * s + z * c, y
    c2, s2 = np.cos(po), np.sin(po)
    xg, yg, zg = x1 * c2 - y1 * s2, x1 * s2 + y1 * c2, z1
    return (np.degrees(np.arctan2(zg, np.hypot(xg, yg))),
            np.degrees(np.arctan2(yg, xg)))


def solve_lam0(theta_o, phi_o, plat, plon):
    """Refining grid search for the single rotated-longitude origin."""
    gl = vgac_rotation(0.0, 0.0, theta_o, phi_o)[1]

    def err(low):
        g = rotated_pole(0.0, 0.0, plat, plon, low)[1]
        return abs(((g - gl + 180) % 360) - 180)

    low = 0.0
    grid = np.arange(-180, 180, 0.5)
    low = grid[int(np.argmin([err(x) for x in grid]))]
    for step in (0.05, 0.005, 0.0005, 5e-5, 5e-6):
        cand = np.arange(low - 10 * step, low + 10 * step, step)
        low = cand[int(np.argmin([err(x) for x in cand]))]
    return low


CASES = [(0.0, 261.3), (10.0, 261.3), (-25.0, 98.7), (40.0, 170.0), (-60.0, 300.0)]


def test_rotation_stage_is_rotated_pole():
    worst = 0.0
    for theta_o, phi_o in CASES:
        plat, plon = vgac_rotation(90.0, 0.0, theta_o, phi_o)  # image of the frame pole
        lam0 = solve_lam0(theta_o, phi_o, plat, plon)
        rng = np.random.default_rng(0)
        al = rng.uniform(-14, 14, 3000)
        lm = rng.uniform(0, 360, 3000)
        lat_v, lon_v = vgac_rotation(al, lm, theta_o, phi_o)
        lat_r, lon_r = rotated_pole(al, lm, plat, plon, lam0)
        err = max(np.abs(lat_v - lat_r).max(),
                  (np.abs(((lon_v - lon_r + 180) % 360) - 180) * np.cos(lat_v * D2R)).max())
        worst = max(worst, err)
        print(f"  theta_o={theta_o:6.1f} phi_o={phi_o:6.1f}  "
              f"rotated pole=({plat:.3f},{plon:.3f}) lam0={lam0:.4f}  max err {err:.2e} deg")
    print(f"worst-case agreement: {worst:.2e} deg "
          f"({worst * D2R * 6.378137e6 * 1e3:.2f} mm on the ground), solver-limited")
    assert worst < 1e-4, "rotation stage does not reduce to a rotated-pole transform"


if __name__ == "__main__":
    print("VGAC rotation stage vs a rotated-pole transform (pole derived, not fitted):")
    test_rotation_stage_is_rotated_pole()
    print("done")
