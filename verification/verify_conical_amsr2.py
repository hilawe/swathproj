"""Verify the conical model against a real CSU AMSR2 granule, a second platform and agency.

Run from the repo root with the project venv:

    .venv/bin/python verification/verify_conical_amsr2.py

Needs a CSU AMSR2 FCDR granule at data/amsr2_sample.nc (about 71 MB, gitignored):

    B=https://www.ncei.noaa.gov/data/ssmis-brightness-temperature-csu/access/FCDR/2022
    curl -o data/amsr2_sample.nc \\
      $B/CSU_AMSR2_FCDR_V02R00_GCOMW1_D20220101_S0005_E0144_R051195.nc

Verify the download by Content-Length, per the CSU playbook. Skips cleanly when absent.

WHY A SECOND CONICAL INSTRUMENT. The SSMIS verification established that the orbit model is
shared and that both the cone radius and the along-track angular spacing must be carried per
scan. Those conclusions were drawn from one platform, DMSP, whose orbit is comparatively
eccentric. This checks whether they generalise. AMSR2 flies on GCOM-W1, a different platform
operated by a different agency, at a different altitude and with a less eccentric orbit.

They do generalise. The de-rotated ground track flattens onto the rotated equator to within
about 75 m, the conical forward map reproduces the file's own coordinates to about 3 km, and
holding either quantity constant degrades that materially. The per-scan requirement is
therefore a property of orbital eccentricity in general rather than a peculiarity of DMSP.

This script also exists because the proposal in docs/CF_PROPOSAL_APPENDIX_F.md cites these
numbers in its evidence table, and every number in that table should be reproducible by running
the suite rather than taken from a session record.
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
DATA = REPO / "data" / "amsr2_sample.nc"
D2R = np.pi / 180.0
EARTH_KM = 6371.0


def great_circle_km(lat1, lon1, lat2, lon2):
    a1, a2 = lat1 * D2R, lat2 * D2R
    dlon = (lon2 - lon1) * D2R
    h = np.sin((a2 - a1) / 2) ** 2 + np.cos(a1) * np.cos(a2) * np.sin(dlon / 2) ** 2
    return 2 * np.arcsin(np.sqrt(np.clip(h, 0, 1))) * EARTH_KM


def main():
    if not DATA.exists():
        print(f"SKIP: {DATA} not present. See this file's docstring for the fetch command.")
        return 0

    import netCDF4
    from swathproj.conical import ConicalGeometry
    from swathproj.geometry import _forward_rotation, _zyz_to_cf_pole
    from swathproj.rotated_pole import geographic_to_rotated

    ds = netCDF4.Dataset(str(DATA))
    sub_lat = np.asarray(ds["spacecraft_lat"][:], float)
    sub_lon = np.asarray(ds["spacecraft_lon"][:], float)
    alt = np.asarray(ds["spacecraft_alt"][:], float)
    hours = np.asarray(ds["scan_time"][:], float)
    hours = (hours - hours[0]) / 3600.0
    lat = np.asarray(ds["lat_tb18"][:], float)
    lon = np.asarray(ds["lon_tb18"][:], float)
    # the incidence angle carries a polarisation dimension on AMSR2, unlike SSMIS
    eia = np.nanmean(np.asarray(ds["eia_tb18"][:], float), axis=2)
    n_scan, n_pixel = lat.shape
    print(f"granule: {n_scan} scans by {n_pixel} samples, altitude {alt.min():.1f} to "
          f"{alt.max():.1f} km (range {alt.max() - alt.min():.1f} km)")

    def track_rms(heading):
        pole = _zyz_to_cf_pole(_forward_rotation(sub_lat[0], heading, sub_lon[0]))
        rl, _ = geographic_to_rotated(sub_lat, sub_lon + 15.0 * hours, *pole)
        return float(np.sqrt(np.mean(rl ** 2)))

    heading = min(np.arange(0.0, 360.0, 0.5), key=track_rms)
    for step in (0.05, 0.005, 0.0005):
        heading = min(np.arange(heading - 10 * step, heading + 10 * step, step), key=track_rms)
    pole = _zyz_to_cf_pole(_forward_rotation(sub_lat[0], heading, sub_lon[0]))
    _, rot_lon = geographic_to_rotated(sub_lat, sub_lon + 15.0 * hours, *pole)
    along = np.unwrap(rot_lon * D2R) / D2R
    if np.diff(along).mean() < 0:
        heading = (heading + 180.0) % 360.0
        pole = _zyz_to_cf_pole(_forward_rotation(sub_lat[0], heading, sub_lon[0]))
        _, rot_lon = geographic_to_rotated(sub_lat, sub_lon + 15.0 * hours, *pole)
        along = np.unwrap(rot_lon * D2R) / D2R
    advance = float(np.diff(along).mean())
    track_m = track_rms(heading) * 111320
    print(f"1: ground track flattens onto the rotated equator at heading {heading:.3f} deg, "
          f"residual {track_m:.0f} m")
    assert track_m < 200.0, "the shared orbit model no longer fits the AMSR2 track"

    scan_eia = np.nanmean(eia, axis=1)
    radius = scan_eia - np.degrees(
        np.arcsin(EARTH_KM * np.sin(scan_eia * D2R) / (EARTH_KM + alt)))
    corr = float(np.corrcoef(radius, alt)[0, 1])
    print(f"2: cone radius correlates with altitude at {corr:+.3f}")
    assert corr > 0.9, "cone radius no longer tracks altitude on AMSR2"

    def build(cone=None, sub=None):
        return ConicalGeometry(
            ref_lat=sub_lat[0], ref_lon=sub_lon[0], heading=heading, n_scan=n_scan,
            n_pixel=n_pixel, scan_angle_deg=advance,
            scan_longitude_deg=along if sub is None else sub,
            cone_radius_deg=radius if cone is None else cone,
            azimuth_origin_deg=psi0, azimuth_step_deg=dpsi,
            scan_time_hours=hours, full_revolution=True)

    probe = n_scan // 3
    plat, plon = geographic_to_rotated(lat[probe, :], lon[probe, :] + 15.0 * hours[probe], *pole)
    dl = (plon - along[probe] + 180.0) % 360.0 - 180.0
    psi = np.degrees(np.arctan2(np.sin(dl * D2R) * np.cos(plat * D2R), np.sin(plat * D2R)))
    coef = np.polyfit(np.arange(n_pixel), np.unwrap(psi * D2R) / D2R, 1)
    psi0, dpsi = float(coef[1]), float(coef[0])

    ii, jj = (a.ravel() for a in np.meshgrid(np.arange(0, n_pixel, 5), np.arange(0, n_scan, 50)))
    truth_lat, truth_lon = lat[jj, ii], lon[jj, ii]

    def residual(geo):
        mlat, mlon = geo.forward(ii, jj)
        return great_circle_km(truth_lat, truth_lon, mlat, mlon)

    full = residual(build())
    fixed = residual(build(cone=float(radius.mean())))
    uniform = residual(build(sub=along[0] + np.arange(n_scan) * advance))
    print(f"3: conical forward vs the file: median {np.median(full):.2f} km "
          f"(max {full.max():.2f}), with a FIXED cone radius {np.median(fixed):.2f} km, "
          f"with a UNIFORM along-track step {np.median(uniform):.2f} km")
    assert np.median(full) < 5.0, "the conical model no longer reproduces the AMSR2 file"
    assert np.median(fixed) > 1.5 * np.median(full), \
        "the per-scan cone radius stopped mattering on AMSR2"
    assert np.median(uniform) > 3.0 * np.median(full), \
        "the per-scan along-track spacing stopped mattering on AMSR2"

    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
