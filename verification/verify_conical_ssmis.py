"""Verify the conical scan model against a real CSU SSMIS granule.

Run from the repo root with the project venv:

    .venv/bin/python verification/verify_conical_ssmis.py

Needs a CSU SSMIS FCDR granule at data/ssmis_sample.nc (about 22 MB, gitignored). Fetch one
from NCEI's public archive, for example:

    B=https://www.ncei.noaa.gov/data/ssmis-brightness-temperature-csu/access/FCDR/2022
    curl -o data/ssmis_sample.nc \\
      $B/CSU_SSMIS_FCDR_V02R00_F17_D20220101_S0029_E0211_R078226.nc

Verify the download by Content-Length rather than by non-emptiness, since these archives
serve partial files during long transfers. This script needs one granule for a geometry
check and skips cleanly when the file is absent.

WHY THIS EXISTS. The project's requirements put conical scanners in scope alongside cross-track
and push-broom. That decision rested on a claim that had only been checked against a synthetic
cone: that a conical instrument reuses the same orbit model, with the rotated pole and the
Earth-rotation term placing the SUB-SATELLITE POINT exactly as for a cross-track instrument,
and only the per-pixel offset locus differing. This checks that claim against a real
instrument's own published geolocation.

The granule is well suited to it because it carries spacecraft_lat, spacecraft_lon and
spacecraft_alt per scan, so the sub-satellite track is given rather than inferred, and it
carries a per-pixel Earth incidence angle.

RESULT 1, the defining conical property holds. Every feed horn's pixels sit at a near-constant
angular radius from the sub-satellite point, about 8.26 to 8.43 degrees depending on the horn,
with an Earth incidence angle near 53 degrees. A cross-track or push-broom scan line would
instead spread zero along the track, so this is the geometric distinction the requirements
describe, measured on real data.

RESULT 2, and this one changed a design assumption. The roughly 0.10 degree scatter in that
radius is NOT noise. It is the orbit: spacecraft altitude varies about 37 km within this single
pass, and the cone radius tracks it with a correlation of about 0.97. Predicting the radius from
the two per-scan observables, Earth incidence angle and altitude, on a spherical Earth,
reproduces the observed radius to about 0.023 degrees, roughly 2.5 km, with the residual
attributable to the spherical approximation.

The design consequence is concrete: a conical mapping cannot use one fixed cone constant for a
whole pass. Doing so would misplace pixels by up to about 43 km here. Either the altitude or the
cone angle has to be carried per scan. This matters because the cross-track case CAN assume a
constant nominal altitude, since its coordinate is the scan angle itself, so the two geometries
differ in what they must record even though they share the orbit model.
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
DATA = REPO / "data" / "ssmis_sample.nc"
D2R = np.pi / 180.0
EARTH_KM = 6371.0
FEEDS = ["env1", "env2", "img1", "img2", "las", "uas"]


def great_circle_deg(lat1, lon1, lat2, lon2):
    a1, a2 = lat1 * D2R, lat2 * D2R
    dlon = (lon2 - lon1) * D2R
    h = np.sin((a2 - a1) / 2) ** 2 + np.cos(a1) * np.cos(a2) * np.sin(dlon / 2) ** 2
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(h, 0, 1))))


def main():
    if not DATA.exists():
        print(f"SKIP: {DATA} not present. See this file's docstring for the fetch command.")
        return 0

    import netCDF4
    ds = netCDF4.Dataset(str(DATA))
    sub_lat = np.asarray(ds["spacecraft_lat"][:], float)
    sub_lon = np.asarray(ds["spacecraft_lon"][:], float)
    alt = np.asarray(ds["spacecraft_alt"][:], float)
    print(f"granule: {len(sub_lat)} scans, altitude {alt.min():.1f} to {alt.max():.1f} km")

    # RESULT 1: the defining conical property, on every feed horn
    radii = {}
    for feed in FEEDS:
        lat = np.asarray(ds["lat_" + feed][:], float)
        lon = np.asarray(ds["lon_" + feed][:], float)
        eia = np.asarray(ds["eia_" + feed][:], float)
        good = np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) <= 90) & np.isfinite(eia)
        scan = np.arange(lat.shape[0])[:, None] * np.ones((1, lat.shape[1]), int)
        rho = great_circle_deg(sub_lat[scan][good], sub_lon[scan][good], lat[good], lon[good])
        radii[feed] = (float(rho.mean()), float(rho.std()), float(eia[good].mean()))
        assert 7.0 < rho.mean() < 10.0, f"{feed}: cone radius {rho.mean():.3f} deg is not conical"
        assert rho.std() < 0.2, f"{feed}: cone radius scatter {rho.std():.3f} deg is too large"
        assert 50.0 < eia[good].mean() < 56.0, f"{feed}: Earth incidence angle is not SSMIS-like"
    summary = ", ".join(f"{f} {radii[f][0]:.2f}" for f in FEEDS)
    print(f"1: every feed horn is at a near-constant angular radius from the sub-satellite "
          f"point (deg): {summary}")

    # RESULT 2: the residual scatter is the orbit, not noise
    lat = np.asarray(ds["lat_env1"][:], float)
    lon = np.asarray(ds["lon_env1"][:], float)
    eia = np.asarray(ds["eia_env1"][:], float)
    good = np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) <= 90) & np.isfinite(eia)
    scan = np.arange(lat.shape[0])[:, None] * np.ones((1, lat.shape[1]), int)
    rho_obs = great_circle_deg(sub_lat[scan][good], sub_lon[scan][good], lat[good], lon[good])
    height = alt[scan][good]
    incidence = eia[good]
    rho_pred = incidence - np.degrees(
        np.arcsin(EARTH_KM * np.sin(incidence * D2R) / (EARTH_KM + height)))
    residual = rho_obs - rho_pred
    corr = float(np.corrcoef(rho_obs, height)[0, 1])
    spread_km = float(rho_obs.max() - rho_obs.min()) * 111.32
    print(f"2: cone radius correlates with altitude at {corr:+.3f}; predicting it from "
          f"incidence angle and altitude leaves {residual.std():.4f} deg "
          f"({residual.std() * 111.32:.2f} km)")
    print(f"   a single fixed cone constant would misplace pixels by up to {spread_km:.1f} km "
          f"across this pass, so altitude or cone angle must be carried per scan")
    assert corr > 0.9, "cone radius no longer tracks altitude, re-examine the model"
    assert residual.std() < 0.05, "incidence angle plus altitude no longer predicts the radius"
    assert spread_km > 10.0, "the altitude effect vanished, which would change the design finding"

    # RESULT 3: the model reproduces the file's geolocation, and the two per-scan
    # requirements are isolated by turning each one off in turn
    from swathproj.conical import ConicalGeometry
    from swathproj.geometry import _forward_rotation, _zyz_to_cf_pole
    from swathproj.rotated_pole import geographic_to_rotated

    hours = np.asarray(ds["scan_time"][:], float)
    hours = (hours - hours[0]) / 3600.0
    n_scan, n_pixel = lat.shape

    # Fit the orbit frame the way a producer would: choose the heading that flattens the
    # de-rotated ground track onto the rotated equator.
    def track_rms(heading):
        m = _forward_rotation(sub_lat[0], heading, sub_lon[0])
        pole = _zyz_to_cf_pole(m)
        rl, _ = geographic_to_rotated(sub_lat, sub_lon + 15.0 * hours, *pole)
        return float(np.sqrt(np.mean(rl ** 2)))

    heading = min(np.arange(0.0, 360.0, 0.5), key=track_rms)
    for step in (0.05, 0.005, 0.0005):
        heading = min(np.arange(heading - 10 * step, heading + 10 * step, step), key=track_rms)
    pole = _zyz_to_cf_pole(_forward_rotation(sub_lat[0], heading, sub_lon[0]))
    _, rot_lon = geographic_to_rotated(sub_lat, sub_lon + 15.0 * hours, *pole)
    along = np.unwrap(rot_lon * D2R) / D2R
    if np.diff(along).mean() < 0:                 # keep the along-track advance positive
        heading = (heading + 180.0) % 360.0
        pole = _zyz_to_cf_pole(_forward_rotation(sub_lat[0], heading, sub_lon[0]))
        _, rot_lon = geographic_to_rotated(sub_lat, sub_lon + 15.0 * hours, *pole)
        along = np.unwrap(rot_lon * D2R) / D2R
    advance = float(np.diff(along).mean())
    print(f"3: ground track flattens onto the rotated equator at heading {heading:.3f} deg, "
          f"residual {track_rms(heading) * 111320:.0f} m")
    assert track_rms(heading) * 111320 < 200.0, "the shared orbit model no longer fits the track"

    scan_eia = np.nanmean(np.asarray(ds["eia_env1"][:], float), axis=1)
    radius = scan_eia - np.degrees(
        np.arcsin(EARTH_KM * np.sin(scan_eia * D2R) / (EARTH_KM + alt)))

    def build(psi0, dpsi, cone=None, sub=None):
        return ConicalGeometry(
            ref_lat=sub_lat[0], ref_lon=sub_lon[0], heading=heading, n_scan=n_scan,
            n_pixel=n_pixel, scan_angle_deg=advance,
            scan_longitude_deg=along if sub is None else sub,
            cone_radius_deg=radius if cone is None else cone,
            azimuth_origin_deg=psi0, azimuth_step_deg=dpsi,
            scan_time_hours=hours, full_revolution=True)

    # the two azimuth constants are fitted from one scan, then used everywhere
    probe = n_scan // 3
    plat, plon = geographic_to_rotated(
        np.asarray(ds["lat_env1"][:], float)[probe, :],
        np.asarray(ds["lon_env1"][:], float)[probe, :] + 15.0 * hours[probe], *pole)
    dl = (plon - along[probe] + 180.0) % 360.0 - 180.0
    psi = np.degrees(np.arctan2(np.sin(dl * D2R) * np.cos(plat * D2R), np.sin(plat * D2R)))
    coef = np.polyfit(np.arange(n_pixel), np.unwrap(psi * D2R) / D2R, 1)
    psi0, dpsi = float(coef[1]), float(coef[0])

    ii, jj = (a.ravel() for a in np.meshgrid(np.arange(0, n_pixel, 3), np.arange(0, n_scan, 40)))
    truth_lat = np.asarray(ds["lat_env1"][:], float)[jj, ii]
    truth_lon = np.asarray(ds["lon_env1"][:], float)[jj, ii]

    def residual_km(geo):
        mlat, mlon = geo.forward(ii, jj)
        return great_circle_deg(truth_lat, truth_lon, mlat, mlon) * 111.32

    full = residual_km(build(psi0, dpsi))
    fixed_cone = residual_km(build(psi0, dpsi, cone=float(radius.mean())))
    uniform = residual_km(build(psi0, dpsi, sub=along[0] + np.arange(n_scan) * advance))
    print(f"4: conical forward vs the file: median {np.median(full):.2f} km "
          f"(max {full.max():.2f}), with a FIXED cone radius {np.median(fixed_cone):.2f} km, "
          f"with a UNIFORM along-track step {np.median(uniform):.2f} km")
    assert np.median(full) < 4.0, "the conical model no longer reproduces the file"
    assert np.median(fixed_cone) > 2 * np.median(full), \
        "the per-scan cone radius stopped mattering, re-check the design finding"
    assert np.median(uniform) > 4 * np.median(full), \
        "the per-scan along-track angle stopped mattering, re-check the design finding"

    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
