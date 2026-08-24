"""Verify the conical model against a real CSU SSM/I granule, a third conical instrument.

Run from the repo root with the project venv:

    .venv/bin/python verification/verify_conical_ssmi.py

Needs a CSU SSM/I FCDR granule at data/ssmi_sample.nc (about 6 MB, gitignored). Fetch one from
NCEI's public archive, for example:

    B=https://www.ncei.noaa.gov/data/ssmis-brightness-temperature-csu/access/FCDR/1998
    curl -o data/ssmi_sample.nc \\
      $B/CSU_SSMI_FCDR_V02R00_F13_D19980101_S0104_E0246_R014311.nc

Verify the download by Content-Length rather than by non-emptiness, since these archives serve
partial files during long transfers. Skips cleanly when the file is absent.

WHY SSM/I, AND WHY IT IS NOT SSMIS. SSM/I (DMSP F8 to F15) is the predecessor of SSMIS, and its
FCDR is laid out DIFFERENTLY, which this script does not paper over. SSMIS packs one scan geometry
with named sounding and imaging feed horns (env, img, las, uas). SSM/I is an imager only and stores
TWO separate swaths: a low-resolution swath of 64 pixels carrying 19, 22 and 37 GHz, and a
high-resolution swath of 128 pixels carrying 85 GHz sampled at twice the scan rate. Each swath has
its own scan dimension, its own per-scan spacecraft position and time, its own per-pixel Earth
incidence angle, and its own geolocation, so the SSMIS reader cannot read this file, and its
variable names, feed-horn list and single-swath assumption do not carry over. The high-resolution
swath has exactly twice the scans of the low-resolution one, and its even scans coincide with the
low-resolution scans (the documented two-to-one pairing).

Measured differences from SSMIS on this F13 granule, for the record and NOT assumed equal:
Earth incidence angle about 52.95 degrees (SSMIS about 53), cone radius about 8.29 degrees (SSMIS
8.26 to 8.43), spacecraft altitude varying about 21 km within the pass (SSMIS about 37), and the
per-scan spacecraft latitude does NOT repeat here (the playbook's repeating warning is for the
SSM/I TDR product, not this FCDR), so the sub-satellite track is usable directly.

WHAT THIS CHECKS. Each swath is validated on its own as a conical scan. The defining conical
property (every pixel at a near-constant angular radius from the sub-satellite point), the finding
that the small residual radius scatter is the orbit rather than noise (the radius tracks altitude
and is predicted by incidence angle plus altitude), and that the shared rotated-pole plus
Earth-rotation orbit model reproduces the file's own geolocation, with the per-scan cone radius and
the per-scan along-track spacing each shown to matter by turning it off. Validating BOTH swaths
also confirms the model does not depend on the 64-pixel or 128-pixel sampling, which is the
SSM/I-specific structure a single-swath check would miss.
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
DATA = REPO / "data" / "ssmi_sample.nc"
D2R = np.pi / 180.0
EARTH_KM = 6371.0


def great_circle_deg(lat1, lon1, lat2, lon2):
    a1, a2 = lat1 * D2R, lat2 * D2R
    dlon = (lon2 - lon1) * D2R
    h = np.sin((a2 - a1) / 2) ** 2 + np.cos(a1) * np.cos(a2) * np.sin(dlon / 2) ** 2
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(h, 0, 1))))


def validate_swath(ds, swath):
    """Validate one SSM/I swath (``lores`` or ``hires``) as a conical scan, and return the
    median forward-reproduction residual in km. Reads only that swath's own variables."""
    from swathproj.conical import ConicalGeometry
    from swathproj.geometry import _forward_rotation, _zyz_to_cf_pole
    from swathproj.rotated_pole import geographic_to_rotated

    sub_lat = np.asarray(ds["spacecraft_lat_" + swath][:], float)
    sub_lon = np.asarray(ds["spacecraft_lon_" + swath][:], float)
    alt = np.asarray(ds["spacecraft_alt_" + swath][:], float)
    lat = np.asarray(ds["lat_" + swath][:], float)
    lon = np.asarray(ds["lon_" + swath][:], float)
    eia = np.asarray(ds["eia_" + swath][:], float)
    hours = np.asarray(ds["scan_time_" + swath][:], float)
    hours = (hours - hours[0]) / 3600.0
    n_scan, n_pixel = lat.shape
    print(f"\n[{swath}] {n_scan} scans by {n_pixel} pixels, altitude {alt.min():.1f} to "
          f"{alt.max():.1f} km (range {alt.max() - alt.min():.1f} km)")

    good = np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) <= 90) & np.isfinite(eia)
    scan = np.arange(n_scan)[:, None] * np.ones((1, n_pixel), int)

    # RESULT 1: the defining conical property, every pixel near a constant radius from nadir
    rho = great_circle_deg(sub_lat[scan][good], sub_lon[scan][good], lat[good], lon[good])
    mean_eia = float(np.nanmean(eia[good]))
    print(f"  1: pixels sit at a near-constant angular radius, mean {rho.mean():.3f} deg "
          f"(scatter {rho.std():.4f}), Earth incidence angle mean {mean_eia:.3f} deg")
    assert 7.0 < rho.mean() < 10.0, f"{swath}: cone radius {rho.mean():.3f} deg is not conical"
    assert rho.std() < 0.2, f"{swath}: cone radius scatter {rho.std():.3f} deg is too large"
    assert 50.0 < mean_eia < 56.0, f"{swath}: incidence angle {mean_eia:.2f} deg is not SSM/I-like"

    # RESULT 2: the residual scatter is the orbit, not noise
    height = alt[scan][good]
    rho_pred = eia[good] - np.degrees(
        np.arcsin(EARTH_KM * np.sin(eia[good] * D2R) / (EARTH_KM + height)))
    residual = rho - rho_pred
    corr = float(np.corrcoef(rho, height)[0, 1])
    # The misplacement a single mean cone would cause is the largest departure of the observed
    # per-pixel radius from its mean, NOT the peak-to-peak spread (which double-counts, being the
    # sum of the two one-sided departures).
    max_dev_km = float(np.max(np.abs(rho - rho.mean()))) * 111.32
    print(f"  2: cone radius correlates with altitude at {corr:+.3f}; incidence angle plus "
          f"altitude predicts it to {residual.std():.4f} deg ({residual.std() * 111.32:.2f} km); "
          f"a single mean cone would misplace pixels by up to {max_dev_km:.1f} km")
    assert corr > 0.85, f"{swath}: cone radius no longer tracks altitude"
    assert residual.std() < 0.05, f"{swath}: incidence plus altitude no longer predicts the radius"
    assert max_dev_km > 5.0, f"{swath}: the altitude effect vanished, the design finding changes"

    # RESULT 3: the shared orbit model reproduces the file's own geolocation
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
    if np.diff(along).mean() < 0:                 # keep the along-track advance positive
        heading = (heading + 180.0) % 360.0
        pole = _zyz_to_cf_pole(_forward_rotation(sub_lat[0], heading, sub_lon[0]))
        _, rot_lon = geographic_to_rotated(sub_lat, sub_lon + 15.0 * hours, *pole)
        along = np.unwrap(rot_lon * D2R) / D2R
    advance = float(np.diff(along).mean())
    track_m = track_rms(heading) * 111320
    print(f"  3: ground track flattens onto the rotated equator at heading {heading:.3f} deg, "
          f"residual {track_m:.0f} m")
    assert track_m < 200.0, f"{swath}: the shared orbit model no longer fits the track"

    scan_eia = np.nanmean(eia, axis=1)
    radius = scan_eia - np.degrees(
        np.arcsin(EARTH_KM * np.sin(scan_eia * D2R) / (EARTH_KM + alt)))

    # the two azimuth constants are fitted from one scan, then used everywhere
    probe = n_scan // 3
    plat, plon = geographic_to_rotated(lat[probe, :], lon[probe, :] + 15.0 * hours[probe], *pole)
    dl = (plon - along[probe] + 180.0) % 360.0 - 180.0
    psi = np.degrees(np.arctan2(np.sin(dl * D2R) * np.cos(plat * D2R), np.sin(plat * D2R)))
    coef = np.polyfit(np.arange(n_pixel), np.unwrap(psi * D2R) / D2R, 1)
    psi0, dpsi = float(coef[1]), float(coef[0])

    def build(cone=None, sub=None):
        return ConicalGeometry(
            ref_lat=sub_lat[0], ref_lon=sub_lon[0], heading=heading, n_scan=n_scan,
            n_pixel=n_pixel, scan_angle_deg=advance,
            scan_longitude_deg=along if sub is None else sub,
            cone_radius_deg=radius if cone is None else cone,
            azimuth_origin_deg=psi0, azimuth_step_deg=dpsi,
            scan_time_hours=hours, full_revolution=True)

    # The scan stride is coprime to two (41, not an even 40) so the sample hits BOTH scan parities.
    # On the high-resolution swath the odd scans are the interleaved half-rate scans, which even
    # strides would skip entirely, so an odd-scan timing or along-track error could pass unseen.
    ii, jj = (a.ravel() for a in
              np.meshgrid(np.arange(0, n_pixel, 3), np.arange(0, n_scan, 41)))
    truth_lat, truth_lon = lat[jj, ii], lon[jj, ii]

    def residual_km(geo):
        mlat, mlon = geo.forward(ii, jj)
        return great_circle_deg(truth_lat, truth_lon, mlat, mlon) * 111.32

    full = residual_km(build())
    fixed_cone = residual_km(build(cone=float(radius.mean())))
    uniform = residual_km(build(sub=along[0] + np.arange(n_scan) * advance))
    print(f"  4: conical forward vs the file: median {np.median(full):.2f} km (max {full.max():.2f}), "
          f"FIXED cone radius {np.median(fixed_cone):.2f} km, UNIFORM along-track step "
          f"{np.median(uniform):.2f} km")
    assert np.median(full) < 4.0, f"{swath}: the conical model no longer reproduces the file"
    # Both per-scan quantities must still matter, but the SIZE of the cone-radius effect scales with
    # orbital eccentricity, and this F13 pass varies only about 21 km in altitude against SSMIS F17's
    # 37 km, so the fixed-cone penalty here (about 1.9x the full residual) is smaller than SSMIS's.
    # The thresholds are set below the measured ratios (about 1.9x and 5.5x) with headroom, matching
    # how the less-eccentric AMSR2 orbit was calibrated, so a real degradation still fires them.
    assert np.median(fixed_cone) > 1.5 * np.median(full), \
        f"{swath}: the per-scan cone radius stopped mattering"
    assert np.median(uniform) > 3.0 * np.median(full), \
        f"{swath}: the per-scan along-track angle stopped mattering"
    return float(np.median(full))


def main():
    if not DATA.exists():
        print(f"SKIP: {DATA} not present. See this file's docstring for the fetch command.")
        return 0

    import netCDF4
    ds = netCDF4.Dataset(str(DATA))

    # Confirm the SSM/I two-swath structure before validating, rather than assuming it, and check
    # the documented two-to-one pairing. This is the structural fact that separates SSM/I from
    # SSMIS. The pairing means the WHOLE per-scan state of every even high-resolution scan equals
    # the matching low-resolution scan, so it is checked across time, position and altitude rather
    # than latitude alone, and the pixel counts are asserted too.
    for req in ("spacecraft_lat_lores", "lat_hires", "eia_lores", "scan_time_hires"):
        if req not in ds.variables:
            raise ValueError(f"{DATA} is missing {req}, so it is not the expected SSM/I FCDR layout")
    assert ds["lat_lores"].shape[1] == 64 and ds["lat_hires"].shape[1] == 128, \
        "SSM/I pixel counts are not the expected 64 (lores) and 128 (hires)"
    n_lo = ds["spacecraft_lat_lores"].shape[0]
    n_hi = ds["spacecraft_lat_hires"].shape[0]
    assert n_hi == 2 * n_lo, "high-resolution swath is not twice the low-resolution one"
    for stem in ("scan_time", "spacecraft_lat", "spacecraft_lon", "spacecraft_alt"):
        lo = np.asarray(ds[stem + "_lores"][:], float)
        hi = np.asarray(ds[stem + "_hires"][:], float)[0::2]
        assert np.isfinite(lo).all() and np.isfinite(hi).all(), f"{stem}: missing paired values"
        assert np.array_equal(lo, hi), \
            f"{stem}: even hires scans do not match lores, so the 2:1 pairing is broken"
    print(f"SSM/I two-swath granule: lores {n_lo} scans (64 px), hires {n_hi} scans (128 px); "
          f"every even hires scan's time, position and altitude match the lores scan (2:1 pairing)")

    med_lores = validate_swath(ds, "lores")
    med_hires = validate_swath(ds, "hires")
    print(f"\ndone: both SSM/I swaths follow the shared conical orbit model "
          f"(lores {med_lores:.2f} km, hires {med_hires:.2f} km), confirming it does not depend on "
          f"the 64- or 128-pixel sampling")
    return 0


if __name__ == "__main__":
    sys.exit(main())
