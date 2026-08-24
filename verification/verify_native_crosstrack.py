"""Verify the cross-track slant-range geometry against real granules (native and one resampled).

Run from the repo root with the project venv:

    .venv/bin/python verification/verify_native_crosstrack.py

Validates EVERY cross-track granule present in data/, and passes only when all of them
pass. It reads a CF-netCDF AVHRR GAC L1C (EUMETSAT C3S FCDR, a clean uniform-step whiskbroom), a
C3S ATMS FCDR L1C (a uniform-step microwave sounder, with coarser footprints), an operational
VIIRS SDR geolocation file (GMTCO/GITCO HDF5 from NOAA CLASS, an aggregated instrument), a NASA
VJ103MOD/VNP03MOD netCDF, and a Sentinel-3 OLCI L1B EFR product (a push-broom instrument's resampled
product). It skips with a clear message when no granule is present.

Two regimes, and the OLCI finding. The whiskbroom and aggregated products (AVHRR, ATMS, VIIRS) are
NATIVE slant-range swaths whose cross-track ground spacing grows away from nadir, so the equal-angle
model must FAIL on them and the slant-range relation must hold. OLCI is different, and the difference
is measured rather than assumed. Its delivered cross-track sampling is uniform in GROUND angle (the
central angle from nadir is linear in column to about 0.13 km), so it is the equal_angle kind, like
the resampled VGAC, not a slant-range native swath. This shows cross_track_kind is a producer's
DELIVERY choice, since a push-broom instrument's operational product can be an equal_angle grid. Two
honesty limits are load-bearing. First, OLCI L1B EFR is a RESAMPLED product rather than native
detector samples, because its per-pixel frame_offset and repeated detector_index (fewer unique
detectors than columns) show the delivered grid is assembled from multiple acquisition frames. It
therefore neither tests a native slant-range push-broom (none here does) nor demonstrates simultaneous
across-track acquisition, and no intra_scan_time_span claim is made. Second, this is a single granule
and a one-dimensional radial-profile-and-consistency check, not a two-dimensional geolocation
reproduction. OLCI takes an inverted test. The equal-angle model must FIT, the view zenith must AGREE
with the geolocation to a tight cap (using a sign-corrected nadir, since the stored OZA is an unsigned
magnitude that plateaus at nadir), and the fitted altitude must be near Sentinel-3's real value. For the slant-range families the absolute residual differs by instrument, sub-km for the AVHRR
and VIIRS visible-infrared pixels and a few km for ATMS, so the pass criteria are RELATIVE (each
model must beat the equal-angle model by a wide margin) with family-specific absolute caps. The ATMS few-km residual is NOT claimed as a footprint
floor. It is only partly nadir-index quantization (ATMS has an even 96-beam count, so nadir falls
between beams, and fitting a fractional nadir removes under 1 km of it). The remainder is where the
spherical slant-range model departs from the ATMS FCDR's full geolocation. What holds for every
instrument is that the slant-range model beats equal-angle by roughly an order of magnitude, which
is the point of the native check. Tested against Metop-C AVHRR, SNPP ATMS, and NOAA-20 VIIRS granules.

Why native, and what this settles. The VGAC check (verify_real_vgac.py) uses a RESAMPLED product
whose grid is a uniform division of ground angle, so the equal-angle model fits it perfectly. That
cannot test a native whiskbroom, whose cross-track ground spacing grows away from nadir by the
slant-range relation gamma(theta) = arcsin(k sin theta) - theta, k = (R + h) / R, where theta is
the look angle from nadir and gamma the Earth central angle from the sub-satellite point. The same
relation, in terms of the satellite zenith epsilon, is sin(epsilon) = k sin(theta), gamma = epsilon
- theta.

METHOD. For a sample of scan lines it takes the observed cross-track profile, the signed central
angle from the line's nadir pixel (the smallest satellite zenith) to every pixel, which is the
great-circle distance divided by the Earth radius. It fits THREE models to that profile and reports
each residual in kilometres.

  equal-angle    gamma(i) = a * (i - nadir)                            linear in index
  pixel-index    gamma(i) = slant_range((i - nadir) * look_step, h)    from index + 2 params
  file-zenith    gamma(i) = epsilon(i) - arcsin(sin epsilon(i) / k)    from the file's own zenith

What each establishes, from weakest to strongest claim.
  1. The equal-angle FAILURE is the direct, non-circular evidence that the native pixels are not a
     uniform angular grid, unlike a resampled product. This is the point the native check turns on.
  2. The file-zenith fit is a CONSISTENCY check. The satellite zenith and the coordinates are
     co-derived by the producer, so it confirms the profile obeys the slant-range relation but does
     not independently reproduce it.
  3. The pixel-index fit is the LESS-CIRCULAR check. It reproduces the observed radial central-angle
     profile from the array index and two physical parameters (look-angle step and altitude), with
     no per-pixel angle in the fit itself, and recovers a consistent step and a physical altitude.
     The zenith is used only to locate the nadir pixel, not in the fit. This is stronger evidence
     for the NativeCrossTrackGeometry mapping than the zenith check, but it remains a
     one-dimensional radial-profile test. It does not check that the pixels lie on the required
     cross-track great circle, so it is not a full two-dimensional geolocation reproduction. It
     applies only to instruments that sample uniformly in scan angle (AVHRR and ATMS). An aggregated
     instrument (VIIRS) is not uniform in index and is not asserted against it.

The nadir is the integer argmin-zenith pixel, so a fractional sub-satellite position contributes a
subpixel-scale residual that is left in rather than tuned out. This is a one-dimensional profile
test, not a full two-dimensional geolocation reproduction, and the conclusion is worded
accordingly. The fit logic is exercised on synthetic data in verification/test_native.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from swathproj.geometry import EARTH_RADIUS_KM  # noqa: E402
from swathproj.native_geometry import (  # noqa: E402
    central_angle_to_look_angle,
    look_angle_to_central_angle,
)

D2R = np.pi / 180.0

# Expected sampling per recognized instrument family. The absolute residual floor differs by
# footprint scale, so the caps are family-specific, and the primary check is always relative. The
# zenith and pixel-index tests get SEPARATE caps: the zenith consistency check is tight (it uses
# the file's own per-pixel geometry), while the pixel-index check is looser because it also carries
# the interpolated GAC geolocation and the integer-nadir quantization. step_deg is the nominal
# cross-track scan step (degrees per beam), used to confirm the fit recovered the real instrument
# step, not just one that happens to fit.
FAMILY = {
    "AVHRR": dict(uniform=True, step_deg=0.27, zenith_cap_km=1.0, index_cap_km=6.0),
    "ATMS": dict(uniform=True, step_deg=1.11, zenith_cap_km=12.0, index_cap_km=12.0),
    "VIIRS": dict(uniform=False, step_deg=None, zenith_cap_km=3.0, index_cap_km=None),
    # OLCI L1B EFR is a push-broom instrument's RESAMPLED product, and MEASUREMENT (not assumption)
    # shows it is distributed on a uniform cross-track GROUND-angle grid: its central angle from
    # nadir is linear in column to about 0.13 km, so it is the equal_angle kind, like the resampled
    # VGAC, NOT a slant-range native swath. It therefore takes the INVERTED test (equal-angle must
    # FIT), with the view-zenith consistency asserted tight (about 0.16 km with the signed-OZA nadir)
    # and the fitted altitude near Sentinel-3's real value.
    "OLCI": dict(uniform=False, expect_equal_angle=True, equal_fit_cap_km=1.0,
                 zenith_consistency_cap_km=0.5, alt_band_km=(760.0, 870.0),
                 step_deg=None, zenith_cap_km=None, index_cap_km=None),
}


def _find_granules():
    """Every recognized cross-track granule under data/, matched case-insensitively and
    recursively, de-duplicated by resolved path, in a stable order: AVHRR (visible-infrared, the
    tightest geolocation), then VIIRS (aggregated), then ATMS (microwave), then OLCI (the push-broom).
    The suite passes when every one of them passes. Only recognized families are matched, so a
    resampled or conical file in data/ (VGAC, SSMIS, AMSR2) is correctly ignored."""
    d = REPO / "data"
    files = [p.resolve() for p in sorted(d.rglob("*")) if p.is_file()]

    def match(pred):
        return [p for p in files if pred(p.name.lower())]

    avhrr = match(lambda n: "avhr" in n and n.endswith(".nc"))
    viirs = (match(lambda n: (n.startswith("gmtco_") or n.startswith("gitco_")) and n.endswith(".h5"))
             + match(lambda n: ("vj103mod" in n or "vnp03" in n) and n.endswith(".nc")))
    atms = match(lambda n: "atms" in n and n.endswith(".nc"))
    # OLCI L1B EFR is a directory product (.SEN3) whose geolocation is split across sibling files.
    # The granule handle is its geo_coordinates.nc, matched only when the tie_geometries.nc that
    # carries the view zenith sits beside it, so an unrelated geo_coordinates.nc is not picked up.
    olci = [p for p in files if p.name == "geo_coordinates.nc"
            and (p.parent / "tie_geometries.nc").is_file()]
    legacy = [p for p in files if p.name == "native_viirs_geo.nc"]

    seen, out = set(), []
    for p in [*avhrr, *viirs, *atms, *olci, *legacy]:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def great_circle_deg(lat1, lon1, lat2, lon2):
    """Central angle in degrees between two geographic points."""
    a1, a2 = lat1 * D2R, lat2 * D2R
    dlon = (np.asarray(lon2, float) - np.asarray(lon1, float)) * D2R
    h = np.sin((a2 - a1) / 2) ** 2 + np.cos(a1) * np.cos(a2) * np.sin(dlon / 2) ** 2
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(h, 0, 1))))


def central_from_zenith(zenith_deg, altitude_km, earth_radius_km=EARTH_RADIUS_KM):
    """Central angle from the sub-satellite point predicted from the satellite zenith angle.

    gamma = epsilon - arcsin(sin epsilon / k), k = (R + h) / R. Uses the granule's own per-pixel
    zenith, so it needs no assumption about how the instrument samples across track.
    """
    eps = np.asarray(zenith_deg, float) * D2R
    k = (earth_radius_km + np.asarray(altitude_km, float)) / earth_radius_km
    theta = np.arcsin(np.clip(np.sin(eps) / k, -1.0, 1.0))
    return np.degrees(eps - theta)


def fit_crosstrack_profile(lat_row, lon_row, zenith_row, nadir_index, altitude_km=None):
    """Fit the equal-angle and slant-range models to one scan line's cross-track profile.

    Returns residuals (km, root mean square across the row) and the fitted altitude. Uses the
    per-pixel satellite zenith for the slant-range model, so aggregated sampling is handled.
    ``altitude_km`` fixes the altitude when known. Pass None to fit it.
    """
    lat_row = np.asarray(lat_row, float)
    lon_row = np.asarray(lon_row, float)
    zen = np.abs(np.asarray(zenith_row, float))
    off = (np.arange(lat_row.size) - nadir_index).astype(float)
    side = np.sign(off)

    gamma = great_circle_deg(lat_row[nadir_index], lon_row[nadir_index], lat_row, lon_row) * side

    # Only jointly finite pixels enter the fit and the residuals. A single fill pixel would
    # otherwise turn the least-squares sums into NaN and silently drop the whole row.
    valid = np.isfinite(gamma) & np.isfinite(zen)

    # equal-angle model: gamma = a * off, least squares through the origin
    a = float(np.sum((gamma * off)[valid]) / np.sum((off * off)[valid]))
    rms_equal_km = _rms_km((gamma - a * off)[valid])

    # slant-range model: fit the single altitude that best matches the observed profile
    def resid(h):
        return _rms_km((gamma - central_from_zenith(zen, h) * side)[valid])

    if altitude_km is not None:
        h_best = float(np.mean(altitude_km))
    else:
        h_lo, h_hi = 300.0, 1400.0
        for _ in range(60):  # golden-free bisection-style narrowing on a unimodal residual
            m1, m2 = h_lo + (h_hi - h_lo) / 3, h_hi - (h_hi - h_lo) / 3
            if resid(m1) < resid(m2):
                h_hi = m2
            else:
                h_lo = m1
        h_best = 0.5 * (h_lo + h_hi)

    return {
        "rms_equal_km": rms_equal_km,
        "rms_native_km": resid(h_best),
        "fitted_altitude_km": h_best,
        "half_swath_deg": float(np.nanmax(np.abs(gamma))),
    }


def fit_crosstrack_index(lat_row, lon_row, nadir_index):
    """Fit the uniform-step slant-range model to a scan line's cross-track profile from PIXEL
    INDEX alone, using no per-pixel angle from the file.

    gamma(i) = slant_range((i - nadir) * look_step, h), fitting the look-angle step and the
    altitude. This is the less-circular check. It reproduces the observed central angles from the
    array index and two physical parameters, so passing it validates the NativeCrossTrackGeometry
    class against real geolocation rather than only confirming the file's own zenith is
    self-consistent. It applies to instruments that sample uniformly in scan angle (AVHRR and ATMS). An
    aggregated instrument (VIIRS) is not uniform in index and leaves a large residual here, which
    is itself informative.
    """
    lat_row = np.asarray(lat_row, float)
    lon_row = np.asarray(lon_row, float)
    off = (np.arange(lat_row.size) - nadir_index).astype(float)
    gamma = great_circle_deg(lat_row[nadir_index], lon_row[nadir_index], lat_row, lon_row) * np.sign(off)
    valid = np.isfinite(gamma)
    g, o = gamma[valid], off[valid]
    if g.size < 8:
        return {"rms_km": np.inf, "look_step_deg": np.nan, "fitted_altitude_km": np.nan}

    # initialise the look-angle step from the edge central angle at a nominal altitude, then
    # narrow a 2-D grid over (look_step, altitude), both windows shrinking each pass.
    step_c = abs(central_angle_to_look_angle(np.abs(g).max(), 830.0)) / max(np.abs(o).max(), 1.0)
    step_w, h_c, h_w = step_c * 0.6, 800.0, 500.0   # first pass searches ~300-1300 km altitude
    best = None
    for _ in range(8):
        steps = np.linspace(max(1e-3, step_c - step_w), step_c + step_w, 31)
        for h in np.linspace(max(400.0, h_c - h_w), h_c + h_w, 25):
            model = look_angle_to_central_angle(o[None, :] * steps[:, None], h)
            rms = np.sqrt(np.nanmean((g[None, :] - model) ** 2, axis=1))
            rms = np.where(np.all(np.isfinite(model), axis=1), rms, np.inf)
            j = int(np.argmin(rms))
            if best is None or rms[j] < best[0]:
                best = (float(rms[j]), float(steps[j]), float(h))
        step_c, h_c = best[1], best[2]
        step_w *= 0.3
        h_w *= 0.3
    return {"rms_km": best[0] * D2R * EARTH_RADIUS_KM,
            "look_step_deg": best[1], "fitted_altitude_km": best[2]}


def _rms_km(residual_deg):
    r = np.asarray(residual_deg, float)
    r = r[np.isfinite(r)]
    return float(np.sqrt(np.mean(r ** 2)) * D2R * EARTH_RADIUS_KM) if r.size else np.inf


def load(path):
    """Load latitude, longitude and satellite zenith from a cross-track granule.

    Several layouts are handled. A CF-netCDF AVHRR GAC L1C and a C3S ATMS FCDR L1C keep latitude,
    longitude and a satellite/sensor zenith at the root. An operational VIIRS SDR (GMTCO/GITCO)
    keeps Latitude, Longitude and SatelliteZenithAngle under a terrain-corrected geolocation group
    /All_Data/VIIRS-*-GEO-TC_All, with a per-scan spacecraft ECEF SCPosition. A NASA VIIRS product
    (VJ103MOD/VNP03MOD) keeps them under a geolocation_data group. Fill values and out-of-range
    coordinates become NaN. The reader raises rather than guessing when a required field is missing
    or a geolocation group is ambiguous, returns the granule's geodetic spacecraft altitude when
    present, and reports the instrument family so the caller knows whether the uniform-step test
    applies. The FILENAME is included in the family match, because the ATMS FCDR carries a
    mislabelled 'Microwave Humidity Sounder' title.
    """
    import netCDF4
    if Path(str(path)).name == "geo_coordinates.nc":
        return _load_olci(Path(str(path)).parent)
    true_alt = None
    with netCDF4.Dataset(str(path)) as ds:
        grp = None
        if "All_Data" in ds.groups:
            geo = [s for n, s in ds["All_Data"].groups.items() if "GEO-TC" in n] \
                or [s for n, s in ds["All_Data"].groups.items() if "GEO" in n]
            if len(geo) > 1:
                raise ValueError("more than one geolocation group found, so specify which to use")
            grp = geo[0] if geo else None
        if grp is None:
            grp = ds.groups.get("geolocation_data", ds)
        title = (getattr(ds, "title", "") or getattr(ds, "Platform_Short_Name", "")
                 or Path(str(path)).name)
        # Determine the instrument FAMILY from the layout, the metadata, and the filename together,
        # and refuse an unknown one rather than defaulting to a regime. Defaulting would silently
        # skip the uniform-step assertion for a sparsely-labelled uniform instrument.
        meta = (" ".join(str(getattr(ds, a, "")) for a in ("instrument", "platform", "title"))
                + " " + Path(str(path)).name).upper()
        viirs_layout = "All_Data" in ds.groups or "geolocation_data" in ds.groups
        if "AVHRR" in meta:
            family = "AVHRR"
        elif "ATMS" in meta:
            family = "ATMS"
        elif "VIIRS" in meta or viirs_layout:
            family = "VIIRS"
        else:
            raise ValueError(
                "cannot determine the instrument family (AVHRR, ATMS or VIIRS) from layout, "
                "metadata or filename, so the sampling regime is unknown and the uniform-step test "
                "cannot be gated safely")

        def get(*names):
            for n in names:
                if n in grp.variables:
                    return np.ma.filled(grp[n][:].astype(float), np.nan)
            raise KeyError(f"none of {names} found, so confirm the granule layout")

        lat = get("Latitude", "latitude")
        lon = get("Longitude", "longitude")
        zen = get("SatelliteZenithAngle", "sensor_zenith_angle", "satellite_zenith_angle",
                  "sensor_zenith", "satellite_zenith")
        if "SCPosition" in grp.variables:
            scp = np.ma.filled(grp["SCPosition"][:].astype(float), np.nan)  # ECEF metres per scan
            if scp.ndim == 2 and scp.shape[1] == 3 and np.isfinite(scp).all():
                true_alt = _geodetic_altitude_km(scp)

    # fill values (large negatives) and physically impossible coordinates -> NaN
    for a in (lat, lon, zen):
        a[a < -200.0] = np.nan
    lat[np.abs(lat) > 90.0] = np.nan
    lon[np.abs(lon) > 180.0] = np.nan
    zen[(zen < 0.0) | (zen > 90.0)] = np.nan
    if not (lat.shape == lon.shape == zen.shape and lat.ndim == 2):
        raise ValueError(f"lat/lon/zenith shapes disagree or are not 2-D: "
                         f"{lat.shape}, {lon.shape}, {zen.shape}")
    return lat, lon, zen, str(title), true_alt, family


def _load_olci(sen3_dir):
    """Load a Sentinel-3 OLCI L1B EFR product (a .SEN3 directory) as a RESAMPLED push-broom swath.

    The EFR product is a delivered, resampled grid, not raw detector samples: its instrument_data.nc
    carries a per-pixel frame_offset and its detector_index repeats (fewer unique detectors than
    output columns), so it is treated as a processed product, not a native swath.

    Latitude and longitude are full-resolution in geo_coordinates.nc (rows along-track, columns
    across-track). The observation zenith angle OZA and its azimuth OAA are in tie_geometries.nc on
    a coarser across-track TIE grid (the along-track dimension is full resolution). OZA is an
    UNSIGNED magnitude, so at nadir it has a false plateau where OAA flips by 180 degrees, and
    interpolating the magnitude directly would mis-place nadir by tens of columns and inflate the
    residual. Instead the cross-track SIGN is taken from OAA (which side of the sub-satellite track
    each tie point is on), the signed zenith is interpolated so it passes smoothly through zero at
    nadir, and its magnitude is returned. The tie subsampling is checked against both the dimension
    arithmetic and the ac_subsampling_factor attribute. Fill values become NaN. Returns the same
    tuple shape as the other loaders, with family 'OLCI' and no spacecraft-position altitude.
    """
    import netCDF4
    with netCDF4.Dataset(str(sen3_dir / "geo_coordinates.nc")) as g:
        lat = np.ma.filled(g["latitude"][:].astype(float), np.nan)
        lon = np.ma.filled(g["longitude"][:].astype(float), np.nan)
    n_row, n_col = lat.shape
    with netCDF4.Dataset(str(sen3_dir / "tie_geometries.nc")) as t:
        oza_tie = np.ma.filled(t["OZA"][:].astype(float), np.nan)   # (tie_rows, tie_columns), unsigned
        oaa_tie = np.ma.filled(t["OAA"][:].astype(float), np.nan)   # view azimuth, degrees
        has_ac = "ac_subsampling_factor" in t.ncattrs()
        ac_factor = int(getattr(t, "ac_subsampling_factor", 0)) if has_ac else None
    n_tie_row, n_tie_col = oza_tie.shape
    if n_tie_row != n_row:
        raise ValueError(f"OLCI tie rows {n_tie_row} != geolocation rows {n_row}")
    if n_tie_col < 2:
        raise ValueError("OLCI needs at least two across-track tie columns")
    # The tie columns must be an exact equal-spaced subsampling spanning the full swath, so full
    # column j maps to tie coordinate j/step. Refuse anything else rather than mis-locate the angle,
    # and require the dimension-derived step to agree with the product's declared subsampling factor.
    if (n_col - 1) % (n_tie_col - 1) != 0:
        raise ValueError(f"OLCI tie columns {n_tie_col} do not evenly subsample {n_col} columns")
    step = (n_col - 1) // (n_tie_col - 1)
    # When the product declares the across-track subsampling factor, it MUST agree with the
    # dimension-derived step (and be non-zero). A genuinely absent attribute is tolerated, while a
    # present one is never allowed to disagree or to be zero.
    if has_ac and ac_factor != step:
        raise ValueError(f"OLCI ac_subsampling_factor {ac_factor} disagrees with derived step {step}")
    j = np.arange(n_col)
    tf = j / step
    lo = np.clip(np.floor(tf).astype(int), 0, n_tie_col - 2)
    frac = tf - lo

    # Sign each tie point by which side of the track it is on, from the view azimuth relative to one
    # edge, so the signed zenith crosses zero at nadir instead of plateauing at the unsigned minimum.
    ref = oaa_tie[:, 0:1]
    dazi = np.mod(oaa_tie - ref + 180.0, 360.0) - 180.0
    signed = np.where(np.abs(dazi) < 90.0, 1.0, -1.0) * oza_tie
    signed_full = signed[:, lo] * (1.0 - frac) + signed[:, lo + 1] * frac
    zen = np.abs(signed_full)

    for a in (lat, lon, zen):
        a[a < -200.0] = np.nan
    lat[np.abs(lat) > 90.0] = np.nan
    lon[np.abs(lon) > 180.0] = np.nan
    zen[(zen < 0.0) | (zen > 90.0)] = np.nan
    title = sen3_dir.name
    return lat, lon, zen, str(title), None, "OLCI"


def _geodetic_altitude_km(scposition_m):
    """Median WGS84 geodetic altitude (km) from per-scan spacecraft ECEF positions.

    The satellite is over high latitude in a typical granule, where height above the equatorial
    radius is not the true altitude, so the ECEF position is converted to a geodetic height above
    the WGS84 ellipsoid. This is the physical altitude the fitted spherical-model value is checked
    against, allowing for the spherical-Earth and terrain approximations in the model itself.
    """
    from pyproj import Transformer
    ecef_to_geodetic = Transformer.from_crs("EPSG:4978", "EPSG:4979", always_xy=True)
    _, _, height_m = ecef_to_geodetic.transform(
        scposition_m[:, 0], scposition_m[:, 1], scposition_m[:, 2])
    return float(np.nanmedian(np.asarray(height_m)) / 1000.0)


def validate_granule(path):
    """Load one native granule, run the three models over a sample of scan lines, print the
    residuals, and assert the slant-range claim with family-specific thresholds. Raises AssertionError,
    tagged with the granule path, on any failure."""
    rel = path.relative_to(REPO) if REPO in path.parents else path
    lat, lon, zen, title, true_alt, family = load(path)
    cfg = FAMILY[family]
    uniform, nominal_step = cfg["uniform"], cfg["step_deg"]
    zenith_cap, index_cap = cfg["zenith_cap_km"], cfg["index_cap_km"]
    expect_equal_angle = cfg.get("expect_equal_angle", False)
    n_line, n_pixel = lat.shape
    kind = "resampled push-broom, equal-angle" if expect_equal_angle else (
        "uniform-step" if uniform else "aggregated")
    print(f"\n{rel}")
    print(f"  {title!r} [{family}, {kind}]: {n_line} lines x {n_pixel} pixels")
    if true_alt is not None:
        print(f"  geodetic spacecraft altitude (from SCPosition): {true_alt:.1f} km")
    assert n_line >= 50 and n_pixel >= 90, f"{rel}: too small to be a swath"

    rng = np.random.default_rng(0)
    rows = rng.choice(n_line, size=min(300, n_line), replace=False)
    # The pixel-index (uniform-step) fit is only meaningful for a uniform slant-range swath, and it
    # is by far the most expensive fit, so it is skipped for an equal-angle push-broom.
    need_index = not expect_equal_angle
    eq, zn, ix, zn_alt, ix_alt, ix_step, half, rejected = [], [], [], [], [], [], [], 0
    for r in rows:
        good = np.isfinite(lat[r]) & np.isfinite(lon[r]) & np.isfinite(zen[r])
        if np.count_nonzero(good) < n_pixel * 0.9:
            rejected += 1
            continue
        # the nadir pixel is the smallest satellite zenith among the JOINTLY VALID pixels. The
        # zenith is used ONLY to locate nadir, not in the pixel-index fit itself.
        nadir = int(np.argmin(np.where(good, np.abs(zen[r]), np.inf)))
        f = fit_crosstrack_profile(lat[r], lon[r], zen[r], nadir)   # equal-angle + zenith relation
        if np.isfinite(f["rms_native_km"]) and np.isfinite(f["rms_equal_km"]):
            eq.append(f["rms_equal_km"])
            zn.append(f["rms_native_km"])
            zn_alt.append(f["fitted_altitude_km"])
            half.append(f["half_swath_deg"])
            if need_index:
                u = fit_crosstrack_index(lat[r], lon[r], nadir)     # pixel-index, no fit-time zenith
                ix.append(u["rms_km"])
                ix_alt.append(u["fitted_altitude_km"])
                ix_step.append(u["look_step_deg"])
    eq, zn = np.array(eq), np.array(zn)
    ix = np.array(ix) if need_index else np.array([np.nan])
    zn_alt, half = np.array(zn_alt), np.array(half)
    ix_alt = np.array(ix_alt) if need_index else np.array([np.nan])
    ix_step = np.array(ix_step) if need_index else np.array([np.nan])
    print(f"  usable lines: {eq.size} of {len(rows)} sampled ({rejected} rejected for fill)")
    # require a strong majority usable, so a median cannot be carried by a minority of good lines
    assert eq.size >= 0.8 * len(rows), f"{rel}: only {eq.size}/{len(rows)} lines usable"

    m_eq, m_zn, m_ix = np.median(eq), np.median(zn), np.median(ix)
    p95_zn = np.percentile(zn, 95)
    print(f"  equal-angle (linear in index) : median {m_eq:6.2f} km")
    if need_index:
        p95_ix = np.percentile(ix, 95)
        print(f"  pixel-index (no fit zenith)   : median {m_ix:6.2f} km "
              f"(95th {p95_ix:5.2f}, step {np.median(ix_step):.4f} deg, altitude {np.median(ix_alt):.0f} km)")
    print(f"  file-zenith (consistency)     : median {m_zn:6.2f} km (95th {p95_zn:.2f} km)")

    if expect_equal_angle:
        # This product is delivered on a uniform cross-track GROUND-angle grid, so the honest tests
        # are inverted from the slant-range case. (a) The equal-angle model must FIT, which is what
        # places it in the equal_angle cross_track_kind. (b) It must be a genuinely wide swath, so
        # the equal-angle fit is not trivial. (c) Its own view zenith must both AGREE with the
        # geolocation to within a tight cap and imply a physical altitude near the platform's real
        # value, confirming the geometry independently of the equal-angle sampling.
        half_swath = float(np.median(half))
        m_alt = np.median(zn_alt)
        lo, hi = cfg["alt_band_km"]
        print(f"  -> equal-angle push-broom PRODUCT (this granule, 1-D profile): equal-angle fits "
              f"to {m_eq:.2f} km over a {half_swath:.1f} deg half-swath, view zenith agrees to "
              f"{m_zn:.2f} km, implying {m_alt:.0f} km altitude")
        assert m_eq < cfg["equal_fit_cap_km"], (
            f"{rel}: equal-angle does not fit ({m_eq:.2f} km), so this product is not the "
            f"equal_angle kind after all")
        assert half_swath > 3.0, f"{rel}: half-swath {half_swath:.1f} deg too narrow to be decisive"
        assert m_zn < cfg["zenith_consistency_cap_km"] and p95_zn < 2.0 * cfg["zenith_consistency_cap_km"], (
            f"{rel}: view zenith does not agree with the geolocation "
            f"(median {m_zn:.2f} km, 95th {p95_zn:.2f} km)")
        assert lo < m_alt < hi, (
            f"{rel}: view-zenith altitude {m_alt:.0f} km outside the physical band {lo}-{hi} km")
        return

    # 1. Equal-angle failure: the direct, non-circular evidence that native pixels are not
    #    uniform in central angle by index. The absolute floor is instrument-independent.
    assert m_eq > 5.0, f"{rel}: equal-angle fits ({m_eq:.1f} km), so this may be resampled"
    # 2. The slant-range relation via file zenith must clearly beat equal-angle (relative), stay
    #    inside the family's absolute cap (median and tail), because the floor is set by the
    #    instrument's footprint scale.
    assert m_zn < 0.3 * m_eq and m_zn < zenith_cap and p95_zn < 2.0 * zenith_cap, (
        f"{rel}: the slant-range relation does not hold via file zenith "
        f"(median {m_zn:.2f} km, 95th {p95_zn:.2f} km, cap {zenith_cap} km)")
    if uniform:
        # 3. The PIXEL-INDEX fit reproduces the radial central-angle profile from the array index and
        #    two physical parameters, with no per-pixel angle in the fit, recovering the KNOWN
        #    instrument step and a physical altitude. This is LESS CIRCULAR than the zenith check,
        #    but still a one-dimensional radial-profile test, not a full two-dimensional reproduction.
        assert m_ix < 0.3 * m_eq and m_ix < index_cap and p95_ix < 2.0 * index_cap, (
            f"{rel}: the pixel-index model did not reproduce this uniform swath "
            f"(median {m_ix:.2f} km, 95th {p95_ix:.2f} km, cap {index_cap} km)")
        assert abs(np.median(ix_step) - nominal_step) < 0.15 * nominal_step \
            and np.std(ix_step) < 0.1 * nominal_step, (
            f"{rel}: fitted step {np.median(ix_step):.4f} deg (std {np.std(ix_step):.4f}) does not "
            f"match the known {family} step {nominal_step} deg")
        assert 350.0 < np.median(ix_alt) < 1200.0, (
            f"{rel}: pixel-index altitude {np.median(ix_alt):.0f} km is not physical LEO")
        print("  -> pixel-index reproduces the profile from index and physical parameters alone "
              "(the less-circular check)")
    else:
        print("  -> pixel-index not asserted (aggregated across track, so the file zenith carries "
              "the real sampling)")
    if true_alt is not None:
        offset = abs(np.median(zn_alt) - true_alt)
        assert offset < 30.0, (
            f"{rel}: fitted altitude {np.median(zn_alt):.1f} km differs from the geodetic "
            f"{true_alt:.1f} km by {offset:.1f} km, beyond the spherical approximation")
        print(f"  -> fitted altitude within {offset:.1f} km of the geodetic spacecraft altitude")


def main():
    granules = _find_granules()
    if not granules:
        print("SKIP: no cross-track granule present in data/. See the module docstring for "
              "the sources (a CF-netCDF AVHRR GAC L1C or ATMS FCDR L1C, or a native VIIRS granule).")
        return 0
    print(f"validating {len(granules)} cross-track granule(s)")
    failures = []
    for path in granules:
        rel = path.relative_to(REPO) if REPO in path.parents else path
        try:
            validate_granule(path)          # prints its own header and results
        except (AssertionError, KeyError, ValueError, OSError) as exc:
            # continue through the remaining granules, then fail once with all of them named
            print(f"  FAILED: {exc}")
            failures.append(str(rel))
    if failures:
        raise AssertionError(f"{len(failures)} of {len(granules)} native granule(s) failed: "
                             + "; ".join(failures))
    print(f"\ndone: {len(granules)} granule(s) validated. The whiskbroom and aggregated swaths are "
          "native and non-uniform (the equal-angle model fails) and follow the slant-range geometry; "
          "the OLCI push-broom product is a resampled equal-angle grid whose view zenith is "
          "consistent with Sentinel-3's geometry.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
