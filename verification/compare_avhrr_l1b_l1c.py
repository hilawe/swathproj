"""Compare native AVHRR GAC geolocation between the NOAA Level 1B and the EUMETSAT L1C.

Run from the repo root with the project venv:

    .venv/bin/python verification/compare_avhrr_l1b_l1c.py

The CF-netCDF granule used for the native cross-track evidence (data/avhrr_l1c/*.nc, EUMETSAT C3S
FCDR AVHRR GAC L1C) derives its geolocation from NOAA Level 1B GAC files. This script reads raw L1B
granules (data/avhrr_gac_l1b/NSS.GHRR.*.SV, NOAA KLM format, ordered from NOAA CLASS) and runs
the SAME three cross-track fits on the L1B's own STORED anchor-point geolocation, then the L1C
granule through the standard harness, and compares the recovered physics. The value of the
comparison is that the L1B anchors are the upstream source, at the original 51-points-per-scan
sampling with no interpolation, in a different format and from a different archive, so agreement
in the recovered step and altitude checks the native cross-track result at its source.

The granules are different orbits (the L1C is 2024, the L1B order is 2026), so the comparison is
of recovered instrument physics (per-pixel look-angle step, orbit altitude, the equal-angle
failure), not a pixel-by-pixel match.

L1B format facts used here follow the NOAA KLM User's Guide layout as encoded by the pygac
project's GAC KLM reader (the same reader named in the L1C filename). The file has an optional
512-byte archive header, then 4608-byte records (one header record, then one record per scan).
Each scan record holds 51 anchor points at every 8th GAC pixel starting at pixel 5 (1-based),
with latitude and longitude as big-endian int32 scaled by 1e-4 degree, an angular-relationships
block of 153 big-endian int16 scaled by 1e-2 degree holding solar zenith, satellite zenith and
relative azimuth for the anchors, and a spacecraft altitude in 0.1 km. The angular block's layout
is not assumed. The satellite-zenith slot is identified empirically as the unique slot with the
V-shaped, near-symmetric cross-track profile, and the script refuses to proceed if zero or
several slots qualify.

Scope. This compares sampled ONE-DIMENSIONAL RADIAL PROFILES, not two-dimensional geolocation,
and the two products share a geophysical lineage. What is independent is the archive, the
format, and the sampling (stored anchors against interpolated pixels). The nadir reference is
the nearest stored anchor, so sub-anchor nadir displacement stays inside the pixel-index
residual rather than being tuned out.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from verification.verify_native_crosstrack import (  # noqa: E402
    fit_crosstrack_index,
    fit_crosstrack_profile,
    load as load_l1c,
)

RECORD = 4608
ANCHOR_FIRST_PIXEL_1BASED = 5
ANCHOR_STEP_PIXELS = 8
N_ANCHORS = 51

# Reduced record layout: only the navigation fields this comparison needs, addressed by byte
# offset within the 4608-byte scan record. Offsets follow the full KLM GAC scanline layout
# (validated below by the itemsize assertion on the full prefix).
_PREFIX = np.dtype([
    ("scan_line_number", ">u2"),
    ("scan_line_year", ">u2"),
    ("scan_line_day_of_year", ">u2"),
    ("satellite_clock_drift_delta", ">i2"),
    ("scan_line_utc_time_of_day", ">u4"),
    ("scan_line_bit_field", ">u2"),
    ("zero_fill0", ">i2", (5,)),
    ("quality_indicator_bit_field", ">u4"),
    ("scan_line_quality_flags", ">u1", (4,)),
    ("calibration_quality_flags", ">u2", (3,)),
    ("count_of_bit_errors_in_frame_sync", ">u2"),
    ("zero_fill1", ">i4", (2,)),
    ("calibration_coefficients", ">i4", (63,)),   # 15 x 3 visible + 6 x 3 infrared
    ("computed_yaw_steering", ">i2", (3,)),
    ("total_applied_attitude_correction", ">i2", (3,)),
    ("navigation_status_bit_field", ">u4"),
    ("time_associated_with_tip_euler_angles", ">u4"),
    ("tip_euler_angles", ">i2", (3,)),
    ("spacecraft_altitude", ">u2"),          # 0.1 km above the reference ellipsoid
    ("angular_relationships", ">i2", (153,)),
    ("zero_fill3", ">i2", (3,)),
    ("earth_location", [("lat", ">i4"), ("lon", ">i4")], (N_ANCHORS,)),
])


def read_l1b(path):
    """Read one NSS.GHRR L1B granule's stored anchor geolocation and navigation fields.

    Returns lat, lon (degrees, n_scans x 51), the angular-relationships block
    (n_scans x 153, degrees), the per-scan spacecraft altitude (km), and the scan quality
    bit field. The Level 1B header record is validated (KLM format version, declared record
    and block sizes, header-record count) rather than assumed, and the declared number of
    header records is skipped.
    """
    # Validate the reduced layout against the documented KLM octets (329 and 641, one-based)
    # before trusting any parse built on it.
    assert _PREFIX.itemsize == 1048
    assert _PREFIX.fields["angular_relationships"][1] == 328
    assert _PREFIX.fields["earth_location"][1] == 640

    raw = Path(path).read_bytes()
    if len(raw) % RECORD == 512:
        raw = raw[512:]                      # archive (ARS) header
    if len(raw) % RECORD != 0:
        raise ValueError(f"{path}: size {len(raw)} is not a whole number of {RECORD}-byte records")
    n_rec = len(raw) // RECORD
    if n_rec < 3:
        raise ValueError(f"{path}: only {n_rec} records")

    # Level 1B header record: creation site (3 ASCII), format version (u2 at byte 4), then the
    # declared logical record length, block size and header-record count (u2 at 10, 12, 14).
    head = np.frombuffer(raw[:16], dtype=">u2")
    site = raw[:3].decode("ascii", "replace")
    version, lrl, blocksize, n_head = int(head[2]), int(head[5]), int(head[6]), int(head[7])
    if not site.isalnum() or version < 2 or lrl != RECORD or blocksize != RECORD or n_head < 1:
        raise ValueError(f"{path}: unexpected L1B header (site {site!r}, format version {version}, "
                         f"record {lrl}, block {blocksize}, header records {n_head})")
    if n_rec <= n_head:
        raise ValueError(f"{path}: {n_rec} records but {n_head} header records")
    body = np.frombuffer(raw, dtype=np.uint8).reshape(n_rec, RECORD)[n_head:]
    scans = np.frombuffer(body[:, : _PREFIX.itemsize].tobytes(), dtype=_PREFIX)

    lat = scans["earth_location"]["lat"].astype(float) / 1e4
    lon = scans["earth_location"]["lon"].astype(float) / 1e4
    ang = scans["angular_relationships"].astype(float) / 1e2
    alt_km = scans["spacecraft_altitude"].astype(float) / 10.0
    return lat, lon, ang, alt_km, scans["quality_indicator_bit_field"]


def identify_satellite_zenith(ang):
    """Pick the satellite-zenith slot of the angular block empirically.

    Candidates are the three slots of an interleaved (51, 3) layout and of a blocked (3, 51)
    layout. The satellite zenith is the unique candidate whose median cross-track profile is
    V-shaped: minimum within a few anchors of the swath center, both edges above 45 degrees,
    all values within [0, 75], and near-symmetric. Anything else (solar zenith, azimuth) fails
    at least one of these on a full orbit. Raises unless exactly one candidate qualifies.
    """
    n = ang.shape[0]
    cands = {}
    tri = ang.reshape(n, N_ANCHORS, 3)
    blk = ang.reshape(n, 3, N_ANCHORS)
    for k in range(3):
        cands[f"interleaved[{k}]"] = tri[:, :, k]
        cands[f"blocked[{k}]"] = blk[:, k, :]

    winners = []
    for name, z in cands.items():
        prof = np.nanmedian(z, axis=0)
        if not np.all(np.isfinite(prof)):
            continue
        c = int(np.argmin(prof))
        ok = (abs(c - N_ANCHORS // 2) <= 3
              and prof.min() >= 0.0 and prof.max() <= 75.0
              and prof[0] > 45.0 and prof[-1] > 45.0
              and abs(prof[0] - prof[-1]) < 5.0)
        if ok:
            winners.append(name)
    if len(winners) != 1:
        raise ValueError(f"satellite-zenith slot ambiguous or absent: candidates {winners}")
    return cands[winners[0]], winners[0]


def anchor_stats(lat, lon, zen, alt_km, quality, sample=300):
    """Run the three cross-track fits over a sample of L1B scans, on the stored anchors.

    Scans flagged fatal in the quality indicator bit field are rejected before fitting: bit 31
    (do not use for product generation), bit 30 (time sequence error) and bit 27 (earth
    location not available), which can corrupt geolocation while the decoded numbers still
    look plausible. Calibration-only flags do not reject a geolocation comparison.
    """
    FATAL_QUALITY = (1 << 31) | (1 << 30) | (1 << 27)
    n = lat.shape[0]
    rng = np.random.default_rng(0)
    rows = rng.choice(n, size=min(sample, n), replace=False)
    eq, zn, ix, ix_step, ix_alt, zn_alt = [], [], [], [], [], []
    n_quality_rejected = 0
    for r in rows:
        if int(quality[r]) & FATAL_QUALITY:
            n_quality_rejected += 1
            continue
        la, lo, ze = lat[r], lon[r], zen[r]
        if not (np.all(np.isfinite(la)) and np.all(np.abs(la) <= 90.0)
                and np.all(np.abs(lo) <= 180.0) and np.all((ze >= 0) & (ze <= 90))):
            continue
        nadir = int(np.argmin(ze))
        if not 5 <= nadir <= N_ANCHORS - 6:
            continue
        f = fit_crosstrack_profile(la, lo, ze, nadir)
        u = fit_crosstrack_index(la, lo, nadir)
        if np.isfinite(f["rms_native_km"]) and np.isfinite(f["rms_equal_km"]):
            eq.append(f["rms_equal_km"])
            zn.append(f["rms_native_km"])
            zn_alt.append(f["fitted_altitude_km"])
            ix.append(u["rms_km"])
            ix_step.append(u["look_step_deg"])
            ix_alt.append(u["fitted_altitude_km"])
    if len(eq) < 0.8 * len(rows):
        raise ValueError(f"only {len(eq)} of {len(rows)} sampled scans usable "
                         f"({n_quality_rejected} quality-rejected), so a median could be carried "
                         f"by a minority of good scans")
    return {
        "n_quality_rejected": n_quality_rejected,
        "n_used": len(eq),
        "n_sampled": len(rows),
        "equal_km": float(np.median(eq)),
        "zenith_km": float(np.median(zn)),
        "zenith_alt_km": float(np.median(zn_alt)),
        "index_km": float(np.median(ix)),
        "index_step_deg": float(np.median(ix_step)),
        "index_alt_km": float(np.median(ix_alt)),
        "file_alt_km": float(np.median(alt_km[np.isfinite(alt_km) & (alt_km > 0)])),
    }


def l1c_stats(path, sample=300):
    """The same three fits on the L1C granule via the standard loader, at full 409-pixel
    resolution, for the side-by-side comparison."""
    lat, lon, zen, _title, _alt, family = load_l1c(path)
    if family != "AVHRR":
        raise ValueError(f"{path}: expected an AVHRR L1C, got {family}")
    n = lat.shape[0]
    rng = np.random.default_rng(0)
    rows = rng.choice(n, size=min(sample, n), replace=False)
    eq, zn, ix, ix_step, ix_alt = [], [], [], [], []
    for r in rows:
        good = np.isfinite(lat[r]) & np.isfinite(lon[r]) & np.isfinite(zen[r])
        if np.count_nonzero(good) < lat.shape[1] * 0.9:
            continue
        nadir = int(np.argmin(np.where(good, np.abs(zen[r]), np.inf)))
        f = fit_crosstrack_profile(lat[r], lon[r], zen[r], nadir)
        u = fit_crosstrack_index(lat[r], lon[r], nadir)
        if np.isfinite(f["rms_native_km"]) and np.isfinite(f["rms_equal_km"]):
            eq.append(f["rms_equal_km"])
            zn.append(f["rms_native_km"])
            ix.append(u["rms_km"])
            ix_step.append(u["look_step_deg"])
            ix_alt.append(u["fitted_altitude_km"])
    if len(eq) < 0.8 * len(rows):
        raise ValueError(f"L1C: only {len(eq)} of {len(rows)} sampled lines usable")
    return {
        "n_used": len(eq),
        "equal_km": float(np.median(eq)),
        "zenith_km": float(np.median(zn)),
        "index_km": float(np.median(ix)),
        "index_step_deg": float(np.median(ix_step)),
        "index_alt_km": float(np.median(ix_alt)),
    }


def main():
    l1b_files = sorted((REPO / "data" / "avhrr_gac_l1b").glob("NSS.GHRR.*"))
    l1c_files = sorted((REPO / "data" / "avhrr_l1c").glob("*.nc"))
    if not l1b_files or not l1c_files:
        print("SKIP: needs data/avhrr_gac_l1b/NSS.GHRR.* (NOAA CLASS L1B) and "
              "data/avhrr_l1c/*.nc (EUMETSAT C3S FCDR L1C)")
        return 0

    results = []
    for p in l1b_files:
        lat, lon, ang, alt_km, quality = read_l1b(p)
        zen, slot = identify_satellite_zenith(ang)
        s = anchor_stats(lat, lon, zen, alt_km, quality)
        results.append((p.name, s))
        print(f"\n{p.name}: {lat.shape[0]} scans, satellite zenith at {slot}")
        print(f"  usable sampled scans          : {s['n_used']} of {s['n_sampled']} "
              f"({s['n_quality_rejected']} rejected by fatal quality bits)")
        print(f"  equal-angle (linear in index) : median {s['equal_km']:7.2f} km")
        print(f"  pixel-index (anchors)         : median {s['index_km']:7.2f} km "
              f"(step {s['index_step_deg']:.4f} deg/anchor = {s['index_step_deg']/ANCHOR_STEP_PIXELS:.4f} "
              f"deg/GAC pixel, altitude {s['index_alt_km']:.0f} km)")
        print(f"  file-zenith (consistency)     : median {s['zenith_km']:7.2f} km "
              f"(altitude {s['zenith_alt_km']:.0f} km)")
        print(f"  file's own spacecraft altitude: {s['file_alt_km']:.1f} km")

    lc = l1c_stats(l1c_files[0])
    print(f"\n{l1c_files[0].name}:")
    print(f"  equal-angle                   : median {lc['equal_km']:7.2f} km")
    print(f"  pixel-index                   : median {lc['index_km']:7.2f} km "
          f"(step {lc['index_step_deg']:.4f} deg/GAC pixel, altitude {lc['index_alt_km']:.0f} km)")
    print(f"  file-zenith                   : median {lc['zenith_km']:7.2f} km")
    # The L1C side must itself pass before its numbers can anchor the comparison, at the same
    # AVHRR caps the standard harness applies (measured 98.9 / 3.67 / 0.12 km).
    assert lc["equal_km"] > 50.0, f"L1C: equal-angle fits ({lc['equal_km']:.1f} km)"
    assert lc["index_km"] < 6.0, f"L1C: pixel-index residual {lc['index_km']:.2f} km too large"
    assert lc["zenith_km"] < 1.0, f"L1C: file-zenith residual {lc['zenith_km']:.2f} km too large"

    # Assertions. Thresholds are set from the first measured run (equal-angle ~97 km, index fit
    # ~3.6 km at step 0.2700 deg/GAC pixel and altitude 828-829 km, file zenith ~2.9 km,
    # onboard altitude 830.0 km) with headroom, per the set-from-clean-values rule. The L1B
    # zenith cap is far looser than the L1C's because the L1B stores angles quantized to
    # 0.01 degree, which alone contributes kilometers at the swath edge.
    print("\nComparison of recovered physics, L1B anchors (per GAC pixel) vs L1C pixels:")
    for name, s in results:
        step_pix = s["index_step_deg"] / ANCHOR_STEP_PIXELS
        print(f"  {name}: step {step_pix:.4f} vs {lc['index_step_deg']:.4f} deg "
              f"({100 * abs(step_pix - lc['index_step_deg']) / lc['index_step_deg']:.2f}% apart), "
              f"altitude {s['index_alt_km']:.0f} vs {lc['index_alt_km']:.0f} km, "
              f"fitted-vs-onboard altitude {s['zenith_alt_km']:.0f} vs {s['file_alt_km']:.0f} km")
        assert s["equal_km"] > 50.0, f"{name}: equal-angle fits ({s['equal_km']:.1f} km), not native?"
        assert s["index_km"] < 8.0 and s["index_km"] < 0.15 * s["equal_km"], (
            f"{name}: pixel-index fit {s['index_km']:.2f} km does not reproduce the anchors")
        assert s["zenith_km"] < 6.0, (
            f"{name}: stored-zenith consistency {s['zenith_km']:.2f} km exceeds the "
            f"quantization-limited cap")
        assert abs(step_pix - 0.2709) < 0.005, (
            f"{name}: recovered step {step_pix:.4f} deg/GAC pixel is not the AVHRR GAC step")
        assert abs(step_pix - lc["index_step_deg"]) < 0.005, (
            f"{name}: L1B step {step_pix:.4f} disagrees with L1C {lc['index_step_deg']:.4f}")
        assert abs(s["index_alt_km"] - lc["index_alt_km"]) < 15.0, (
            f"{name}: L1B altitude {s['index_alt_km']:.0f} km disagrees with L1C "
            f"{lc['index_alt_km']:.0f} km")
        assert abs(s["zenith_alt_km"] - s["file_alt_km"]) < 10.0, (
            f"{name}: fitted altitude {s['zenith_alt_km']:.0f} km disagrees with the granule's "
            f"own onboard value {s['file_alt_km']:.1f} km")

    print("\ndone: on sampled one-dimensional radial profiles, the raw L1B anchor geolocation "
          "gives the same recovered physics as the L1C (step, altitude, equal-angle failure), "
          "from an independent archive and format though a shared geophysical lineage, and the "
          "fitted altitude matches the granule's own onboard navigation value. The nadir anchor "
          "is the nearest stored point, so sub-anchor nadir displacement remains inside the "
          "pixel-index residual.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
