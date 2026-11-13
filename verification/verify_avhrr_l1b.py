"""Fit the native cross-track models to AVHRR GAC Level 1B granules from NOAA CLASS.

Run from the repo root with the project venv:

    .venv/bin/python verification/verify_avhrr_l1b.py

Reads every AVHRR GAC Level 1B granule under data/avhrr_gac_l1b/ (NSS.GHRR.*, NOAA KLM format,
ordered from NOAA CLASS) and runs the three cross-track fits of verify_native_crosstrack.py on
each granule's STORED anchor-point geolocation. Level 1B keeps latitude and longitude at 51 anchor
points per scan, so the fits see the producer's own navigation at its original sampling, with no
interpolation. Each granule must pass on its own, and the script passes only when all of them do.

What each fit shows, from the weakest claim to the strongest:
  1. The equal-angle model (central angle linear in anchor index) must FAIL by tens of kilometers.
     This is the direct evidence that the native samples are not a uniform angular grid.
  2. The file-zenith fit is a CONSISTENCY check, because the producer derives the satellite zenith
     and the coordinates together. Its cap is looser than a per-pixel product would need,
     because the residual on these anchors is about 2.9 km, and the source of that residual was
     not examined.
  3. The anchor-index fit reproduces the radial profile from the anchor index and two physical
     parameters, the look-angle step and the altitude, with the zenith used only to locate nadir.
     The recovered step must match the AVHRR GAC step and the recovered altitude must match the
     spacecraft altitude the granule itself records.

L1B format facts used here follow the NOAA KLM User's Guide layout as encoded by the pygac
project's GAC KLM reader. The file has an optional
512-byte archive header, then 4608-byte records (one header record, then one record per scan).
Each scan record holds 51 anchor points at every 8th GAC pixel starting at pixel 5 (1-based),
with latitude and longitude as big-endian int32 scaled by 1e-4 degree, an angular-relationships
block of 153 big-endian int16 scaled by 1e-2 degree holding solar zenith, satellite zenith and
relative azimuth for the anchors, and a spacecraft altitude in 0.1 km. The angular block's layout
is not assumed. The satellite-zenith slot is identified empirically as the unique slot with the
V-shaped, near-symmetric cross-track profile, and the script refuses to proceed if zero or
several slots qualify.

Scope. These are sampled ONE-DIMENSIONAL RADIAL PROFILES, not two-dimensional geolocation. They
do not check that the anchors lie on the cross-track great circle or test their along-track
positions. The nadir reference is the nearest stored anchor, so sub-anchor nadir displacement
stays inside the anchor-index residual rather than being tuned out.
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
        "zenith_p95_km": float(np.percentile(zn, 95)),
        "zenith_alt_km": float(np.median(zn_alt)),
        "index_km": float(np.median(ix)),
        "index_p95_km": float(np.percentile(ix, 95)),
        "index_step_deg": float(np.median(ix_step)),
        "index_alt_km": float(np.median(ix_alt)),
        "file_alt_km": float(np.median(alt_km[np.isfinite(alt_km) & (alt_km > 0)])),
    }


def main():
    l1b_files = sorted((REPO / "data" / "avhrr_gac_l1b").glob("NSS.GHRR.*"))
    if not l1b_files:
        print("SKIP: needs data/avhrr_gac_l1b/NSS.GHRR.* (AVHRR GAC Level 1B from NOAA CLASS)")
        return 0

    # Thresholds. The equal-angle floor, the anchor-index cap and the step tolerance were set from
    # the first measured runs (equal-angle about 97 km, anchor-index fit about 3.6 km at a step of
    # 0.2700 degrees per GAC pixel and an altitude of 828 to 829 km, file zenith about 2.9 km,
    # onboard altitude 830.0 km) with headroom, per the set-from-clean-values rule. The 0.2709 degree
    # reference step and its 0.005 degree tolerance are carried unchanged from those runs.
    failures = []
    for p in l1b_files:
        lat, lon, ang, alt_km, quality = read_l1b(p)
        zen, slot = identify_satellite_zenith(ang)
        s = anchor_stats(lat, lon, zen, alt_km, quality)
        step_pix = s["index_step_deg"] / ANCHOR_STEP_PIXELS
        print(f"\n{p.name}: {lat.shape[0]} scans, satellite zenith at {slot}")
        print(f"  usable sampled scans          : {s['n_used']} of {s['n_sampled']} "
              f"({s['n_quality_rejected']} rejected by fatal quality bits)")
        print(f"  equal-angle (linear in index) : median {s['equal_km']:7.2f} km")
        print(f"  anchor-index (no fit zenith)  : median {s['index_km']:7.2f} km "
              f"(95th {s['index_p95_km']:.2f} km, step {s['index_step_deg']:.4f} deg/anchor = "
              f"{step_pix:.4f} deg/GAC pixel, altitude {s['index_alt_km']:.0f} km)")
        print(f"  file-zenith (consistency)     : median {s['zenith_km']:7.2f} km "
              f"(95th {s['zenith_p95_km']:.2f} km, altitude {s['zenith_alt_km']:.0f} km)")
        print(f"  file's own spacecraft altitude: {s['file_alt_km']:.1f} km")
        checks = [
            (s["equal_km"] > 50.0, f"equal-angle fits ({s['equal_km']:.1f} km), not native?"),
            (s["index_km"] < 8.0 and s["index_km"] < 0.15 * s["equal_km"],
             f"anchor-index fit {s['index_km']:.2f} km does not reproduce the anchors"),
            (s["zenith_km"] < 6.0,
             f"stored-zenith consistency {s['zenith_km']:.2f} km exceeds its 6 km cap"),
            (abs(step_pix - 0.2709) < 0.005,
             f"recovered step {step_pix:.4f} deg/GAC pixel is not the AVHRR GAC step"),
            (abs(s["zenith_alt_km"] - s["file_alt_km"]) < 10.0,
             f"fitted altitude {s['zenith_alt_km']:.0f} km disagrees with the granule's own "
             f"spacecraft altitude {s['file_alt_km']:.1f} km"),
            (abs(s["index_alt_km"] - s["file_alt_km"]) < 10.0,
             f"anchor-index altitude {s['index_alt_km']:.0f} km disagrees with the granule's own "
             f"spacecraft altitude {s['file_alt_km']:.1f} km"),
        ]
        bad = [msg for ok, msg in checks if not ok]
        for msg in bad:
            print(f"  FAILED: {msg}")
        if bad:
            failures.append(p.name)
    if failures:
        raise AssertionError(f"{len(failures)} of {len(l1b_files)} Level 1B granule(s) failed: "
                             + "; ".join(failures))

    print(f"\ndone: {len(l1b_files)} Level 1B granule(s). On sampled one-dimensional radial "
          "profiles of the stored anchors, the equal-angle model fails, the anchor-index fit "
          "recovers the AVHRR GAC step and an altitude matching the granule's own spacecraft "
          "altitude, and the stored zenith is consistent with the slant-range relation. The nadir "
          "anchor is the nearest stored point, so sub-anchor nadir displacement remains inside the "
          "anchor-index residual.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
