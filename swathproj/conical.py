"""Conical scanning geometry, sharing the orbit model with the cross-track case.

This module exists to make the proposal's central structural claim executable: the orbit half
of the mapping is IDENTICAL for every in-scope scan geometry, and only the per-pixel offset
locus differs.

    shared      rotated pole + Earth-rotation shear  ->  sub-satellite point for a scan
    differs     the offset from that point to a pixel

For a cross-track or push-broom instrument that offset runs along a meridian of the rotated
frame, which is what ``SwathGeometry`` implements. For a conical instrument the instrument
holds a constant Earth incidence angle, so the offset traces a SMALL CIRCLE of fixed angular
radius about the sub-satellite point, and a pixel is picked out by its azimuth around that
circle. That is the whole difference, and it is why both are one grid mapping with a
``scan_geometry`` discriminator rather than two unrelated proposals.

The cone radius may be a SCALAR or an array of one value per scan. That is not a convenience.
Measured on a real CSU SSMIS granule, spacecraft altitude varies about 37 km within a single
pass and the cone radius tracks it at a correlation of +0.97; a single fixed radius would
misplace pixels by up to about 43 km across the pass, so the DMSP series needs a per-scan
cone radius.

The SAME orbital eccentricity affects a second quantity: the along-track ANGULAR RATE varies with altitude, at a correlation
of -0.91, exactly as Kepler's second law requires. On the SSMIS granule a constant along-track
step, even the best-fit one, leaves a 16.6 km rms and 38.5 km maximum along-track error. So
``scan_longitude_deg`` may carry the along-track angle per scan instead. A synthetic resampled
product such as VGAC does not need it, because its grid is DEFINED as a uniform division of 360
degrees, whereas a native instrument swath has no such guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import D2R, _validate_axis, _zyz_to_cf_pole, _forward_rotation
from .rotated_pole import geographic_to_rotated, rotated_to_geographic


@dataclass
class ConicalGeometry:
    """A conically scanning instrument on a near-circular orbit.

    Orbit parameters (shared with SwathGeometry, same meanings):
        ref_lat, ref_lon    sub-satellite position at the reference time, degrees
        heading             orbit trajectory angle, degrees
        n_scan              number of scans (rotations sampled) in the file
        scan_angle_deg      along-track angular advance per scan, degrees
        scan_origin_deg     rotated longitude of scan 0
        rotation_rate       Earth rotation, degrees of longitude per hour
        scan_time_hours     observation time of each scan, hours since the reference

    Cone parameters (specific to this geometry):
        n_pixel             samples per rotation retained in the file
        cone_radius_deg     angular radius from the sub-satellite point. Scalar, or one value
                            per scan when the orbit is not tight enough for a constant.
        azimuth_origin_deg  cone azimuth of pixel 0, measured from the rotated-frame north
        azimuth_step_deg    azimuth advance per pixel
    """

    ref_lat: float
    ref_lon: float
    heading: float
    n_scan: int
    n_pixel: int
    scan_angle_deg: float
    cone_radius_deg: float | np.ndarray
    azimuth_origin_deg: float
    azimuth_step_deg: float
    scan_time_hours: np.ndarray | None = None
    scan_origin_deg: float = 0.0
    # Per-scan along-track angle, overriding scan_origin + j * scan_angle when supplied.
    # Needed whenever the orbit is not circular enough for a constant step: see the class
    # docstring.
    scan_longitude_deg: np.ndarray | None = None
    rotation_rate: float = 15.0
    # present so the SHARED validation can run unchanged. A conical instrument has no
    # cross-track cell size, and the cross-track extent check is deliberately not applied:
    # n_pixel counts azimuth samples around the cone, not cells across a track.
    cell_size_deg: float = 1.0
    nadir_index: int = 0
    full_revolution: bool | None = None

    def __post_init__(self):
        if self.scan_time_hours is None:
            raise ValueError(
                "scan_time_hours is required: the Earth-rotation term needs each scan's "
                "observation time and it cannot be inferred from n_scan.")
        self.scan_time_hours = np.asarray(self.scan_time_hours, float)
        if self.scan_time_hours.shape != (self.n_scan,):
            raise ValueError(
                f"scan_time_hours must have shape ({self.n_scan},), "
                f"got {self.scan_time_hours.shape}")
        self._scan_angd = float(self.scan_angle_deg)
        _validate_axis(self)
        if self.azimuth_step_deg == 0.0 or not np.isfinite(self.azimuth_step_deg):
            raise ValueError(
                f"azimuth_step_deg must be finite and non-zero, got {self.azimuth_step_deg!r}")
        radius = np.asarray(self.cone_radius_deg, float)
        if radius.ndim == 0:
            radius = np.full(self.n_scan, float(radius))
        if radius.shape != (self.n_scan,):
            raise ValueError(
                f"cone_radius_deg must be a scalar or have shape ({self.n_scan},), "
                f"got {radius.shape}")
        if not np.all(np.isfinite(radius)) or np.any(radius <= 0) or np.any(radius >= 90):
            raise ValueError("cone_radius_deg must be finite and in (0, 90) degrees")
        self._radius = radius
        if self.scan_longitude_deg is None:
            self._sub_lon = (self.scan_origin_deg
                             + np.arange(self.n_scan, dtype=float) * self._scan_angd)
        else:
            self._sub_lon = np.asarray(self.scan_longitude_deg, float)
            if self._sub_lon.shape != (self.n_scan,):
                raise ValueError(
                    f"scan_longitude_deg must have shape ({self.n_scan},), "
                    f"got {self._sub_lon.shape}")
            if not np.all(np.isfinite(self._sub_lon)):
                raise ValueError("scan_longitude_deg must be finite")
        self._M = _forward_rotation(self.ref_lat, self.heading, self.ref_lon)

    def cf_rotated_pole(self):
        """The CF rotated_latitude_longitude parameters, derived exactly as for the
        cross-track case. This is the shared half of the mapping."""
        return _zyz_to_cf_pole(self._M)

    def forward(self, i, j):
        """Pixel (i azimuth index, j scan index) to geographic lat/lon in degrees.

        j must be an integer scan index, for the same reason as in the cross-track case: a
        scan carries its own observation time, so a fractional scan has no cell to belong to.
        """
        j_arr = np.asarray(j)
        if not np.all(np.equal(np.mod(np.asarray(j_arr, float), 1.0), 0.0)):
            raise ValueError("j must be an integer scan index")
        in_range = (j_arr >= 0) & (j_arr < self.n_scan)
        safe = np.where(in_range, j_arr, 0).astype(np.int64)

        rot_lat, rot_lon = self._cone_offset(i, safe)
        pole_lat, pole_lon, npgl = self.cf_rotated_pole()
        lat, lon = rotated_to_geographic(rot_lat, rot_lon, pole_lat, pole_lon, npgl)
        lon = lon - self.rotation_rate * self.scan_time_hours[safe]
        lat = np.where(in_range, lat, np.nan)
        lon = np.where(in_range, lon, np.nan)
        return lat, (lon + 180.0) % 360.0 - 180.0

    def _cone_offset(self, i, scan):
        """Offset from the sub-satellite point to pixel i, in the rotated frame.

        The sub-satellite point of scan j sits on the rotated equator at
        scan_origin + j * scan_angle. A pixel lies at angular distance rho from it, at azimuth
        psi measured from rotated north. This is the standard spherical offset, and it is the
        ONLY place this class differs from the cross-track geometry.
        """
        sub_lon = self._sub_lon[np.asarray(scan, np.int64)]
        psi = (self.azimuth_origin_deg + np.asarray(i, float) * self.azimuth_step_deg) * D2R
        rho = self._radius[np.asarray(scan, np.int64)] * D2R
        rot_lat = np.degrees(np.arcsin(np.clip(np.sin(rho) * np.cos(psi), -1.0, 1.0)))
        rot_lon = sub_lon + np.degrees(np.arctan2(np.sin(psi) * np.sin(rho), np.cos(rho)))
        return rot_lat, rot_lon

    def inverse(self, lat, lon):
        """Geographic lat/lon to conical pixel indices. Not yet available.

        The conical FORWARD map is verified against real files to a few km
        (verification/verify_conical_ssmis.py, verify_conical_amsr2.py). A robust conical
        inverse is future work. The cross-track inverse in SwathGeometry is complete.
        """
        raise NotImplementedError(
            "the conical inverse is not yet implemented. The conical forward map is "
            "available and verified")
