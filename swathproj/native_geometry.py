"""Native (unresampled) cross-track whiskbroom geometry.

``SwathGeometry`` models the EQUAL-ANGLE cross-track case: rotated latitude is uniform in the
sample index, rot_lat(i) = (i - nadir) * cell_size. That is exactly right for a resampled
product such as VGAC, whose grid is DEFINED as a uniform division of ground angle. It is wrong
for a NATIVE whiskbroom, which steps at a uniform instrument LOOK ANGLE (the mirror angle from
nadir), whose ground central angle grows away from nadir by the slant-range geometry.

For a satellite at altitude ``h`` above a spherical Earth of radius ``R``, a look angle ``theta``
(from nadir, measured at the satellite) reaches the ground at Earth central angle

    gamma(theta) = arcsin( ((R + h) / R) * sin(theta) ) - theta

measured from the sub-satellite point. This is the rotated latitude of the pixel, because the
cross-track offset runs along the rotated meridian and moving along a meridian by an angle
changes rotated latitude by that angle. ``gamma`` is odd in ``theta``, so one signed expression
covers both sides of nadir. Near nadir d(gamma)/d(theta) = h / R, and it grows toward the swath
edge, so uniform look-angle steps give NON-uniform rotated-latitude steps. On a Low Earth Orbit
whiskbroom the edge step is several times the nadir step, which is the effect a resampled grid
hides and the equal-angle formula cannot reproduce.

The inverse is closed form. Writing k = (R + h) / R and gamma for the rotated latitude,

    theta(gamma) = arctan2( sin(gamma), k - cos(gamma) )

which follows from sin(gamma + theta) = k * sin(theta).

Altitude varies within a pass (about 37 km on the SSMIS granule measured in the conical work),
and at the swath edge d(gamma)/d(h) is large enough that a constant altitude misplaces edge
pixels by tens of kilometres, so ``altitude_km`` may be one value per scan, exactly as
``cone_radius_deg`` is per scan for the conical geometry.

The rotated pole, the Earth-rotation shear, and the uniform along-track handling follow
``SwathGeometry``. This class keeps its own limb and altitude validation and does not provide the
self-overlap ``inverse_all``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import (
    D2R,
    EARTH_RADIUS_KM,
    _forward_rotation,
    _validate_axis,
    _zyz_to_cf_pole,
)
from .rotated_pole import geographic_to_rotated, rotated_to_geographic


def look_angle_to_central_angle(look_deg, altitude_km, earth_radius_km=EARTH_RADIUS_KM):
    """Instrument look angle (from nadir) to Earth central angle from the sub-satellite point.

    gamma(theta) = arcsin(k sin theta) - theta, k = (R + h) / R. Odd in theta, so the sign of the
    look angle carries through to which side of nadir the pixel lies. Only the physical branch is
    returned. A ray reaches the Earth only for |theta| <= arcsin(1/k), the limb look angle. Beyond
    that the ray misses the sphere and the result is NaN. Angles past the limb whose sine happens
    to satisfy k sin theta <= 1 again (for example 130 degrees) are non-physical and also NaN,
    which a bare arcsin would silently accept.
    """
    theta = np.asarray(look_deg, float) * D2R
    k = (earth_radius_km + np.asarray(altitude_km, float)) / earth_radius_km
    theta_limb = np.arcsin(np.clip(1.0 / k, -1.0, 1.0))
    gamma = np.arcsin(np.clip(k * np.sin(theta), -1.0, 1.0)) - theta
    return np.degrees(np.where(np.abs(theta) <= theta_limb, gamma, np.nan))


def central_angle_to_look_angle(central_deg, altitude_km, earth_radius_km=EARTH_RADIUS_KM):
    """Inverse of ``look_angle_to_central_angle``, closed form, physical branch only.

    theta = arctan2(sin gamma, k - cos gamma). arctan2 keeps the sign of gamma, so a negative
    rotated latitude returns a negative look angle. A central angle is reachable only for
    |gamma| <= arccos(1/k), the value at the limb. Beyond that no look angle produces it and the
    result is NaN, rather than a spurious index-valued answer.
    """
    gamma = np.asarray(central_deg, float) * D2R
    k = (earth_radius_km + np.asarray(altitude_km, float)) / earth_radius_km
    gamma_max = np.arccos(np.clip(1.0 / k, -1.0, 1.0))
    theta = np.arctan2(np.sin(gamma), k - np.cos(gamma))
    return np.degrees(np.where(np.abs(gamma) <= gamma_max, theta, np.nan))


@dataclass
class NativeCrossTrackGeometry:
    """A native cross-track (whiskbroom) instrument on a near-circular orbit.

    Orbit parameters (shared with SwathGeometry, same meanings):
        ref_lat, ref_lon    sub-satellite position at the reference time, degrees
        heading             orbit trajectory angle, degrees
        n_scan              number of scans in the file
        scan_angle_deg      along-track angular advance per scan, degrees
        scan_origin_deg     rotated longitude of scan 0
        rotation_rate       Earth rotation, degrees of longitude per hour
        scan_time_hours     observation time of each scan, hours since the reference

    Native cross-track parameters:
        n_pixel             samples across the track
        look_angle_step_deg instrument look-angle advance per sample (from nadir)
        nadir_index         0-based sample index of nadir (look angle zero)
        altitude_km         spacecraft altitude, scalar or one value per scan
        earth_radius_km     spherical Earth radius used by the slant-range projection
    """

    ref_lat: float
    ref_lon: float
    heading: float
    n_scan: int
    n_pixel: int
    scan_angle_deg: float
    look_angle_step_deg: float
    nadir_index: int
    altitude_km: float | np.ndarray
    scan_time_hours: np.ndarray | None = None
    scan_origin_deg: float = 0.0
    rotation_rate: float = 15.0
    earth_radius_km: float = EARTH_RADIUS_KM
    # present so the SHARED axis validation runs unchanged. The native cross-track offset does
    # not use a fixed cell size, and the [-90, 90] cross-track extent check is applied directly
    # against the computed rotated latitudes below rather than through cell_size_deg.
    cell_size_deg: float = 1.0
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

        if not np.isfinite(self.look_angle_step_deg) or self.look_angle_step_deg <= 0.0:
            raise ValueError(
                f"look_angle_step_deg must be finite and positive, got "
                f"{self.look_angle_step_deg!r}. A zero step collapses the cross-track axis.")
        if not np.isfinite(self.earth_radius_km) or self.earth_radius_km <= 0.0:
            raise ValueError(
                f"earth_radius_km must be finite and positive, got {self.earth_radius_km!r}")

        altitude = np.asarray(self.altitude_km, float)
        if altitude.ndim == 0:
            altitude = np.full(self.n_scan, float(altitude))
        if altitude.shape != (self.n_scan,):
            raise ValueError(
                f"altitude_km must be a scalar or have shape ({self.n_scan},), "
                f"got {altitude.shape}")
        if not np.all(np.isfinite(altitude)) or np.any(altitude <= 0.0):
            raise ValueError("altitude_km must be finite and positive at every scan")
        self._altitude = altitude

        self._sub_lon = (self.scan_origin_deg
                         + np.arange(self.n_scan, dtype=float) * self._scan_angd)

        # A declared full revolution must close to exactly 360 degrees, so a wrong step cannot be
        # waved through by the flag, matching SwathGeometry.
        if self.full_revolution:
            spans = abs(self.n_scan * self._scan_angd - 360.0) <= 8.0 * np.spacing(360.0)
            if not spans:
                raise ValueError(
                    f"full_revolution=True but {self.n_scan} scans of {self._scan_angd} deg span "
                    f"{self.n_scan * self._scan_angd} deg, not 360")

        # The FOOTPRINT boundaries, half a sample beyond the outer pixel centers, must stay inside
        # the Earth limb at every scan's altitude and inside [-90, 90] of rotated latitude, or
        # forward and inverse stop being inverses. Checking the centers alone lets a geometry
        # construct whose outermost cell reaches past the limb.
        outer = max(self.nadir_index + 0.5, self.n_pixel - 0.5 - self.nadir_index)
        edge_look = self.look_angle_step_deg * outer
        gamma_edge = look_angle_to_central_angle(edge_look, self._altitude.max(),
                                                 self.earth_radius_km)
        if not np.isfinite(gamma_edge):
            raise ValueError(
                f"the outer footprint boundary at look angle {edge_look:.3f} deg misses the "
                f"Earth at altitude {self._altitude.max():.1f} km, passing beyond the limb. "
                "Reduce look_angle_step_deg, n_pixel, or nadir_index.")
        if abs(gamma_edge) >= 90.0:
            raise ValueError(
                f"the outer footprint boundary reaches rotated latitude {gamma_edge:.3f} deg, "
                "outside [-90, 90], where cross-track cells fold across a pole and alias.")

        self._M = _forward_rotation(self.ref_lat, self.heading, self.ref_lon)

    def cf_rotated_pole(self):
        """The CF rotated_latitude_longitude parameters, derived exactly as for the
        equal-angle case. This is the shared half of the mapping."""
        return _zyz_to_cf_pole(self._M)

    def _cross_track_rot_lat(self, i, scan):
        """Rotated latitude of sample i on the given scan, by the slant-range projection.

        rot_lat = gamma(look angle), look angle = (i - nadir) * look_angle_step, at that scan's
        altitude. This cross-track offset is the substantive difference from the equal-angle case.
        """
        look = (np.asarray(i, float) - self.nadir_index) * self.look_angle_step_deg
        return look_angle_to_central_angle(look, self._altitude[np.asarray(scan, np.int64)],
                                           self.earth_radius_km)

    def forward(self, i, j):
        """Sample/scan indices (i cross-track, j along-track) to geographic lat/lon, degrees.

        j must be an integer scan index, for the same reason as the equal-angle case: a scan
        carries its own observation time, so a fractional scan has no cell to belong to.
        """
        j_arr = np.asarray(j)
        if not np.all(np.equal(np.mod(np.asarray(j_arr, float), 1.0), 0.0)):
            raise ValueError("j must be an integer scan index")
        in_range = (j_arr >= 0) & (j_arr < self.n_scan)
        safe = np.where(in_range, j_arr, 0).astype(np.int64)

        rot_lat = self._cross_track_rot_lat(i, safe)
        rot_lon = self._sub_lon[safe]
        pole_lat, pole_lon, npgl = self.cf_rotated_pole()
        lat, lon = rotated_to_geographic(rot_lat, rot_lon, pole_lat, pole_lon, npgl)
        lon = lon - self.rotation_rate * self.scan_time_hours[safe]
        lat = np.where(in_range, lat, np.nan)
        lon = np.where(in_range, lon, np.nan)
        return lat, (lon + 180.0) % 360.0 - 180.0

    def inverse(self, lat, lon):
        """Geographic lat/lon to the sample/scan (i, j) whose footprint contains the point.

        Exhaustive over scans, as in the equal-angle case: each scan's own observation time
        fixes its Earth-rotation correction, so a scan is tested exactly with no iteration. The
        only change is the cross-track step: the point's rotated latitude is turned back into a
        look angle by the closed-form inverse, then into a sample index.
        """
        lat_in = np.asarray(lat, float)
        lon_in = np.asarray(lon, float)
        scalar = lat_in.ndim == 0 and lon_in.ndim == 0
        shape = np.broadcast(lat_in, lon_in).shape
        lat, lon = (np.ravel(a) for a in np.broadcast_arrays(lat_in, lon_in))
        pole_lat, pole_lon, npgl = self.cf_rotated_pole()

        first_i = np.full(lat.shape, -1, dtype=np.int64)
        first_j = np.full(lat.shape, -1, dtype=np.int64)
        half = 0.5 * self._scan_angd

        for scan in range(self.n_scan):
            dlon = self.rotation_rate * float(self.scan_time_hours[scan])
            rot_lat, rot_lon = geographic_to_rotated(lat, lon + dlon, pole_lat, pole_lon, npgl)
            delta = (rot_lon - self._sub_lon[scan] + half) % 360.0 - half
            on_scan = (delta >= -half) & (delta < half)
            # cross-track: rotated latitude -> look angle -> sample index
            look = central_angle_to_look_angle(rot_lat, self._altitude[scan], self.earth_radius_km)
            cross = look / self.look_angle_step_deg + self.nadir_index
            finite = np.isfinite(cross) & np.isfinite(delta)
            i = np.floor(np.where(finite, cross, 0.0) + 0.5).astype(np.int64)
            hit = (finite & on_scan
                   & (cross >= -0.5) & (cross < self.n_pixel - 0.5)
                   & (first_j < 0))
            first_i = np.where(hit, i, first_i)
            first_j = np.where(hit, scan, first_j)

        out = tuple(a.reshape(shape) for a in (first_i, first_j))
        if scalar:
            out = tuple(a.reshape(())[()] for a in out)
        return out
