"""The swath geometry: rotated-pole transform composed with an Earth-rotation shear.

``SwathGeometry`` holds the per-orbit parameters and provides the two operations a consumer
needs: cell indices to geographic position (forward), and geographic position to cell indices
(inverse). It also exposes the CF ``rotated_latitude_longitude`` grid-mapping parameters
derived from the orbit parameters, which is what lets the proposal reuse an existing CF
construct for the rotation and add only the time term.

Parameter names follow the published VGAC files, with plain-language aliases:

    ref_time     proj_time0   orbit reference time
    ref_lat      proj_lat0    sub-satellite latitude at the reference (theta_o), degrees
    ref_lon      proj_lon0    sub-satellite longitude at the reference (lambda_o), degrees
    heading      proj_rot     orbit trajectory angle (phi_o), degrees

The cross-track axis (pixel index i) is the rotated latitude and the along-track axis (scan
index j) is the rotated longitude of the satellite scanning frame. The mapping is spherical,
matching the published formulation, and the ellipsoid is not used here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rotated_pole import geographic_to_rotated, rotated_to_geographic

D2R = np.pi / 180.0
EARTH_RADIUS_KM = 6378.0
DEFAULT_CELL_KM = 3.9
DEFAULT_ROTATION_RATE = 15.0  # degrees of longitude per hour


@dataclass
class SwathGeometry:
    ref_lat: float                 # proj_lat0, theta_o
    ref_lon: float                 # proj_lon0, lambda_o
    heading: float                 # proj_rot, phi_o
    n_scan: int                    # along-track cells (N_L)
    n_pixel: int = 801             # cross-track cells (N_P)
    nadir_index: int = 400         # 0-based cross-track index of nadir (see _cell_to_angles)
    cell_size_deg: float = DEFAULT_CELL_KM / EARTH_RADIUS_KM / D2R
    rotation_rate: float = DEFAULT_ROTATION_RATE
    scan_time_hours: np.ndarray | None = None  # hours since ref_time, per scan (required)
    scan_angle_deg: float | None = None        # along-track angular step per scan
    scan_origin_deg: float = 0.0               # rotated longitude of scan index 0
    full_revolution: bool | None = None        # declare that the array closes on itself

    def __post_init__(self):
        if self.scan_time_hours is None:
            raise ValueError(
                "scan_time_hours is required: the Earth-rotation term needs the observation "
                "time of each scan and it cannot be inferred from n_scan. Pass the per-scan "
                "times in hours since the orbit reference, or pass np.zeros(n_scan) to state "
                "explicitly that no Earth-rotation correction is wanted.")
        self.scan_time_hours = np.asarray(self.scan_time_hours, float)
        if self.scan_time_hours.shape != (self.n_scan,):
            raise ValueError(
                f"scan_time_hours must have shape ({self.n_scan},), "
                f"got {self.scan_time_hours.shape}")
        # The along-track axis is METADATA, never inferred from how many scans the array
        # happens to hold. Two quantities describe it: the angular step per scan, and the
        # rotated longitude of scan 0. Deriving the step as 360/n_scan is correct only for a
        # complete revolution, so the caller must SAY which case this is. Silently guessing
        # displaced a cropped orbit's pixels by 19000 km, and a step alone still cannot place
        # an interior crop, which is why the origin exists.
        if self.scan_angle_deg is None:
            if not self.full_revolution:
                raise ValueError(
                    "the along-track axis is under-specified: pass scan_angle_deg (the "
                    "angular step per scan), or full_revolution=True to declare that these "
                    f"{self.n_scan} scans span exactly one revolution. Guessing 360/n_scan "
                    "is only right for a complete orbit and is catastrophically wrong for a "
                    "subset, so it is not done silently.")
            self.scan_angle_deg = 360.0 / self.n_scan
        self._scan_angd = float(self.scan_angle_deg)
        _validate_axis(self)
        # Whether the axis closes on itself decides if scan indices may wrap. An explicit
        # declaration wins, and otherwise it is inferred from the span, which is exact metadata.
        # Closure must be EXACT to within floating-point noise, not a geographic tolerance.
        # A 1e-6 relative window accepted a 359.99982 degree axis as closed, leaving a 20 m
        # gap at the equator that the wrap then silently papered over.
        spans = abs(self.n_scan * self._scan_angd - 360.0) <= 8.0 * np.spacing(360.0)
        self._closes = spans if self.full_revolution is None else bool(self.full_revolution)
        if self._closes and not spans:
            raise ValueError(
                f"full_revolution=True but {self.n_scan} scans of {self._scan_angd} deg span "
                f"{self.n_scan * self._scan_angd} deg, not 360")
        _validate_cross_track_extent(self)
        self._M = _forward_rotation(self.ref_lat, self.heading, self.ref_lon)

    # -- cell axes <-> satellite-frame angles -------------------------------------------
    def _cell_to_angles(self, i, j):
        """Cell indices to satellite-frame angles, both axes centre-based.

        Cell i has its CENTRE at rotated latitude (i - nadir_index) * cell_size_deg, with
        nadir_index a ZERO-BASED array index. This convention is not a choice: it is what the
        published VGAC files themselves use, established by fitting the offset against a real
        orbit's stored latitude and longitude (verification/verify_real_vgac.py). The residual
        minimises sharply at i - 400 (0.34 km, which is the sub-cell scatter expected because
        the stored coordinates are pixel means) and degrades linearly at 0.97 km per quarter
        cell either side.

        Two nearby conventions are wrong by whole fractions of a cell and are worth naming so
        they are not reintroduced. An earlier version of this code used (i - 401 + 0.5), which
        is half a cell out (1.95 km). The paper's Sect. 3.2 formula, alpha = (i - 401) * beta,
        is a full cell out (3.89 km) when i is read as a zero-based array index, and is
        consistent only if i is one-based.
        """
        rot_lat = (np.asarray(i, float) - self.nadir_index) * self.cell_size_deg
        rot_lon = self.scan_origin_deg + np.asarray(j, float) * self._scan_angd
        return rot_lat, rot_lon

    # -- forward: cell -> geographic ----------------------------------------------------
    def forward(self, i, j):
        """Cell indices (i cross-track, j along-track) to geographic lat/lon in degrees.

        j must be an INTEGER scan index. A scan is a discrete observation with its own
        observation time, so a fractional j has no physical referent: interpolating the time
        between two scans places a point on a surface no cell owns, and inverse() then
        correctly reports it off-swath. An earlier version accepted fractional j and
        interpolated, which produced exactly that confusing pair of behaviours, so it is now
        refused. i remains any real value, since it is simply a rotated latitude.

        j must be in [0, n_scan). Outside that the Earth-rotation term has no observation
        time to use, so both latitude and longitude come back NaN rather than a half-valid
        position a caller might use by accident. Cell FOOTPRINTS still extend half a cell
        either side of a scan centre, but that is a property of the cell, not of the index.
        """
        j_arr = np.asarray(j)
        if not np.all(np.equal(np.mod(np.asarray(j_arr, float), 1.0), 0.0)):
            raise ValueError(
                "j must be an integer scan index. A scan carries its own observation time, so "
                "a fractional j has no cell to belong to: the interpolated point lies on a "
                "surface no observation owns and inverse() reports it off-swath.")
        rot_lat, rot_lon = self._cell_to_angles(i, j)
        pole_lat, pole_lon, npgl = self.cf_rotated_pole()
        lat, lon = rotated_to_geographic(rot_lat, rot_lon, pole_lat, pole_lon, npgl)
        lon = lon - self.rotation_rate * self._scan_time(j)  # ref_lon is inside the rotation
        lat = np.where(self._scan_in_range(j), lat, np.nan)
        return lat, (lon + 180.0) % 360.0 - 180.0

    # -- inverse: geographic -> cell ----------------------------------------------------
    def inverse(self, lat, lon):
        """Geographic lat/lon to the cell indices (i, j) whose footprint contains the point.

        The contract is CONTAINMENT. A cell is a rectangle in the satellite scanning frame,
        so "which cell covers this point" has an exact answer: fix a scan, which fixes that
        scan's Earth-rotation correction, rotate into the frame, and floor. A point outside
        the swath returns (-1, -1). Where the orbit laps itself two scans genuinely contain
        the point. This returns the smaller scan index, and inverse_all() returns both.

        This is deliberately EXHAUSTIVE: it tests every scan, so it is O(n_scan) per point.
        That is the whole design. An earlier seeded fixed-point solver was about 250 times
        faster but harder to get right, so the exhaustive form is preferred here for being
        obviously correct: a full verification against a real 10313-scan orbit takes about 13
        seconds. A performance-sensitive reader should implement a solver and check it against
        this.
        """
        i, j, _, _ = self._containing(lat, lon)
        return i, j

    def inverse_all(self, lat, lon):
        """As inverse(), but returns both containing cells: (i, j, i2, j2).

        Where the orbit laps itself a point is imaged twice, and the two cells differ in the
        cross-track index as well as the scan, because the Earth-rotation correction moves
        rotated latitude too. The second cell is therefore returned in full and cannot be
        reconstructed as (i, j2). Both are -1 where there is only one containing cell.

        SCOPE LIMIT, so the name does not overpromise: at most TWO cells are reported. For the
        near-circular orbits this targets, a ground point is imaged at most twice per
        revolution, so two is the physical maximum. A contrived scan-time array can make more
        than two cells coincide (for example timing that exactly cancels each scan's
        along-track motion), and only the two smallest scans are returned in that case.
        """
        return self._containing(lat, lon)

    def _containing(self, lat, lon):
        """Every scan is tested for containment, and the two smallest hits are kept."""
        lat_in = np.asarray(lat, float)
        lon_in = np.asarray(lon, float)
        scalar = lat_in.ndim == 0 and lon_in.ndim == 0
        shape = np.broadcast(lat_in, lon_in).shape
        lat, lon = (np.ravel(a) for a in np.broadcast_arrays(lat_in, lon_in))
        pole_lat, pole_lon, npgl = self.cf_rotated_pole()

        first_i = np.full(lat.shape, -1, dtype=np.int64)
        first_j = np.full(lat.shape, -1, dtype=np.int64)
        second_i = np.full(lat.shape, -1, dtype=np.int64)
        second_j = np.full(lat.shape, -1, dtype=np.int64)

        half = 0.5 * self._scan_angd
        for scan in range(self.n_scan):
            # this scan's own observation time fixes the Earth-rotation correction
            dlon = self.rotation_rate * float(self.scan_time_hours[scan])
            rot_lat, rot_lon = geographic_to_rotated(lat, lon + dlon, pole_lat, pole_lon, npgl)
            # along-track: is the point inside THIS scan's half-open cell?
            delta = (rot_lon - self.scan_origin_deg - scan * self._scan_angd + half) % 360.0 - half
            on_scan = (delta >= -half) & (delta < half)
            # cross-track: which cell, and is it on the swath?
            cross = rot_lat / self.cell_size_deg + self.nadir_index
            finite = np.isfinite(cross) & np.isfinite(delta)
            i = np.floor(np.where(finite, cross, 0.0) + 0.5).astype(np.int64)
            hit = (finite & on_scan
                   & (cross >= -0.5) & (cross < self.n_pixel - 0.5))
            take_first = hit & (first_j < 0)
            first_i = np.where(take_first, i, first_i)
            first_j = np.where(take_first, scan, first_j)
            take_second = hit & ~take_first & (second_j < 0) & (first_j != scan)
            second_i = np.where(take_second, i, second_i)
            second_j = np.where(take_second, scan, second_j)

        out = tuple(a.reshape(shape) for a in (first_i, first_j, second_i, second_j))
        if scalar:
            out = tuple(a.reshape(())[()] for a in out)
        return out

    def _scan_in_range(self, j):
        """Scans are the integers [0, n_scan). Cell footprints still extend half a cell either
        side of a scan centre, but that is a property of the CELL, not of the scan index."""
        j = np.asarray(j, float)
        return (j >= 0) & (j < self.n_scan)

    def _scan_time(self, j):
        """Observation time (hours) for integer scan j, NaN outside [0, n_scan).

        No interpolation: forward() requires an integer scan, so there is nothing between two
        scans to interpolate. An earlier version interpolated and extrapolated here to serve
        fractional j, which existed only to place points no cell owns.
        """
        j = np.asarray(j)
        in_range = self._scan_in_range(j)
        safe = np.where(in_range, j, 0).astype(np.int64)
        return np.where(in_range, self.scan_time_hours[safe], np.nan)

    # -- CF grid-mapping parameters -----------------------------------------------------
    def cf_rotated_pole(self):
        """(grid_north_pole_latitude, grid_north_pole_longitude, north_pole_grid_longitude).

        Derived from the orbit parameters by a ZYZ decomposition of the rotation matrix, so
        the rotation stage is expressed exactly as CF's rotated_latitude_longitude mapping.
        """
        return _zyz_to_cf_pole(self._M)


def _validate_cross_track_extent(geo):
    """Cross-track cells are rotated LATITUDES, so the swath must stay inside [-90, 90].

    Specific to the cross-track and push-broom geometries. A conical instrument's pixel axis
    counts azimuth samples around a cone, not cross-track cells, so this check does not apply
    to it and applying it anyway rejects perfectly good files. AMSR2, with 243 samples per
    rotation, is one such file.
    """
    lo = (-0.5 - geo.nadir_index) * geo.cell_size_deg
    hi = (geo.n_pixel - 0.5 - geo.nadir_index) * geo.cell_size_deg
    if lo < -90.0 or hi > 90.0:
        raise ValueError(
            f"the cross-track extent spans rotated latitude {lo:.3f} to {hi:.3f} degrees, "
            "which leaves [-90, 90]. Beyond a pole the cells fold back and alias, so forward "
            "and inverse stop being inverses of each other.")


def _validate_axis(geo):
    """Reject a geometry that would construct successfully but map nonsensically.

    Each check here corresponds to a way a caller could previously build an object whose
    forward map was silently degenerate: a zero step collapsing the along-track axis, a
    non-finite parameter making every result NaN, a cross-track extent folding across a pole
    and aliasing cells, or a non-integer count producing floating-point array indices.
    """
    if int(geo.n_scan) != geo.n_scan or geo.n_scan < 1:
        raise ValueError(f"n_scan must be a positive integer, got {geo.n_scan!r}")
    if int(geo.n_pixel) != geo.n_pixel or geo.n_pixel < 1:
        raise ValueError(f"n_pixel must be a positive integer, got {geo.n_pixel!r}")
    geo.n_scan, geo.n_pixel = int(geo.n_scan), int(geo.n_pixel)

    if not np.isfinite(geo._scan_angd) or geo._scan_angd <= 0.0:
        raise ValueError(
            f"scan_angle_deg must be finite and positive, got {geo.scan_angle_deg!r}. A zero "
            "or non-finite step collapses the whole along-track axis onto one point.")
    if not np.isfinite(geo.cell_size_deg) or geo.cell_size_deg <= 0.0:
        raise ValueError(
            f"cell_size_deg must be finite and positive, got {geo.cell_size_deg!r}")
    if not np.isfinite(geo.nadir_index) or int(geo.nadir_index) != geo.nadir_index:
        raise ValueError(
            f"nadir_index must be a finite integer, got {geo.nadir_index!r}. A non-finite "
            "value degenerates every operation while still constructing.")
    geo.nadir_index = int(geo.nadir_index)

    for name in ("ref_lat", "ref_lon", "heading", "rotation_rate", "scan_origin_deg"):
        value = float(getattr(geo, name))
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")

    if not np.all(np.isfinite(geo.scan_time_hours)):
        bad = int(np.count_nonzero(~np.isfinite(geo.scan_time_hours)))
        raise ValueError(
            f"scan_time_hours contains {bad} non-finite value(s). Every scan's own time places "
            "its cells, so a missing one silently loses that whole scan. A caller must fill or "
            "drop them deliberately rather than have them propagate.")

    if geo.n_scan * geo._scan_angd > 360.0 + 1e-9 and not bool(geo.full_revolution):
        raise ValueError(
            f"{geo.n_scan} scans of {geo._scan_angd} deg span "
            f"{geo.n_scan * geo._scan_angd} deg, which is more than one revolution. Either "
            "the step is wrong or this is a full orbit. A partial axis must span at most 360 "
            "degrees so its scans are unambiguous.")


def _forward_rotation(ref_lat, heading, ref_lon):
    """Rotated-frame vector to geographic vector.

    M = Rz(ref_lon) @ Ry(-ref_lat) @ Rx(-heading). The reference longitude is part of the
    ROTATION, not a separate shift applied afterwards. Folding it in is what makes the
    exposed CF parameters sufficient on their own: a consumer that reads
    grid_north_pole_latitude, grid_north_pole_longitude and north_pole_grid_longitude and
    applies only the Earth-rotation term reproduces the mapping exactly. When ref_lon was
    applied outside the rotation, such a consumer was wrong by exactly ref_lon.
    """
    a = -heading * D2R
    b = -ref_lat * D2R
    c = ref_lon * D2R
    rx = np.array([[1.0, 0.0, 0.0],
                   [0.0, np.cos(a), -np.sin(a)],
                   [0.0, np.sin(a), np.cos(a)]])
    ry = np.array([[np.cos(b), 0.0, np.sin(b)],
                   [0.0, 1.0, 0.0],
                   [-np.sin(b), 0.0, np.cos(b)]])
    rz = np.array([[np.cos(c), -np.sin(c), 0.0],
                   [np.sin(c), np.cos(c), 0.0],
                   [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def _zyz_to_cf_pole(m):
    """Decompose the rotation matrix M into CF rotated_latitude_longitude parameters.

    CF's convention is M = Rz(pole_lon) Ry(90-pole_lat) Rz(180 - npgl), so this extracts the
    ZYZ Euler angles (pole_lon, 90-pole_lat, gamma) and returns npgl = 180 - gamma. The two
    gimbal-lock cases are handled separately: at the north-pole singularity (beta ~ 0,
    m[2,2] ~ +1) only pole_lon + gamma is determined, and at the south-pole singularity
    (beta ~ pi, m[2,2] ~ -1) only pole_lon - gamma is determined. Splitting the two is
    necessary because a single formula gives an order-one error at beta ~ pi.
    """
    sin_beta = np.hypot(m[0, 2], m[1, 2])
    beta = np.arctan2(sin_beta, m[2, 2])
    pole_lat = 90.0 - np.degrees(beta)
    if sin_beta < 1e-9:
        # Gimbal lock: fix pole_lon = 0 and put the whole rotation into gamma.
        pole_lon = 0.0
        if m[2, 2] > 0.0:      # beta ~ 0, M ~ Rz(pole_lon + gamma)
            gamma = np.arctan2(m[1, 0], m[0, 0])
        else:                  # beta ~ pi, M ~ Rz(pole_lon - gamma) @ diag(1,-1,-1)
            gamma = np.arctan2(m[1, 0], -m[0, 0])
    else:
        pole_lon = np.arctan2(m[1, 2], m[0, 2])
        gamma = np.arctan2(m[2, 1], -m[2, 0])
    npgl = 180.0 - np.degrees(gamma)
    npgl = (npgl + 180.0) % 360.0 - 180.0
    return pole_lat, np.degrees(pole_lon), npgl
