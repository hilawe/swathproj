"""Stage 1: the rotated-pole transform, CF's rotated_latitude_longitude grid mapping.

This module contains only the rotation. It is the part of the swath mapping that CF already
standardizes, kept separate so the proposal can say precisely what is reused and what is new.

Conventions follow CF's ``rotated_latitude_longitude`` grid mapping. A rotated coordinate
system is defined by the geographic position of its north pole
(``grid_north_pole_latitude``, ``grid_north_pole_longitude``) and, optionally, a
``north_pole_grid_longitude`` fixing the rotation about that pole. A point is described by its
rotated latitude and rotated longitude, and the transform returns its true geographic
latitude and longitude.

The maths is a rotation of the unit vector on the sphere. It carries no ellipsoid and no
Earth radius, because a rotation is scale-free, and the sphere versus ellipsoid choice enters
only when angles are turned into ground positions, handled elsewhere.
"""

from __future__ import annotations

import numpy as np

D2R = np.pi / 180.0


def _unit_vector(lat_deg, lon_deg):
    # Broadcast first so a scalar latitude pairs with a vector longitude (and vice versa).
    lat, lon = np.broadcast_arrays(np.asarray(lat_deg, float) * D2R,
                                   np.asarray(lon_deg, float) * D2R)
    cos_lat = np.cos(lat)
    return np.stack([cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat)], axis=0)


def _lat_lon(vec):
    x, y, z = vec[0], vec[1], vec[2]
    return np.degrees(np.arctan2(z, np.hypot(x, y))), np.degrees(np.arctan2(y, x))


def _pole_rotation(pole_lat, pole_lon, north_pole_grid_lon):
    """Rotation matrix taking rotated-frame vectors to geographic vectors.

    Built as Rz(pole_lon) @ Ry(90 - pole_lat) @ Rz(180 - north_pole_grid_lon). This matches
    the CF rotated_latitude_longitude grid mapping exactly, where north_pole_grid_longitude
    is the longitude of the true north pole in the rotated frame rather than a free spin
    angle, so the third rotation is 180 - npgl, not npgl. The convention and the 180 - npgl
    term are verified against PROJ, an independent CF implementation, in the test suite.
    """
    a = (90.0 - pole_lat) * D2R
    b = pole_lon * D2R
    c = (180.0 - north_pole_grid_lon) * D2R
    ca, sa = np.cos(a), np.sin(a)
    cb, sb = np.cos(b), np.sin(b)
    cc, sc = np.cos(c), np.sin(c)
    ry = np.array([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]])
    rz_b = np.array([[cb, -sb, 0.0], [sb, cb, 0.0], [0.0, 0.0, 1.0]])
    rz_c = np.array([[cc, -sc, 0.0], [sc, cc, 0.0], [0.0, 0.0, 1.0]])
    return rz_b @ ry @ rz_c


def rotated_to_geographic(rot_lat, rot_lon, pole_lat, pole_lon, north_pole_grid_lon=0.0):
    """Rotated latitude/longitude to true geographic latitude/longitude (degrees)."""
    rot = _pole_rotation(pole_lat, pole_lon, north_pole_grid_lon)
    vec = _unit_vector(rot_lat, rot_lon)
    geo = np.einsum("ij,j...->i...", rot, vec)
    lat, lon = _lat_lon(geo)
    return lat, (lon + 180.0) % 360.0 - 180.0


def geographic_to_rotated(lat, lon, pole_lat, pole_lon, north_pole_grid_lon=0.0):
    """True geographic latitude/longitude to rotated latitude/longitude (degrees)."""
    rot = _pole_rotation(pole_lat, pole_lon, north_pole_grid_lon)
    vec = _unit_vector(lat, lon)
    rframe = np.einsum("ji,j...->i...", rot, vec)  # transpose of the forward rotation
    rlat, rlon = _lat_lon(rframe)
    return rlat, (rlon + 180.0) % 360.0 - 180.0
