# swathproj

Analytic geolocation for Low Earth Orbit satellite scanning swaths.

By Hilawe Semunegus, NOAA NCEI.

Most polar-orbiting swath products record a latitude and longitude for every pixel. That says
where each observation is, but not the relationship between array position and ground position,
so software cannot ask which pixel is nearest a location, subset by geography, or check the stored
coordinates against the geometry that produced them without reading the whole coordinate array.

`swathproj` provides that relationship as a small analytic model. The mapping is a rotated-pole
transform, whose equator follows the satellite ground track, composed with a longitude shear
proportional to elapsed observation time for Earth rotation during the orbit. The rotation
stage is exactly the netCDF Climate and Forecast `rotated_latitude_longitude` grid mapping, so
its pole parameters can be emitted directly and read by any CF-aware tool.

## Scan geometries

| geometry | offset from the sub-satellite point | status |
|---|---|---|
| cross-track (whiskbroom or push-broom) | along a rotated meridian | forward and inverse |
| conical (spinning) | around a small circle of fixed angular radius | forward |

A push-broom shares the cross-track offset, because its detectors sit at the same fixed
cross-track look angles. The cross-track offset comes in two kinds, a uniform ground-angle grid
(a resampled product) and a slant-range spacing (a native swath, whose ground spacing grows
toward the swath edge).

## Verification

The `verification/` scripts fit the model to a published file's own stored coordinates and report
the residual. Each downloads nothing, and points at a local file (paths in each script's
docstring). Measured against real products:

| instrument | platform | geometry | test | median residual |
|---|---|---|---|---|
| VGAC | NOAA-20 | cross-track, resampled | full mapping | 0.3 m |
| AVHRR GAC L1C | Metop-C | cross-track, native | radial profile | 3.7 km |
| VIIRS SDR | NOAA-20 | cross-track, native | radial profile, file zenith | 0.3 km |
| ATMS FCDR L1C | Suomi-NPP | cross-track, native | radial profile | 7.2 km |
| SSM/I | DMSP F13 | conical | full mapping | 2.1 km |
| SSMIS | DMSP F17 | conical | full mapping | 2.3 km |
| AMSR2 | GCOM-W1 | conical | full mapping | 3.1 km |

Read the rows at the level each was tested, because they are not equivalent.

The full-mapping rows compare the complete transformation against the file's stored coordinates.
The native cross-track rows are one-dimensional radial-profile checks of the slant-range offset,
not full two-dimensional reproductions, so they are reported at that level.

The VIIRS row takes the instrument zenith angle from the file, so it is a consistency check rather
than a reproduction. Predicting VIIRS geolocation from scan index alone gives about 73 km, because
onboard aggregation makes the sampling non-uniform across the scan, and that figure is not claimed
as a result. Every other row is a reproduction.

The VGAC row is in metres because the stored coordinates for this resampled product are
consistent with the analytic mapping once one orbit-specific rotation rate is fitted. Fitting that
rate on the first half of the orbit and scoring on the second gives a held-out median of 0.4 m, and
scoring all 8260713 valid pixels gives 0.3 m, against a float32 storage quantum of 0.4 m.

Read the two VGAC numbers as answers to different questions. 0.3 m is the reconstruction error
after fitting one scalar from the orbit itself. 0.3326 km is the error using the nominal 15.0
degrees per hour, where this orbit wants 14.994075 and the difference accumulates 1.1 km of
longitude over the pass.

What this does not show is how the product was generated. Agreement at storage precision is
consistent with the coordinates having been produced by this mapping, and it does not establish
that they were, nor that 14.994075 is a parameter the producer holds explicitly rather than a value
the fit uses to absorb longitudinal drift. Confirming that requires the production algorithm. What
is certain either way is that the file does not store the rate, so its coordinates cannot be
reproduced from the file alone. See `verification/verify_vgac_rotation_rate.py`.

## Install

```
pip install -e .
```

`numpy` is the only runtime dependency. The verification scripts also need `pyproj` and
`netCDF4`.

## Example

```python
import numpy as np
from swathproj import SwathGeometry

geo = SwathGeometry(
    ref_lat=0.0, ref_lon=-75.0, heading=261.28, n_scan=10313,
    scan_time_hours=np.linspace(0, 101 / 60, 10313), full_revolution=True,
)
lat, lon = geo.forward(400, 5000)     # pixel (cross-track, along-track) to Earth
i, j = geo.inverse(lat, lon)          # Earth back to the containing pixel
```

## Contributors

Ken Knapp (Knapp WeatherSat Services LLC) is a key contributor. The rotated-pole and
Earth-rotation core of this implementation is the projection he developed for the VGAC dataset
(Knapp et al., 2024, doi:10.25921/gsef-pg81), which this work generalizes to the three scan
geometries.

## License

Dedicated to the public domain under CC0 1.0. No rights reserved.

## Status

Early release of a reference implementation. The API may change.
