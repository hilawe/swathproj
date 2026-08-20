# swathproj

Analytic geolocation for Low Earth Orbit satellite scanning swaths.

By Hilawe Semunegus, NOAA NCEI.

Most polar-orbiting swath products record a latitude and longitude for every pixel. That says
where each observation is, but not the relationship between array position and ground position,
so software cannot ask which pixel covers a location, subset by geography, or check the stored
coordinates against the geometry that produced them without reading the whole coordinate array.

`swathproj` provides that relationship as a small analytic model. The mapping is a rotated-pole
transform, whose equator follows the satellite ground track, composed with a longitude shear
proportional to elapsed observation time for Earth rotation during the orbit. The rotation
stage is exactly the netCDF Climate and Forecast `rotated_latitude_longitude` grid mapping, so
its pole parameters can be emitted directly and read by any CF-aware tool.

## Scan geometries

| geometry | offset from the sub-satellite point | status |
|---|---|---|
| cross-track (whiskbroom) | along a rotated meridian | forward and inverse |
| push-broom | along a rotated meridian | forward and inverse |
| conical (spinning) | around a small circle of fixed angular radius | forward |

## Verification

The `verification/` scripts fit the model to a published file's own stored coordinates and
report the residual. Each downloads nothing, and points at a local file (paths in each script's
docstring). Measured against real products:

| instrument | platform | geometry | median residual |
|---|---|---|---|
| VIIRS GAC | NOAA-20 | cross-track (resampled) | 0.34 km |
| SSMIS | DMSP F17 | conical | 2.3 km |
| AMSR2 | GCOM-W1 | conical | 3.1 km |

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

## License

Dedicated to the public domain under CC0 1.0. No rights reserved.

## Status

Early release of a reference implementation. The API may change.
