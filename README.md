# swathproj

Analytic geolocation for Low Earth Orbit satellite scanning swaths.

By Hilawe Semunegus, NOAA NCEI.

Many polar-orbiting swath products store latitude and longitude for each pixel. These arrays
locate the observations. An analytic description can also express how array position relates to
ground position, using compact grid parameters and per-scan inputs.

`swathproj` implements a spherical model that composes a rotated-pole transform with a longitude
shear proportional to elapsed observation time. The rotation stage corresponds to the network
Common Data Form (netCDF) Climate and Forecast (CF) `rotated_latitude_longitude` grid mapping.
Its pole parameters can be used by software that implements that mapping. The time-dependent
composition requires additional logic and is not yet a standardized CF mapping.

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
| VGAC | NOAA-20 | cross-track, resampled | full mapping | 0.18 m |
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

None of these figures is corroborated by continuous integration. The CI suite is synthetic: it
covers the mathematics and the package install, and installs neither netCDF4 nor any granule. The
table comes from running the `verification/` scripts against real files locally.

The VGAC row is in meters because the stored coordinates for this resampled product are
consistent with the analytic mapping once ONE along-track degree of freedom is fitted. Fitting it
on the first half of the orbit and scoring on the second gives a held-out median of 0.18 m, and
scoring all 8260713 valid pixels gives 0.18 m. The same script measures a storage-spacing scale
on 400000 sampled positions, with seed 23. A positive half-unit-in-the-last-place (ULP)
perturbation in both float32 angular coordinates produces about 0.43 m median displacement.
This is a scale for comparison, not a lower bound on residuals or a geolocation accuracy estimate.

Read the two VGAC numbers as answers to different questions. 0.18 m is the reconstruction error
after fitting that one parameter from the orbit. 0.3326 km is the error with the along-track term
left at the nominal 15 degrees per hour.

Two limits on what that shows, both worth stating plainly.

It does not show how the product was generated. Agreement at storage precision is consistent with
the coordinates having been produced by this mapping and does not establish that they were.

The fitted parameter is separable from the per-scan angular step, which is the other along-track
quantity it could be confused with. Written abstractly as `j*scan_step - rate*t(j)`, with both
terms in one angular variable and a time axis linear in scan index, the two would be degenerate.
This model does not have that form: the scan step is a rotated-frame longitude applied BEFORE the
rotated-pole transformation, so changing it moves a cell along the ground track and alters latitude
as well, while the rate is subtracted from geographic longitude AFTER it, a pure zonal shear.
Latitude separates them. On the reference orbit the fitted rate gives 0.18 m, leaving the rate
nominal gives 329.44 m, and the compensating step the abstract algebra prescribes gives 803.44 m,
worse than doing nothing. See `verification/verify_vgac_alongtrack.py`, which asserts all three.

What the file does not carry is any value for that along-track term. Its stored coordinates cannot
be regenerated from its stated mapping parameters alone, without either fitting against the stored
latitude and longitude or obtaining the parameter externally.

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
