"""swathproj: a reference implementation of the LEO swath grid mapping.

This is the reference code that accompanies the proposal to give netCDF-CF an analytic
geolocation for Low Earth Orbit scanning instruments. It is deliberately small and readable,
because its job is to let a reviewer run the mapping against real data and see the numbers,
not to be a production library.

The mapping decomposes into two stages, and the code keeps them separate on purpose:

1. A rotated-pole transform between the satellite scanning frame and geographic coordinates.
   This is exactly CF's existing ``rotated_latitude_longitude`` grid mapping. The rotated
   pole is derived from the orbit parameters, not fitted.
2. A longitude shear proportional to elapsed time, ``-omega * t``, accounting for Earth
   rotation during the orbit. This is the single element CF does not already have.

``SwathGeometry.forward`` is where that claim lives, and it is a handful of lines. The inverse
is deliberately exhaustive rather than clever: it tests every scan for containment. An earlier
seeded solver was far faster but harder to get right, so the exhaustive form is preferred
here for being obviously correct, and a full verification against a real orbit takes about 13
seconds.

The decomposition is verified in ``verification/rotated_pole_decomposition.py`` and the CF
parameters are checked against PROJ in the test suite. The formulation follows Knapp et al. (VGAC, ESSD preprint
essd-2026-339).

This implementation adopts the spherical, angular-cell model of the published formulation. Its
inverse answers containment (which cell footprint covers a point) and reports points outside
the swath as such. Its one stated limit is that exactness is in the flooring sense: a point
lying mathematically on a cell boundary can fall to either side, because the rotation is
floating-point trigonometry. It differs
from the tutorial's ``vgac_to_earth`` forward by up to about 30 km. Measured against a
production NOAA-20 VGAC orbit's own per-cell coordinates, this spherical model lands within
0.34 km while ``vgac_to_earth`` is 17.8 km out (``verification/verify_real_vgac.py``), so the
spherical angular model is the one consistent with the published data. The difference is a
property of that helper code, which mixes an ellipsoidal path into an otherwise spherical model
and assumes a different nadir index.

Read that 0.34 km as a comparison baseline, not as this model's accuracy. It is what the
along-track term costs when the Earth-rotation rate is left at its nominal 15 degrees per hour.
Fitting one along-track degree of freedom against the same orbit brings the agreement to about
0.3 m, which is the float32 quantum of the stored coordinates
(``verification/verify_vgac_alongtrack.py``). That fitted quantity IS separable from the per-scan
angular step it might be confused with, because the step is applied as a rotated-frame longitude
before the rotated-pole transformation and so moves latitude as well, while the rate is a pure
zonal shear applied after it. Substituting a compensating step at the nominal rate fits worse than
leaving the rate nominal, 804 m against 329 m, so the parameters are not interchangeable here.
"""

from .geometry import SwathGeometry
from .rotated_pole import geographic_to_rotated, rotated_to_geographic

__all__ = ["SwathGeometry", "rotated_to_geographic", "geographic_to_rotated"]
__version__ = "0.1.0"
