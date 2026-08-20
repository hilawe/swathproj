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
from the tutorial's ``vgac_to_earth`` forward by up to about 30 km. That question is SETTLED,
not open: measured against a production NOAA-20 VGAC orbit's own per-cell coordinates, this
spherical model has a median error of 0.34 km while ``vgac_to_earth`` has 17.8 km
(``verification/verify_real_vgac.py``). The spherical angular model is the one that reproduces
the published data, and the difference is a property of that helper code, which mixes an
ellipsoidal path into an otherwise spherical model and assumes a different nadir index.
"""

from .geometry import SwathGeometry
from .rotated_pole import geographic_to_rotated, rotated_to_geographic

__all__ = ["SwathGeometry", "rotated_to_geographic", "geographic_to_rotated"]
__version__ = "0.1.0"
