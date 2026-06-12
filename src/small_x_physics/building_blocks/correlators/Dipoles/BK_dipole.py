# This file reads a datafile for a BK evolved dipole amplitude evaluated based on some input parameters.
# It has been tested against the ICDipole amplitude at Y = 0.0 and it works

import numpy as np

from scipy.interpolate import RectBivariateSpline

class BKDipole:
    def __init__(self, bkfile, Y= None):
        self.Y = Y
         # Read files
        with open(bkfile) as f:
            lines = f.readlines()

        # Find all ### markers
        markers = [i for i, line in enumerate(lines)
                if line.startswith("###")]

        # Read grid information
        rmin = float(lines[markers[0]][3:])
        rmult = float(lines[markers[1]][3:])
        nr = int(float(lines[markers[2]][3:]))

        # Construct r_grid
        r_grid = rmin * rmult**np.arange(nr)

        # Construct Y grid and N grid
        Y_values = []
        N_rows = []

        for m in markers[4:]:
            Y = float(lines[m][3:])
            Y_values.append(Y)

            values = np.array([float(x) for x in lines[m+1:m+1+nr]])

            N_rows.append(values)

        Y_values = np.array(Y_values)
        N_grid = np.array(N_rows)

        self.spline = RectBivariateSpline(Y_values, r_grid, N_grid, kx=3, ky=3)


    def radius(self, x, y):
        """Compute the dipole size r = |x - y|."""
        return np.linalg.norm(x - y, axis=-1) #np.linalg.norm(np.asarray(x) - np.asarray(y), axis=-1)
    
    def N(self,r, Y):
        """Evaluate the dipole amplitude N(r, Y) using the spline interpolation."""
        return self.spline(Y, r, grid=False)

    def BK_evolved_MV_model_S2(self, x, y, **_kwargs):
        """Evaluate the BK-evolved S2 for given transverse coordinates x and y."""
        if self.Y is None:
            raise ValueError(
                "BKDipole was created without a fixed Y. "
                "Use BK_evolved_MV_model_S2_Y instead."
            )
        return self.BK_evolved_MV_model_S2_Y(x, y, self.Y)
    
    def BK_evolved_MV_model_S2_Y(self, x, y, Y, **_kwargs):
        """Evaluate the BK-evolved S2 for given transverse coordinates x and y at rapidity Y."""
        r = np.clip(self.radius(x, y), 1e-6, 100.0)

        N = self.N(r, np.clip(Y, 0, 16.0))

        return np.maximum(1.0 - N, 1e-14)
    

   