### Contains dipole in the MV model ###

import numpy as np
from physicslib.constants import LambdaQCD


class Dipole:
    def __init__(self, Qs0, gamma, ec, LambdaQCD=LambdaQCD):
        self.Qs0 = Qs0
        self.gamma = gamma
        self.ec = ec
        self.LambdaQCD = LambdaQCD
    
    def radius(self, x, y):
        """
        Compute dipole size |x - y|.
        Supports scalar or array inputs.
        """
        diff = np.asarray(x) - np.asarray(y)
        if diff.ndim == 0:
            return abs(diff)
        return np.linalg.norm(diff, axis=-1)
    
    def S_r(self, r, bk=None, Y=None):
        """Return S(r) — analytic or BK-evolved.
        
        Parameters
        ----------
        r : scalar or array
            Dipole separation.
        bk : RCBKData instance, optional
            If provided, returns BK-evolved S(r,Y) from the data.
            If None, returns analytic MV S(r).
        Y : float, optional
            Rapidity for BK evaluation (required if bk is provided).
        
        Returns
        -------
        S(r) : scalar or array
            Either analytic or BK-evolved dipole amplitude.
        """
        eps = 1e-12
        if bk is not None:
            # Return BK-evolved S(Y, r)
            if Y is None:
                raise ValueError("Y rapidity required for BK evaluation")
            return bk.S(Y, r + eps)
        # Return analytic MV S(r)
        return np.exp(-((r**2 * self.Qs0**2)**self.gamma) / 4 * np.log(1/(r * self.LambdaQCD) + self.ec * np.exp(1)))

    def S_xy(self, x, y, bk=None, Y=None):
        """Return S(|x-y|) — analytic or BK-evolved.
        
        Parameters
        ----------
        x, y : array-like or scalar
            Transverse coordinates (can be 2D vectors or scalars).
        bk : RCBKData instance, optional
            If provided, returns BK-evolved S via RCBKData.
        Y : float, optional
            Rapidity for BK evaluation (required if bk is provided).
        
        Returns
        -------
        S : scalar or array
            Either analytic or BK-evolved dipole amplitude at separation |x-y|.
        """
        r = self.radius(x, y)
        return self.S_r(r, bk=bk, Y=Y)

    def S(self, r):
        """
        Radial dipole S(r).
        
        Convenience alias for S_r(r), used by integrands that work
        directly with a radial coordinate.
        """
        return self.S_r(r)




        