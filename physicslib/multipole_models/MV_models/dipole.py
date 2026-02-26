### Contains dipole in the MV model ###

import numpy as np
from physicslib.constants import LambdaQCD

class Dipole:
    def __init__(self, Qs0, gamma, ec, LambdaQCD=LambdaQCD):
        self.Qs0 = Qs0
        self.gamma = gamma
        self.ec = ec
        self.LambdaQCD = LambdaQCD
        self.eps = 1e-14
    
    def radius(self, x, y):
        """
        Compute dipole size |x - y|.
        Supports scalar or array inputs.
        """
        diff = np.asarray(x) - np.asarray(y)
        if diff.ndim == 0:
            return abs(diff)
        return np.linalg.norm(diff, axis=-1)
    
    def S_r(self, r):
        """
        Radial dipole: S(r) with r = |x - y|
        """
        return np.exp(-((r**2 * self.Qs0**2)**self.gamma) / 4 * np.log(1/(r * self.LambdaQCD) + self.ec * np.exp(1)))


    def S_xy(self, x, y):
        """
        Return the dipole value: D(x,y) = exp(exponent(x,y))
        """
        r = self.radius(x, y) + self.eps
        return np.exp(-((r**2 * self.Qs0**2) ** self.gamma) / 4 * np.log(1 / (r * self.LambdaQCD) + self.ec * np.exp(1)))
    
    def BK_evolved_S_xy(self, x, y, Y):
        """
        Placeholder for BK-evolved S(x,y) at rapidity Y.

        In a full implementation, this would solve the BK equation to evolve
        the dipole from its initial condition at Y=0 to the desired rapidity Y.
        For now, it simply returns the unevolved S(x,y).
        """

        

    def S(self, r):
        """
        Radial dipole S(r).

        Small convenience alias for S_r(r), used by integrands that work
        directly with a radial coordinate (e.g. u, up in polar variables).
        """
        return self.S_r(r)

        