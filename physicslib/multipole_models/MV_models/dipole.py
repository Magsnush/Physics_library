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
    
    def exponent_r(self, r):
        r = np.asarray(r) + self.eps
        return -((r**2 * self.Qs0**2)**self.gamma) / 4 * np.log(
            1/(r * self.LambdaQCD) + self.ec * np.e
        )

    def exponent(self, x, y):
        """
        Compute the exponent of the dipole.
        """
        r = self.radius(x, y) + self.eps
        return -((r**2 * self.Qs0**2) ** self.gamma) / 4 * np.log(1 / (r * self.LambdaQCD) + self.ec * np.exp(1))
    
    def S_r(self, r):
        """
        Radial dipole: S(r) with r = |x - y|
        """
        return np.exp(self.exponent_r(r))


    def S_xy(self, x, y):
        """
        Return the dipole value: D(x,y) = exp(exponent(x,y))
        """
        return np.exp(self.exponent(x, y))

    def S(self, r):
        """
        Radial dipole S(r).

        Small convenience alias for S_r(r), used by integrands that work
        directly with a radial coordinate (e.g. u, up in polar variables).
        """
        return self.S_r(r)

        