import numpy as np
from scipy.integrate import quad


class CFDPIntegrand:
    """
    Convolution of hard coefficient with WW distribution.
    """

    def __init__(
        self,
        WW_distribution,
        Q2,
        mu2,
        alphaS,
        Zf,
    ):
        self.WW = WW_distribution
        self.Q2 = Q2
        self.mu2 = mu2
        self.alphaS = alphaS
        self.Zf = Zf

    # --------------------------------------------------
    # Hard coefficient
    # --------------------------------------------------

    def hard_coeff(self, z):
        """
        Gluon → quark hard coefficient.
        """
        P_gq = z**2 + (1.0 - z)**2
        Cg = (
            P_gq * np.log((1.0 - z) / z)
            + 3.0 * z * (1.0 - z)
            - P_gq
        )
        return (
            self.alphaS / (2.0 * np.pi)
            * (np.log(self.Q2 / self.mu2) * P_gq + Cg)
        )

    # --------------------------------------------------
    # K integrand
    # --------------------------------------------------

    def _K_integrand(self, K, z):
        """
        Integrand for fixed z.
        """
        return self.WW(K)

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def integrate_K(self, z, K_max):
        """
        Perform the K integral for fixed z.
        """
        prefactor = 2.0 * self.Zf**2 * (1.0 / z)
        hard = self.hard_coeff(z)

        val, err = quad(
            self._K_integrand,
            0.0,
            K_max,
            args=(z,),
            limit=200,
        )

        return prefactor * hard * val


