import numpy as np
from scipy.special import jv

# NOTE: This file is part of the physicslib package. If you see stale
# behavior at runtime, delete the corresponding `__pycache__` entry.


class LODISIntegrand4D:
    def __init__(
        self,
        quark_masses,           # quark mass parameter, flavor dependent
        photon_wf,              # LO_FE_PhotonWF_squared instance
        sigma0,                 # parameter for transverse area of target
        dipole_model,           # model for dipole
        quadrupole_model,       # model for quadrupole
        polarization="T",       # Polarization of photon "T", "L", or "TL"
        largeNc=False,          # Set true if using large Nc model
    ):
        self.mf = quark_masses
        self.wf = photon_wf
        self.dipole = dipole_model
        self.quadrupole = quadrupole_model
        self.pol = polarization
        self.largeNc = largeNc
        self.sigma0 = sigma0

    # Define the type of target interaction (i.e. the optical theorem or the finite energy)
    def FE_target_interaction_polar(self, u, up, z, theta):
        S2 = self.dipole.S(u)
        S2prime = self.dipole.S(up)
        S4 = self.quadrupole(u, up, z, theta)
        return 1 - S2 - S2prime + S4

    def relative_momentum_integral(self, u, up, z, theta, Msq_max, flavor):
        """
        Relative momentum integral over k (already performed analytically),
        leaving a 4D integral over (u, up, z, theta).
        """
        mf = self.mf[flavor]

        arg = Msq_max * z * (1 - z) - mf**2
        if arg <= 0:
            return 0.0

        r2 = u**2 + up**2 - 2 * u * up * np.cos(theta)
        if r2 <= 0:
            return 0.0

        zeta = np.sqrt(arg * r2)
        # J1 Bessel kernel for the k-integral
        return zeta * jv(1, zeta) / (2 * np.pi * r2)

    def FE_integrand(self, Q, Msq_max, u, up, z, theta, flavor=0):
        """
        Finite-energy (FE) integrand.

        Expects `self.wf` to be a LO_FE_PhotonWF_squared-like object
        exposing psi_T_squared / psi_L_squared.
        """
        if self.pol in ("T", "TL"):
            psi_T_sq = self.wf.psi_T_squared(Q, u, up, z, theta, flavor)
        else:
            psi_T_sq = 0.0

        if self.pol in ("L", "TL"):
            psi_L_sq = self.wf.psi_L_squared(Q, u, up, z, theta, flavor)
        else:
            psi_L_sq = 0.0

        target_interaction = self.FE_target_interaction_polar(u, up, z, theta)

        I_P = self.relative_momentum_integral(u, up, z, theta, Msq_max, flavor)

        # --- Normalization and Jacobian ---
        NormFactor = 1 / (4 * np.pi)
        Jac = ((u * up) / (z * (1 - z))) * 2 * np.pi

        return (
            self.sigma0/2.0
            * NormFactor
            * Jac
            * (psi_T_sq + psi_L_sq)
            * target_interaction
            * I_P
        )

