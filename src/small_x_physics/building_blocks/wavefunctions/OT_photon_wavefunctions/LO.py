### CONTAINS LEADING ORDER OPTICAL THEOREM PHOTON LIGHTCONE WAVEFUNCTIONS ###

import numpy as np
from scipy.special import kv
from small_x_physics.building_blocks.constants import Nc, alpha_em

class LO_OT_PhotonWF_squared:
    """
    Leading-order finite-energy constrained photon lightcone wavefunction squared.

    Methods
    -------
    psi_T_squared(Q, r, z, flavor)
        Returns squared transverse photon wavefunction.
    psi_L_squared(Q, r, z, flavor)
        Returns squared longitudinal photon wavefunction.

    Attributes
    ----------
    mf : array_like
        Quark masses.
    Zf : array_like
        Quark charges.
    Nc : int
        Number of colors.
    alpha_em : float
        Fine structure constant.
    """


    def __init__(self, Q, quark_masses, quark_charges, Nc = Nc, alpha_em = alpha_em):
        self.Q = Q
        self.mf = quark_masses
        self.Zf = quark_charges
        self.Nc = Nc
        self.alpha_em = alpha_em


    def psi_T_squared(self, Q, r, z): 

        mf = self.mf
        zf = self.Zf

        epsilon_sq = Q**2 * z * (1 - z) + mf**2
        epsilon_sq = np.maximum(epsilon_sq, 1e-12)  # Avoid numerical issues with very small epsilon_sq
        Cf_T = 2 * Nc * alpha_em * zf**2 /np.pi

        # Bessel functions that represent coordinate space version of the LCWFs
        bessel_arg = r * np.sqrt(epsilon_sq)
        K0_sq = kv(0,bessel_arg)**2
        K1_sq = kv(1,bessel_arg)**2

        term1 = (z**2 + (1 - z)**2) * epsilon_sq * K1_sq
        term2 = mf**2 * K0_sq

        return Cf_T * z * (1-z) * (term1 + term2)
    
    def psi_L_squared(self, Q, r, z):

        mf = self.mf
        zf = self.Zf

        epsilon_sq = Q**2 * z * (1 - z) + mf**2
        epsilon_sq = np.maximum(epsilon_sq, 1e-12)  # Avoid numerical issues with very small epsilon_sq
        Cf_L = 8 * Nc * alpha_em * zf**2 /np.pi

        # Bessel functions that represent coordinate space version of the LCWFs
        bessel_arg = r * np.sqrt(epsilon_sq)
        K0_sq = kv(0,bessel_arg)**2

        return Cf_L * Q**2 * z**3 * (1-z)**3 * K0_sq

