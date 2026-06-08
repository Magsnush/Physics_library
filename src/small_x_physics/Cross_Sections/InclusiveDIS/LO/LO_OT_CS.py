# Leading order cross section in DIS based on the optical theorem. Tested and works well.

import numpy as np
from scipy.integrate import dblquad
from small_x_physics.numerics.totalDIS import LO
from small_x_physics.building_blocks.wavefunctions.OT_photon_wavefunctions.LO import LO_OT_PhotonWF_squared
from small_x_physics.building_blocks.correlators.Dipoles.IC_dipole import ICDipole   
from small_x_physics.building_blocks.constants import Nc, alpha_em, LambdaQCD

class OT_CrossSection_LO:
    """
    Leading-order DIS cross section based on the optical theorem.

    Methods
    -------
    compute_cross_section(Q, mf, Zf, sigma0, Qs0, gamma, ec, r_min, r_max, z_min, z_max)
        Computes the longitudinal and transverse cross sections by integrating the LO optical theorem integrand.

    """
    def __init__(self, Q, mf, Zf, sigma0, Qs0, gamma, ec):
        self.Q = Q
        self.mf = mf
        self.Zf = Zf
        self.sigma0 = sigma0
        self.Qs0 = Qs0
        self.gamma = gamma
        self.ec = ec
        
    def OT_cross_section(self, r_min, r_max, z_min, z_max):

        def integrand(r, z, polarization):
            # Photon wavefunction squared
            photon_wavefunction_squared = LO_OT_PhotonWF_squared(self.Q, self.mf, self.Zf, Nc=Nc, alpha_em=alpha_em)

            Long_wf_sq = photon_wavefunction_squared.psi_L_squared(self.Q, r, z)

            Trans_wf_sq = photon_wavefunction_squared.psi_T_squared(self.Q, r, z)

            # Dipole amplitude: 2*(1 - S(r))

            icdipole = ICDipole(self.Qs0, self.gamma, self.ec, LambdaQCD=LambdaQCD)
            IC_S2 = icdipole.MV_model_S2(np.stack([r, 0]), np.array([0, 0]))

            DipoleAmp = 2 * (1 - IC_S2)

            # Jacobian factor
            Jac = 2 * np.pi * r / (z * (1 - z))

            if polarization == "L":
                Long_integrand = (self.sigma0 / 2) * 1 / (4 * np.pi) * Jac * Long_wf_sq * DipoleAmp
                return Long_integrand
            elif polarization == "T":
                Trans_integrand = (self.sigma0 / 2) * 1 / (4 * np.pi) * Jac * Trans_wf_sq * DipoleAmp
                return Trans_integrand

            return Long_integrand, Trans_integrand

        # Use adaptive quadrature for analytic dipoles
        # The integrand expects signature integrand.OT_integrand(r, z, flavor).
        long_cs, long_err = dblquad(lambda z, r: integrand(r, z, "L"),
                r_min,
                r_max,
                z_min,
                z_max,
            )

        trans_cs, trans_err = dblquad(lambda z, r: integrand(r, z, "T"),
                r_min,
                r_max,
                z_min,
                z_max,
            )

        return long_cs, long_err, trans_cs, trans_err
    
    
