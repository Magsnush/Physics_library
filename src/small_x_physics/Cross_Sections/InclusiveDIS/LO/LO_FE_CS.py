# Leading order cross section in DIS based on the finite energy constrained cross section found in https://arxiv.org/pdf/2601.07302. Tested and works well.
import numpy as np
import vegas
from scipy.special import jv
from numba import njit

# Local imports
from small_x_physics.building_blocks.constants import Nc, alpha_em, LambdaQCD
from small_x_physics.building_blocks.wavefunctions.FE_photon_wavefunctions.LO import LO_FE_PhotonWF_squared
from small_x_physics.building_blocks.correlators.Dipoles.IC_dipole import ICDipole
from small_x_physics.building_blocks.correlators.Quadrupoles.QuadrupoleCorrelator import QuadrupoleCorrelatorModel 

# An integrand wrapper that allow me to use multiple processesors with vegas. 
@vegas.rbatchintegrand
class FEIntegrandWrapper:

    def __init__(self, cross_section, polarization):
        self.cs = cross_section
        self.pol = polarization

    def __call__(self, x):

        u = x[0, :]
        up = x[1, :]
        z = x[2, :]
        theta = x[3, :]

        return self.cs.integrand(
            u,
            up,
            z,
            theta,
            self.pol,
        )
    
class FE_CrossSection_LO:
    """
    Leading-order finite-energy constrained inclusive DIS cross section.
    """

    def __init__(self, Q, xB, mf, Zf, sigma0, Qs0, gamma, ec):
        self.Q = Q
        self.xB = xB
        self.mf = mf
        self.Zf = Zf
        self.sigma0 = sigma0
        self.Qs0 = Qs0
        self.gamma = gamma
        self.ec = ec

        self.photon_wavefunction_squared = LO_FE_PhotonWF_squared(self.mf, self.Zf, Nc=Nc, alpha_em=alpha_em)
        self.icdipole = ICDipole(self.Qs0, self.gamma, self.ec, LambdaQCD=LambdaQCD)
        self.quad_model_ic = QuadrupoleCorrelatorModel(Nc=Nc, LambdaQCD=LambdaQCD,dipole_model=self.icdipole.MV_model_S2)


    def integrand(self, u, up, z, theta, polarization):

        Long_wf_sq = self.photon_wavefunction_squared.psi_L_squared(
            self.Q, u, up, z, theta
        )

        Trans_wf_sq = self.photon_wavefunction_squared.psi_T_squared(
            self.Q, u, up, z, theta
        )

        IC_S2 = self.icdipole.MV_model_S2(
            np.stack([u, np.zeros_like(u)], axis=-1),
            np.array([0.0, 0.0]),
        )

        IC_S2_conj = self.icdipole.MV_model_S2(
            np.stack([up, np.zeros_like(up)], axis=-1),
            np.array([0.0, 0.0]),
        )

        IC_S4 = self.quad_model_ic.quadrupole_polar(
            u, up, z, theta
        )

        TargetAmp = 1 - IC_S2 - IC_S2_conj + IC_S4

        Msq_max = self.Q**2 * (1 - self.xB) / self.xB

        arg = Msq_max * z * (1.0 - z) - self.mf**2

        r2 = u**2 + up**2 - 2*u*up*np.cos(theta)

        valid = (arg > 0) & (r2 > 0)

        I_P = np.zeros_like(arg)

        if np.any(valid):
            zeta = np.sqrt(arg[valid] * r2[valid])

            I_P[valid] = (
                zeta * jv(1, zeta)
                / (2*np.pi*r2[valid])
            )

        NormFactor = 1/(4*np.pi)
        Jac = ((u*up)/(z*(1-z))) * 2*np.pi

        if polarization == "L":
            wf_sq = Long_wf_sq
        else:
            wf_sq = Trans_wf_sq

        return (
            (self.sigma0/2)
            * NormFactor
            * Jac
            * wf_sq
            * TargetAmp
            * I_P
        )
        
    def compute_cross_section_FE(
            self,
            r_min,
            r_max,
            z_min,
            z_max,
            theta_min,
            theta_max,
        ):

            L_integrand = FEIntegrandWrapper(self, "L")
            T_integrand = FEIntegrandWrapper(self, "T")

            warm = dict(nitn=5, neval=10000, min_neval_batch=1000)
            full = dict(nitn=20, neval=100000, min_neval_batch=5000)

            integ = vegas.Integrator(
                [
                    [r_min, r_max],
                    [r_min, r_max],
                    [z_min, z_max],
                    [theta_min, theta_max],
                ],
                nproc=8,
            )

            integ(L_integrand, **warm)
            integ(T_integrand, **warm)

            result_L = integ(L_integrand, **full)
            result_T = integ(T_integrand, **full)

            return (
                result_L.mean,
                result_L.sdev,
                result_T.mean,
                result_T.sdev,
            )
            
