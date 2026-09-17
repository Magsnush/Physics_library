# Leading order cross section in DIS based on the finite energy constrained cross section found in https://arxiv.org/pdf/2601.07302. Tested and works well.
import numpy as np
import vegas
from scipy.special import jv
import os
import multiprocessing


# Local imports
from small_x_physics.building_blocks.constants import Nc, alpha_em, LambdaQCD
from small_x_physics.building_blocks.wavefunctions.FE_photon_wavefunctions.LO import LO_FE_PhotonWF_squared
from small_x_physics.building_blocks.correlators.Dipoles.IC_dipole import ICDipole
from small_x_physics.building_blocks.correlators.Quadrupoles.QuadrupoleCorrelator import QuadrupoleCorrelatorModel 



# A class that computes the finite-energy constrained inclusive DIS cross section at leading order.
@vegas.rbatchintegrand
class FE_CrossSection_LO_z_is_zero_correlator:
    """
    Leading-order finite-energy constrained inclusive DIS cross section.
    Specify dipole model by passing either the string "MV" or "GBW" to the dipole_model argument. 
    """

    def __init__(self, Q, xB, mf, Zf, sigma0, Qs0, gamma, ec, mcpoints, polarization, dipole_model):
        # Initialize the parameters for the cross section calculation. 
        self.Q = Q
        self.xB = xB
        self.mf = mf
        self.Zf = Zf
        self.sigma0 = sigma0
        self.Qs0 = Qs0
        self.gamma = gamma
        self.ec = ec
        self.mcpoints = mcpoints
        self.polarization = polarization
        self.dipole_model = dipole_model


       # Initialize the wavefunctions and correlators used in the cross section calculation.
        self.photon_wavefunction_squared = LO_FE_PhotonWF_squared(self.mf, self.Zf, Nc=Nc, alpha_em=alpha_em)
        self.icdipole = ICDipole(self.Qs0, self.gamma, self.ec, LambdaQCD=LambdaQCD)
        if dipole_model == "MV":
            self.dipole_model = self.icdipole.MV_model_S2

        elif dipole_model == "GBW":
            self.dipole_model = self.icdipole.GBW_model_S2

        else:
            raise ValueError(
                f"Unknown dipole model '{dipole_model}'. "
                "Choose 'MV' or 'GBW'."
            )
        self.quad_model_ic = QuadrupoleCorrelatorModel(Nc=Nc, LambdaQCD=LambdaQCD,dipole_model=self.dipole_model)

    # Define the integrand for the cross section calculation.
    def _integrand(self, u, up, z, theta):
        """Construct the 5D-integrand for the finite-energy constrained DIS cross section with BK evolution in xB."""

        Long_wf_sq = self.photon_wavefunction_squared.psi_L_squared(self.Q, u, up, z, theta)
        Trans_wf_sq = self.photon_wavefunction_squared.psi_T_squared(self.Q, u, up, z, theta)

        IC_S2 = self.dipole_model(np.stack([u, np.zeros_like(u)], axis=-1),np.array([0.0, 0.0]),)
        IC_S2_conj = self.dipole_model(np.stack([up, np.zeros_like(up)], axis=-1),np.array([0.0, 0.0]),)
        IC_S4 = self.quad_model_ic.quadrupole_polar(u, up, 0.0, theta)
        TargetAmp = 1 - IC_S2 - IC_S2_conj + IC_S4

        Msq_max = self.Q**2 * (1 - self.xB) / self.xB
        arg = Msq_max * z * (1-z) - self.mf**2
        r2 = u**2 + up**2 - 2*u*up*np.cos(theta)
        I_P = np.zeros_like(r2)
        if arg > 0:
            valid = r2 > 0
            zeta = np.sqrt(arg * r2[valid])
            I_P[valid] = (zeta * jv(1, zeta)/ (2*np.pi*r2[valid]))

        NormFactor = 1/(4*np.pi)
        Jac = ((u*up)/(z*(1-z))) * 2*np.pi

        if self.polarization == "L":
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
    
    def __call__(self, x):

        u = x[0,:]
        up = x[1,:]
        z = x[2,:]
        theta = x[3,:]

        return self._integrand(u, up, z, theta)
        
    @vegas.rbatchintegrand
    class FixedZIntegrand:

        def __init__(self, parent, z):
            self.parent = parent
            self.z = z

        def __call__(self, x):

            u = x[0,:]
            up = x[1,:]
            theta = x[2,:]

            return self.parent._integrand(
                u, up, self.z, theta
            )

    def dsigma_dz(self, 
            z,
            r_min,
            r_max,
            theta_min,
            theta_max,
        ):
        """
        Compute the differential cross section dσ/dz for a given value of z.
        """
        # Bounded chunk-targeted batch heuristic:
        # target ~4 chunks per core, but clamp to a safe range.
        self.z = z
        n_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count()))
        target_chunks_per_core = 4
        batch_min = 1000
        batch_max = 50000
        raw_batch = int(self.mcpoints // (target_chunks_per_core * max(1, n_cores)))
        min_neval_batch = max(batch_min, min(batch_max, raw_batch))


        sensible_nproc = min(n_cores, max(1, int(self.mcpoints // min_neval_batch)))

        warm = dict(nitn=10, neval=int(self.mcpoints//10), min_neval_batch=min_neval_batch)
        full = dict(nitn=20, neval=int(self.mcpoints), min_neval_batch=min_neval_batch)

        integ = vegas.Integrator([[r_min, r_max],[r_min, r_max],[theta_min, theta_max]],nproc=sensible_nproc)

        fixed_z_integrand = self.FixedZIntegrand(self, z)

        integ(fixed_z_integrand, **warm)
        result = integ(fixed_z_integrand, **full)

        return (
            result.mean,
            result.sdev,
        )