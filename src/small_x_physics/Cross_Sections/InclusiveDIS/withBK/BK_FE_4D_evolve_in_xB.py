# Finite energy constrained cross section in DIS found in https://arxiv.org/abs/2601.07302 based on dipole picture with a BK-evolved dipole. This works now.

import numpy as np
import vegas
import os
import multiprocessing
from scipy.special import jv
from small_x_physics.building_blocks.wavefunctions.FE_photon_wavefunctions.LO import LO_FE_PhotonWF_squared
from small_x_physics.building_blocks.correlators.Dipoles.BK_dipole import BKDipole  
from small_x_physics.building_blocks.correlators.Quadrupoles.QuadrupoleCorrelator import QuadrupoleCorrelatorModel 
from small_x_physics.building_blocks.constants import Nc, alpha_em, LambdaQCD

@vegas.rbatchintegrand  
class FE_CrossSection_BK_4D:
    """
    Leading-order DIS cross section based on the optical theorem.

    Methods
    -------
    compute_cross_section(Q, mf, Zf, sigma0, Qs0, gamma, ec, r_min, r_max, z_min, z_max)
        Computes the longitudinal and transverse cross sections by integrating the LO optical theorem integrand.

    """
    def __init__(self, Q, xB, mf, Zf, sigma0, Qs0, gamma, ec, bkfile, x0, mcpoints, polarization):
        self.Q = Q
        self.xB = xB
        self.mf = mf
        self.Zf = Zf
        self.sigma0 = sigma0
        self.Qs0 = Qs0
        self.gamma = gamma
        self.ec = ec
        self.bkfile = bkfile
        self.x0 = x0
        self.mcpoints = mcpoints
        self.polarization = polarization

        self.photon_wavefunction_squared = LO_FE_PhotonWF_squared(self.mf, self.Zf, Nc=Nc, alpha_em=alpha_em)
        self.Y = np.log(self.x0 / self.xB)
        self.BKdipole = BKDipole(self.bkfile, self.Y)
        self.quad_model_ic = QuadrupoleCorrelatorModel(Nc=Nc, LambdaQCD=LambdaQCD, dipole_model=self.BKdipole.BK_evolved_MV_model_S2)

    def __call__(self, x):
        """Construct the 5D-integrand for the finite-energy constrained DIS cross section with BK evolution in xB."""
        u = x[0, :]
        up = x[1, :]
        z = x[2, :]
        theta = x[3, :]

        # Photon wavefunction squared
        Long_wf_sq = self.photon_wavefunction_squared.psi_L_squared(self.Q, u, up, z, theta)
        Trans_wf_sq = self.photon_wavefunction_squared.psi_T_squared(self.Q, u, up, z, theta)

        # Target amplitude: 1 - S(u) - S(up) + S4(u, up)
        BK_S2 = self.BKdipole.BK_evolved_MV_model_S2(np.stack([u, np.zeros_like(u)], axis=-1), np.array([0, 0]))
        BK_S2_conj = self.BKdipole.BK_evolved_MV_model_S2(np.stack([up, np.zeros_like(up)], axis=-1), np.array([0.0, 0.0]))
        IC_S4 = self.quad_model_ic.quadrupole_polar(u, up, z, theta)
        TargetAmp = 1 - BK_S2 - BK_S2_conj + IC_S4

        # Phase space integral
        Msq_max = self.Q**2 * (1 - self.xB) / self.xB
        arg = Msq_max * z * (1.0 - z) - self.mf**2
        r2 = u**2 + up**2 - 2*u*up*np.cos(theta)
        valid = (arg > 0) & (r2 > 0)
        I_P = np.zeros_like(arg)
        if np.any(valid):
            zeta = np.sqrt(arg[valid] * r2[valid])
            I_P[valid] = zeta * jv(1, zeta) / (2*np.pi*r2[valid])

        NormFactor = 1/(4*np.pi)
        Jac = ((u*up)/(z*(1-z))) * 2*np.pi

        if self.polarization == "L":
            wf_sq = Long_wf_sq
        elif self.polarization == "T":
            wf_sq = Trans_wf_sq

        return (
            (self.sigma0/2)
            * NormFactor
            * Jac
            * wf_sq
            * TargetAmp
            * I_P
        )

        
    def BK_FE_cross_section_4D(self, r_min, r_max, z_min, z_max, theta_min, theta_max):

        # Bounded chunk-targeted batch heuristic:
        # target ~4 chunks per core, but clamp to a safe range.
        n_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count()))
        target_chunks_per_core = 4
        batch_min = 1000
        batch_max = 100000
        raw_batch = int(self.mcpoints // (target_chunks_per_core * max(1, n_cores)))
        min_neval_batch = max(batch_min, min(batch_max, raw_batch))
        

        sensible_nproc = min(n_cores, max(1, int(self.mcpoints // min_neval_batch)))

        warm = dict(nitn=10, neval=int(self.mcpoints//10), min_neval_batch=min_neval_batch)
        full = dict(nitn=20, neval=int(self.mcpoints), min_neval_batch=min_neval_batch)

        integ = vegas.Integrator([[r_min, r_max],[r_min, r_max],[z_min, z_max],[theta_min, theta_max]],nproc=sensible_nproc)

        integ(self, **warm)

        result = integ(self, **full)

        return (
            result.mean,
            result.sdev,
        )