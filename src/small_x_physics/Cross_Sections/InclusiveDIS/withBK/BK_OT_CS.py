# Cross section in DIS based on the optical theorem and dipole picture with a BK-evolved dipole. This works now.

import numpy as np
import vegas
import os
import multiprocessing
from scipy.integrate import dblquad
from small_x_physics.numerics.totalDIS import LO
from small_x_physics.building_blocks.wavefunctions.OT_photon_wavefunctions.LO import LO_OT_PhotonWF_squared
from small_x_physics.building_blocks.correlators.Dipoles.BK_dipole import BKDipole   
from small_x_physics.building_blocks.constants import Nc, alpha_em, LambdaQCD

@vegas.rbatchintegrand
class BKOTIntegrandWrapper:

    def __init__(self, cross_section, polarization):
        self.cs = cross_section
        self.pol = polarization

    def __call__(self, x):

        r = x[0, :]
        z = x[1, :]

        return self.cs.BK_integrand(
            r,
            z,
            self.pol,
        )
    
class OT_CrossSection_BK:
    """
    Leading-order DIS cross section based on the optical theorem.

    Methods
    -------
    compute_cross_section(Q, mf, Zf, sigma0, Qs0, gamma, ec, r_min, r_max, z_min, z_max)
        Computes the longitudinal and transverse cross sections by integrating the LO optical theorem integrand.

    """
    def __init__(self, Q, xB, mf, Zf, sigma0, Qs0, gamma, ec, bkfile, x0, mcpoints):
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

        self.photon_wavefunction_squared = LO_OT_PhotonWF_squared(self.Q, self.mf, self.Zf, Nc=Nc, alpha_em=alpha_em)
        self.Y = np.log(self.x0 / self.xB)
        self.icdipole = BKDipole(self.bkfile, Y=self.Y)

    def BK_integrand(self, r, z, polarization):

        # Photon wavefunction squared
        Long_wf_sq = self.photon_wavefunction_squared.psi_L_squared(self.Q, r, z)

        Trans_wf_sq = self.photon_wavefunction_squared.psi_T_squared(self.Q, r, z)

        # Dipole amplitude: 2*(1 - S(r))
        BK_S2 = self.icdipole.BK_evolved_MV_model_S2(np.stack([r, np.zeros_like(r)], axis=-1), np.array([0, 0]))

        DipoleAmp = 2 * (1 - BK_S2)

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
        
    def BK_OT_cross_section(self, r_min, r_max, z_min, z_max):

        L_integrand = BKOTIntegrandWrapper(self, "L")
        T_integrand = BKOTIntegrandWrapper(self, "T")

        # Bounded chunk-targeted batch heuristic:
        # target ~4 chunks per core, but clamp to a safe range.
        n_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count()))
        target_chunks_per_core = 4
        batch_min = 1000
        batch_max = 50000
        raw_batch = int(self.mcpoints // (target_chunks_per_core * max(1, n_cores)))
        min_neval_batch = max(
            1,
            min(
                int(self.mcpoints),
                max(batch_min, min(batch_max, raw_batch)),
            ),
        )

        sensible_nproc = min(n_cores, max(1, int(self.mcpoints // min_neval_batch)))

        warm = dict(nitn=10, neval=int(self.mcpoints//10), min_neval_batch=min_neval_batch)
        full = dict(nitn=20, neval=int(self.mcpoints), min_neval_batch=min_neval_batch)

        integ = vegas.Integrator(
            [
                [r_min, r_max],
                [z_min, z_max],
            ],
            nproc=sensible_nproc,
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
