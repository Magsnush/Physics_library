# 

import numpy as np
import vegas
import os
import multiprocessing
from scipy.special import j0
from small_x_physics.building_blocks.wavefunctions.FE_photon_wavefunctions.LO import LO_FE_PhotonWF_squared
from small_x_physics.building_blocks.correlators.Dipoles.BK_dipole import BKDipole
from small_x_physics.building_blocks.correlators.Dipoles.BK_extrapolations import (
    ExponentialBackwardBKDipole, LargeXFalloffBKDipole)
from small_x_physics.building_blocks.correlators.Quadrupoles.QuadrupoleCorrelator import QuadrupoleCorrelatorModel
from small_x_physics.building_blocks.constants import Nc, alpha_em, LambdaQCD

EXTRAPOLATIONS = ("frozen", "exponential", "falloff", "truncated")

@vegas.rbatchintegrand
class FE_CrossSection_BK_5D:
    """
    Leading-order DIS cross section based on a finite-energy constraint together with BK evolution for a invariant mass dependent rapidity.

    The dipole is evaluated at x_P = (M^2 + Q^2)/(W^2 + Q^2) = xB (1 + M^2/Q^2), i.e.
    at Y = ln(x0/x_P), which lies above x0 (Y < 0) for M^2 > Q^2 (x0/xB - 1), where
    the BK solution has no data. `extrapolation` chooses what is assumed there:

        "frozen"       N(r, x) = N(r, x0): the initial condition (the default)
        "exponential"  N(r, x) = N(r, x0) (x0/x)^lambda(r), lambda(r) the solution's
                       own d ln N/dY at x0 (ExponentialBackwardBKDipole)
        "falloff"      N(r, x) = N(r, x0) [(1 - x)/(1 - x0)]^falloff_n
                       (LargeXFalloffBKDipole)
        "truncated"    no extrapolation: only M^2 <= Q^2 (x0/xB - 1), i.e.
                       x_P <= x0, is integrated

    Below x0 all four use the same BK solution.

    Methods
    -------
    __call__(x)
        Evaluates the 5D integrand for the finite-energy constrained DIS cross section with BK evolution in xP at the given point(s) x.

    """
    def __init__(self, Q, xB, mf, Zf, sigma0, Qs0, gamma, ec, bkfile, x0, mcpoints, polarization, largeNc=False,
                 extrapolation="frozen", falloff_n=None):
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
        if polarization not in ("L", "T"):
            raise ValueError(f"polarization must be 'L' or 'T', got {polarization!r}")
        self.polarization = polarization
        self.largeNc = largeNc
        if extrapolation not in EXTRAPOLATIONS:
            raise ValueError(f"extrapolation must be one of {EXTRAPOLATIONS}, got {extrapolation!r}")
        if (extrapolation == "falloff") != (falloff_n is not None):
            raise ValueError("falloff_n is needed for extrapolation='falloff' and for nothing else")
        self.extrapolation = extrapolation
        self.falloff_n = falloff_n
        # self.Mqq_sq = None
        # self.xP = None

        self.photon_wavefunction_squared = LO_FE_PhotonWF_squared(self.mf, self.Zf, Nc=Nc, alpha_em=alpha_em)
        if extrapolation == "exponential":
            self.BKdipole = ExponentialBackwardBKDipole(self.bkfile, warn_on_clamp=False)
        elif extrapolation == "falloff":
            self.BKdipole = LargeXFalloffBKDipole(self.bkfile, n=falloff_n, warn_on_clamp=False)
        else:
            self.BKdipole = BKDipole(self.bkfile)
        # x0 defines the rapidity Y = ln(x0/x_P) at which the dipole is read, so it
        # has to be the x0 the bk file's evolution starts from.
        if not np.isclose(self.x0, self.BKdipole.x0, rtol=1e-6):
            raise ValueError(f"x0 = {self.x0:g}, but {self.bkfile} starts its evolution at "
                             f"x0 = {self.BKdipole.x0:g}")
        self.quad_model_ic = QuadrupoleCorrelatorModel(Nc=Nc, LambdaQCD=LambdaQCD, dipole_model=self.BKdipole.BK_evolved_MV_model_S2_Y)

    def __call__(self, x):
        """Construct the 5D-integrand for the finite-energy constrained DIS cross section with BK evolution in xP."""
        # disentangle arguments
        u = x[0, :]
        up = x[1, :]
        z = x[2, :]
        theta = x[3, :]
        Mqq_sq = x[4, :]
        # Mqq_sq = self.Mqq_sq
        #xP = self.xP

        # Kinematics
        W2 = self.Q**2 * (1/self.xB - 1.0)
        xP = (Mqq_sq + self.Q**2) / (W2 + self.Q**2)
        Y = np.log(self.x0 / xP)
        #Mqq_sq = xP * (W2 + self.Q**2) - self.Q**2
        
        # Photon wavefunction squared, only for the polarization this object integrates
        if self.polarization == "L":
            wf_sq = self.photon_wavefunction_squared.psi_L_squared(self.Q, u, up, z, theta)
        else:
            wf_sq = self.photon_wavefunction_squared.psi_T_squared(self.Q, u, up, z, theta)

        # Target amplitude: 1 - S(u) - S(up) + S4(u, up). S(u) and S(up) are two of
        # the six pair dipoles the quadrupole is built from, so all three come from
        # six dipole evaluations instead of sixteen. The dipole is the quadrupole
        # model's, which is what a caller swapping in another dipole replaces.
        S_u, S_up, S4 = self.quad_model_ic.polar_correlators(
            u, up, z, theta, dipole_args={"Y": Y}, largeNc=self.largeNc)
        TargetAmp = 1 - S_u - S_up + S4

        # Phase space integral
        arg = z*(1-z) * Mqq_sq - self.mf**2
        r2 = u**2 + up**2 - 2*u*up*np.cos(theta)

        # Compute kinematic upper bound on Msq_qq
        Msq_min = self.mf**2 / (z*(1-z))
        #xP_min = (Msq_min + self.Q**2) / (W2 + self.Q**2)

        # Enforce kinematic bounds: return 0 if outside physical region
        valid = (Msq_min <= Mqq_sq) #& (arg > 0) & (r2 > 0)
        #valid = (xP_min <= xP) & (arg > 0) & (r2 > 0)
        if self.extrapolation == "truncated":
            valid &= xP <= self.x0

        I_P = np.zeros_like(arg, dtype=float)
        if np.any(valid):
            zeta = np.sqrt(arg[valid] * r2[valid])
            # J0 Bessel kernel for the k-integral
            I_P_valid = z[valid]*(1-z[valid])*j0(zeta) / (4.0 * np.pi)      # The factor of 1/4pi comes expressing the phase space integral in terms of the 0th order Bessel function.
            I_P[valid] = I_P_valid

        # In the paper the normalization factor and the factor 1/(z*(1-z)) has been absorbed into the definition of the wavefunction squared, but here we keep it explicit.
        NormFactor = 1/(4*np.pi)
        Jac = ((u*up)/(z*(1-z))) * 2*np.pi #* (W2 + self.Q**2)  # The factor of (W2 + Q^2) comes from the change of variables from Mqq_sq to xP .

        return (
            (self.sigma0/2)
            * NormFactor
            * Jac
            * wf_sq
            * TargetAmp
            * I_P
        )

        
    def BK_FE_cross_section_5D(self, r_min, r_max, z_min, z_max, theta_min, theta_max, Mqq_sq_min, Mqq_sq_max):
        """Integrate the cross section; returns (mean, sdev).

        With extrapolation="truncated" the M^2 range is cut at x_P = x0, i.e. at
        M^2 = Q^2 (x0/xB - 1), so that no points are spent where the integrand is
        zero. When that leaves no range, as for every xB above about x0, the cross
        section is zero and (0.0, 0.0) is returned without integrating.
        """
        if self.extrapolation == "truncated":
            Mqq_sq_max = min(Mqq_sq_max, self.Q**2 * (self.x0 / self.xB - 1))
            if Mqq_sq_max <= Mqq_sq_min:
                return 0.0, 0.0

        # Bounded chunk-targeted batch heuristic:
        # target ~4 chunks per core, but clamp to a safe range.
        n_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count()))
        target_chunks_per_core = 4
        batch_min = 1000
        batch_max = 50000
        raw_batch = int(self.mcpoints // (target_chunks_per_core * max(1, n_cores)))
        min_neval_batch = max(batch_min, min(batch_max, raw_batch))
        

        sensible_nproc = min(n_cores, max(1, int(self.mcpoints // min_neval_batch)))

        warm = dict(nitn=10, neval=int(self.mcpoints//10), min_neval_batch=min_neval_batch)
        full = dict(nitn=20, neval=int(self.mcpoints), min_neval_batch=min_neval_batch)

        integ = vegas.Integrator([[r_min, r_max],[r_min, r_max],[z_min, z_max],[theta_min, theta_max], [Mqq_sq_min, Mqq_sq_max]],nproc=sensible_nproc)

        integ(self, **warm)

        result = integ(self, **full)

        return (
            result.mean,
            result.sdev,
       )
    
    # def dsigma_dMsq(self,
    #     Mqq_sq,
    #     r_min,
    #     r_max,
    #     z_min,
    #     z_max,
    #     theta_min,
    #     theta_max,
    # ):

    #     self.Mqq_sq = Mqq_sq
    #     n_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count()))
    #     target_chunks_per_core = 4
    #     batch_min = 1000
    #     batch_max = 50000
    #     raw_batch = int(self.mcpoints // (target_chunks_per_core * max(1, n_cores)))
    #     min_neval_batch = max(batch_min, min(batch_max, raw_batch))
        

    #     sensible_nproc = min(n_cores, max(1, int(self.mcpoints // min_neval_batch)))

    #     warm = dict(nitn=10, neval=int(self.mcpoints//10), min_neval_batch=min_neval_batch)
    #     full = dict(nitn=20, neval=int(self.mcpoints), min_neval_batch=min_neval_batch)

    #     # setup VEGAS exactly as before, but only 4 dimensions

    #     integ = vegas.Integrator([
    #         [r_min, r_max],
    #         [r_min, r_max],
    #         [z_min, z_max],
    #         [theta_min, theta_max]]
    #         ,nproc=sensible_nproc)

    #     integ(self, **warm)
    #     result = integ(self, **full)

    #     return result.mean, result.sdev
    
    # def dsigma_dxP(self,
    #     xP,
    #     r_min,
    #     r_max,
    #     z_min,
    #     z_max,
    #     theta_min,
    #     theta_max,
    # ):

    #     self.xP = xP
    #     n_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count()))
    #     target_chunks_per_core = 4
    #     batch_min = 1000
    #     batch_max = 50000
    #     raw_batch = int(self.mcpoints // (target_chunks_per_core * max(1, n_cores)))
    #     min_neval_batch = max(batch_min, min(batch_max, raw_batch))
        

    #     sensible_nproc = min(n_cores, max(1, int(self.mcpoints // min_neval_batch)))

    #     warm = dict(nitn=10, neval=int(self.mcpoints//10), min_neval_batch=min_neval_batch)
    #     full = dict(nitn=20, neval=int(self.mcpoints), min_neval_batch=min_neval_batch)

    #     # setup VEGAS exactly as before, but only 4 dimensions

    #     integ = vegas.Integrator([
    #         [r_min, r_max],
    #         [r_min, r_max],
    #         [z_min, z_max],
    #         [theta_min, theta_max]]
    #         ,nproc=sensible_nproc)

    #     integ(self, **warm)
    #     result = integ(self, **full)

    #     return result.mean, result.sdev










