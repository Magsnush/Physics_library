# 

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
class FE_CrossSection_BK_5D:
    """
    Leading-order DIS cross section based on a finite-energy constraint together with BK evolution for a invariant mass dependent rapidity.

    Methods
    -------
    __call__(x)
        Evaluates the 5D integrand for the finite-energy constrained DIS cross section with BK evolution in xP at the given point(s) x.

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
        self.BKdipole = BKDipole(self.bkfile)
        self.quad_model_ic = QuadrupoleCorrelatorModel(Nc=Nc, LambdaQCD=LambdaQCD, dipole_model=self.BKdipole.BK_evolved_MV_model_S2_Y)

    def __call__(self, x):
        """Construct the 5D-integrand for the finite-energy constrained DIS cross section with BK evolution in xP."""
        # disentangle arguments
        u = x[0, :]
        up = x[1, :]
        z = x[2, :]
        theta = x[3, :]
        Mqq_sq = x[4, :]

        # Kinematics
        W2 = self.Q**2 * (1/self.xB - 1.0)
        xP = (Mqq_sq + self.Q**2) / (W2 + self.Q**2)
        Y = np.log(self.x0 / xP)
        
        # Photon wavefunction squared
        Long_wf_sq = self.photon_wavefunction_squared.psi_L_squared(self.Q, u, up, z, theta)
        Trans_wf_sq = self.photon_wavefunction_squared.psi_T_squared(self.Q, u, up, z, theta)

        # Target amplitude: 1 - S(u) - S(up) + S4(u, up)
        BK_S2 = self.BKdipole.BK_evolved_MV_model_S2_Y(np.stack([u, np.zeros_like(u)], axis=-1), np.array([0.0, 0.0]), Y)
        BK_S2_conj = self.BKdipole.BK_evolved_MV_model_S2_Y(np.stack([up, np.zeros_like(up)], axis=-1), np.array([0.0, 0.0]), Y)
        IC_S4 = self.quad_model_ic.quadrupole_polar(u, up, z, theta, dipole_args={"Y": Y})
        TargetAmp = 1 - BK_S2 - BK_S2_conj + IC_S4

        # Phase space integral
        arg = z*(1-z)*Mqq_sq - self.mf**2

        r2 = u**2 + up**2 - 2*u*up*np.cos(theta)

        valid = (arg > 0) & (r2 > 0)

        # Compute kinematic upper bound on Msq_qq
        Msq_min = self.mf**2 / (z*(1-z))

        # Enforce kinematic bounds: return 0 if outside physical region
        valid = Msq_min <= Mqq_sq

        I_P = np.zeros_like(arg, dtype=float)
        if np.any(valid):
            zeta = np.sqrt(arg[valid] * r2[valid])
            # J0 Bessel kernel for the k-integral (scipy.special.jv supports arrays)
            I_P_valid = z*(1-z)*jv(0, zeta) / (4.0 * np.pi)      # The factor of 1/4pi comes expressing the phase space integral in terms of the 0th order Bessel function.
            I_P[valid] = I_P_valid
        

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

        
    def BK_FE_cross_section_5D(self, r_min, r_max, z_min, z_max, theta_min, theta_max, Mqq_sq_min, Mqq_sq_max):

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
    

# Plot the unintegrated cross section as a function of Mqq_sq for some Q^2 and xB values.
def plot_unintegrated_cross_section(Q, xB):
    import matplotlib.pyplot as plt
    mf = 0.14  # GeV
    Zf = np.sqrt(2/3)
    sigma0 = 2 * 2.57 * 18.81  # GeV
    Qs0 = np.sqrt(0.104)    # GeV
    gamma = 1.0
    ec = 1.0
    bkfile = "/home/ermabert/Academia/Research/Analysis/Project-1-Finite-energy-constraint-LO-dipole-picture-total-cross-section/Papers/Paper3_Inferrning_BK_IC__with_FEC/BK_data_files/mv.dat"
    x0 = 0.01
    mcpoints = 10000

    cross_section_L = FE_CrossSection_BK_5D(Q, xB, mf, Zf, sigma0, Qs0, gamma, ec, bkfile, x0, mcpoints, "L")
    cross_section_T = FE_CrossSection_BK_5D(Q, xB, mf, Zf, sigma0, Qs0, gamma, ec, bkfile, x0, mcpoints, "T")

    Mqq_sq_values = np.logspace(np.log10(mf**2/4), np.log10((Q**2  * (1/xB - 1))), 100)
    F2_values = []

    u_values = [10.0]
    up_values = [10.0]
    z_values = [0.001, 0.5]
    theta_values = [0, np.pi/6, np.pi/4, np.pi/2]

    fig, axes = plt.subplots(3, 4, figsize=(16, 12), sharex=True, sharey=True)

    axes = axes.flatten()

    plot_idx = 0

    for u in u_values:
        for up in up_values:
            for z in z_values:
                for theta in theta_values:

                    F2_values = []

                    for Mqq_sq in Mqq_sq_values:
                        x = np.array([[u], [up], [z], [theta], [Mqq_sq]])

                        F2_value = (
                            Q**2 / (4 * np.pi**2 * alpha_em)
                            * (cross_section_L(x) + cross_section_T(x))
                        )

                        F2_values.append(F2_value.item())

                    ax = axes[plot_idx]

                    ax.plot(Mqq_sq_values, F2_values)
                    ax.set_xscale("log")

                    ax.set_title(
                        rf"$u={u}$, $u'={up}$" "\n"
                        rf"$z={z}$, $\theta={theta:.2f}$",
                        fontsize=10
                    )

                    ax.grid(True)

                    plot_idx += 1

    for ax in axes[-4:]:
        ax.set_xlabel(r"$M_{q\bar q}^2$")

    for ax in axes[::4]:
        ax.set_ylabel(r"$F_2$")

    plt.tight_layout()
    plt.show()


# Example usage
Q = np.sqrt(45)  # GeV
xB = 1e-2

plot_unintegrated_cross_section(Q, xB)







