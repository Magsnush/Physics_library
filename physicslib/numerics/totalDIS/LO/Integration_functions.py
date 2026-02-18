from theoretical_building_blocks.PhotonProtonCrossSectionLib import LOPhotonProtonCrossSection
import vegas
import multiprocessing 
import numpy as np
from scipy.integrate import dblquad
from time import perf_counter
import sys

n_cores = multiprocessing.cpu_count()


######    OPTICAL THEOREM INTEGRATION   ######

def OT_L_Cross_Section(Q, m, r_min, r_max, z_min, z_max):

    # Cross section class from which we obtain the optical theorem calculation as well as the kinematically constrained result. Paramter values are obtained from https://arxiv.org/pdf/1309.6963

    cs = LOPhotonProtonCrossSection(Q = Q, m = m, Zf = np.sqrt(6/9), Nc = 3, Qs0= np.sqrt(0.104), gamma = 1.0, LambdaQCD = 0.241, ec = 1.0, sigma0 = 2.57*2*18.81)

    # Definition of the Optical Theorem cross section integrands

    # Longitudinal polarization "L"

    def OT_L_integrand(r, z):
        return cs.OT_integrand(r, z, "L")
    
    OT_cs_L_result, OT_cs_L_error = dblquad(
    lambda z, r: OT_L_integrand(r,z), r_min, r_max, z_min, z_max, 
    )

    return OT_cs_L_result, OT_cs_L_error

def OT_T_Cross_Section(Q, m, r_min, r_max, z_min, z_max):

    # Cross section class from which we obtain the optical theorem calculation as well as the kinematically constrained result. Paramter values are obtained from https://arxiv.org/pdf/1309.6963

    cs = LOPhotonProtonCrossSection(Q = Q, m = m, Zf = np.sqrt(6/9), Nc = 3, Qs0= np.sqrt(0.104), gamma = 1.0, LambdaQCD = 0.241, ec = 1.0, sigma0 = 2.57*2*18.81)

    # Definition of the Optical Theorem cross section integrands

    # Longitudinal polarization "L"

    def OT_T_integrand(r, z):
        return cs.OT_integrand(r, z, "T")
    
    OT_cs_T_result, OT_cs_T_error = dblquad(
    lambda z, r: OT_T_integrand(r,z), r_min, r_max, z_min, z_max, 
    )

    return OT_cs_T_result, OT_cs_T_error



######   KINEMATICALLY CONSTRAINED INTEGRATION   #######

@vegas.rbatchintegrand
class KC_L_integrand:
    def __init__(self, Msq, cs):
        self.Msq = Msq
        self.cs = cs
    def __call__(self, x):
        u, up, z, alpha = x
        return self.cs.KC_HypGeom_integrand(u, up, z, alpha, self.Msq, "L", largeNc=False)


@vegas.rbatchintegrand
class KC_T_integrand:
    def __init__(self, Msq, cs):
        self.Msq = Msq
        self.cs = cs
    def __call__(self, x):
        u, up, z, alpha = x
        return self.cs.KC_HypGeom_integrand(u, up, z, alpha, self.Msq, "T", largeNc=False)


@vegas.rbatchintegrand
class Large_Nc_KC_L_integrand:
    def __init__(self, Msq, cs):
        self.Msq = Msq
        self.cs = cs
    def __call__(self, x):
        u, up, z, alpha = x
        return self.cs.KC_HypGeom_integrand(u, up, z, alpha, self.Msq, "L", largeNc=True)


@vegas.rbatchintegrand
class Large_Nc_KC_T_integrand:
    def __init__(self, Msq, cs):
        self.Msq = Msq
        self.cs = cs
    def __call__(self, x):
        u, up, z, alpha = x
        return self.cs.KC_HypGeom_integrand(u, up, z, alpha, self.Msq, "T", largeNc=True)



# ============================================================
#   Dictionary mapping (mode, polarization) → integrand class
# ============================================================

CROSS_SECTIONS = {
    ("KC",       "L"): KC_L_integrand,
    ("KC",       "T"): KC_T_integrand,
    ("LargeNc",  "L"): Large_Nc_KC_L_integrand,
    ("LargeNc",  "T"): Large_Nc_KC_T_integrand,
}



# ============================================================
#    Generic VEGAS integrator for all cross section types
# ============================================================

def run_vegas_integral(
    integrand_class, Q, xB, m, Zf,
    umin, umax, upmin, upmax, zmin, zmax, alphamin, alphamax,
    mcpoints, n_cores
):

    # Construct cross section object
    cs = LOPhotonProtonCrossSection(
        Q=Q, m=m, Zf=Zf, Nc=3,
        Qs0=np.sqrt(0.104), gamma=1.0,
        LambdaQCD=0.241, ec=1.0,
        sigma0=2.57 * 2 * 18.81
    )

    # Kinematic invariant
    Msq = Q**2 * (1/xB - 1)

    # VEGAS integrator
    integ = vegas.Integrator(
        [[umin, umax], [upmin, upmax], [zmin, zmax], [alphamin, alphamax]],
        nproc=n_cores
    )

    warm = dict(nitn=30, neval=mcpoints//10, min_neval_batch=1e5)
    full = dict(nitn=20, neval=mcpoints,     min_neval_batch=1e5)

    # Warm-up adaptation
    integ(integrand_class(Msq, cs), **warm)

    # Full integration
    result = integ(integrand_class(Msq, cs), **full)

    return result.mean, result.sdev



# ============================================================
#   Single unified API
# ============================================================

def compute_cross_section(
    mode, pol, Q, xB, m, Zf,
    umin, umax, upmin, upmax, zmin, zmax,
    alphamin, alphamax, mcpoints, n_cores
):
    """
    mode: "KC" or "LargeNc"
    pol:  "L" or "T"
    """
    integrand_class = CROSS_SECTIONS[(mode, pol)]

    return run_vegas_integral(
        integrand_class,
        Q, xB, m, Zf,
        umin, umax, upmin, upmax, zmin, zmax,
        alphamin, alphamax,
        mcpoints, n_cores
    )




