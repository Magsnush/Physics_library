# The DIS reduced cross section sigma_r, and F_L, F_T, F_2, for each of the LO
# dipole-picture cross sections in this library.
#
# Every function here integrates the longitudinal and transverse photon-proton
# cross sections for one quark flavour (group) at one (Q^2, xB), and converts them:
#
#     F_{L,T} = Q^2 / (4 pi^2 alpha_em) sigma_{L,T},      F_2 = F_L + F_T,
#     sigma_r = F_2 - f(y) F_L,   f(y) = y^2 / (1 + (1 - y)^2),   y = Q^2 / (s xB).
#
# L and T come from separate integrations, so their errors are independent, and
# the error of sigma_r = F_T + (1 - f) F_L is sqrt(dF_T^2 + (1 - f)^2 dF_L^2).
#
# Units: Q^2 in GeV^2, sqrt_s and the quark mass in GeV, sigma0 in GeV^-2.
#
# Integration limits default to those used in production: dipole sizes from 1e-6
# to 30 GeV^-1, z from 1e-8 to 1 - 1e-8, the angle over [0, 2 pi], and for the 5D
# cross section the invariant mass from 4 mf^2 to W^2 = Q^2 (1/xB - 1). The dipole
# sizes and z can be changed with the keyword arguments of the same names.
#
# The Monte Carlo cross sections run VEGAS on SLURM_CPUS_PER_TASK processes, or on
# every core if that is unset; run them under `if __name__ == "__main__":`.

from dataclasses import dataclass

import numpy as np

from small_x_physics.building_blocks.constants import alpha_em
from small_x_physics.building_blocks.correlators.Dipoles.BK_dipole import BKDipole
from small_x_physics.Cross_Sections.InclusiveDIS.LO.LO_OT_CS import OT_CrossSection_LO
from small_x_physics.Cross_Sections.InclusiveDIS.LO.LO_FE_CS import FE_CrossSection_LO
from small_x_physics.Cross_Sections.InclusiveDIS.withBK.BK_OT_CS import OT_CrossSection_BK
from small_x_physics.Cross_Sections.InclusiveDIS.withBK.BK_FE_4D_evolve_in_xB import FE_CrossSection_BK_4D
from small_x_physics.Cross_Sections.InclusiveDIS.withBK.BK_FE_5D_evolve_in_xP import (
    EXTRAPOLATIONS, FE_CrossSection_BK_5D)

SQRT_S_HERA = 318.0
LIMITS = dict(r_min=1e-6, r_max=30.0, z_min=1e-8, z_max=1 - 1e-8)
THETA = (0.0, 2 * np.pi)


@dataclass(frozen=True)
class StructureFunctions:
    """F_L, F_T, F_2 and the reduced cross section at one (Q^2, xB), with errors."""
    Q2: float
    xB: float
    y: float
    FL: float
    FL_err: float
    FT: float
    FT_err: float
    F2: float
    F2_err: float
    sigma_r: float
    sigma_r_err: float


def structure_functions(Q2, xB, sigma_L, sigma_L_err, sigma_T, sigma_T_err, sqrt_s=SQRT_S_HERA):
    """StructureFunctions from the longitudinal and transverse cross sections (GeV^-2).

    The errors of sigma_L and sigma_T are taken to be independent.
    """
    y = Q2 / (sqrt_s**2 * xB)
    if y > 1:
        raise ValueError(f"y = Q^2/(s xB) = {y:.3g} > 1 at Q^2 = {Q2:g}, xB = {xB:g}, "
                         f"sqrt(s) = {sqrt_s:g}: not a physical point")
    to_F = Q2 / (4 * np.pi**2 * alpha_em)
    FL, FL_err = to_F * sigma_L, to_F * sigma_L_err
    FT, FT_err = to_F * sigma_T, to_F * sigma_T_err
    f = y**2 / (1 + (1 - y)**2)
    return StructureFunctions(
        Q2=Q2, xB=xB, y=y,
        FL=FL, FL_err=FL_err, FT=FT, FT_err=FT_err,
        F2=FL + FT, F2_err=float(np.hypot(FL_err, FT_err)),
        sigma_r=FT + (1 - f) * FL, sigma_r_err=float(np.hypot(FT_err, (1 - f) * FL_err)),
    )


def _limits(overrides):
    unknown = set(overrides) - set(LIMITS)
    if unknown:
        raise TypeError(f"unknown integration limit(s): {', '.join(sorted(unknown))}; "
                        f"the adjustable ones are {', '.join(LIMITS)}")
    return {**LIMITS, **overrides}


def _bk_parameters(bkfile):
    """Initial-condition parameters and x0 from the bk file's header.

    The BK cross sections take Qs0, gamma and ec for a uniform interface but use the
    initial condition the file was computed with, and x0 must be the one the file's
    evolution starts at, so all four are read from the file rather than passed in.
    """
    d = BKDipole(bkfile, warn_on_clamp=False)
    Qs0 = None if d.Qs0_sq is None else float(np.sqrt(d.Qs0_sq))
    return dict(Qs0=Qs0, gamma=d.gamma, ec=d.ec, x0=d.x0)


# ---------------------------------------------------------------------------
# Analytic dipole (MV or GBW), no evolution
# ---------------------------------------------------------------------------

def sigma_r_OT_LO(Q2, xB, mf, Zf, sigma0, Qs0, gamma, ec, sqrt_s=SQRT_S_HERA, **limits):
    """Optical-theorem cross section with the MV dipole, by adaptive quadrature.

    The dipole does not depend on x, so xB enters only through y. The errors are the
    quadrature's error estimates.
    """
    lim = _limits(limits)
    cs = OT_CrossSection_LO(Q=np.sqrt(Q2), mf=mf, Zf=Zf, sigma0=sigma0, Qs0=Qs0, gamma=gamma, ec=ec)
    sL, eL, sT, eT = cs.OT_cross_section(lim["r_min"], lim["r_max"], lim["z_min"], lim["z_max"])
    return structure_functions(Q2, xB, sL, eL, sT, eT, sqrt_s)


def sigma_r_FE_LO(Q2, xB, mf, Zf, sigma0, Qs0, gamma, ec, mcpoints, dipole_model="MV",
                  sqrt_s=SQRT_S_HERA, **limits):
    """Finite-energy constrained cross section (4D) with the MV or GBW dipole."""
    lim = _limits(limits)
    out = {}
    for pol in ("L", "T"):
        cs = FE_CrossSection_LO(Q=np.sqrt(Q2), xB=xB, mf=mf, Zf=Zf, sigma0=sigma0, Qs0=Qs0,
                                gamma=gamma, ec=ec, mcpoints=mcpoints, polarization=pol,
                                dipole_model=dipole_model)
        out[pol] = cs.cross_section(lim["r_min"], lim["r_max"], lim["z_min"], lim["z_max"], *THETA)
    return structure_functions(Q2, xB, *out["L"], *out["T"], sqrt_s)


# ---------------------------------------------------------------------------
# rcBK-evolved dipole
# ---------------------------------------------------------------------------

def sigma_r_OT_BK(Q2, xB, mf, Zf, sigma0, bkfile, mcpoints, sqrt_s=SQRT_S_HERA, **limits):
    """Optical-theorem cross section with the BK dipole at Y = ln(x0/xB)."""
    lim = _limits(limits)
    cs = OT_CrossSection_BK(Q=np.sqrt(Q2), xB=xB, mf=mf, Zf=Zf, sigma0=sigma0, bkfile=bkfile,
                            mcpoints=mcpoints, **_bk_parameters(bkfile))
    sL, eL, sT, eT = cs.BK_OT_cross_section(lim["r_min"], lim["r_max"], lim["z_min"], lim["z_max"])
    return structure_functions(Q2, xB, sL, eL, sT, eT, sqrt_s)


def sigma_r_FE_BK_4D(Q2, xB, mf, Zf, sigma0, bkfile, mcpoints, largeNc=False,
                     sqrt_s=SQRT_S_HERA, **limits):
    """Finite-energy constrained cross section (4D), BK dipole and quadrupole at Y = ln(x0/xB)."""
    lim = _limits(limits)
    params = _bk_parameters(bkfile)
    out = {}
    for pol in ("L", "T"):
        cs = FE_CrossSection_BK_4D(Q=np.sqrt(Q2), xB=xB, mf=mf, Zf=Zf, sigma0=sigma0, bkfile=bkfile,
                                   mcpoints=mcpoints, polarization=pol, largeNc=largeNc, **params)
        out[pol] = cs.BK_FE_cross_section_4D(lim["r_min"], lim["r_max"], lim["z_min"], lim["z_max"], *THETA)
    return structure_functions(Q2, xB, *out["L"], *out["T"], sqrt_s)


def sigma_r_FE_BK_5D(Q2, xB, mf, Zf, sigma0, bkfile, mcpoints, extrapolation="frozen",
                     falloff_n=None, largeNc=False, sqrt_s=SQRT_S_HERA, **limits):
    """Finite-energy constrained cross section (5D), the dipoles read at x_P = xB (1 + M^2/Q^2).

    x_P exceeds the x0 of the BK solution for M^2 > Q^2 (x0/xB - 1), and
    `extrapolation` sets what is assumed there: "frozen" (the initial condition),
    "exponential", "falloff" (with `falloff_n`) or "truncated" (that region left
    out); see FE_CrossSection_BK_5D. With "truncated", every point with xB above
    about x0 has no region left and comes out exactly zero.
    """
    if extrapolation not in EXTRAPOLATIONS:
        raise ValueError(f"extrapolation must be one of {EXTRAPOLATIONS}, got {extrapolation!r}")
    lim = _limits(limits)
    params = _bk_parameters(bkfile)
    Msq_min, Msq_max = 4 * mf**2, Q2 * (1 / xB - 1)
    out = {}
    for pol in ("L", "T"):
        cs = FE_CrossSection_BK_5D(Q=np.sqrt(Q2), xB=xB, mf=mf, Zf=Zf, sigma0=sigma0, bkfile=bkfile,
                                   mcpoints=mcpoints, polarization=pol, largeNc=largeNc,
                                   extrapolation=extrapolation, falloff_n=falloff_n, **params)
        out[pol] = cs.BK_FE_cross_section_5D(lim["r_min"], lim["r_max"], lim["z_min"], lim["z_max"],
                                             *THETA, Msq_min, Msq_max)
    return structure_functions(Q2, xB, *out["L"], *out["T"], sqrt_s)
