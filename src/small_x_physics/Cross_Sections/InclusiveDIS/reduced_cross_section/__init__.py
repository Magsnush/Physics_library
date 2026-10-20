"""The DIS reduced cross section sigma_r, with F_L, F_T and F_2, for every LO cross section.

    from small_x_physics.Cross_Sections.InclusiveDIS.reduced_cross_section import sigma_r_FE_BK_5D

    sf = sigma_r_FE_BK_5D(Q2=8.5, xB=2e-4, mf=0.14, Zf=(2/3)**0.5, sigma0=96.68,
                          bkfile="data/mv.dat", mcpoints=1_000_000,
                          extrapolation="falloff", falloff_n=5)
    sf.sigma_r, sf.sigma_r_err, sf.F2, sf.FL, ...

See sigma_r.py for the conventions.
"""

from .sigma_r import (
    EXTRAPOLATIONS,
    SQRT_S_HERA,
    StructureFunctions,
    sigma_r_FE_BK_4D,
    sigma_r_FE_BK_5D,
    sigma_r_FE_LO,
    sigma_r_OT_BK,
    sigma_r_OT_LO,
    structure_functions,
)

__all__ = [
    "EXTRAPOLATIONS",
    "SQRT_S_HERA",
    "StructureFunctions",
    "sigma_r_FE_BK_4D",
    "sigma_r_FE_BK_5D",
    "sigma_r_FE_LO",
    "sigma_r_OT_BK",
    "sigma_r_OT_LO",
    "structure_functions",
]
