from tracemalloc import start
import numpy as np

from small_x_physics.Cross_Sections.InclusiveDIS.withBK.BK_OT_CS import OT_CrossSection_BK 
from small_x_physics.Cross_Sections.InclusiveDIS.withBK.BK_FE_4D_evolve_in_xB import FE_CrossSection_BK_4D
from small_x_physics.Cross_Sections.InclusiveDIS.withBK.BK_FE_5D_evolve_in_xP import FE_CrossSection_BK_5D
from small_x_physics.building_blocks.constants import alpha_em
import time

import argparse

parser = argparse.ArgumentParser(description="Compute leading-order DIS cross sections.")
parser.add_argument("--Q", type=float, default=np.sqrt(45), help="Photon virtuality")
parser.add_argument("--xB", type=float, default=1e-2, help="Bjorken x")
parser.add_argument("--mf", type=float, default=0.14, help="Quark mass")
parser.add_argument("--Zf", type=float, default=np.sqrt(2/3), help="Quark charge")
parser.add_argument("--sigma0", type=float, default=2*2.57*18.81, help="Reference cross section")
parser.add_argument("--Qs0", type=float, default=np.sqrt(0.104), help="Initial saturation scale")
parser.add_argument("--gamma", type=float, default=1.0, help="Anomalous dimension")
parser.add_argument("--ec", type=float, default=1.0, help="Energy scale")
parser.add_argument("--bkfile", type=str, default=None, help="Path to BK evolution file")
parser.add_argument("--x0", type=float, default=0.01, help="Initial x for evolution")
parser.add_argument("--mcpoints", type=int, default=10000, help="Number of Monte Carlo points for integration")

args = parser.parse_args()

r_min = 1e-6
r_max = 10.0
z_min = 1e-8
z_max = 1.0 - z_min
theta_min = 0.0
theta_max = 2*np.pi
Mqq_sq_min = args.mf**2
Mqq_sq_max = (args.Q**2 / 4) *(1/args.xB - 1)

start_time_1 = time.time()

# OT_cross_section = OT_CrossSection_BK(Q=args.Q, xB=args.xB, mf=args.mf, Zf=args.Zf, sigma0=args.sigma0, Qs0=args.Qs0, gamma=args.gamma, ec=args.ec, bkfile=args.bkfile, x0=args.x0, mcpoints=args.mcpoints)

# OT_CS_L, OT_CS_L_err, OT_CS_T, OT_CS_T_err = (
#     OT_cross_section.BK_OT_cross_section(
#         r_min,
#         r_max,
#         z_min,
#         z_max,
#     )
# )

# OT_FL = args.Q**2 / (4 * np.pi**2 * alpha_em) * OT_CS_L
# OT_FL_err = args.Q**2 / (4 * np.pi**2 * alpha_em) * OT_CS_L_err

# OT_FT = args.Q**2 / (4 * np.pi**2 * alpha_em) * OT_CS_T
# OT_FT_err = args.Q**2 / (4 * np.pi**2 * alpha_em) * OT_CS_T_err

# OT_F2 =  (OT_FL + OT_FT)
# OT_F2_err = np.sqrt(OT_FL_err**2 + OT_FT_err**2)

# print(f"F2 = {OT_F2:.6e} ± {OT_F2_err:.6e}")

FE_cross_section_4D_L = FE_CrossSection_BK_4D(Q=args.Q, xB=args.xB, mf=args.mf, Zf=args.Zf, sigma0=args.sigma0, Qs0=args.Qs0, gamma=args.gamma, ec=args.ec, bkfile=args.bkfile, x0=args.x0, mcpoints=args.mcpoints, polarization="L")

FE_CS_L_4D, FE_CS_L_err_4D = (
    FE_cross_section_4D_L.BK_FE_cross_section_4D(
        r_min,
        r_max,
        z_min,
        z_max,
        theta_min,
        theta_max,
    )
)

FE_cross_section_4D_T = FE_CrossSection_BK_4D(Q=args.Q, xB=args.xB, mf=args.mf, Zf=args.Zf, sigma0=args.sigma0, Qs0=args.Qs0, gamma=args.gamma, ec=args.ec, bkfile=args.bkfile, x0=args.x0, mcpoints=args.mcpoints, polarization="T")

FE_CS_T_4D, FE_CS_T_err_4D = (
    FE_cross_section_4D_T.BK_FE_cross_section_4D(
        r_min,
        r_max,
        z_min,
        z_max,
        theta_min,
        theta_max,
    )
)

FE_FL_4D = args.Q**2 / (4 * np.pi**2 * alpha_em) * FE_CS_L_4D
FE_FL_err_4D = args.Q**2 / (4 * np.pi**2 * alpha_em) * FE_CS_L_err_4D

FE_FT_4D = args.Q**2 / (4 * np.pi**2 * alpha_em) * FE_CS_T_4D
FE_FT_err_4D = args.Q**2 / (4 * np.pi**2 * alpha_em) * FE_CS_T_err_4D

FE_F2_4D =  (FE_FL_4D + FE_FT_4D)
FE_F2_err_4D = np.sqrt(FE_FL_err_4D**2 + FE_FT_err_4D**2)

end_time_1 = time.time()

print(f"F2 = {FE_F2_4D:.6e} ± {FE_F2_err_4D:.6e}")
print(f"Computation time: {end_time_1 - start_time_1:.2f} seconds")

start_time_2 = time.time()

FE_cross_section_5D_L = FE_CrossSection_BK_5D(Q=args.Q, xB=args.xB, mf=args.mf, Zf=args.Zf, sigma0=args.sigma0, Qs0=args.Qs0, gamma=args.gamma, ec=args.ec, bkfile=args.bkfile, x0=args.x0, mcpoints=args.mcpoints, polarization="L")

FE_CS_L_5D, FE_CS_L_err_5D = (
    FE_cross_section_5D_L.BK_FE_cross_section_5D(
        r_min,
        r_max,
        z_min,
        z_max,
        theta_min,
        theta_max,
        Mqq_sq_min,
        Mqq_sq_max,
    )
)

FE_cross_section_5D_T = FE_CrossSection_BK_5D(Q=args.Q, xB=args.xB, mf=args.mf, Zf=args.Zf, sigma0=args.sigma0, Qs0=args.Qs0, gamma=args.gamma, ec=args.ec, bkfile=args.bkfile, x0=args.x0, mcpoints=args.mcpoints, polarization="T")

FE_CS_T_5D, FE_CS_T_err_5D = (
    FE_cross_section_5D_T.BK_FE_cross_section_5D(
        r_min,
        r_max,
        z_min,
        z_max,
        theta_min,
        theta_max,
        Mqq_sq_min,
        Mqq_sq_max,
    )
)

FE_FL_5D = args.Q**2 / (4 * np.pi**2 * alpha_em) * FE_CS_L_5D
FE_FL_err_5D = args.Q**2 / (4 * np.pi**2 * alpha_em) * FE_CS_L_err_5D

FE_FT_5D = args.Q**2 / (4 * np.pi**2 * alpha_em) * FE_CS_T_5D
FE_FT_err_5D = args.Q**2 / (4 * np.pi**2 * alpha_em) * FE_CS_T_err_5D

FE_F2_5D =  (FE_FL_5D + FE_FT_5D)
FE_F2_err_5D = np.sqrt(FE_FL_err_5D**2 + FE_FT_err_5D**2)

end_time_2 = time.time()

print(f"F2 = {FE_F2_5D:.6e} ± {FE_F2_err_5D:.6e}")
print(f"Computation time: {end_time_2 - start_time_2:.2f} seconds")

