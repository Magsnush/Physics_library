import numpy as np

from small_x_physics.Cross_Sections.InclusiveDIS.LO.LO_OT_CS import OT_CrossSection_LO
from small_x_physics.building_blocks.constants import alpha_em

import argparse

parser = argparse.ArgumentParser(description="Compute leading-order DIS cross sections.")
parser.add_argument("--Q", type=float, default=np.sqrt(45), help="Photon virtuality")
parser.add_argument("--xB", type=float, default=0.01, help="Bjorken x")
parser.add_argument("--mf", type=float, default=0.14, help="Quark mass")
parser.add_argument("--Zf", type=float, default=np.sqrt(2/3), help="Quark charge")
parser.add_argument("--sigma0", type=float, default=2*2.57*18.81, help="Reference cross section")
parser.add_argument("--Qs0", type=float, default=np.sqrt(0.104), help="Initial saturation scale")
parser.add_argument("--gamma", type=float, default=1.0, help="Anomalous dimension")
parser.add_argument("--ec", type=float, default=1.0, help="Energy scale")

args = parser.parse_args()

OT_cross_section = OT_CrossSection_LO(Q=args.Q, mf=args.mf, Zf=args.Zf, sigma0=args.sigma0, Qs0=args.Qs0, gamma=args.gamma, ec=args.ec)

r_min = 1e-6
r_max = 10.0
z_min = 1e-8
z_max = 1.0 - z_min

theta_min = 0.0
theta_max = 2 * np.pi

OT_long_integrand, OT_long_err, OT_trans_integrand, OT_trans_err = OT_cross_section.OT_cross_section(r_min, r_max, z_min, z_max)

OT_FL = args.Q**2 / (4 * np.pi**2 * alpha_em) * OT_long_integrand
OT_FL_err = args.Q**2 / (4 * np.pi**2 * alpha_em) * OT_long_err

OT_FT = args.Q**2 / (4 * np.pi**2 * alpha_em) * OT_trans_integrand
OT_FT_err = args.Q**2 / (4 * np.pi**2 * alpha_em) * OT_trans_err

OT_F2 = OT_FL + OT_FT
OT_F2_err = np.sqrt(OT_FL_err**2 + OT_FT_err**2)

print(f"F2 = {OT_F2:.6e} ± {OT_F2_err:.6e}")

from small_x_physics.Cross_Sections.InclusiveDIS.LO.LO_FE_CS import FE_CrossSection_LO

FE_CrossSection = FE_CrossSection_LO(Q=args.Q, xB=args.xB, mf=args.mf, Zf=args.Zf, sigma0=args.sigma0, Qs0=args.Qs0, gamma=args.gamma, ec=args.ec)

CS_L, CS_L_err, CS_T, CS_T_err = (
    FE_CrossSection.compute_cross_section_FE(
        r_min,
        r_max,
        z_min,
        z_max,
        theta_min,
        theta_max,
    )
)

FE_L = args.Q**2 / (4 * np.pi**2 * alpha_em) * CS_L
FE_L_err = args.Q**2 / (4 * np.pi**2 * alpha_em) * CS_L_err

FE_T = args.Q**2 / (4 * np.pi**2 * alpha_em) * CS_T
FE_T_err = args.Q**2 / (4 * np.pi**2 * alpha_em) * CS_T_err

FE_F2 =  (FE_L + FE_T)
FE_F2_err = np.sqrt(FE_L_err**2 + FE_T_err**2)

print(f"F2 = {FE_F2:.6e} ± {FE_F2_err:.6e}")
