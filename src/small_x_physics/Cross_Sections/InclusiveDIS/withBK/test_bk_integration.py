import numpy as np

from small_x_physics.Cross_Sections.InclusiveDIS.withBK.BK_OT_CS import OT_CrossSection_BK
from small_x_physics.Cross_Sections.InclusiveDIS.withBK.BK_FE_4D_evolve_in_xB import FE_CrossSection_BK_4D
from small_x_physics.Cross_Sections.InclusiveDIS.withBK.BK_FE_5D_evolve_in_xP import FE_CrossSection_BK_5D
from small_x_physics.building_blocks.constants import alpha_em

import argparse

parser = argparse.ArgumentParser(description="Compute leading-order DIS cross sections.")
parser.add_argument("--Q", type=float, default=np.sqrt(2.0), help="Photon virtuality")
parser.add_argument("--xB", type=float, default=1e-2, help="Bjorken x")
parser.add_argument("--mf", type=float, default=0.14, help="Quark mass")
parser.add_argument("--Zf", type=float, default=np.sqrt(2/3), help="Quark charge")
parser.add_argument("--sigma0", type=float, default=2*2.57*18.81, help="Reference cross section")
parser.add_argument("--Qs0", type=float, default=np.sqrt(0.104), help="Initial saturation scale")
parser.add_argument("--gamma", type=float, default=1.0, help="Anomalous dimension")
parser.add_argument("--ec", type=float, default=1.0, help="Energy scale")
parser.add_argument("--bkfile", type=str, required=True, help="Path to BK evolution file")
parser.add_argument("--x0", type=float, default=0.01, help="Initial x for evolution")
parser.add_argument("--mcpoints", type=int, default=500000, help="Number of Monte Carlo points for integration")
parser.add_argument("--largeNc", action="store_true", help="Use the large-Nc quadrupole instead of the finite-Nc one (affects the FE rows only; OT has no quadrupole)")

args = parser.parse_args()

r_min = 1e-6
r_max = 10.0
z_min = 1e-8
z_max = 1.0 - z_min
theta_min = 0.0
theta_max = 2*np.pi
Mqq_sq_min = args.mf**2 * 4
Mqq_sq_max = args.Q**2  *(1/args.xB - 1)

# Parameters shared by all three cross sections.
common = dict(Q=args.Q, xB=args.xB, mf=args.mf, Zf=args.Zf, sigma0=args.sigma0,
              Qs0=args.Qs0, gamma=args.gamma, ec=args.ec, bkfile=args.bkfile,
              x0=args.x0, mcpoints=args.mcpoints)

# sigma -> structure function
prefactor = args.Q**2 / (4 * np.pi**2 * alpha_em)

rows = []


def record(calc, sigma_L, sigma_L_err, sigma_T, sigma_T_err):
    """Convert a pair of cross sections into structure functions and store a CSV row."""
    FL = prefactor * sigma_L
    FL_err = prefactor * sigma_L_err
    FT = prefactor * sigma_T
    FT_err = prefactor * sigma_T_err
    F2 = FL + FT
    F2_err = np.sqrt(FL_err**2 + FT_err**2)
    rows.append(
        f"{calc}, {args.Q**2}, {args.xB}, {args.mf}, {int(args.largeNc)}, {args.mcpoints}, "
        f"{FL}, {FL_err}, {FT}, {FT_err}, {F2}, {F2_err}"
    )


# --- Optical theorem -------------------------------------------------------
# No quadrupole enters here, so the large-Nc flag has no effect on this row.
OT_cross_section = OT_CrossSection_BK(**common)

OT_CS_L, OT_CS_L_err, OT_CS_T, OT_CS_T_err = OT_cross_section.BK_OT_cross_section(
    r_min,
    r_max,
    z_min,
    z_max,
)

record("OT", OT_CS_L, OT_CS_L_err, OT_CS_T, OT_CS_T_err)


# --- Finite-energy constrained, 4D, BK evolved in xB -----------------------
FE_4D = {}
for pol in ("L", "T"):
    cross_section = FE_CrossSection_BK_4D(polarization=pol, largeNc=args.largeNc, **common)
    FE_4D[pol] = cross_section.BK_FE_cross_section_4D(
        r_min,
        r_max,
        z_min,
        z_max,
        theta_min,
        theta_max,
    )

record("FE_4D", FE_4D["L"][0], FE_4D["L"][1], FE_4D["T"][0], FE_4D["T"][1])


# --- Finite-energy constrained, 5D, BK evolved in xP -----------------------
FE_5D = {}
for pol in ("L", "T"):
    cross_section = FE_CrossSection_BK_5D(polarization=pol, largeNc=args.largeNc, **common)
    FE_5D[pol] = cross_section.BK_FE_cross_section_5D(
        r_min,
        r_max,
        z_min,
        z_max,
        theta_min,
        theta_max,
        Mqq_sq_min,
        Mqq_sq_max,
    )

record("FE_5D", FE_5D["L"][0], FE_5D["L"][1], FE_5D["T"][0], FE_5D["T"][1])


print("calc, Q2, xB, m, largeNc, mcpoints, FL, FL_err, FT, FT_err, F2, F2_err")
for row in rows:
    print(row)
