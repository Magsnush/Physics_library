import numpy as np

from small_x_physics.building_blocks.correlators.Dipoles.BK_dipole import BKDipole
from small_x_physics.building_blocks.correlators.Dipoles.IC_dipole import ICDipole
from small_x_physics.building_blocks.correlators.Quadrupoles.QuadrupoleCorrelator import QuadrupoleCorrelatorModel

import matplotlib.pyplot as plt
from pathlib import Path

# Bundled test data, resolved relative to the repository root so this runs from
# any working directory and on any machine.
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
MV_FILE = REPO_ROOT / "data" / "mv.dat"

r = np.linspace(0.01, 100.0, 1000)  # GeV^-1
x = np.array([0.0, 0.0])
y = np.stack([r, np.zeros_like(r)], axis=1)
Y = 0.0

bkdipole = BKDipole(MV_FILE)

# print("min BK_S2 =", np.min(BK_S2(x, y)))
# print("max BK_S2 =", np.max(BK_S2(x, y)))

icdipole = ICDipole(Qs0=np.sqrt(0.104), gamma=1.0, ec=1.0, LambdaQCD=0.241)
IC_MV = lambda x, y: icdipole.MV_model_S2(x, y)
IC_GBW = lambda x, y: icdipole.GBW_model_S2(x, y)

MV_quad_model_ic = QuadrupoleCorrelatorModel(Nc=3, LambdaQCD=0.241, dipole_model=IC_MV)
GBW_quad_model_ic = QuadrupoleCorrelatorModel(Nc=3, LambdaQCD=0.241, dipole_model=IC_GBW)
# MV_quad_S = MV_quad_model_ic.quadrupole_polar(r, 0, 0, np.pi)
# GBW_quad_S = GBW_quad_model_ic.quadrupole_polar(r, 0, 0, np.pi)

# for _ in range(1000):
#     quad_model_bk = QuadrupoleCorrelatorModel(Nc=3, LambdaQCD=0.241, dipole_model=bkdipole.BK_evolved_MV_model_S2_Y)
#     quad_S_bk = quad_model_bk.quadrupole_polar(r, r, 1/2, np.pi/2, dipole_args={"Y": Y})

# plt.figure(figsize=(8, 5))
# # plt.plot(r, BK_S2, label='BK-evolved S2', color='blue')
# # plt.plot(r, IC_S2, label='IC S2', color='red', linestyle='--')
# plt.plot(r, MV_quad_S, label='Quadrupole S (MV)', color='green', linestyle='-.')
# plt.plot(r, GBW_quad_S, label='Quadrupole S (GBW)', color='orange', linestyle='--')
# #plt.plot(r, quad_S_bk, label='Quadrupole S (BK)', color='purple', linestyle=':')
# plt.xlabel('r (GeV$^{-1}$)', fontsize=12)
# plt.ylabel('S', fontsize=12)
# plt.title('Quadrupoles', fontsize=14)
# plt.xscale('log')   
# plt.grid(True, which='both', linestyle='--', alpha=0.5)
# plt.legend(fontsize=11)
# plt.tight_layout()
# plt.show()

angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

z0 = 0.0
zhalf = 0.5
for theta in angles:

    MV_quad_S = MV_quad_model_ic.quadrupole_polar(r, r, zhalf, theta) - MV_quad_model_ic.quadrupole_polar(r, r, z0, theta)
    GBW_quad_S = GBW_quad_model_ic.quadrupole_polar(r, r, zhalf, theta) - GBW_quad_model_ic.quadrupole_polar(r, r, z0, theta)

    axes[0].plot(
        r,
        MV_quad_S,
        label=fr'$\theta={theta/np.pi:.2f}\pi$'
    )

    axes[1].plot(
        r,
        GBW_quad_S,
        label=fr'$\theta={theta/np.pi:.2f}\pi$'
    )

axes[0].set_title(f"MV diff")
axes[1].set_title(f"GBW diff")

for ax in axes:
    ax.set_xscale('log')
    ax.set_xlabel(r'$r$ (GeV$^{-1}$)')
    ax.set_ylabel(r'$S$')
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()

