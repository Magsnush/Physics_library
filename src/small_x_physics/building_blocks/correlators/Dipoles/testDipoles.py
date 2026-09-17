from small_x_physics.building_blocks.correlators.Dipoles.BK_dipole import BKDipole
from small_x_physics.building_blocks.correlators.Dipoles.IC_dipole import ICDipole

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Bundled test data, resolved relative to the repository root so this runs from
# any working directory and on any machine.
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
MV_FILE = REPO_ROOT / "data" / "mv.dat"

r = np.linspace(0.01, 100.0, 1000)  # GeV^-1
x = np.array([0.0, 0.0])
y = np.stack([r, np.zeros_like(r)], axis=1)

bkdipole = BKDipole(MV_FILE, Y=4.0)
for _ in range(10000):
    BK_S2 = bkdipole.BK_evolved_MV_model_S2(x, y)

icdipole = ICDipole(Qs0=np.sqrt(0.104), gamma=1.0, ec=1.0, LambdaQCD=0.241)
IC_MV = icdipole.MV_model_S2(x=x, y=y)
IC_GBW = icdipole.GBW_model_S2(x=x, y=y)

# print(type(BK_S2), np.shape(BK_S2))
# print(type(IC_S2), np.shape(IC_S2))

# print(f"BK-evolved S2: {BK_S2}")
# print(f"IC S2: {IC_S2}")

plt.figure(figsize=(8, 5))
plt.plot(r, BK_S2, label='BK-evolved S2', color='blue')
plt.plot(r, IC_MV, label='IC MV S2', color='red', linestyle='--')
plt.plot(r, IC_GBW, label='IC GBW S2', color='green', linestyle='-.')
plt.xlabel('r (GeV$^{-1}$)', fontsize=12)
plt.ylabel('S2', fontsize=12)
plt.title('Dipole S2 Comparison', fontsize=14)
plt.xscale('log')
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()