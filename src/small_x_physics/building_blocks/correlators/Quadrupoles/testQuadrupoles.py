import numpy as np

from small_x_physics.building_blocks.correlators.Dipoles.BK_dipole import BKDipole
from small_x_physics.building_blocks.correlators.Dipoles.IC_dipole import ICDipole
from small_x_physics.building_blocks.correlators.Quadrupoles.QuadrupoleCorrelator import QuadrupoleCorrelatorModel

import matplotlib.pyplot as plt

r = np.linspace(0.01, 100.0, 1000)  # GeV^-1
x = np.array([0.0, 0.0])
y = np.stack([r, np.zeros_like(r)], axis=1) 
Y = 0.0

bkdipole = BKDipole("/home/ermabert/Academia/Research/Physics_code_library/src/small_x_physics/building_blocks/correlators/Quadrupoles/mv.dat")

# print("min BK_S2 =", np.min(BK_S2(x, y)))
# print("max BK_S2 =", np.max(BK_S2(x, y)))

icdipole = ICDipole(Qs0=np.sqrt(0.104), gamma=1.0, ec=1.0, LambdaQCD=0.241)
IC_S2 = lambda x, y: icdipole.MV_model_S2(x, y)

quad_model_ic = QuadrupoleCorrelatorModel(Nc=3, LambdaQCD=0.241, dipole_model=IC_S2)
quad_S = quad_model_ic.quadrupole_polar(r, r, 1/2, np.pi)

for _ in range(1000):
    quad_model_bk = QuadrupoleCorrelatorModel(Nc=3, LambdaQCD=0.241, dipole_model=bkdipole.BK_evolved_MV_model_S2_Y)
    quad_S_bk = quad_model_bk.quadrupole_polar(r, r, 1/2, np.pi/2, dipole_args={"Y": Y})

# plt.figure(figsize=(8, 5))
# # plt.plot(r, BK_S2, label='BK-evolved S2', color='blue')
# # plt.plot(r, IC_S2, label='IC S2', color='red', linestyle='--')
# plt.plot(r, quad_S, label='Quadrupole S', color='green', linestyle='-.')
# plt.plot(r, quad_S_bk, label='Quadrupole S (BK)', color='purple', linestyle=':')
# plt.xlabel('r (GeV$^{-1}$)', fontsize=12)
# plt.ylabel('S', fontsize=12)
# plt.title('Dipole and Quadrupole Correlators', fontsize=14)
# plt.xscale('log')   
# plt.grid(True, which='both', linestyle='--', alpha=0.5)
# plt.legend(fontsize=11)
# plt.tight_layout()
# plt.show()

