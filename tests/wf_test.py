from physicslib.wavefunctions.FE_photon_wavefunctions.LO import LO_FE_PhotonWF_squared
from physicslib.wavefunctions.OT_photon_wavefunctions.LO import LO_OT_PhotonWF_squared

fe_wf = LO_FE_PhotonWF_squared(
    quark_masses=[0.14, 0.14, 0.14],
    quark_charges=[2/3, -1/3, 2/3]
)

ot_wf = LO_OT_PhotonWF_squared(
    quark_masses=[0.14, 0.14, 0.14],
    quark_charges=[2/3, -1/3, 2/3]
)

Q = 10.0
u = 0.2
up = 0.2
z = 0.4
theta = 0.0

print(fe_wf.psi_T_squared(Q, u, up, z, theta, flavor=0))
print(ot_wf.psi_T_squared(Q, u, z, flavor=0))
# psiTsquared = psiTsquared