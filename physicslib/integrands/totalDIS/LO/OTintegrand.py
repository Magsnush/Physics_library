import numpy as np

class OTIntegrand:
    def __init__(
        self,
        quark_masses,           # quark mass parameter, flavor dependent
        photon_wf,              # LO_FE_PhotonWF_squared instance
        sigma0,                 # parameter for transverse area of target
        dipole_model,           # model for dipole
        polarization="T",       # Polarization of photon "T", "L", or "TL"
    ):
        self.mf = quark_masses
        self.wf = photon_wf
        self.dipole = dipole_model
        self.pol = polarization
        self.sigma0 = sigma0

    def OT_integrand(self, r, Q, z, flavor):

        # signature: r is the quadrature variable; Q, z, flavor are passed via
        # quad(..., args=(Q, z, flavor)). The photon wavefunction expects
        # (Q, r, z, flavor).
        OTWavefunctionSq = self.wf(Q, r, z, flavor)

        DipoleAmp = 2 * (1 - self.dipole(r, 0))

        Jac = 2 * np.pi * r / (z * (1 - z))

        return (self.sigma0 / 2) * 1 / (4 * np.pi) * Jac * OTWavefunctionSq * DipoleAmp