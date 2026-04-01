import numpy as np
from scipy.special import jv

# NOTE: This file is part of the physicslib package. If you see stale
# behavior at runtime, delete the corresponding `__pycache__` entry.


class LODISIntegrand5D:
    def __init__(
        self,
        quark_masses,           # quark mass parameter, flavor dependent
        photon_wf,              # LO_FE_PhotonWF_squared instance
        sigma0,                 # parameter for transverse area of target
        dipole_model,           # model for dipole
        quadrupole_model,       # model for quadrupole
        Q,                      # Photon virtuality (GeV), REQUIRED
        xB,                     # Bjorken-x value, REQUIRED (enables kinematic bounds)
        polarization="T",       # Polarization of photon "T", "L", or "TL"
        largeNc=False,          # Set true if using large Nc model
    ):
        """
        Initialize the 5D finite-energy DIS integrand with kinematic bounds.
        
        Parameters
        ----------
        quark_masses : array-like
            Quark mass parameter, flavor dependent.
        photon_wf : LO_FE_PhotonWF_squared instance
            Photon wavefunction squared object.
        sigma0 : float
            Parameter for transverse area of target.
        dipole_model : object
            Model for dipole cross section.
        quadrupole_model : object
            Model for quadrupole interaction.
        Q : float
            Photon virtuality (GeV). REQUIRED - used for kinematic bound calculation.
        xB : float
            Bjorken-x value. REQUIRED - enables z-dependent kinematic bounds.
            Enforces: Msq_qq <= W²·z·(1-z) where W² = Q²(1/xB - 1).
        polarization : str, default "T"
            Photon polarization: "T" (transverse), "L" (longitudinal), or "TL" (both).
        largeNc : bool, default False
            If True, use large-Nc quadrupole approximation.
        """
        self.mf = quark_masses
        self.wf = photon_wf
        self.dipole = dipole_model
        self.quadrupole = quadrupole_model
        self.pol = polarization
        self.largeNc = largeNc
        self.sigma0 = sigma0
        self.Q = Q
        self.xB = xB

    # Define the type of target interaction (i.e. the optical theorem or the finite energy)
    def FE_target_interaction_polar(self, u, up, z, theta):
        S2 = self.dipole.S(u)
        S2prime = self.dipole.S(up)
        S4 = self.quadrupole(u, up, z, theta)
        return 1 - S2 - S2prime + S4

    def relative_momentum_integral(self, u, up, z, theta, Msq_qqtilde, flavor):
        """
        Relative momentum integral over P. Has been transformed so that integral bounds are independent of z.
        """
        mf = self.mf[flavor]
        # Vectorize support: u, up, z, theta may be scalars or arrays. Use
        # numpy broadcasting so this function can return an array of values
        # when called with vector inputs.
        u_a, up_a, theta_a = np.broadcast_arrays(u, up, theta)

        arg = Msq_qqtilde - mf ** 2
        r2 = u_a ** 2 + up_a ** 2 - 2.0 * u_a * up_a * np.cos(theta_a)

        valid = (arg > 0) & (r2 > 0)

        I_P = np.zeros_like(arg, dtype=float)
        if np.any(valid):
            zeta = np.sqrt(arg[valid] * r2[valid])
            # J1 Bessel kernel for the k-integral (scipy.special.jv supports arrays)
            I_P_valid = jv(0, zeta) / (4.0 * np.pi )
            I_P[valid] = I_P_valid

        # If inputs were scalars, return a scalar
        if I_P.shape == ():
            return float(I_P) 
        return I_P

    def FE_integrand(self, Q, Msq_qq, u, up, z, theta, flavor=0):
        """
        Finite-energy (FE) integrand with z-dependent kinematic bounds.

        Enforces the kinematic constraint: Msq_qq <= W²·z·(1-z)
        where W² = Q²(1/xB - 1). Returns 0 if constraint is violated,
        ensuring physically consistent results.
        
        Parameters
        ----------
        Q : float
            Photon virtuality (GeV).
        Msq_qq : float or array
            Invariant mass squared of the quark-antiquark pair.
        u, up, z, theta : float or array
            Integration variables.
        flavor : int, default 0
            Quark flavor index.
        
        Returns
        -------
        float or array
            Integrand value, or 0 if kinematic constraint is violated.
        """
        W_squared = self.Q**2 * (1.0/self.xB - 1.0)
        Msq_max = W_squared * z * (1 - z)
        
        # Return zero if outside kinematic bounds
        if np.any(Msq_qq > Msq_max):
            # Handle both scalar and array cases
            out = np.asarray(Msq_qq > Msq_max, dtype=float)
            return np.where(out, 0.0, self._FE_integrand_impl(Q, Msq_qq, u, up, z, theta, flavor))
        
        return self._FE_integrand_impl(Q, Msq_qq, u, up, z, theta, flavor)
    
    def _FE_integrand_impl(self, Q, Msq_qq, u, up, z, theta, flavor=0):
        """
        Implementation of the finite-energy integrand (internal method).
        """
        if self.pol in ("T", "TL"):
            psi_T_sq = self.wf.psi_T_squared(Q, u, up, z, theta, flavor)
        else:
            psi_T_sq = 0.0

        if self.pol in ("L", "TL"):
            psi_L_sq = self.wf.psi_L_squared(Q, u, up, z, theta, flavor)
        else:
            psi_L_sq = 0.0

        target_interaction = self.FE_target_interaction_polar(u, up, z, theta)

        I_P = self.relative_momentum_integral(u, up, z, theta, Msq_qq, flavor)

        # --- Normalization and Jacobian ---
        NormFactor = 1 / (4 * np.pi) # (In paper absorbed into wavefunction squared, but we keep it here for clarity)
        Jac = ((u * up) / (z * (1 - z))) * 2 * np.pi

        return (
            self.sigma0/2.0
            * NormFactor
            * Jac
            * (psi_T_sq + psi_L_sq)
            * target_interaction
            * I_P
        )