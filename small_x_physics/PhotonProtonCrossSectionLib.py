import numpy as np
from scipy.special import kv, jv, hyp0f1  #Modified Bessel function of the second kind and hypergeometric function
from .wavefunctions.WaveFunctions import OT_photon_proton_wave_function_sq, KC_photon_proton_wave_function_sq
from .target_models.Target_models_lib import dipole, quadrupole_uu, dipole_1d, WW_distribution, Dipole_distribution


######### UNPOLARIZED LIGHTCONE WAVEFUNCTIONS #########

### Longitudinal case ###

# Square of the standard LCWF (convention from Kovchegov and Levin)

class LOPhotonProtonCrossSection:
    def __init__(self, 
                 Q, 
                 m, 
                 Zf, 
                 Nc, 
                 Qs0, 
                 gamma, 
                 LambdaQCD, 
                 ec, 
                 sigma0, 
                 alpha_EM, 
                 alphaS, 
                 r_max):
        """
        Lightcone Wavefunction Parameters (GeV):
        Q : photon virtuality 
        m : quark mass
        alphaEM : Fine structure
        Zf : Quark charge for flavor f
        Nc : Number of colors
        
        Model parameters (GeV):
        Qs0 : Saturation scale: float
        gamma : anomalous dimension: float
        LambdaQCD : QCD scale : float
        ec : scaling parameter: float
        """
        self.r_max = r_max

        self.Q = Q
        self.m = m
        self.Zf = Zf
        self.Nc = Nc 

        self.Qs0 = Qs0
        self.gamma = gamma
        self.LambdaQCD = LambdaQCD  
        self.ec = ec
        self.sigma0 = sigma0
        self.alpha_EM = alpha_EM
        self.alphaS = alphaS

        self.WW = WW_distribution(
            sigma0=self.sigma0,
            alphaS=self.alphaS,
            r_max=self.r_max
        )
    
    # Define differential cross section to be numerically integrated over

    # Optical theorem
    def OT_integrand(self, r, z, polarization):
        
        OTWavefunctionSq = OT_photon_proton_wave_function_sq(r, z, polarization, self.Q, self.m, self.Zf, self.Nc)

        DipoleAmp = 2 * (1 - dipole(r, 0))

        Jac = 2 * np.pi * r / (z*(1-z))

        return (self.sigma0/2)*1/(4*np.pi) * Jac * OTWavefunctionSq * DipoleAmp
    
    
    # Finite energy constrained. Eq. (24) in https://arxiv.org/abs/2601.07302

    def KC_HypGeom_integrand(self, u, up, z, alpha, Msq_up, polarization, largeNc=False):
        """
        Unified kinematically constrained photon-proton integrand.
        
        Parameters
        ----------
        largeNc : bool
            If True → use Large-Nc quadrupole approximation
            If False → use full finite-Nc quadrupole
        """

        # --- Hypergeometric factor ---
        a1 = 0.25 * (self.m**2 - Msq_up * z * (1 - z)) * (u**2 + up**2 - 2*u*up*np.cos(alpha))
        hyperSum = (1 / (4*np.pi)) * (hyp0f1(2, a1) * (Msq_up * z * (1 - z) - self.m**2))

        #print(hyperSum)

        # --- Photon wave function ---
        KCWaveFunctionSq = KC_photon_proton_wave_function_sq(
            u, up, z, alpha, polarization, self.Q, self.m, self.Zf, self.Nc
        )

        # # --- Dipole amplitudes ---
        # uDipole  = dipole(np.array([u]),np.zeros_like(u))
        # upDipole = dipole(np.array([up]),np.zeros_like(up))

        uDipole = dipole_1d(u)
        upDipole = dipole_1d(up)


        # --- Quadrupole ---
        Qpole = quadrupole_uu(
            u, up, z, alpha, largeNc
        )

        # --- Target amplitude ---
        TargetAmplitude = 1 - uDipole - upDipole + Qpole
        #print(TargetAmplitude)

        # --- Normalization and Jacobian ---
        NormFactor = 1 / (4 * np.pi)
        Jac = ((u * up) / (z * (1 - z))) * 2 * np.pi

        # --- Final integrand ---
        integrand = NormFactor * self.sigma0 / 2 * Jac * KCWaveFunctionSq * TargetAmplitude * hyperSum

        return integrand

    # Treat quarks as massless. K is the magnitude of the total transverse momentum of the dipole, xi is the 
    # momentum fraction of a quark w.r.t to a gluon and alpha is the angle between K and the x-axis in polar coordinates.

    def WW_distribution_integrand(self, K, xi):

        # Flavor dependent constant
        CL_f = (8/np.pi) * self.Nc * self.alpha_EM * self.Zf**2

        # Gluon distribution
        gluon_distribution = self.WW(K)

        jacobian = 1/(2*np.pi)**2 * K 

        integrand = (1/(4*np.pi)) * (1/self.Q**4) * CL_f * jacobian * gluon_distribution * xi*(1-xi)

        return integrand
    
    def Dipole_distribution_integrand():
        return