### Contains a gaussian approximation to the quadrupole in the MV model ###

import numpy as np
from physicslib.constants import Nc, LambdaQCD, CF


class GaussianQuadrupole:
    def __init__(self, dipole_model, Nc=Nc, LambdaQCD=LambdaQCD, CF=CF):
        self.Nc = Nc
        self.LambdaQCD = LambdaQCD
        self.CF = CF
        self.dipole = dipole_model

    def FNc_quadrupole(self, x1, x2, x2p, x1p):

        CF = self.CF
        #Always ensure broadcasting works
        x1  = np.array(x1)
        x2  = np.array(x2)
        x1p = np.array(x1p)               # <---- THIS IS OK
        x2p = np.array(x2p)

        f = self.dipole.exponent
        
        # Functions that appear in quadrupole in terms of dipole exponential
        def F(x1,x2,x2p,x1p):
            return (1/CF)*(f(x1,x2p) 
                + f(x2,x1p) 
                - f(x1,x1p) 
                - f(x2,x2p))              # <----- THIS IS OK
        
        F1 = F(x1,x2p,x2,x1p)
        F2 = F(x1,x2,x2p,x1p) 
        F3 = F(x1,x1p,x2p,x2)                # <----- THIS IS OK

        # Shared dipole S(r) factors
        SuSup = np.exp(f(x1, x2) + f(x2p, x1p)) # <---- THIS IS OK
    

        # Discriminant (avoid tiny negative numerical noise)
        Delta = F1**2 + (4 / Nc**2) * F2 * F3
        sqrt_Delta = np.sqrt(Delta)             # <----- THIS IS OK


        # Avoid 0/0
        good = sqrt_Delta >0
        term1 = np.zeros_like(sqrt_Delta)
        term2 = np.zeros_like(sqrt_Delta)

        # Compute the two terms only where valid. A 1/Nc**2 factor can be introduced here if one is thinking of a dipole-dipole correlator as 
        term1[good] = ((sqrt_Delta[good] + F1[good]) / (2 * sqrt_Delta[good]) - F2[good] / sqrt_Delta[good])* np.exp( Nc * sqrt_Delta[good] / 4) 

        term2[good] = ((sqrt_Delta[good] - F1[good]) / (2 * sqrt_Delta[good]) + F2[good] / sqrt_Delta[good])* np.exp(-Nc * sqrt_Delta[good] / 4)    

        BigFactor = term1 + term2       # <----- THIS IS OK

        # Final finite-Nc quadrupole expression
        return SuSup * BigFactor * np.exp((-Nc/4)*F1 + (1/(2*Nc))*F2)
    
    def LNc_quadrupole(self, x1, x2, x2p, x1p):

        CF = self.CF

        
        #Always ensure broadcasting works
        x1  = np.array(x1)
        x2  = np.array(x2)
        x1p = np.array(x1p)               # <---- THIS IS OK
        x2p = np.array(x2p)

        f = self.dipole.exponent
        
        # Functions that appear in quadrupole in terms of dipole exponential
        def F(x1,x2,x2p,x1p):
            return (1/CF)*(f(x1,x2p) 
                + f(x2,x1p) 
                - f(x1,x1p) 
                - f(x2,x2p))              # <----- THIS IS OK
        
        F1 = F(x1,x2p,x2,x1p)
        F2 = F(x1,x2,x2p,x1p) 

        # Shared dipole S(u) factors
        SuSup = np.exp(f(x1, x2) + f(x2p, x1p)) # <---- THIS IS OK
        
        Su_mixed = np.exp(f(x1, x1p) + f(x2p, x2))

        return SuSup - (F2 / (F1 + 1e-12)) * (SuSup - Su_mixed)
    
    def quadrupole(self, x1, x2, x2p, x1p, largeNc=False):
        """
        Gaussian quadrupole correlator.

        Parameters
        ----------
        largeNc : bool
            If True, use large-Nc approximation.
        """
        if largeNc:
            return self.LNc_quadrupole(x1, x2, x2p, x1p)
        return self.FNc_quadrupole(x1, x2, x2p, x1p)

    
    def quadrupole_polar(self, u, up, z, theta, largeNc = False):
        x1  = np.stack([(1 - z) * u, np.zeros_like(u)], axis=-1)
        x2  = np.stack([-z * u, np.zeros_like(u)], axis=-1)
        x1p = np.stack([(1 - z) * up * np.cos(theta), (1 - z) * up * np.sin(theta)], axis=-1)
        x2p = np.stack([-z * up * np.cos(theta), -z * up * np.sin(theta)], axis=-1)
        
        return self.quadrupole(x1, x2, x2p, x1p, largeNc = largeNc)