### Contains a gaussian approximation to the quadrupole in the MV model ###

import numpy as np

class QuadrupoleCorrelatorModel:
    def __init__(self, dipole_model, Nc, LambdaQCD):
        self.Nc = Nc
        self.CF = (self.Nc**2 - 1) / (2 * self.Nc)
        self.LambdaQCD = LambdaQCD
        self.dipole = dipole_model

    def log_dipole(self, x, y, dipole_args):
        """Compute the logarithm of the dipole S-matrix, ensuring numerical stability."""
        S2 = self.dipole(x, y, **dipole_args) + 1e-14  # Avoid log(0)
        return np.log(S2)
    
    # Functions that appear in quadrupole in terms of dipole exponential
    def F(self, x1,x2,x2p,x1p, dipole_args):
        """See below eq. B13 of https://journals.aps.org/prd/pdf/10.1103/PhysRevD.83.105005 for the definition of F."""
        return (1/self.CF)*(self.log_dipole(x1,x2p, dipole_args) + self.log_dipole(x2,x1p, dipole_args) - self.log_dipole(x1,x1p, dipole_args) - self.log_dipole(x2,x2p, dipole_args))              # <----- THIS IS OK

    def FNc_quadrupole(self, x1, x2, x2p, x1p, dipole_args):
        """Compute the finite-Nc quadrupole correlator using the formula from Dominguez et al. (2011) https://journals.aps.org/prd/pdf/10.1103/PhysRevD.83.105005, eq. B21."""
        F1 = self.F(x1,x2p,x2,x1p, dipole_args)
        F2 = self.F(x1,x2,x2p,x1p, dipole_args)
        F3 = self.F(x1,x1p,x2p,x2, dipole_args)                # <----- THIS IS OK

        # Shared dipole S(r) factors that enter into expression B21 that forms the gaussian quadrupole.
        SuSup = np.exp(self.log_dipole(x1, x2, dipole_args) + self.log_dipole(x2p, x1p, dipole_args)) # <---- THIS IS OK
    
        # Discriminant (avoid tiny negative numerical noise)
        Delta = F1**2 + (4 / self.Nc**2) * F2 * F3
        sqrt_Delta = np.sqrt(Delta)             # <----- THIS IS OK

        # Avoid 0/0
        good = sqrt_Delta >0
        term1 = np.zeros_like(sqrt_Delta)
        term2 = np.zeros_like(sqrt_Delta)

        # Compute the two terms only where valid. A 1/Nc**2 factor can be introduced here if one is thinking of a dipole-dipole correlator as (eq. B21)
        term1[good] = ((sqrt_Delta[good] + F1[good]) / (2 * sqrt_Delta[good]) - F2[good] / sqrt_Delta[good])* np.exp(self.Nc * sqrt_Delta[good] / 4)
        term2[good] = ((sqrt_Delta[good] - F1[good]) / (2 * sqrt_Delta[good]) + F2[good] / sqrt_Delta[good])* np.exp(-self.Nc * sqrt_Delta[good] / 4)

        BigFactor = term1 + term2       # <----- THIS IS OK

        # Final finite-Nc quadrupole expression
        return SuSup * BigFactor * np.exp((-self.Nc/4)*F1 + (1/(2*self.Nc))*F2)

    def LNc_quadrupole(self, x1, x2, x2p, x1p, dipole_args):
        """Compute the large-Nc limit of the same Dominguez et al. expression.

        This takes dipole_args like FNc_quadrupole does, so it works with a
        rapidity-dependent dipole (e.g. BKDipole.BK_evolved_MV_model_S2_Y) and
        not only with an analytic one.
        """
        F1 = self.F(x1, x2p, x2, x1p, dipole_args)
        F2 = self.F(x1, x2, x2p, x1p, dipole_args)

        # Shared dipole S(r) factors
        SuSup = np.exp(self.log_dipole(x1, x2, dipole_args) + self.log_dipole(x2p, x1p, dipole_args))

        Su_mixed = np.exp(self.log_dipole(x1, x1p, dipole_args) + self.log_dipole(x2p, x2, dipole_args))

        return SuSup - (F2 / (F1 + 1e-12)) * (SuSup - Su_mixed)

    def quadrupole_polar(self, u, up, z, theta, dipole_args = None, largeNc = False):
        """Quadrupole in polar integration variables.

        Parameters
        ----------
        largeNc : bool, default False
            If True use the large-Nc expression, otherwise the finite-Nc one.
        """
        if dipole_args is None:
            dipole_args = {}
        x1  = np.stack([(1 - z) * u, np.zeros_like(u)], axis=-1)
        x2  = np.stack([-z * u, np.zeros_like(u)], axis=-1)
        x1p = np.stack([(1 - z) * up * np.cos(theta), (1 - z) * up * np.sin(theta)], axis=-1)
        x2p = np.stack([-z * up * np.cos(theta), -z * up * np.sin(theta)], axis=-1)

        if largeNc:
            return self.LNc_quadrupole(x1, x2, x2p, x1p, dipole_args)
        return self.FNc_quadrupole(x1, x2, x2p, x1p, dipole_args)

