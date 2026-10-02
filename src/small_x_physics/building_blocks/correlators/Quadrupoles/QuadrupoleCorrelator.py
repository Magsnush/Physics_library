### Contains a gaussian approximation to the quadrupole in the MV model ###

import numpy as np

# Added to every dipole before its log is taken, so that log(S) stays finite.
_LOG_FLOOR = 1e-14


class QuadrupoleCorrelatorModel:
    def __init__(self, dipole_model, Nc, LambdaQCD):
        self.Nc = Nc
        self.CF = (self.Nc**2 - 1) / (2 * self.Nc)
        self.LambdaQCD = LambdaQCD
        self.dipole = dipole_model

    def log_dipole(self, x, y, dipole_args):
        """Compute the logarithm of the dipole S-matrix, ensuring numerical stability."""
        S2 = self.dipole(x, y, **dipole_args) + _LOG_FLOOR  # Avoid log(0)
        return np.log(S2)

    # Functions that appear in quadrupole in terms of dipole exponential
    def F(self, x1,x2,x2p,x1p, dipole_args):
        """See below eq. B13 of https://journals.aps.org/prd/pdf/10.1103/PhysRevD.83.105005 for the definition of F."""
        return (1/self.CF)*(self.log_dipole(x1,x2p, dipole_args) + self.log_dipole(x2,x1p, dipole_args) - self.log_dipole(x1,x1p, dipole_args) - self.log_dipole(x2,x2p, dipole_args))              # <----- THIS IS OK

    # ------------------------------------------------------------------
    # The six pair dipoles
    # ------------------------------------------------------------------
    #
    # Four positions make six distinct pairs, and every term of the Gaussian
    # quadrupole is built from the dipoles of those six. Spelled out term by term
    # the finite-Nc expression evaluates the dipole 14 times; evaluating each pair
    # once and sharing it gives the same numbers from six evaluations. For a
    # BK-evolved dipole that is most of the cost of the finite-energy integrands.
    #
    # Keys name the pair: "12" is (x1, x2), "1p2p" is (x1', x2'), "12p" is
    # (x1, x2'), and so on.

    def pair_dipoles(self, x1, x2, x2p, x1p, dipole_args):
        """Dipole S-matrix of each of the six distinct pairs of the four positions."""
        return {
            "12": self.dipole(x1, x2, **dipole_args),
            "1p2p": self.dipole(x2p, x1p, **dipole_args),
            "11p": self.dipole(x1, x1p, **dipole_args),
            "22p": self.dipole(x2, x2p, **dipole_args),
            "12p": self.dipole(x1, x2p, **dipole_args),
            "21p": self.dipole(x2, x1p, **dipole_args),
        }

    @staticmethod
    def _logs(S):
        return {pair: np.log(value + _LOG_FLOOR) for pair, value in S.items()}

    def _F_terms(self, L):
        """F1, F2, F3 of Dominguez et al. (below eq. B13) from the six log-dipoles.

        Each is exactly F(...) above with the argument orders FNc_quadrupole used,
        summed in the same order:
            F1 = F(x1, x2', x2, x1'),  F2 = F(x1, x2, x2', x1'),  F3 = F(x1, x1', x2', x2).
        """
        F1 = (1/self.CF)*(L["12"] + L["1p2p"] - L["11p"] - L["22p"])
        F2 = (1/self.CF)*(L["12p"] + L["21p"] - L["11p"] - L["22p"])
        F3 = (1/self.CF)*(L["12p"] + L["21p"] - L["12"] - L["1p2p"])
        return F1, F2, F3

    def _FNc_from_logs(self, L):
        """Finite-Nc quadrupole, Dominguez et al. (2011) eq. B21, from the log-dipoles."""
        F1, F2, F3 = self._F_terms(L)

        # Shared dipole S(r) factors that enter into expression B21 that forms the gaussian quadrupole.
        SuSup = np.exp(L["12"] + L["1p2p"])

        # Discriminant (avoid tiny negative numerical noise)
        Delta = F1**2 + (4 / self.Nc**2) * F2 * F3
        sqrt_Delta = np.sqrt(Delta)

        # Avoid 0/0
        good = sqrt_Delta >0
        term1 = np.zeros_like(sqrt_Delta)
        term2 = np.zeros_like(sqrt_Delta)

        # Compute the two terms only where valid. A 1/Nc**2 factor can be introduced here if one is thinking of a dipole-dipole correlator as (eq. B21)
        term1[good] = ((sqrt_Delta[good] + F1[good]) / (2 * sqrt_Delta[good]) - F2[good] / sqrt_Delta[good])* np.exp(self.Nc * sqrt_Delta[good] / 4)
        term2[good] = ((sqrt_Delta[good] - F1[good]) / (2 * sqrt_Delta[good]) + F2[good] / sqrt_Delta[good])* np.exp(-self.Nc * sqrt_Delta[good] / 4)

        BigFactor = term1 + term2

        # At Delta = 0 both terms are 0/0. Their sum is
        # cosh(a) + (F1/2 - F2) (Nc/2) sinh(a)/a with a = Nc sqrt(Delta)/4, which
        # tends to 1 + Nc (F1/2 - F2)/2. That happens when all six dipoles are
        # equal, e.g. for a fully transparent target (S = 1 for every pair), where
        # the quadrupole must reduce to S(u) S(u') and not to 0.
        degenerate = ~good
        BigFactor[degenerate] = 1 + self.Nc * (F1[degenerate] / 2 - F2[degenerate]) / 2

        # Final finite-Nc quadrupole expression
        return SuSup * BigFactor * np.exp((-self.Nc/4)*F1 + (1/(2*self.Nc))*F2)

    def _LNc_from_logs(self, L):
        """Large-Nc limit of the same expression, from the log-dipoles."""
        F1, F2, _ = self._F_terms(L)

        # Shared dipole S(r) factors
        SuSup = np.exp(L["12"] + L["1p2p"])
        Su_mixed = np.exp(L["11p"] + L["22p"])

        return SuSup - (F2 / (F1 + 1e-12)) * (SuSup - Su_mixed)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def FNc_quadrupole(self, x1, x2, x2p, x1p, dipole_args):
        """Compute the finite-Nc quadrupole correlator using the formula from Dominguez et al. (2011) https://journals.aps.org/prd/pdf/10.1103/PhysRevD.83.105005, eq. B21."""
        return self._FNc_from_logs(self._logs(self.pair_dipoles(x1, x2, x2p, x1p, dipole_args)))

    def LNc_quadrupole(self, x1, x2, x2p, x1p, dipole_args):
        """Compute the large-Nc limit of the same Dominguez et al. expression.

        This takes dipole_args like FNc_quadrupole does, so it works with a
        rapidity-dependent dipole (e.g. BKDipole.BK_evolved_MV_model_S2_Y) and
        not only with an analytic one.
        """
        return self._LNc_from_logs(self._logs(self.pair_dipoles(x1, x2, x2p, x1p, dipole_args)))

    @staticmethod
    def positions_polar(u, up, z, theta):
        """Quark and antiquark positions of amplitude and conjugate, photon at the origin."""
        x1  = np.stack([(1 - z) * u, np.zeros_like(u)], axis=-1)
        x2  = np.stack([-z * u, np.zeros_like(u)], axis=-1)
        x1p = np.stack([(1 - z) * up * np.cos(theta), (1 - z) * up * np.sin(theta)], axis=-1)
        x2p = np.stack([-z * up * np.cos(theta), -z * up * np.sin(theta)], axis=-1)
        return x1, x2, x2p, x1p

    def polar_correlators(self, u, up, z, theta, dipole_args=None, largeNc=False):
        """S(u), S(u') and the quadrupole S4, together, from six dipole evaluations.

        The finite-energy target amplitude is 1 - S(u) - S(u') + S4. The two
        dipoles are the pairs (x1, x2) and (x1', x2') of the quadrupole, so they
        come out of the same six evaluations rather than costing two more.

        Returns
        -------
        S_u, S_up, S4 : arrays
        """
        if dipole_args is None:
            dipole_args = {}
        S = self.pair_dipoles(*self.positions_polar(u, up, z, theta), dipole_args)
        L = self._logs(S)
        S4 = self._LNc_from_logs(L) if largeNc else self._FNc_from_logs(L)
        return S["12"], S["1p2p"], S4

    def quadrupole_polar(self, u, up, z, theta, dipole_args = None, largeNc = False):
        """Quadrupole in polar integration variables.

        Parameters
        ----------
        largeNc : bool, default False
            If True use the large-Nc expression, otherwise the finite-Nc one.
        """
        return self.polar_correlators(u, up, z, theta, dipole_args, largeNc)[2]
