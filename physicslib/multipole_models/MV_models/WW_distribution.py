import numpy as np
from pathlib import Path
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.special import jv

from physicslib.constants import Nc

# In the MV-model, we have (eq. (5) in https://arxiv.org/pdf/1101.0715)

class WWDistribution:
    """
    Weizsäcker–Williams gluon distribution
    with radial integration + cached interpolation.
    """

    def __init__(
        self,
        dipole_1d,
        sigma0,
        alphaS,
        r_max,
        K_min=0.0,
        K_max=10.0,
        nK=200,
        cache_dir="ww_cache",
    ):
        """
        Parameters
        ----------
        dipole_1d : callable
            Function S(r) for the dipole.
        sigma0 : float
            Dipole cross section normalization.
        alphaS : float
            Strong coupling.
        r_max : float
            Upper cutoff of radial integral.
        K_min, K_max : float
            Momentum range.
        nK : int
            Number of grid points in K.
        cache_dir : str
            Directory for cached grids.
        """
        self.dipole_1d = dipole_1d
        self.sigma0 = sigma0
        self.alphaS = alphaS
        self.r_max = r_max
        self.K_min = K_min
        self.K_max = K_max
        self.nK = nK

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        self._build_prefactor()
        self._load_or_precompute()
        self._build_interpolator()

    # ------------------------------------------------------------------
    # Physics pieces
    # ------------------------------------------------------------------

    def _build_prefactor(self):
        S_perp = self.sigma0 / 2.0
        self.prefactor = (
            S_perp / (self.alphaS * np.pi**2)
        ) * (Nc**2 - 1) / Nc

    def _integrand(self, r, K):
        """
        Radial integrand for fixed K.
        """
        jacobian = r / (2.0 * np.pi)**2
        phase = 2.0 * np.pi * jv(0, K * r)
        return jacobian * phase * (1.0 / r**2) * (1.0 - self.dipole_1d(r))

    def _compute_WW_at_K(self, K):
        """
        Compute WW(K) by radial integration.
        """
        val, err = quad(
            self._integrand,
            0.0,
            self.r_max,
            args=(K,),
            limit=200,
        )
        return self.prefactor * val

    # ------------------------------------------------------------------
    # Caching and interpolation
    # ------------------------------------------------------------------

    def _cache_tag(self):
        return (
            f"sigma0={self.sigma0}_alphaS={self.alphaS}_"
            f"rmax={self.r_max}_nK={self.nK}"
        )

    def _load_or_precompute(self):
        tag = self._cache_tag()
        self.K_file = self.cache_dir / f"K_grid_{tag}.npy"
        self.WW_file = self.cache_dir / f"WW_grid_{tag}.npy"

        if self.K_file.exists() and self.WW_file.exists():
            self.K_grid = np.load(self.K_file)
            self.WW_grid = np.load(self.WW_file)
            return

        print("Precomputing WW distribution...")

        self.K_grid = np.linspace(self.K_min, self.K_max, self.nK)
        self.WW_grid = np.array(
            [self._compute_WW_at_K(K) for K in self.K_grid]
        )

        np.save(self.K_file, self.K_grid)
        np.save(self.WW_file, self.WW_grid)

    def _build_interpolator(self):
        self._interp = interp1d(
            self.K_grid,
            self.WW_grid,
            kind="cubic",
            bounds_error=False,
            fill_value=0.0,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(self, K):
        """
        Evaluate WW(K) via interpolation.
        """
        return self._interp(K)
