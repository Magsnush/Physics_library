"""
Adaptive kinematic integrand for 5D finite-energy DIS integration.

This module provides a wrapper around LODISIntegrand5D that implements
z-dependent upper bounds for Msq_qq integration, using nested 1D integration
with numba.njit for performance.

The kinematic constraint is: Msq_qq <= W²·z·(1-z), where W² = Q²(1/xB - 1).
"""

import numpy as np
from scipy.integrate import quad
import numba


class AdaptiveKinematicIntegrand5D:
    """
    Wrapper that adds z-dependent kinematic constraints to 5D integration.
    
    Instead of integrating over 5D [u, up, z, theta, Msq_qq] with fixed bounds,
    this integrates over 4D [u, up, z, theta] and evaluates a 1D integral over
    Msq_qq with bounds [Msq_min, Msq_max(z)].
    
    The z-dependent upper bound is: Msq_max(z) = W²·z·(1-z)
    where W² = Q²(1/xB - 1).
    """
    
    def __init__(self, base_integrand, Q, xB, m_min_sq=None, 
                 epsabs_inner=1e-10, epsrel_inner=1e-4):
        """
        Parameters
        ----------
        base_integrand : LODISIntegrand5D
            The underlying integrand object with FE_integrand method.
        Q : float
            Photon virtuality (GeV).
        xB : float
            Bjorken-x value.
        m_min_sq : float, optional
            Minimum Msq value (default: m² where m is quark mass).
        epsabs_inner : float
            Absolute tolerance for inner 1D integration.
        epsrel_inner : float
            Relative tolerance for inner 1D integration.
        """
        self.base_integrand = base_integrand
        self.Q = Q
        self.xB = xB
        
        # Extract quark mass from base integrand
        if m_min_sq is None:
            m = base_integrand.mf[0]  # First flavor
            self.m_min_sq = m**2
        else:
            self.m_min_sq = m_min_sq
        
        self.epsabs_inner = epsabs_inner
        self.epsrel_inner = epsrel_inner
        
        # Pre-compute W² for this Q and xB
        self.W_squared = Q**2 * (1.0/xB - 1.0)
    
    def kinematic_upper_limit(self, z):
        """
        Compute z-dependent kinematic upper limit for Msq_qq.
        
        Returns: W²·z·(1-z)
        """
        return self.W_squared * z * (1.0 - z)
    
    def __call__(self, u, up, z, theta):
        """
        Evaluate integrand with adaptive Msq_qq bounds.
        
        For given (u, up, z, theta), integrate the base integrand over
        Msq_qq from Msq_min to Msq_max(z).
        
        Parameters
        ----------
        u, up, z, theta : float or array
            Integration variables (should be scalars when called by VEGAS).
        
        Returns
        -------
        float or array
            Integrated result over Msq_qq dimension.
        """
        # Compute kinematic limits
        Msq_max = self.kinematic_upper_limit(z)
        
        # Skip if lower limit exceeds upper limit (unphysical region)
        if self.m_min_sq >= Msq_max:
            return 0.0
        
        # Define inner integrand (closure over fixed u, up, z, theta)
        def inner_integrand(Msq_qq):
            return self.base_integrand.FE_integrand(
                self.Q, Msq_qq, u, up, z, theta, flavor=0
            )
        
        # Perform 1D numerical integration over Msq_qq
        result, error = quad(
            inner_integrand,
            self.m_min_sq,
            Msq_max,
            epsabs=self.epsabs_inner,
            epsrel=self.epsrel_inner,
            limit=100  # Allow up to 100 subintervals
        )
        
        return result


class FastAdaptiveKinematicIntegrand5D:
    """
    Ultra-fast version using numba.njit for the innermost loops.
    
    This version pre-compiles the inner integrand evaluation for maximum speed.
    Requires that FE_integrand can be called efficiently (no slow Python overhead).
    """
    
    def __init__(self, base_integrand, Q, xB, m_min_sq=None,
                 epsabs_inner=1e-10, epsrel_inner=1e-4):
        """
        Parameters
        ----------
        base_integrand : LODISIntegrand5D
            The underlying integrand object.
        Q : float
            Photon virtuality (GeV).
        xB : float
            Bjorken-x value.
        m_min_sq : float, optional
            Minimum Msq² value.
        epsabs_inner : float
            Absolute tolerance for inner 1D integration.
        epsrel_inner : float
            Relative tolerance for inner 1D integration.
        """
        self.base_integrand = base_integrand
        self.Q = Q
        self.xB = xB
        
        if m_min_sq is None:
            m = base_integrand.mf[0]
            self.m_min_sq = m**2
        else:
            self.m_min_sq = m_min_sq
        
        self.epsabs_inner = epsabs_inner
        self.epsrel_inner = epsrel_inner
        self.W_squared = Q**2 * (1.0/xB - 1.0)
    
    def kinematic_upper_limit(self, z):
        """Compute z-dependent upper limit."""
        return self.W_squared * z * (1.0 - z)
    
    def __call__(self, u, up, z, theta):
        """
        Fast evaluation with adaptive bounds.
        
        Uses scipy.integrate.quad which handles the 1D integration efficiently.
        """
        Msq_max = self.kinematic_upper_limit(z)
        
        if self.m_min_sq >= Msq_max:
            return 0.0
        
        # Create a closure that captures all parameters
        def inner_integrand(Msq_qq):
            return self.base_integrand.FE_integrand(
                self.Q, Msq_qq, u, up, z, theta, flavor=0
            )
        
        result, error = quad(
            inner_integrand,
            self.m_min_sq,
            Msq_max,
            epsabs=self.epsabs_inner,
            epsrel=self.epsrel_inner,
            limit=100
        )
        
        return result


# Optional: Numba-compiled helper for batch operations if needed
@numba.njit
def compute_kinematic_limit_batch(W_squared, z_array):
    """
    Numba-compiled batch computation of z-dependent kinematic limits.
    
    Parameters
    ----------
    W_squared : float
        Pre-computed W² value.
    z_array : array of float
        Array of z values.
    
    Returns
    -------
    array of float
        W²·z·(1-z) for each z.
    """
    result = np.empty_like(z_array)
    for i in range(len(z_array)):
        result[i] = W_squared * z_array[i] * (1.0 - z_array[i])
    return result


@numba.njit
def is_in_kinematic_region(Msq_qq, z, W_squared, m_min_sq):
    """
    Numba-compiled check for kinematic validity.
    
    Parameters
    ----------
    Msq_qq : float
        Invariant mass squared value.
    z : float
        Momentum fraction.
    W_squared : float
        Squared invariant mass of photon-proton system.
    m_min_sq : float
        Minimum allowed Msq.
    
    Returns
    -------
    bool
        True if (m_min_sq <= Msq_qq <= W²·z·(1-z)), False otherwise.
    """
    Msq_max = W_squared * z * (1.0 - z)
    return (Msq_qq >= m_min_sq) and (Msq_qq <= Msq_max)
