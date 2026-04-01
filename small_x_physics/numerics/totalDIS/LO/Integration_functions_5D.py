"""
VEGAS integration utilities for the 5D finite-energy LO DIS integrand.

This module mirrors the older `Integration_functions.py` but is built
directly on top of `LODISIntegrand5D` from
`small_x_physics.integrands.totalDIS.LO.integrand5D`.
"""

import multiprocessing
import os

import numpy as np
import vegas

from small_x_physics.integrands.totalDIS.LO.integrand5D import LODISIntegrand5D
from small_x_physics.wavefunctions.FE_photon_wavefunctions.LO import LO_FE_PhotonWF_squared
from small_x_physics.multipole_models.MV_models.dipole import Dipole
from small_x_physics.multipole_models.MV_models.gaussian_quadrupole import GaussianQuadrupole

from typing import Optional


class BKDipoleWrapper:
    """Wrapper to make BK-evolved dipole pickleable for multiprocessing.
    
    This class wraps the original S_xy and S_r methods to inject BK provider
    and rapidity, while being serializable for multiprocessing.
    """
    def __init__(self, original_method, bk_provider, bk_Y):
        self.original_method = original_method
        self.bk_provider = bk_provider
        self.bk_Y = bk_Y
    
    def __call__(self, *args, **kwargs):
        # Only inject bk and Y if not already present
        if 'bk' not in kwargs:
            kwargs['bk'] = self.bk_provider
        if 'Y' not in kwargs:
            kwargs['Y'] = self.bk_Y
        return self.original_method(*args, **kwargs)




class QuadrupolePolarWrapper:
    """Wraps quadrupole model and optionally overrides z-target for testing z→0 limit."""
    def __init__(self, quad_model, largeNc: bool, z_override=None):
        self.quad_model = quad_model
        self.largeNc = largeNc
        self.z_override = z_override

    def __call__(self, u, up, z, theta):
        z_for_target = (
            self.z_override
            if self.z_override is not None
            else z
        )
        return self.quad_model.quadrupole_polar(
            u, up, z_for_target, theta, largeNc=self.largeNc
        )


def _build_lodis5d(
    Q,
    xB,
    m,
    Zf,
    polarization: str = "T",
    largeNc: bool = False,
    z_target_override=None,
    bk_provider=None,
    bk_Y=None,
):
    """
    Helper to construct a LODISIntegrand5D with standard parameters.

    Parameters
    ----------
    Q : float
        Photon virtuality (GeV).
    xB : float
        Bjorken-x value. REQUIRED - enables z-dependent kinematic bounds on Msq_qq
        to enforce physical constraint: Msq_qq <= W²·z·(1-z) where W² = Q²(1/xB - 1).
    m : float
        Quark mass (GeV). Currently treated as flavor-independent.
    Zf : float
        Quark charge factor for this flavor.
    polarization : {"T", "L", "TL"}
        Photon polarization.
    largeNc : bool
        If True, use large-Nc quadrupole approximation.
    bk_provider : RCBKData instance, optional
        If provided, use BK-evolved dipole instead of analytic MV.
        Must be an instance with S(Y, r) method (e.g., from rcbk_adapter).
    bk_Y : float, optional
        Rapidity for BK evaluation. Required if bk_provider is provided.
    """
    quark_masses = np.array([m])
    quark_charges = np.array([Zf])

    photon_wf = LO_FE_PhotonWF_squared(
        quark_masses=quark_masses,
        quark_charges=quark_charges,
    )

    # MV dipole parameters
    Qs0 = np.sqrt(0.104)
    gamma = 1.0
    ec = 1.0
    dipole_model = Dipole(Qs0=Qs0, gamma=gamma, ec=ec)

    # Apply BK-evolved dipole if provider is supplied
    if bk_provider is not None:
        if bk_Y is None:
            raise ValueError("bk_Y (rapidity) required when using BK provider")
        dipole_model.S_xy = BKDipoleWrapper(dipole_model.S_xy, bk_provider, bk_Y)
        dipole_model.S_r = BKDipoleWrapper(dipole_model.S_r, bk_provider, bk_Y)

    quad_model = GaussianQuadrupole(dipole_model)

    sigma0 = 2.57 * 2 * 18.81

    quadrupole_polar = QuadrupolePolarWrapper(
        quad_model=quad_model,
        largeNc=largeNc,
        z_override=z_target_override,
    )

    lodis_integrand = LODISIntegrand5D(
        quark_masses=quark_masses,
        photon_wf=photon_wf,
        sigma0=sigma0,
        dipole_model=dipole_model,
        quadrupole_model=quadrupole_polar,
        polarization=polarization,
        largeNc=largeNc,
        Q=Q,
        xB=xB,
    )
    
    return lodis_integrand


@vegas.rbatchintegrand
class LODIS5D_VegasIntegrand:
    """
    VEGAS-ready wrapper around LODISIntegrand5D.FE_integrand.
    """

    def __init__(self, Q, lodis5d, flavor: int = 0):
        self.Q = Q
        self.integrand = lodis5d
        self.flavor = flavor
    def __call__(self, x):
        """
        Batched/vectorized integrand for VEGAS.

        Accepts either a single point of shape (5,) and returns a scalar,
        or a batch of points of shape (N,5) and returns a 1D array of length N.
        """
        arr = np.asarray(x)

        if arr.ndim == 1:
            if arr.size != 5:
                raise ValueError(f"Expected 5 coordinates, got shape {arr.shape}")
            u, up, z, theta, Msq_qq = arr
            Q = getattr(self.integrand.wf, "Q", None)
            return self.integrand.FE_integrand(Q, Msq_qq, u, up, z, theta, flavor=self.flavor)

        if arr.shape[-1] == 5:
            batch = arr.reshape(-1, 5)
        elif arr.shape[0] == 5:
            batch = arr.T.reshape(-1, 5)
        else:
            try:
                batch = arr.reshape(-1, 5)
            except Exception:
                raise ValueError(f"Cannot interpret VEGAS batch shape {arr.shape} as points of length 5")

        u = batch[:, 0]
        up = batch[:, 1]
        z = batch[:, 2]
        theta = batch[:, 3]
        Msq_qq = batch[:, 4]

        return self.integrand.FE_integrand(self.Q, Msq_qq, u, up, z, theta, flavor=self.flavor)

class WrappedLODIS5DIntegrand:
    def __init__(self, lodis5d, Q):
        self.lodis5d = lodis5d
        self.Q = Q

    def __call__(self, x):
        u, up, z, theta, Msq_qq = x
        return self.lodis5d.FE_integrand(
            self.Q, Msq_qq, u, up, z, theta, flavor=0
        )


def run_vegas_integral_5D(
    Q,
    xB,
    m,
    Zf,
    polarization: str,
    largeNc: bool,
    umin,
    umax,
    upmin,
    upmax,
    zmin,
    zmax,
    thetamin,
    thetamax,
    Msq_qqtilde_min,
    Msq_qqtilde_max,
    mcpoints: int,
    n_cores: Optional[int] = None,
    z_target_override=None,
    bk_provider=None,
    bk_Y=None,
):
    """
    Perform a 5D VEGAS integration of the finite-energy DIS integrand with kinematic bounds.

    Parameters
    ----------
    Q : float
        Photon virtuality (GeV).
    xB : float
        Bjorken-x value. REQUIRED - enables z-dependent kinematic bounds on Msq_qq.
        The integrand enforces: Msq_qq <= W²·z·(1-z) where W² = Q²(1/xB - 1).
        This constraint ensures physical consistency in the finite-energy formulation.
    m : float
        Quark mass (GeV), treated as a single flavor for now.
    Zf : float
        Quark charge factor for this flavor.
    polarization : {"T", "L", "TL"}
    largeNc : bool
    umin, umax : float
        Integration bounds for u.
    upmin, upmax : float
        Integration bounds for up.
    zmin, zmax : float
        Integration bounds for z.
    thetamin, thetamax : float
        Integration bounds for theta (angle between u and up).
    Msq_qq_min, Msq_qq_max : float
        Integration bounds for invariant mass squared.
    mcpoints : int
        Monte Carlo points per iteration.
    n_cores : int or None
        Number of processes for VEGAS; if None, use all available cores.
    z_target_override : float, optional
        Override z value for testing z->0 limit.
    bk_provider : RCBKData instance, optional
        BK-evolved dipole provider. If provided, uses BK-evolved dipole.
    bk_Y : float, optional
        Rapidity for BK evaluation (required if bk_provider is provided).

    Returns
    -------
    (mean, error) : tuple of floats
        VEGAS estimate and its standard deviation.
    """
    if n_cores is None:
        n_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count()))

    warm_nitn = int(os.environ.get("VEGAS_WARM_NITN", "15"))
    full_nitn = int(os.environ.get("VEGAS_FULL_NITN", "40"))
    min_neval_batch = int(os.environ.get("VEGAS_MIN_NEVAL_BATCH", "50000"))

    lodis5d = _build_lodis5d(
        Q, xB, m, Zf,
        polarization=polarization, 
        largeNc=largeNc, 
        z_target_override=z_target_override,
        bk_provider=bk_provider,
        bk_Y=bk_Y,
    )

    batched_integrand = LODIS5D_VegasIntegrand(Q, lodis5d, flavor=0)

    sensible_nproc = min(n_cores, max(1, int(mcpoints // min_neval_batch)))

    integ = vegas.Integrator(
        [[umin, umax], [upmin, upmax], [zmin, zmax], [thetamin, thetamax], [Msq_qqtilde_min, Msq_qqtilde_max]],
        nproc=sensible_nproc,
    )

    neval_warm = max(int(mcpoints // 10), min_neval_batch)
    warm = dict(nitn=warm_nitn, neval=neval_warm, min_neval_batch=min_neval_batch)
    full = dict(nitn=full_nitn, neval=int(mcpoints), min_neval_batch=min_neval_batch)

    integ(batched_integrand, **warm)
    result = integ(batched_integrand, **full)

    if hasattr(result, '__len__') and len(result) > 0:
        mean_val = result[0].mean
        sdev_val = result[0].sdev
    else:
        mean_val = result.mean
        sdev_val = result.sdev
    
    return mean_val, sdev_val


def compute_cross_section_5D(
    Q,
    xB,
    m,
    Zf,
    polarization: str,
    largeNc: bool,
    umin,
    umax,
    upmin,
    upmax,
    zmin,
    zmax,
    thetamin,
    thetamax,
    Msq_qqtilde_min,
    Msq_qqtilde_max,
    mcpoints: int,
    n_cores: Optional[int] = None,
    z_target_override=None,
    bk_provider=None,
    bk_Y=None,
):
    """
    Convenience wrapper for 5D finite-energy DIS integration with kinematic bounds.
    
    Supports both analytic MV dipole and BK-evolved dipole.
    The z-dependent kinematic bounds on Msq_qq are ALWAYS enforced to ensure
    physical consistency: Msq_qq <= W²·z·(1-z) where W² = Q²(1/xB - 1).
    
    Parameters
    ----------
    Q : float
        Photon virtuality (GeV).
    xB : float
        Bjorken-x value. REQUIRED - enforces kinematic bounds on Msq_qq.
    m : float
        Quark mass (GeV).
    Zf : float
        Quark charge factor.
    polarization : str
        Photon polarization ("T", "L", or "TL").
    largeNc : bool
        Whether to use large-Nc approximation.
    umin, umax, upmin, upmax, zmin, zmax, thetamin, thetamax : float
        Integration bounds (see run_vegas_integral_5D).
    Msq_qqtilde_min, Msq_qqtilde_max : float
        Integration bounds for invariant mass squared.
    mcpoints : int
        Monte Carlo points per integration iteration.
    n_cores : int, optional
        Number of cores for VEGAS.
    z_target_override : float, optional
        Override z value for testing z->0 limit.
    bk_provider : RCBKData instance, optional
        BK-evolved dipole provider. If None, uses analytic MV dipole.
    bk_Y : float, optional
        Rapidity for BK evaluation (required if bk_provider is provided).

    Returns
    -------
    (mean, error) : tuple of floats
        Cross section estimate and its standard deviation.
    """
    return run_vegas_integral_5D(
        Q=Q,
        xB=xB,
        m=m,
        Zf=Zf,
        polarization=polarization,
        largeNc=largeNc,
        umin=umin,
        umax=umax,
        upmin=upmin,
        upmax=upmax,
        zmin=zmin,
        zmax=zmax,
        thetamin=thetamin,
        thetamax=thetamax,
        Msq_qqtilde_min=Msq_qqtilde_min,
        Msq_qqtilde_max=Msq_qqtilde_max,
        mcpoints=mcpoints,
        n_cores=n_cores,
        z_target_override=z_target_override,
        bk_provider=bk_provider,
        bk_Y=bk_Y,
    )
