"""
VEGAS integration utilities for the 5D finite-energy LO DIS integrand.

This module mirrors the older `Integration_functions.py` but is built
directly on top of `LODISIntegrand5D` from
`physicslib.integrands.totalDIS.LO.integrand5D`.
"""

import multiprocessing
import os

import numpy as np
import vegas

from physicslib.integrands.totalDIS.LO.integrand5D import LODISIntegrand5D
from physicslib.wavefunctions.FE_photon_wavefunctions.LO import LO_FE_PhotonWF_squared
from physicslib.multipole_models.MV_models.dipole import Dipole
from physicslib.multipole_models.MV_models.gaussian_quadrupole import GaussianQuadrupole

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


# Defined a quadrupole model that can be used to wrap the quadrupole model and override the z-target if desired to test z->0 limit.
class QuadrupolePolarWrapper:
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

    # MV dipole parameters (taken to match the rest of the codebase)
    Qs0 = np.sqrt(0.104)
    gamma = 1.0
    ec = 1.0
    dipole_model = Dipole(Qs0=Qs0, gamma=gamma, ec=ec)

    # If BK provider is supplied, wrap S_xy to use BK-evolved dipole
    if bk_provider is not None:
        if bk_Y is None:
            raise ValueError("bk_Y (rapidity) required when using BK provider")
        
        # Use pickleable wrapper for multiprocessing
        dipole_model.S_xy = BKDipoleWrapper(dipole_model.S_xy, bk_provider, bk_Y)
        dipole_model.S_r = BKDipoleWrapper(dipole_model.S_r, bk_provider, bk_Y)

    quad_model = GaussianQuadrupole(dipole_model)

    # sigma0 as used in the rest of the project
    sigma0 = 2.57 * 2 * 18.81

    quadrupole_polar = QuadrupolePolarWrapper(
        quad_model=quad_model,
        largeNc=largeNc,
        z_override=z_target_override,
    )

    return LODISIntegrand5D(
        quark_masses=quark_masses,
        photon_wf=photon_wf,
        sigma0=sigma0,
        dipole_model=dipole_model,
        quadrupole_model=quadrupole_polar,
        polarization=polarization,
        largeNc=largeNc,
    )


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

        Accepts either a single point of shape (4,) and returns a scalar,
        or a batch of points of shape (N,4) and returns a 1D array of length N.
        This delegates to the existing `LODISIntegrand4D.FE_integrand`, which
        is written with numpy/scipy and therefore supports array inputs.
        """
        arr = np.asarray(x)

        # Normalize input to shape (N,5) where last axis are coordinates
        if arr.ndim == 1:
            if arr.size != 5:
                raise ValueError(f"Expected 5 coordinates, got shape {arr.shape}")
            # single point -> pass scalars through
            u, up, z, theta, Msq_qq = arr
            Q = getattr(self.integrand.wf, "Q", None)
            return self.integrand.FE_integrand(Q, Msq_qq, u, up, z, theta, flavor=self.flavor)

        # If last axis is 5, good: shape (N,5)
        if arr.shape[-1] == 5:
            batch = arr.reshape(-1, 5)
        # If first axis is 5, assume shape (5,N) and transpose
        elif arr.shape[0] == 5:
            batch = arr.T.reshape(-1, 5)
        else:
            # Try to coerce into (-1,4)
            try:
                batch = arr.reshape(-1, 5)
            except Exception:
                raise ValueError(f"Cannot interpret VEGAS batch shape {arr.shape} as points of length 4")

        u = batch[:, 0]
        up = batch[:, 1]
        z = batch[:, 2]
        theta = batch[:, 3]
        Msq_qq = batch[:, 4]

        # FE_integrand expects Q explicitly; use the stored Q from construction.
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
    Perform a 5D VEGAS integration of the finite-energy DIS integrand.

    Parameters
    ----------
    Q : float
        Photon virtuality (GeV).
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
    # Determine number of CPU cores to use. Prefer SLURM setting when present,
    # otherwise use all local CPUs. Allow override via n_cores argument.
    if n_cores is None:
        n_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count()))

    # VEGAS tuning via environment variables for quick experiments
    warm_nitn = int(os.environ.get("VEGAS_WARM_NITN", "10"))
    full_nitn = int(os.environ.get("VEGAS_FULL_NITN", "20"))
    min_neval_batch = int(os.environ.get("VEGAS_MIN_NEVAL_BATCH", "50000"))

    lodis5d = _build_lodis5d(
        Q, m, Zf, 
        polarization=polarization, 
        largeNc=largeNc, 
        z_target_override=z_target_override,
        bk_provider=bk_provider,
        bk_Y=bk_Y,
    )

    # Use the batched/vectorized integrand that delegates to FE_integrand
    batched_integrand = LODIS5D_VegasIntegrand(Q, lodis5d, flavor=0)

    # Choose a sensible number of worker processes so that each worker receives
    # reasonably large batches (avoids high IPC/process overhead).
    sensible_nproc = min(n_cores, max(1, int(mcpoints // min_neval_batch)))

    integ = vegas.Integrator(
        [[umin, umax], [upmin, upmax], [zmin, zmax], [thetamin, thetamax], [Msq_qqtilde_min, Msq_qqtilde_max]],
        nproc=sensible_nproc,
    )

    neval_warm = max(int(mcpoints // 10), min_neval_batch)
    warm = dict(nitn=warm_nitn, neval=neval_warm, min_neval_batch=min_neval_batch)
    full = dict(nitn=full_nitn, neval=int(mcpoints), min_neval_batch=min_neval_batch)

    # Helpful runtime info for tuning
    # print(
    #     f"[VEGAS] n_cores={n_cores}, nproc={sensible_nproc}, mcpoints={mcpoints}, "
    #     f"warm_nitn={warm_nitn}, full_nitn={full_nitn}, min_neval_batch={min_neval_batch}"
    # )

    # Warm-up adaptation
    integ(batched_integrand, **warm)

    # Full integration
    result = integ(batched_integrand, **full)

    # Extract mean and standard deviation from VEGAS result
    # Vegas with nproc > 0 returns RAvg with .mean and .sdev attributes as floats
    # But sometimes with array-like results, it returns RAvgArray instead
    
    if hasattr(result, '__len__') and len(result) > 0:
        # It's an array result; extract first element
        mean_val = result[0].mean
        sdev_val = result[0].sdev
    else:
        # Scalar result
        mean_val = result.mean
        sdev_val = result.sdev
    
    return mean_val, sdev_val

# To test z->0 limit, we can override the z-target by passing in a value for z_target_override. It should be a float value i.e 0.0
def compute_cross_section_5D(
    Q,
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
    Convenience wrapper for 5D finite-energy DIS integration.
    
    Supports both analytic MV dipole and BK-evolved dipole.
    
    Parameters
    ----------
    ... (see run_vegas_integral_5D for details)
    bk_provider : RCBKData instance, optional
        BK-evolved dipole provider. If None, uses analytic MV dipole.
    bk_Y : float, optional
        Rapidity for BK evaluation (required if bk_provider is provided).
    """
    return run_vegas_integral_5D(
        Q=Q,
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
