"""
VEGAS integration utilities for the 4D finite-energy LO DIS integrand.

This module mirrors the older `Integration_functions.py` but is built
directly on top of `LODISIntegrand4D` from
`physicslib.integrands.totalDIS.LO.integrand4D`.
"""

import multiprocessing
import os

import numpy as np
import vegas

from physicslib.integrands.totalDIS.LO.integrand4D import LODISIntegrand4D
from physicslib.wavefunctions.FE_photon_wavefunctions.LO import LO_FE_PhotonWF_squared
from physicslib.multipole_models.MV_models.dipole import Dipole
from physicslib.multipole_models.MV_models.gaussian_quadrupole import GaussianQuadrupole

from typing import Optional

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


def _build_lodis4d(
    Q,
    m,
    Zf,
    polarization: str = "T",
    largeNc: bool = False,
    z_target_override=None,
):
    """
    Helper to construct a LODISIntegrand4D with standard parameters.

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

    quad_model = GaussianQuadrupole(dipole_model)

    # sigma0 as used in the rest of the project
    sigma0 = 2.57 * 2 * 18.81

    quadrupole_polar = QuadrupolePolarWrapper(
    quad_model=quad_model,
    largeNc=largeNc,
    z_override=z_target_override,
)


    return LODISIntegrand4D(
        quark_masses=quark_masses,
        photon_wf=photon_wf,
        sigma0=sigma0,
        dipole_model=dipole_model,
        quadrupole_model=quadrupole_polar,
        polarization=polarization,
        largeNc=largeNc,
    )


@vegas.rbatchintegrand
class LODIS4D_VegasIntegrand:
    """
    VEGAS-ready wrapper around LODISIntegrand4D.FE_integrand.
    """

    def __init__(self, Q, Msq, lodis4d, flavor: int = 0):
        self.Q = Q
        self.Msq = Msq
        self.integrand = lodis4d
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

        # Normalize input to shape (N,4) where last axis are coordinates
        if arr.ndim == 1:
            if arr.size != 4:
                raise ValueError(f"Expected 4 coordinates, got shape {arr.shape}")
            # single point -> pass scalars through
            u, up, z, theta = arr
            Q = getattr(self.integrand.wf, "Q", None)
            return self.integrand.FE_integrand(Q, self.Msq, u, up, z, theta, flavor=self.flavor)

        # If last axis is 4, good: shape (N,4)
        if arr.shape[-1] == 4:
            batch = arr.reshape(-1, 4)
        # If first axis is 4, assume shape (4,N) and transpose
        elif arr.shape[0] == 4:
            batch = arr.T.reshape(-1, 4)
        else:
            # Try to coerce into (-1,4)
            try:
                batch = arr.reshape(-1, 4)
            except Exception:
                raise ValueError(f"Cannot interpret VEGAS batch shape {arr.shape} as points of length 4")

        u = batch[:, 0]
        up = batch[:, 1]
        z = batch[:, 2]
        theta = batch[:, 3]

        # FE_integrand expects Q explicitly; use the stored Q from construction.
        return self.integrand.FE_integrand(self.Q, self.Msq, u, up, z, theta, flavor=self.flavor)

class WrappedLODIS4DIntegrand:
    def __init__(self, lodis4d, Q, Msq):
        self.lodis4d = lodis4d
        self.Q = Q
        self.Msq = Msq

    def __call__(self, x):
        u, up, z, theta = x
        return self.lodis4d.FE_integrand(
            self.Q, self.Msq, u, up, z, theta, flavor=0
        )




def run_vegas_integral_4D(
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
    mcpoints: int,
    n_cores: Optional[int] = None,
    z_target_override=None,
):
    """
    Perform a 4D VEGAS integration of the finite-energy DIS integrand.

    Parameters
    ----------
    Q : float
        Photon virtuality (GeV).
    xB : float
        Bjorken-x.
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
    mcpoints : int
        Monte Carlo points per iteration.
    n_cores : int or None
        Number of processes for VEGAS; if None, use all available cores.

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
    warm_nitn = int(os.environ.get("VEGAS_WARM_NITN", "5"))
    full_nitn = int(os.environ.get("VEGAS_FULL_NITN", "10"))
    min_neval_batch = int(os.environ.get("VEGAS_MIN_NEVAL_BATCH", "50000"))

    lodis4d = _build_lodis4d(Q, m, Zf, polarization=polarization, largeNc=largeNc, z_target_override=z_target_override)

    # Kinematic invariant (upper limit on the invariant mass squared)
    Msq = Q**2 * (1.0 / xB - 1.0)

    # Use the batched/vectorized integrand that delegates to FE_integrand
    batched_integrand = LODIS4D_VegasIntegrand(Q, Msq, lodis4d, flavor=0)

    # Choose a sensible number of worker processes so that each worker receives
    # reasonably large batches (avoids high IPC/process overhead).
    sensible_nproc = min(n_cores, max(1, int(mcpoints // min_neval_batch)))

    integ = vegas.Integrator(
        [[umin, umax], [upmin, upmax], [zmin, zmax], [thetamin, thetamax]],
        nproc=sensible_nproc,
    )

    neval_warm = max(int(mcpoints // 10), min_neval_batch)
    warm = dict(nitn=warm_nitn, neval=neval_warm, min_neval_batch=min_neval_batch)
    full = dict(nitn=full_nitn, neval=int(mcpoints), min_neval_batch=min_neval_batch)

    # Helpful runtime info for tuning
    print(
        f"[VEGAS] n_cores={n_cores}, nproc={sensible_nproc}, mcpoints={mcpoints}, "
        f"warm_nitn={warm_nitn}, full_nitn={full_nitn}, min_neval_batch={min_neval_batch}"
    )

    # Warm-up adaptation
    integ(batched_integrand, **warm)

    # Full integration
    result = integ(batched_integrand, **full)

    return result.mean, result.sdev

# To test z->0 limit, we can override the z-target by passing in a value for z_target_override. It should be a float value i.e 0.0
def compute_cross_section_4D(
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
    mcpoints: int,
    n_cores: Optional[int] = None,
    z_target_override=None,
):
    """
    Convenience wrapper that mirrors the API of the old
    `compute_cross_section` but for the 4D FE integrand.
    """
    return run_vegas_integral_4D(
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
        mcpoints=mcpoints,
        n_cores=n_cores,
        z_target_override=z_target_override,
    )

