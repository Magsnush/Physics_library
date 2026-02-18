"""
VEGAS integration utilities for the 4D finite-energy LO DIS integrand.

This module mirrors the older `Integration_functions.py` but is built
directly on top of `LODISIntegrand4D` from
`physicslib.integrands.totalDIS.LO.integrand4D`.
"""

import multiprocessing

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

    def __init__(self, Msq, lodis4d, flavor: int = 0):
        self.Msq = Msq
        self.integrand = lodis4d
        self.flavor = flavor

    def __call__(self, x):
        u, up, z, theta = x
        # Q is stored inside the photon wavefunction via construction.
        # We pass it explicitly to keep the API clear.
        Q = self.integrand.wf.Q if hasattr(self.integrand.wf, "Q") else None
        # In this codebase, Q is a parameter to the FE_integrand, so we
        # keep it as an explicit argument in the run function below.
        raise RuntimeError(
            "LODIS4D_VegasIntegrand should not be called directly. "
            "Use `run_vegas_integral_4D` which binds Q explicitly."
        )

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
    if n_cores is None:
        n_cores = multiprocessing.cpu_count()

    lodis4d = _build_lodis4d(Q, m, Zf, polarization=polarization, largeNc=largeNc, z_target_override=z_target_override)

    # Kinematic invariant (upper limit on the invariant mass squared)
    Msq = Q**2 * (1.0 / xB - 1.0)

    wrapped_integrand = WrappedLODIS4DIntegrand(lodis4d, Q, Msq)


    integ = vegas.Integrator(
        [[umin, umax], [upmin, upmax], [zmin, zmax], [thetamin, thetamax]],
        nproc=n_cores,
    )

    warm = dict(nitn=10, neval=mcpoints // 10, min_neval_batch=10_000)
    full = dict(nitn=20, neval=mcpoints, min_neval_batch=10_000)

    # Warm-up adaptation
    integ(wrapped_integrand, **warm)

    # Full integration
    result = integ(wrapped_integrand, **full)

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

